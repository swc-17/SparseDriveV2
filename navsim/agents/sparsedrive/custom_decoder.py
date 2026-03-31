import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.sparsedrive.ops import deformable_format
from navsim.common.dataclasses import Trajectory

from .blocks import DeformableFeatureAggregation
from .scorer.get_pdm_score_v1 import get_pdm_score_para as get_pdm_score_v1
from .scorer.get_pdm_score_v2 import get_pdm_score_para as get_pdm_score_v2


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CustomTransformerDecoder(nn.Module):
    def __init__(self, num_poses, d_model, d_ffn, config):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = nn.ModuleList()
        for i in range(config.decoder_num_layers):
            decoder_layer = CustomTransformerDecoderLayer(
                num_poses=num_poses,
                d_model=d_model,
                d_ffn=d_ffn,
                config=config,
                decoder_idx=i,
            )
            self.layers.append(decoder_layer)
    
    def forward(self, feature, input):
        outputs = {}
        loss_dicts = {}
        for i, mod in enumerate(self.layers):
            feature, output, loss_dict = mod(*feature, *input)
            outputs.update(output)
            loss_dicts.update(loss_dict)
        return outputs, loss_dicts


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, num_poses, d_model, d_ffn, config, decoder_idx):
        super().__init__()
        self._config = config
        self.decoder_idx = decoder_idx
        ## path
        self.p_deform_model = DeformableFeatureAggregation(
            config=config,
            embed_dims=d_model,
            num_groups=8,
            num_levels=self._config.num_levels,
            num_cams=len(config.cams),
            num_pts=self._config.len_path,
            attn_drop=0.0,
            use_deformable_func=True,
            use_camera_embed=True,
            residual_mode="add",
        )
        self.p_attention = nn.MultiheadAttention(
                config.d_model,
                config.num_head,
                dropout=config.dropout,
                batch_first=True,
            )
        self.p_ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ffn),
            nn.ReLU(),
            nn.Linear(config.d_ffn, config.d_model),
        )
        self.p_norm1 = nn.LayerNorm(config.d_model)
        self.p_dropout1 = nn.Dropout(0.1)
        self.p_norm2 = nn.LayerNorm(config.d_model)
        self.p_dropout2 = nn.Dropout(0.1)
        self.path_mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, 1),
        )

        ## vel
        self.v_img_attention = nn.MultiheadAttention(
            config.d_model,
            config.num_head,
            dropout=config.dropout,
            batch_first=True,
        )
        self.v_attention = nn.MultiheadAttention(
                config.d_model,
                config.num_head,
                dropout=config.dropout,
                batch_first=True,
            )
        self.v_ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ffn),
            nn.ReLU(),
            nn.Linear(config.d_ffn, config.d_model),
        )
        self.v_norm1 = nn.LayerNorm(config.d_model)
        self.v_dropout1 = nn.Dropout(0.1)
        self.v_norm2 = nn.LayerNorm(config.d_model)
        self.v_dropout2 = nn.Dropout(0.1)
        self.vel_mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, 1),
        )

        ## traj
        if self.decoder_idx == self._config.decoder_num_layers - 1:
            self.t_deform_model = DeformableFeatureAggregation(
                config=config,
                embed_dims=d_model,
                num_groups=8,
                num_levels=self._config.num_levels,
                num_cams=len(config.cams),
                num_pts=num_poses,
                attn_drop=0.0,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode="add",
            )
            self.t_attention = nn.MultiheadAttention(
                    config.d_model,
                    config.num_head,
                    dropout=config.dropout,
                    batch_first=True,
                )
            self.t_ffn = nn.Sequential(
                nn.Linear(config.d_model, config.d_ffn),
                nn.ReLU(),
                nn.Linear(config.d_ffn, config.d_model),
            )
            self.t_norm1 = nn.LayerNorm(config.d_model)
            self.t_dropout1 = nn.Dropout(0.1)
            self.t_norm2 = nn.LayerNorm(config.d_model)
            self.t_dropout2 = nn.Dropout(0.1)
            self.traj_mlp = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, 1),
            )
            self.metric_heads = nn.ModuleDict()
            for metric in self._config.metrics:
                self.metric_heads[metric] = nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                )

    def forward(self, path_embed, vel_embed, path_vocab, vel_vocab, traj_vocab, traj_mask,
                camera_feature, status_encoding, targets,
    ):
        num_path = path_embed.shape[1]
        num_vel = vel_embed.shape[1]

        img_value = camera_feature["feature_maps"][-1].permute(0, 1, 3, 4, 2).flatten(1, 3)
        deform_value = deformable_format(camera_feature["feature_maps"])

        ## ego status
        path_embed = path_embed + status_encoding.unsqueeze(1)
        vel_embed = vel_embed + status_encoding.unsqueeze(1)

        ## path
        path_vocab_flat = path_vocab[..., :2].flatten(-2)
        path_embed = self.p_deform_model(
            path_embed,
            path_vocab_flat,
            None,
            deform_value,
            camera_feature,
            None,
        )
        path_embed = path_embed + self.p_dropout1(self.p_attention(path_embed, path_embed, path_embed)[0])
        path_embed = self.p_norm1(path_embed)
        path_embed = path_embed + self.p_dropout2(self.p_ffn(path_embed))
        path_embed = self.p_norm2(path_embed)
        path_scores = self.path_mlp(path_embed).squeeze(-1)

        ## velocity
        vel_embed = vel_embed + self.v_img_attention(vel_embed, img_value, img_value)[0]
        vel_embed = vel_embed + self.v_dropout1(self.v_attention(vel_embed, vel_embed, vel_embed)[0])
        vel_embed = self.v_norm1(vel_embed)
        vel_embed = vel_embed + self.v_dropout2(self.v_ffn(vel_embed))
        vel_embed = self.v_norm2(vel_embed)
        vel_scores = self.vel_mlp(vel_embed).squeeze(-1)

        ## corase filter
        filter_traj_vocab = traj_vocab.clone()
        filter_traj_mask = traj_mask.clone()

        if num_path > self._config.path_filter_num[self.decoder_idx]:
            topk_path_scores, topk_path_indices = torch.topk(path_scores, self._config.path_filter_num[self.decoder_idx], dim=1)
            filter_path_embed = torch.gather(path_embed, 1, topk_path_indices.unsqueeze(-1).expand(-1, -1, path_embed.shape[-1]))
            filter_path_vocab = torch.gather(path_vocab, 1, topk_path_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, path_vocab.shape[-2], path_vocab.shape[-1]))
            filter_traj_vocab = torch.gather(filter_traj_vocab, 1, topk_path_indices[:, :, None, None, None].expand(-1, -1, filter_traj_vocab.shape[-3], filter_traj_vocab.shape[-2], filter_traj_vocab.shape[-1]))
            filter_traj_mask = torch.gather(filter_traj_mask, 1, topk_path_indices[:, :, None, None].expand(-1, -1, filter_traj_mask.shape[-2], filter_traj_mask.shape[-1]))
        else:
            filter_path_embed = path_embed
            filter_path_vocab = path_vocab

        if num_vel > self._config.velocity_filter_num[self.decoder_idx]:
            topk_vel_scores, topk_vel_indices = torch.topk(vel_scores, self._config.velocity_filter_num[self.decoder_idx], dim=1)
            filter_vel_embed = torch.gather(vel_embed, 1, topk_vel_indices.unsqueeze(-1).expand(-1, -1, vel_embed.shape[-1]))
            filter_vel_vocab = torch.gather(vel_vocab, 1, topk_vel_indices.unsqueeze(-1).expand(-1, -1, vel_vocab.shape[-1]))
            filter_traj_vocab = torch.gather(filter_traj_vocab, 2, topk_vel_indices[:, None, :, None, None].expand(-1, filter_traj_vocab.shape[-4], -1, filter_traj_vocab.shape[-2], filter_traj_vocab.shape[-1]))
            filter_traj_mask = torch.gather(filter_traj_mask, 2, topk_vel_indices[:, None, :, None].expand(-1, filter_traj_mask.shape[-3], -1, filter_traj_mask.shape[-1]))
        else:
            filter_vel_embed = vel_embed
            filter_vel_vocab = vel_vocab

        ## trajectory reconditioning
        if self.decoder_idx == self._config.decoder_num_layers - 1:
            traj_emed = filter_path_embed.unsqueeze(2) + filter_vel_embed.unsqueeze(1)
            traj_emed = traj_emed.flatten(1, 2)

            filter_traj_vocab_flat = filter_traj_vocab[..., :2].flatten(1, 2).flatten(-2)
            traj_emed = self.t_deform_model(
                traj_emed,
                filter_traj_vocab_flat,
                None,
                deform_value,
                camera_feature,
                None,
            )
            traj_emed = traj_emed + self.t_dropout1(self.t_attention(traj_emed, traj_emed, traj_emed)[0])
            traj_emed = self.t_norm1(traj_emed)
            traj_emed = traj_emed + self.t_dropout2(self.t_ffn(traj_emed))
            traj_emed = self.t_norm2(traj_emed)
            traj_scores = self.traj_mlp(traj_emed).squeeze(-1)
            metric_logit = {}
            for metric in self._config.metrics:
                metric_logit[metric] = self.metric_heads[metric](traj_emed).squeeze(-1)

        loss_dict = {}
        if self.training:
            ## path
            target_path = targets["path"]
            target_path_mask = targets["path_mask"]

            diff = (path_vocab - target_path[:, None])[..., :2]
            dist = diff.pow(2).sum(-1)
            mask = target_path_mask[:, None].float()
            dist = dist * mask

            valid_cnt = mask.sum(-1).clamp(min=1.0)
            dist = dist.sum(-1) / valid_cnt

            dist = dist * self._config.path_sigmas * self._config.len_path
            path_loss = F.cross_entropy(path_scores, (-dist).softmax(1))
            loss_dict[f'path_loss_{self.decoder_idx}'] = path_loss

            ## vel
            target_vel = targets["velocity"]
            dist = (vel_vocab - target_vel[:, None]).abs()
            dist = dist.sum(-1) * self._config.velocity_sigmas
            vel_loss = F.cross_entropy(vel_scores, (-dist).softmax(1))
            loss_dict[f'velocity_loss_{self.decoder_idx}'] = vel_loss

            ## traj
            if self.decoder_idx == self._config.decoder_num_layers - 1:
                ## imi
                target_traj = targets["trajectory"]
                dist = (filter_traj_vocab.flatten(1, 2) - target_traj[:, None])[..., :2] ** 2
                dist = dist.sum((-2, -1)) * self._config.trajectory_sigmas
                traj_loss = F.cross_entropy(traj_scores, (-dist).softmax(1))
                loss_dict[f'traj_loss_{self.decoder_idx}'] = traj_loss

                ## metric
                trajectory = filter_traj_vocab.flatten(1,2)
                pdm_token_paths = []
                for token_path in targets["token_path"]:
                    pdm_token_path = token_path.replace("data_cache_navtrain", f"metric_cache_navtrain{self._config.dataset_version}")
                    pdm_token_path_parts = pdm_token_path.split('/')
                    pdm_token_path_parts.insert(-1, 'unknown')
                    pdm_token_path = '/'.join(pdm_token_path_parts) + "/metric_cache.pkl"
                    pdm_token_paths.append(pdm_token_path)
                if self._config.dataset_version == "v1":
                    sub_scores = get_pdm_score_v1(trajectory, pdm_token_paths)
                elif self._config.dataset_version == "v2":
                    sub_scores = get_pdm_score_v2(trajectory, pdm_token_paths)
                for metric in self._config.metrics:
                    metric_pred = metric_logit[metric]
                    metric_gt = torch.tensor(np.stack([sub_score[metric] for sub_score in sub_scores])).to(metric_pred)
                    metric_gt[metric_gt == 0.5] = 0.0
                    metric_loss = F.binary_cross_entropy_with_logits(metric_pred, metric_gt)
                    loss_dict[f'{metric}_loss_{self.decoder_idx}'] = metric_loss * self._config.metric_loss_weight
        
        output = {}
        if self.decoder_idx == self._config.decoder_num_layers - 1:
            if self._config.dataset_version == "v1":
                scores = (
                    metric_logit["no_at_fault_collisions"].sigmoid() * 
                    metric_logit["drivable_area_compliance"].sigmoid()
                ) * (
                    5 * metric_logit["time_to_collision_within_bound"].sigmoid() +
                    5 * metric_logit["ego_progress"].sigmoid()  +
                    2 * metric_logit["comfort"].sigmoid()
                )
            if self._config.dataset_version == "v2":
                scores = (
                    metric_logit["no_at_fault_collisions"].sigmoid() * 
                    metric_logit["drivable_area_compliance"].sigmoid() *
                    metric_logit["driving_direction_compliance"].sigmoid() *
                    metric_logit["traffic_light_compliance"].sigmoid()
                ) * (
                    5 * metric_logit["time_to_collision_within_bound"].sigmoid() +
                    5 * metric_logit["ego_progress"].sigmoid()  +
                    2 * metric_logit["lane_keeping"].sigmoid() +
                    2 * metric_logit["history_comfort"].sigmoid()
                )

            bs_indices = torch.arange(scores.shape[0], device=scores.device)
            mode_indices = scores.argmax(1)
            trajectory = filter_traj_vocab.flatten(1, 2)[bs_indices, mode_indices] 
            output["trajectory"] = trajectory

        return (filter_path_embed, filter_vel_embed, filter_path_vocab, filter_vel_vocab, filter_traj_vocab, filter_traj_mask), output, loss_dict




