import torch
from torch.autograd.function import Function, once_differentiable

from . import deformable_aggregation_ext
from . import deformable_aggregation_with_depth_ext


class DeformableAggregationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        # output: [bs, num_pts, num_embeds]
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output = deformable_aggregation_ext.deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        deformable_aggregation_ext.deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )

class DeformableAggregationWithDepthFunction(Function):
    @staticmethod
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
        num_depths,
    ):
        # output: [bs, num_pts, num_embeds]
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output = deformable_aggregation_with_depth_ext.deformable_aggregation_with_depth_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            num_depths,
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx._num_depths = num_depths
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        num_depths = ctx._num_depths
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        deformable_aggregation_with_depth_ext.deformable_aggregation_with_depth_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            num_depths,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
            None,
        )


def deformable_aggregation_func(
    mc_ms_feat,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
    depth_prob=None,
    depth=None
):
    if depth_prob is not None and depth is not None:
        mc_ms_feat = torch.cat([mc_ms_feat, depth_prob], dim=-1)
        sampling_location = torch.cat([sampling_location, depth], dim=-1)
        return DeformableAggregationWithDepthFunction.apply(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            depth_prob.shape[-1],
        )
    else:
        return DeformableAggregationFunction.apply(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )

def feature_maps_format(feature_maps, inverse=False):
    if inverse:
        col_feats, spatial_shape, scale_start_index = feature_maps
        num_cams, num_levels = spatial_shape.shape[:2]

        split_size = spatial_shape[..., 0] * spatial_shape[..., 1]
        split_size = split_size.cpu().numpy().tolist()

        idx = 0
        cam_split = [1]
        cam_split_size = [sum(split_size[0])]
        for i in range(num_cams - 1):
            if not torch.all(spatial_shape[i] == spatial_shape[i + 1]):
                cam_split.append(0)
                cam_split_size.append(0)
            cam_split[-1] += 1
            cam_split_size[-1] += sum(split_size[i + 1])
        mc_feat = [
            x.unflatten(1, (cam_split[i], -1))
            for i, x in enumerate(col_feats.split(cam_split_size, dim=1))
        ]

        spatial_shape = spatial_shape.cpu().numpy().tolist()
        mc_ms_feat = []
        shape_index = 0
        for i, feat in enumerate(mc_feat):
            feat = list(feat.split(split_size[shape_index], dim=2))
            for j, f in enumerate(feat):
                feat[j] = f.unflatten(2, spatial_shape[shape_index][j])
                feat[j] = feat[j].permute(0, 1, 4, 2, 3)
            mc_ms_feat.append(feat)
            shape_index += cam_split[i]
        return mc_ms_feat

    if isinstance(feature_maps[0], (list, tuple)):
        formated = [feature_maps_format(x) for x in feature_maps]
        col_feats = torch.cat([x[0] for x in formated], dim=1)
        spatial_shape = torch.cat([x[1] for x in formated], dim=0)
        scale_start_index = torch.cat([x[2] for x in formated], dim=0)
        return [col_feats, spatial_shape, scale_start_index]

    bs, num_cams = feature_maps[0].shape[:2]
    spatial_shape = []

    col_feats = []
    for i, feat in enumerate(feature_maps):
        spatial_shape.append(feat.shape[-2:])
        col_feats.append(
            torch.reshape(feat, (bs, num_cams, feat.shape[2], -1))
        )

    col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
    spatial_shape = [spatial_shape] * num_cams
    spatial_shape = torch.tensor(
        spatial_shape,
        dtype=torch.int64,
        device=col_feats.device,
    )
    scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
    scale_start_index = scale_start_index.flatten().cumsum(dim=0)
    scale_start_index = torch.cat(
        [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
    )
    scale_start_index = scale_start_index.reshape(num_cams, -1)

    feature_maps = [
        col_feats,
        spatial_shape,
        scale_start_index,
    ]
    return feature_maps


def deformable_format(
    feature_maps,
    spatial_shapes=None,
    level_start_index=None,
    flat_batch=False,
    batch_size=None,
):
    if spatial_shapes is None:
        if flat_batch and feature_maps[0].dim() > 4:
            feature_maps = [x.flatten(end_dim=-4) for x in feature_maps]
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(feature_maps):
            spatial_shape = torch._shape_as_tensor(feat)[-2:].to(feat.device)
            feat = feat.flatten(start_dim=-2).transpose(-1, -2)
            feat_flatten.append(feat)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, -2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),  # (num_level)
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        return feat_flatten, spatial_shapes, level_start_index
    else:
        split_size = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        feature_maps = feature_maps.transpose(-1, -2)
        feature_maps = list(torch.split(feature_maps, split_size, dim=-1))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = feature_maps[i].unflatten(
                -1, (spatial_shapes[i, 0], spatial_shapes[i, 1])
            )
            if batch_size is not None:
                if isinstance(batch_size, int):
                    feature_maps[i] = feature_maps[i].unflatten(
                        0, (batch_size, -1)
                    )
                else:
                    feature_maps[i] = feature_maps[i].unflatten(
                        0, batch_size + (-1,)
                    )
        return feature_maps