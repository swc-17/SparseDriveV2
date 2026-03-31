import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork as FPN

from navsim.agents.sparsedrive.ops import deformable_format

from .grid_mask import GridMask


class SparseBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.embed_dims = config.d_model
        self.use_grid_mask = config.use_grid_mask
        self.with_img_neck = config.with_img_neck

        if self.use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

        assert config.image_architecture in ["resnet34"], f"Image architecture {config.image_architecture} not supported."
        self.img_backbone = timm.create_model(
            config.image_architecture, pretrained=True, features_only=True,
            pretrained_cfg_overlay=dict(file=config.bkb_path), 
            out_indices=(1, 2, 3, 4)[-config.num_levels:]
        )
        if self.with_img_neck:
            self.img_neck = FPN(
                in_channels_list=[64,128,256,512][-config.num_levels:],
                out_channels=self.embed_dims,
            )
        else:
            self.img_neck = nn.Conv2d(
                512,
                config.d_model,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            )

    def forward(self, img):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        
        if self.use_grid_mask:
            img = self.grid_mask(img)
        feature_maps = self.img_backbone(img)
        if self.with_img_neck:
            feature_dict = {f"feat_{i}": feature_maps[i] for i in range(len(feature_maps))}
            feature_maps = list(self.img_neck(feature_dict).values())
        else:
            feature_maps = [self.img_neck(feature_maps[-1])]

        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )

        return feature_maps

