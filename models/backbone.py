# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
# from util.misc import is_main_process

from torch import nn

from .position_encoding import build_position_encoding
from models.fpn import FPN
from models.transformer import build_transformer
from models.resnet import resnet50, resnet101, resnet152
import models.convnextv2 as convnextv2
from models.convnext import convnext_base
from models.convnext_2 import ConvNeXt
# from .randomizations import StyleRandomization, StyleRandomization_multi_scale, StyleRandomization_multi_scale_3layer
# from timm.models import create_model

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, position_embedding, transformer):
        super(Backbone, self).__init__()
        self.backbone = resnet50(replace_stride_with_dilation=[False, False, False],
                                 # pretrained=False,
                                 pretrained=True,
                                 norm_layer=FrozenBatchNorm2d)

        # self.backbone = resnet101(replace_stride_with_dilation=[False, False, False],
        #                          # pretrained=False,
        #                          pretrained=True,
        #                          norm_layer=FrozenBatchNorm2d)

        # self.backbone = resnet152(replace_stride_with_dilation=[False, False, False],
        #                          # pretrained=False,
        #                          pretrained=True,
        #                          norm_layer=FrozenBatchNorm2d)

        # self.backbone = convnextv2.__dict__["convnextv2_base"](
        #     num_classes=5,
        #     drop_path_rate=0.1,
        #     head_init_scale=0.001,
        # )

        self.neck = FPN(in_channels=[64, 256, 512, 1024, 2048],
                        out_channels=256,
                        num_outs=6,
                        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'))

        # self.backbone = convnext_base(pretrained=True,
        #                               drop_path_rate=0.5,
        #                               layer_scale_init_value=1.0,
        #                               )
        # self.backbone = ConvNeXt(depths=[3, 3, 9, 3],
        #                          dims=[128, 256, 512, 1024],
        #                          drop_path_rate=0.7,
        #                          layer_scale_init_value=1.0,)
        # self.url = r"/media/zhouzhihao/DeepInformatic_dataset/lihansheng/bxr/code/convnext_base_1k_224.pth"
        # self.url = r"https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth"
        # self.backbone.init_weights(self.url)

        # self.neck = FPN(
        #                 in_channels=[128, 256, 512, 1024],
        #                 # in_channels=[512, 1024],
        #                 out_channels=256,
        #                 num_outs=6,
        #                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'))

        self.position_embedding = position_embedding
        self.transformer = transformer
        

    def forward(self, images):
        xs = list(self.backbone(images))
        mask = torch.zeros_like(xs[-1][:, 0], dtype=torch.bool)
        pos = self.position_embedding(mask)
        out = self.neck(xs, self.transformer, mask, pos)
        return out


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    transformer = build_transformer(args)
    backbone = Backbone(position_embedding, transformer)
    return backbone
