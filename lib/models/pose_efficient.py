import json
from collections import OrderedDict
import logging
import os

import torch.nn as nn
import torch

from config import MODEL_EXTRAS
import nasnet_function.net_builder as mbuilder
import nasnet_function.modeldef as modeldef

from nasnet_function.net_builder import ConvBNRelu, IRFBlock


logger = logging.getLogger(__name__)


def create_builder(cfg, arch=None):
    extra = MODEL_EXTRAS[cfg.MODEL.NAME]
    bn_type = extra.BN_TYPE
    if bn_type == "gn":
        bn_type = (bn_type, 32)
    factor = extra.SCALE_FACTOR
    if not arch:
        arch = modeldef.MODEL_ARCH[cfg.MODEL.ARCH]
    arch_def = mbuilder.unify_arch_def(arch)
    width_divisor = extra.WIDTH_DIVISOR
    dw_skip_bn = extra.DW_CONV_SKIP_BN
    dw_skip_relu = extra.DW_CONV_SKIP_RELU
    logger.info("Building model with arch {}\n".format(arch))
    
    builder = mbuilder.NetBuilder(
        width_ratio=factor,
        bn_type=bn_type,
        width_divisor=width_divisor,
        dw_skip_bn=dw_skip_bn,
        dw_skip_relu=dw_skip_relu,
    )
    
    return builder, arch_def


class NetTrunk(nn.Module):
    def __init__(
        self, builder, arch_def, dim_in,
    ):
        super(NetTrunk, self).__init__()
        num_stages = mbuilder.get_num_stages(arch_def)
        ret = mbuilder.get_blocks(arch_def, stage_indices=range(num_stages))
        self.stages = builder.add_blocks(ret["stages"])

    # return features for each stage
    def forward(self, x):
        y = self.stages(x)
        return y


def _add_blocks(cfg, arch=None, dim_in=3):
    builder, arch_def = create_builder(cfg, arch)
    builder.last_depth = dim_in
    blocks = NetTrunk(builder, arch_def, dim_in)
    blocks.out_channels = builder.last_depth
    return blocks, arch_def


class EfficientPoseNet(nn.Module):
    def __init__(self, cfg, arch=None):
        extra = cfg.MODEL.EXTRA
        super(EfficientPoseNet, self).__init__()
        input_channel = cfg.MODEL.IN_CHANNELS
        self.first = nn.Sequential(
            ConvBNRelu(input_depth=3, output_depth=32, kernel=3, stride=2, pad=3 // 2, no_bias=1, use_relu="relu", bn_type="bn"),
            ConvBNRelu(input_depth=32, output_depth=32, kernel=3, stride=1, no_bias=1, use_relu="relu", bn_type="bn", pad=1, group=32),
            nn.Conv2d(32, input_channel, 1, 1, 0, bias=False), nn.BatchNorm2d(input_channel)
        )
        
        self.blocks, self.block_arch = _add_blocks(cfg, arch, dim_in=input_channel)
        self.inplanes = self.blocks.out_channels
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(
                extra.NUM_DECONV_FILTERS[-1], 
                extra.NUM_DECONV_FILTERS[-1], kernel_size=3, 
                stride=1, padding=1, groups=extra.NUM_DECONV_FILTERS[-1], bias=False),
            
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=extra.NUM_DECONV_FILTERS[-1],
                out_channels=cfg.MODEL.NUM_JOINTS,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
            pretrained_state_dict = torch.load(pretrained)
            self.load_state_dict(pretrained_state_dict)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1)) 
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        x = self.blocks(x)
        x = self.deconv_layers(x)
        x = self.final(x)
        return x


def get_pose_net(cfg, is_train, arch=None):
    model = EfficientPoseNet(cfg, arch)
    '''
    if is_train:
        model.init_weights(pretrained=cfg.MODEL.PRETRAINED)
    '''
    return model



