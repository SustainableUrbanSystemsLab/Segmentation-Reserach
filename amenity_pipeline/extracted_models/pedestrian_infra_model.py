import torch
import os
from collections import namedtuple
import torch.nn as nn
import math
import importlib
from torch.utils import data
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as standard_transforms
import transforms as extended_transforms
import numpy as np  
from PIL import Image  
from collections import OrderedDict
import time  
from pedestrian_infra_utils import AttrDict, Norm2d, ResizeX, Upsample, scale_as



def get_seg_model():
    model = HighResolutionNet()
    model.init_weights()
    #for name, param in model.named_parameters():  
    #    print(name, param.shape)  


    return model

def get_trunk(trunk_name, output_stride=8):
    """
    Retrieve the network trunk and channel counts.
    """
    assert output_stride == 8, 'Only stride8 supported right now'

    if trunk_name == 'hrnetv2':
        backbone = get_seg_model()
        high_level_ch = backbone.high_level_ch
        s2_ch = -1
        s4_ch = -1
    else:
        raise 'unknown backbone {}'.format(trunk_name)

    # logger.debug("Trunk: {}".format(trunk_name))
    return backbone, s2_ch, s4_ch, high_level_ch

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            print(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != (num_channels[branch_index] *
                                                 block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Norm2d(num_channels[branch_index] * block.expansion,
                       momentum=0.1),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        Norm2d(num_inchannels[i], momentum=0.1)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                Norm2d(num_outchannels_conv3x3,
                                       momentum=0.1)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                Norm2d(num_outchannels_conv3x3,
                                       momentum=0.1),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=False)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

class HighResolutionNet(nn.Module):

    def __init__(self, **kwargs):
        extra = AttrDict()
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = Norm2d(64, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = Norm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = AttrDict()
        self.stage1_cfg.NUM_MODULES = 1
        self.stage1_cfg.NUM_RANCHES = 1
        self.stage1_cfg.BLOCK  = 'BOTTLENECK'
        self.stage1_cfg.NUM_BLOCKS = [4]
        self.stage1_cfg.NUM_CHANNELS = [64]
        self.stage1_cfg.FUSE_METHOD = 'SUM'
        num_channels = 64
        block = blocks_dict['BOTTLENECK']
        num_blocks = 4
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        #self.stage2_cfg = AttrDict()
        self.stage2_cfg = AttrDict()
        self.stage2_cfg.NUM_MODULES = 1
        self.stage2_cfg.NUM_BRANCHES = 2
        self.stage2_cfg.BLOCK  = 'BASIC'
        self.stage2_cfg.NUM_BLOCKS = [4, 4]
        self.stage2_cfg.NUM_CHANNELS = [48, 96]
        self.stage2_cfg.FUSE_METHOD = 'SUM'
        num_channels = [48, 96]
        block = blocks_dict['BASIC']
        num_channels = [num_channels[i] * block.expansion
                        for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = AttrDict()
        self.stage3_cfg.NUM_MODULES = 4
        self.stage3_cfg.NUM_BRANCHES = 3
        self.stage3_cfg.BLOCK  = 'BASIC'
        self.stage3_cfg.NUM_BLOCKS = [4, 4, 4]
        self.stage3_cfg.NUM_CHANNELS = [48, 96, 192]
        self.stage3_cfg.FUSE_METHOD = 'SUM'
        num_channels = [48, 96, 192]
        block = blocks_dict['BASIC']
        num_channels = [num_channels[i] * block.expansion
                        for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = AttrDict()
        self.stage4_cfg.NUM_MODULES = 3
        self.stage4_cfg.NUM_BRANCHES = 4
        self.stage4_cfg.BLOCK  = 'BASIC'
        self.stage4_cfg.NUM_BLOCKS = [4, 4, 4, 4]
        self.stage4_cfg.NUM_CHANNELS = [48, 96, 192, 384]
        self.stage4_cfg.FUSE_METHOD = 'SUM'
        num_channels = [48, 96, 192, 384]
        block = blocks_dict['BASIC']
        num_channels = [num_channels[i] * block.expansion
                        for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        self.high_level_ch = int(np.sum(pre_stage_channels))

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        Norm2d(
                            num_channels_cur_layer[i], momentum=0.1),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        Norm2d(outchannels, momentum=0.1),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Norm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=False)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=False)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=False)

        feats = torch.cat([x[0], x1, x2, x3], 1)

        return None, None, feats

    # def init_weights(self, pretrained=cfg.MODEL.HRNET_CHECKPOINT):
    def init_weights(self, pretrained=None):
        if pretrained is None:
            current_path = os.path.abspath(__file__)
            parent_dir = os.path.dirname(current_path)
            checkpoint = 'hrnetv2_w48_imagenet_pretrained.pth' 
            pretrained = os.path.join(parent_dir, checkpoint)

        # logger.debug('=> init weights from normal distribution')
        #logger.info('init weights from normal distribution')
        for name, m in self.named_modules():
            if any(part in name for part in {'cls', 'aux', 'ocr'}):
                # print('skipped', name)
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, torch.nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained,
                                         map_location={'cuda:0': 'cpu'})
            # logger.debug('=> loading pretrained model {}'.format(pretrained))
            #logger.info('loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k.replace('last_layer',
                                         'aux_head').replace('model.', ''): v
                               for k, v in pretrained_dict.items()}
            #print(set(model_dict) - set(pretrained_dict))
            #print(set(pretrained_dict) - set(model_dict))
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.

        Output:
          The correlation of every class map with every feature map
          shape = [n, num_feats, num_classes, 1]


    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, _, _ = probs.size(0), probs.size(1), probs.size(2), \
            probs.size(3)

        # each class image now a vector
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)

        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).unsqueeze(3)
        return ocr_context

class ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature
                            maps (save memory cost)
    Return:
        N X C X H X W
    '''
    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.in_channels),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear',
                                    align_corners=cfg.MODEL.ALIGN_CORNERS)

        return context

class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation
    for each pixel.
    """
    def __init__(self, in_channels, key_channels, out_channels, scale=1,
                 dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock(in_channels,
                                                         key_channels,
                                                         scale)
        if False:
            self.aspp, aspp_out_ch = get_aspp(
                in_channels, bottleneck_ch=256,
                output_stride=8)
            _in_channels = 2 * in_channels + aspp_out_ch
        else:
            _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0,
                      bias=False),
            BNReLU(out_channels),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        if False:
            aspp = self.aspp(feats)
            output = self.conv_bn_dropout(torch.cat([context, aspp, feats], 1))
        else:
            output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output

class OCR_block(nn.Module):
    """
    Some of the code in this class is borrowed from:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
    """
    def __init__(self, high_level_ch):
        super(OCR_block, self).__init__()

        ocr_mid_channels = 512
        ocr_key_channels = 256
        num_classes = 4

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(high_level_ch, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            BNReLU(ocr_mid_channels),
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0,
            bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, high_level_ch,
                      kernel_size=1, stride=1, padding=0),
            BNReLU(high_level_ch),
            nn.Conv2d(high_level_ch, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        INIT_DECODER = False
        if INIT_DECODER:
            initialize_weights(self.conv3x3_ocr,
                               self.ocr_gather_head,
                               self.ocr_distri_head,
                               self.cls_head,
                               self.aux_head)

    def forward(self, high_level_features):
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats

class OCRNet(nn.Module):
    """
    OCR net
    """
    def __init__(self, num_classes, trunk='hrnetv2', criterion=None):
        super(OCRNet, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.ocr = OCR_block(high_level_ch)

    def forward(self, inputs):
        assert 'images' in inputs
        x = inputs['images']

        _, _, high_level_features = self.backbone(x)
        cls_out, aux_out, _ = self.ocr(high_level_features)
        aux_out = scale_as(aux_out, x)
        cls_out = scale_as(cls_out, x)

        if self.training:
            gts = inputs['gts']
            aux_loss = self.criterion(aux_out, gts,
                                      do_rmi=False)
            main_loss = self.criterion(cls_out, gts)
            loss = 0.4 * aux_loss + main_loss
            return loss
        else:
            output_dict = {'pred': cls_out}
            return output_dict

class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, ignore_index=-1,
                 norm=False, upper_bound=1.0, fp16=False):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        # logger.debug("Using Per Image based weighted loss")
        #logger.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(weight, reduction='mean',
                                   ignore_index=ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = True
        self.fp16 = fp16

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        bins = torch.histc(target, bins=self.num_classes, min=0.0,
                           max=self.num_classes)
        hist_norm = bins.float() / bins.sum()
        if self.norm:
            hist = ((bins != 0).float() * self.upper_bound *
                    (1 / hist_norm)) + 1.0
        else:
            hist = ((bins != 0).float() * self.upper_bound *
                    (1. - hist_norm)) + 1.0
        return hist

    def forward(self, inputs, targets, do_rmi=None):

        if self.batch_weights:
            weights = self.calculate_weights(targets)
            self.nll_loss.weight = weights

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(targets)
                if self.fp16:
                    weights = weights.half()
                self.nll_loss.weight = weights

            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0), dim=1),
                                  targets[i].unsqueeze(0),)
        return loss

def make_attn_head(in_ch, out_ch):
    bot_ch = 256
    od = OrderedDict([('conv0', nn.Conv2d(in_ch, bot_ch, kernel_size=3,
                                          padding=1, bias=False)),
                      ('bn0', Norm2d(bot_ch)),
                      ('re0', nn.ReLU(inplace=True))])

    if True:
        od['conv1'] = nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1,
                                bias=False)
        od['bn1'] = Norm2d(bot_ch)
        od['re1'] = nn.ReLU(inplace=True)

    od['conv2'] = nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False)
    od['sig'] = nn.Sigmoid()

    attn_head = nn.Sequential(od)
    # init_attn(attn_head)
    return attn_head

class MscaleOCR(nn.Module):
    """
    OCR net
    """
    def __init__(self, num_classes, trunk='hrnetv2', criterion=None):
        super(MscaleOCR, self).__init__()
        self.criterion = criterion
        self.backbone, _, _, high_level_ch = get_trunk(trunk)
        self.ocr = OCR_block(high_level_ch)
        self.scale_attn = make_attn_head(
            in_ch=512, out_ch=1)

    def _fwd(self, x):
        x_size = x.size()[2:]

        _, _, high_level_features = self.backbone(x)
        #print(high_level_features.shape)
        cls_out, aux_out, ocr_mid_feats = self.ocr(high_level_features)
        attn = self.scale_attn(ocr_mid_feats)

        aux_out = Upsample(aux_out, x_size)
        cls_out = Upsample(cls_out, x_size)
        attn = Upsample(attn, x_size)

        return {'cls_out': cls_out,
                'aux_out': aux_out,
                'logit_attn': attn}

    def nscale_forward(self, inputs, scales):
        """
        Hierarchical attention, primarily used for getting best inference
        results.

        We use attention at multiple scales, giving priority to the lower
        resolutions. For example, if we have 4 scales {0.5, 1.0, 1.5, 2.0},
        then evaluation is done as follows:

              p_joint = attn_1.5 * p_1.5 + (1 - attn_1.5) * down(p_2.0)
              p_joint = attn_1.0 * p_1.0 + (1 - attn_1.0) * down(p_joint)
              p_joint = up(attn_0.5 * p_0.5) * (1 - up(attn_0.5)) * p_joint

        The target scale is always 1.0, and 1.0 is expected to be part of the
        list of scales. When predictions are done at greater than 1.0 scale,
        the predictions are downsampled before combining with the next lower
        scale.

        Inputs:
          scales - a list of scales to evaluate
          inputs - dict containing 'images', the input, and 'gts', the ground
                   truth mask

        Output:
          If training, return loss, else return prediction + attention
        """
        x_1x = inputs['images']

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)

        pred = None
        aux = None
        output_dict = {}

        for s in scales:
            x = ResizeX(x_1x, s)
            outs = self._fwd(x)
            cls_out = outs['cls_out']
            attn_out = outs['logit_attn']
            aux_out = outs['aux_out']

            output_dict[fmt_scale('pred', s)] = cls_out
            if s != 2.0:
                output_dict[fmt_scale('attn', s)] = attn_out

            if pred is None:
                pred = cls_out
                aux = aux_out
            elif s >= 1.0:
                # downscale previous
                pred = scale_as(pred, cls_out)
                pred = attn_out * cls_out + (1 - attn_out) * pred
                aux = scale_as(aux, cls_out)
                aux = attn_out * aux_out + (1 - attn_out) * aux
            else:
                # s < 1.0: upscale current
                cls_out = attn_out * cls_out
                aux_out = attn_out * aux_out

                cls_out = scale_as(cls_out, pred)
                aux_out = scale_as(aux_out, pred)
                attn_out = scale_as(attn_out, pred)

                pred = cls_out + (1 - attn_out) * pred
                aux = aux_out + (1 - attn_out) * aux

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = 0.4 * self.criterion(aux, gts) + \
                self.criterion(pred, gts)
            return loss
        else:
            output_dict['pred'] = pred
            return output_dict

    def two_scale_forward(self, inputs):
        """
        Do we supervised both aux outputs, lo and high scale?
        Should attention be used to combine the aux output?
        Normally we only supervise the combined 1x output

        If we use attention to combine the aux outputs, then
        we can use normal weighting for aux vs. cls outputs
        """
        assert 'images' in inputs
        x_1x = inputs['images']

        x_lo = ResizeX(x_1x, 0.5)
        #print(x_lo.shape)
        lo_outs = self._fwd(x_lo)
        pred_05x = lo_outs['cls_out']
        p_lo = pred_05x
        aux_lo = lo_outs['aux_out']
        logit_attn = lo_outs['logit_attn']
        attn_05x = logit_attn

        hi_outs = self._fwd(x_1x)
        pred_10x = hi_outs['cls_out']
        p_1x = pred_10x
        aux_1x = hi_outs['aux_out']

        p_lo = logit_attn * p_lo
        aux_lo = logit_attn * aux_lo
        p_lo = scale_as(p_lo, p_1x)
        aux_lo = scale_as(aux_lo, p_1x)

        logit_attn = scale_as(logit_attn, p_1x)

        # combine lo and hi predictions with attention
        joint_pred = p_lo + (1 - logit_attn) * p_1x
        joint_aux = aux_lo + (1 - logit_attn) * aux_1x

        if self.training:
            gts = inputs['gts']
            do_rmi = False
            aux_loss = self.criterion(joint_aux, gts, do_rmi=do_rmi)

            # Optionally turn off RMI loss for first epoch to try to work
            # around cholesky errors of singular matrix
            do_rmi_main = True  # cfg.EPOCH > 0
            main_loss = self.criterion(joint_pred, gts, do_rmi=do_rmi_main)
            loss = 0.4 * aux_loss + main_loss

            # Optionally, apply supervision to the multi-scale predictions
            # directly. Turn off RMI to keep things lightweight
            SUPERVISED_MSCALE_WT = 0
            if SUPERVISED_MSCALE_WT:
                scaled_pred_05x = scale_as(pred_05x, p_1x)
                loss_lo = self.criterion(scaled_pred_05x, gts, do_rmi=False)
                loss_hi = self.criterion(pred_10x, gts, do_rmi=False)
                loss += SUPERVISED_MSCALE_WT * loss_lo
                loss += SUPERVISED_MSCALE_WT * loss_hi
            return loss
        else:
            output_dict = {
                'pred': joint_pred,
                'pred_05x': pred_05x,
                'pred_10x': pred_10x,
                'attn_05x': attn_05x,
            }
            return output_dict

    def forward(self, inputs):
        N_SCALES = None
        if N_SCALES and not self.training:
            return self.nscale_forward(inputs, N_SCALES)

        return self.two_scale_forward(inputs)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = Norm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = Norm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm2d(planes, momentum=0.1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = Norm2d(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = Norm2d(planes * self.expansion, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

def BNReLU(ch):
    return nn.Sequential(
        Norm2d(ch),
        nn.ReLU())

def get_model(network, num_classes, criterion):
    """
    Fetch Network Function Pointer
    """
    #module = network[:network.rfind('.')]
    #model = network[network.rfind('.') + 1:]
    #mod = importlib.import_module(module)
    
    #net_func = getattr(mod, model)
    #print(mod)
    #net = net_func(num_classes=num_classes, criterion=criterion)
    net = MscaleOCR(num_classes, trunk='hrnetv2', criterion=criterion)
    return net

def HRNet_Mscale(num_classes, criterion):
    return MscaleOCR(num_classes, trunk='hrnetv2', criterion=criterion)

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}