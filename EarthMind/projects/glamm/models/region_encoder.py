from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple
from torch import Tensor

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv import ops
from mmcv.cnn import ConvModule, Linear
from mmengine.model import BaseModule

class BaseRoIExtractor(BaseModule, metaclass=ABCMeta):
    """Base class for RoI extractor.

    Args:
        roi_layer (:obj:`ConfigDict` or dict): Specify RoI layer type and
            arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (list[int]): Strides of input feature maps.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 roi_layer,
                 out_channels: int,
                 featmap_strides: List[int],
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides

    @property
    def num_inputs(self) -> int:
        """int: Number of input feature maps."""
        return len(self.featmap_strides)

    def build_roi_layers(self, layer_cfg,
                         featmap_strides: List[int]) -> nn.ModuleList:
        """Build RoI operator to extract feature from each level feature map.

        Args:
            layer_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
                config RoI layer operation. Options are modules under
                ``mmcv/ops`` such as ``RoIAlign``.
            featmap_strides (list[int]): The stride of input feature map w.r.t
                to the original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.

        Returns:
            :obj:`nn.ModuleList`: The RoI extractor modules for each level
                feature map.
        """

        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        if isinstance(layer_type, str):
            assert hasattr(ops, layer_type)
            layer_cls = getattr(ops, layer_type)
        else:
            layer_cls = layer_type
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def roi_rescale(self, rois: Tensor, scale_factor: float) -> Tensor:
        """Scale RoI coordinates by scale factor.

        Args:
            rois (Tensor): RoI (Region of Interest), shape (n, 5)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            Tensor: Scaled RoI.
        """

        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        y1 = cy - new_h * 0.5
        y2 = cy + new_h * 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    @abstractmethod
    def forward(self,
                feats: Tuple[Tensor],
                rois: Tensor,
                roi_scale_factor: Optional[float] = None) -> Tensor:
        """Extractor ROI feats.

        Args:
            feats (Tuple[Tensor]): Multi-scale features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            roi_scale_factor (Optional[float]): RoI scale factor.
                Defaults to None.

        Returns:
            Tensor: RoI feature.
        """
        pass


class MLVLFuseModule(nn.Module):
    def __init__(self, input_dims=1024, embed_dims=1024, num_levels=3, num_fuse=4):
        super(MLVLFuseModule, self).__init__()
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_fuse = num_fuse
        self.input_dims = input_dims
        self.shuffle_channles = embed_dims // 4

        # contains the tuple of level indices that will do the interaction
        self.fuse_lvl_list = []
        num_levels = self.num_levels
        for lvl in range(num_levels):
            top_lvl = min(lvl + 1, num_levels - 1)
            dow_lvl = max(lvl - 1, 0)
            tar_lvl = lvl
            self.fuse_lvl_list.append((tar_lvl, top_lvl, dow_lvl))

        self.remain_chs = self.embed_dims - self.shuffle_channles * 2
        self._init_layers()

    def generate_coordinate(self, featmap_sizes, device='cuda'):

        x_range = torch.linspace(-1, 1, featmap_sizes[-1], device=device)
        y_range = torch.linspace(-1, 1, featmap_sizes[-2], device=device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([featmap_sizes[0], 1, -1, -1])
        x = x.expand([featmap_sizes[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        return coord_feat

    def _init_layers(self):
        self.input_conv = nn.ModuleList([nn.Conv2d(self.input_dims + 2,
                                                   self.embed_dims, 1)
                                         for _ in range(self.num_levels)])
        self.fuse_convs = nn.ModuleList()
        for i in range(self.num_fuse):
            self.fuse_convs.append(
                ConvModule(self.embed_dims,
                           self.embed_dims,
                           3,
                           stride=1,
                           padding=3 // 2,
                           conv_cfg=None,
                           norm_cfg=dict(type='GN',
                                         num_groups=64,
                                         requires_grad=True)
                           ))

    def init_weights(self):
        pass

    def _single_shuffle(self, inputs, conv_module):
        if not isinstance(conv_module, (nn.ModuleList, list)):
            conv_module = [conv_module]
        for single_conv_m in conv_module:
            fused_inputs = []
            for fuse_lvl_tuple in self.fuse_lvl_list:
                tar_lvl, top_lvl, dow_lvl = fuse_lvl_tuple
                tar_input = inputs[tar_lvl]
                top_input = inputs[top_lvl]
                down_input = inputs[dow_lvl]
                remain = tar_input[:, :self.remain_chs]
                from_top = top_input[:, self.remain_chs:][:, self.shuffle_channles:]
                from_top = F.interpolate(from_top.to(torch.float32),
                                         size=tar_input.shape[-2:],
                                         mode='bilinear',
                                         align_corners=True)
                from_down = down_input[:, self.remain_chs:][:, :self.shuffle_channles]
                from_down = F.interpolate(from_down.to(torch.float32),
                                          size=tar_input.shape[-2:],
                                          mode='bilinear',
                                          align_corners=True)
                fused_inputs.append(
                    torch.cat([remain, from_top.to(remain.dtype), from_down.to(remain.dtype)], dim=1))
            fused_inputs = [single_conv_m(item) for item in fused_inputs]
            inputs = fused_inputs
        return inputs

    def forward(self, inputs, ):
        feat_size = [item.shape for item in inputs]
        new_inputs = []
        for feat, single_feat_size in zip(inputs, feat_size):
            coord_feat = self.generate_coordinate(
                single_feat_size, device=inputs[0].device)
            # feat = torch.cat([feat, coord_feat], dim=1)
            feat = torch.cat([feat, coord_feat.to(feat.dtype)], dim=1)
            new_inputs.append(feat)
        inputs = new_inputs

        inputs = [self.input_conv[lvl](item)
                  for lvl, item in enumerate(inputs)]

        for conv_m in self.fuse_convs:
            inputs = self._single_shuffle(inputs, [conv_m])
        return inputs


class MlvlRoIExtractor(BaseRoIExtractor):
    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 embed_dims=1024,
                 stride=1,
                 norm_init=True,
                 fuse_level=3,
                 finest_scale=56,
                 init_cfg=None):
        super(MlvlRoIExtractor, self).__init__(roi_layer, out_channels,
                                               featmap_strides, init_cfg)
        self.embed_dims = embed_dims
        self.finest_scale = finest_scale
        self.fuse_level = fuse_level
        self.norm_init = norm_init

        self.pconvs = nn.ModuleList(
            nn.Conv2d(self.embed_dims, self.embed_dims, 3, stride=1, padding=1)
            for _ in range(self.fuse_level))
        self.pos_embedd = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.LayerNorm(1024),
        )
        self.updims = nn.Linear(1024, 4096)

        self.flatten_linear = nn.Linear(
            self.embed_dims * self.roi_layers[0].output_size[0] ** 2, 1024)

        self.norm_init_weights()

    #  self.dtype = torch.float32
    def norm_init_weights(self):
        pass

    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        num_imgs = len(rois)
        # feats = [item for item in feats]
        batch_rois = torch.cat(rois, dim=0).to(feats[0].dtype)
        pos_embedd = self.pos_embedd(batch_rois)
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        if feats[0].dim() == 3:
            h = w = int(math.sqrt(feats[0].shape[1]))
            assert h == 16
            assert w == 16
            b, c = feats[0].shape[0], feats[0].shape[-1]
            feats = [item.reshape(b, h, w, c).permute(0, 3, 1, 2)
                     for item in feats]
        new_rois = []
        for img_id, single_img_roi in enumerate(rois):
            # rescale to original img scale
            single_img_roi = single_img_roi * 224

            roi_img_id = single_img_roi.new_ones(len(single_img_roi)) * img_id
            single_img_roi = torch.cat(
                [roi_img_id[:, None], single_img_roi], dim=1)
            new_rois.append(single_img_roi)
        rois = torch.cat(new_rois)

        roi_feats = feats[0].new_zeros(self.fuse_level,
                                       rois.size(0), self.out_channels, *out_size)

        for i in range(num_levels):
            if len(rois) > 0:
                rois_ = rois
                ori_dtype = feats[i].dtype
                roi_feats_t = self.roi_layers[i](feats[i].to(
                    torch.float32), rois_.to(torch.float32))

                roi_feats[i] = roi_feats_t.to(ori_dtype)

            else:
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.

        fuse_roi_feats = []
        for i in range(self.fuse_level):
            fuse_roi_feats.append(self.pconvs[i](roi_feats[i]))

        fuse_roi_feats = sum(fuse_roi_feats)
        fuse_roi_feats = F.relu(fuse_roi_feats)
        fuse_roi_feats = fuse_roi_feats.flatten(1, -1)
        fuse_roi_feats = self.flatten_linear(fuse_roi_feats)
        fuse_roi_feats = fuse_roi_feats + pos_embedd
        fuse_roi_feats = self.updims(fuse_roi_feats)
        query_feats = []
        for i in range(num_imgs):
            mask = rois[:, 0] == i
            query_feats.append(fuse_roi_feats[mask])

        return query_feats


class MLVLROIQueryModule(nn.Module):
    def __init__(self, embed_dims=1024, out_dims=4096,
                 num_levels=3):
        super(MLVLROIQueryModule, self).__init__()
        self.mlvl_fuse = MLVLFuseModule(input_dims=embed_dims,
                                        embed_dims=embed_dims,
                                        num_levels=num_levels,
                                        num_fuse=5)
        strids = [14 / 8, 14 / 4, 14 / 2, 14]
        assert len(strids) == num_levels
        bbox_roi_extractor = dict(roi_layer=dict(type='RoIAlign',
                                                 output_size=14,
                                                 sampling_ratio=2),
                                  out_channels=embed_dims,
                                  embed_dims=embed_dims,
                                  fuse_level=num_levels,
                                  featmap_strides=strids)

        self.roi_align = MlvlRoIExtractor(**bbox_roi_extractor)

    def forward(self, mlvl_feats, bboxes):
        if mlvl_feats[0].dim() == 3:
            h = w = int(math.sqrt(mlvl_feats[0].shape[1]))
            assert h == 24
            assert w == 24
            b, c = mlvl_feats[0].shape[0], mlvl_feats[0].shape[-1]
            mlvl_feats = [item.reshape(b, h, w, c).permute(0, 3, 1, 2) for item in mlvl_feats]
        base_shape = mlvl_feats[0].shape[-2:]
        num_level = len(mlvl_feats)
        to_shape = [(base_shape[0] * 2 ** level, base_shape[1] * 2 ** level)
                    for level in range(num_level)]
        to_shape = to_shape[::-1]
        for level in range(num_level):
            feat = mlvl_feats[level]
            shape = to_shape[level]
            # feat = feat
            # mlvl_feats[level] = F.interpolate(feat, size=shape, mode='bilinear', align_corners=True)
            # todo: temporary fix for "upsample_bilinear2d_out_frame" not implemented for 'BFloat16'
            feat = feat.to(torch.float32)
            mlvl_feats[level] = F.interpolate(
                feat, size=shape, mode='bilinear', align_corners=True)
            mlvl_feats[level] = mlvl_feats[level].to(torch.bfloat16)

        mlvl_feats = self.mlvl_fuse(mlvl_feats)

        return self.roi_align(mlvl_feats, bboxes)
