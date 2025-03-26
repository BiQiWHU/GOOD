import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmrotate.models.roi_heads.bbox_heads import RotatedConvFCBBoxHead
from torch import Tensor
from mmdet.models.losses import accuracy


@HEADS.register_module()
class DG2RBBoxHeadForAngleVisualization(RotatedConvFCBBoxHead):

    def __init__(self,
                 mlp_head_channels: int = 1024,
                 with_weight_decay: bool = False,
                 loss_contrast: Dict = dict(
                     type='SupervisedContrastiveLossOriginal',
                     temperature=0.1,
                     iou_threshold=0.5,
                     loss_weight=0.2,
                     reweight_type='none'),
                 scale: int = 20,
                 learnable_scale: bool = False,
                 eps: float = 1e-5,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # override the fc_cls in :obj:`ConvFCBBoxHead`
        if self.with_cls:
            self.fc_cls = nn.Linear(
                self.cls_last_dim, self.num_classes + 1, bias=False)

        # learnable global scaling factor
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1) * scale)
        else:
            self.scale = scale
        self.mlp_head_channels = mlp_head_channels
        self.with_weight_decay = with_weight_decay
        self.eps = eps
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = 1.0
        self.gamma = 1
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_out_channels, mlp_head_channels))
        self.contrast_loss = build_loss(copy.deepcopy(loss_contrast))
        self.angle_pred = []
        self.angle_gt = []

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward function.
        Args:
            x (Tensor): Shape of (num_proposals, C, H, W).
        Returns:
            tuple:
                cls_score (Tensor): Cls scores, has shape
                    (num_proposals, num_classes).
                bbox_pred (Tensor): Box energies / deltas, has shape
                    (num_proposals, 4).
                contrast_feat (Tensor): Box features for contrast loss,
                    has shape (num_proposals, C).
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # separate branches
        x_cls = x
        x_reg = x
        x_contra = x
        
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        
        if x_cls.dim() > 2:
            x_cls = torch.flatten(x_cls, start_dim=1)
        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x_cls, p=2, dim=1).unsqueeze(1).expand_as(x_cls)
        x_cls_normalized = x_cls.div(x_norm + self.eps)
        # normalize weight
        with torch.no_grad():
            temp_norm = torch.norm(
                self.fc_cls.weight, p=2,
                dim=1).unsqueeze(1).expand_as(self.fc_cls.weight)
            self.fc_cls.weight.div_(temp_norm + self.eps)
        # calculate and scale cls_score
        cls_score = self.scale * self.fc_cls(
            x_cls_normalized) if self.with_cls else None

        # contrastive branch
        contrast_feat = self.contrastive_head(x_contra)
        x_contra_norm = torch.norm(contrast_feat, p=2, dim=1).unsqueeze(1).expand_as(contrast_feat)
        x_contra_normalized = contrast_feat.div(x_contra_norm + self.eps)
        contrast_feat = x_contra_normalized

        return cls_score, bbox_pred, contrast_feat

    def set_decay_rate(self, decay_rate: float) -> None:
        """Contrast loss weight decay hook will set the `decay_rate` according
        to iterations.
        Args:
            decay_rate (float): Decay rate for weight decay.
        """
        self._decay_rate = decay_rate

    @force_fp32(apply_to=('contrast_feat'))
    def loss_contrast(self,
                      contrast_feat: Tensor,
                      proposal_ious: Tensor,
                      labels: Tensor,
                      reduction_override: Optional[str] = None) -> Dict:
        """Loss for contract.
        Args:
            contrast_feat (tensor): BBox features with shape (N, C)
                used for contrast loss.
            proposal_ious (tensor): IoU between proposal and ground truth
                corresponding to each BBox features with shape (N).
            labels (tensor): Labels for each BBox features with shape (N).
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss. Options
                are "none", "mean" and "sum". Default: None.
        Returns:
            Dict: The calculated loss.
        """

        losses = dict()
        if self.with_weight_decay:
            decay_rate = self._decay_rate
        else:
            decay_rate = None
        losses['loss_contrast'] = self.contrast_loss(
            contrast_feat,
            labels,
            proposal_ious,
            decay_rate=decay_rate,
            reduction_override=reduction_override)
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss function.

        Args:
            cls_score (torch.Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 5).
            rois (torch.Tensor): Boxes to be transformed. Has shape
                (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            labels (torch.Tensor): Shape (n*bs, ).
            label_weights(torch.Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
            bbox_targets(torch.Tensor):Regression target for all
                  proposals, has shape (num_proposals, 5), the
                  last dimension 5 represents [cx, cy, w, h, a].
            bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 5) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 5).
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        
        
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                
                # import ipdb; ipdb.set_trace()
                save_index = bbox_targets[pos_inds.type(torch.bool)][:,4] != 0
                save_pred = pos_bbox_pred[save_index,4]
                save_gt = bbox_targets[pos_inds.type(torch.bool)][save_index,4]
                for angle in save_pred:
                    self.angle_pred.append(angle.detach().cpu().numpy())
                for angle in save_gt:
                    self.angle_gt.append(angle.detach().cpu().numpy())
                
                np.save("angle_visualization/prediction.npy", self.angle_pred)
                np.save("angle_visualization/groundtruth.npy", self.angle_gt)
                print("at present, len of predicted and gt angle is " + str(len(self.angle_gt)))
                
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
                
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
