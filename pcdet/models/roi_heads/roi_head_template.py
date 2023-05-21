import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from .target_assigner.proposal_target_layer import ProposalTargetLayer
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


class RoIHeadTemplate(nn.Module):
    def __init__(self, num_class, model_cfg, predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None
        self.predict_boxes_when_training = predict_boxes_when_training

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        if batch_dict.get('rois', None) is not None:
            return batch_dict

        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict
    
    def update_metrics(self, targets_dict, mask_type='cls', vis_type='pred_gt', pred_type=None):
        metric_registry = targets_dict['metric_registry']
        unlabeled_inds = targets_dict['unlabeled_inds']
        softmatch = targets_dict['softmatch']
        sample_preds, sample_pred_scores, sample_pred_weights = [], [], []
        sample_rois, sample_roi_scores = [], []
        sample_targets, sample_target_scores = [], []
        sample_pls, sample_pl_scores = [], []
        ema_preds_of_std_rois, ema_pred_scores_of_std_rois = [], []
        sample_gts = []
        sample_gt_iou_of_rois = []
        sample_softmatch_weights,sample_objective,sample_valid_mask = [],[],[]
        sample_ori_boxes = []
        for i, uind in enumerate(unlabeled_inds):
            mask = (targets_dict['reg_valid_mask'][uind] > 0) if mask_type == 'reg' else (
                        targets_dict['rcnn_cls_labels'][uind] >= 0)

            # (Proposals) ROI info
            rois = targets_dict['rois'][uind][mask].detach().clone()
            roi_labels = targets_dict['roi_labels'][uind][mask].unsqueeze(-1).detach().clone()
            roi_scores = torch.sigmoid(targets_dict['roi_scores'])[uind][mask].detach().clone()
            roi_labeled_boxes = torch.cat([rois, roi_labels], dim=-1)
            gt_iou_of_rois = targets_dict['gt_iou_of_rois'][uind][mask].unsqueeze(-1).detach().clone()
            softmatch_weights = targets_dict['softmatch_weights'][uind][mask].detach().clone()
            sample_rois.append(roi_labeled_boxes)
            sample_roi_scores.append(roi_scores)
            sample_gt_iou_of_rois.append(gt_iou_of_rois)
            sample_softmatch_weights.append(softmatch_weights)
            # Target info
            target_labeled_boxes = targets_dict['gt_of_rois_src'][uind][mask].detach().clone()
            target_scores = targets_dict['rcnn_cls_labels'][uind][mask].detach().clone()
            sample_targets.append(target_labeled_boxes)
            sample_target_scores.append(target_scores)

            # Pred info
            pred_boxes = targets_dict['batch_box_preds'][uind][mask].detach().clone()
            pred_scores = torch.sigmoid(targets_dict['rcnn_cls']).view_as(targets_dict['rcnn_cls_labels'])[uind][mask].detach().clone()
            pred_labeled_boxes = torch.cat([pred_boxes, roi_labels], dim=-1)
            sample_preds.append(pred_labeled_boxes)
            sample_pred_scores.append(pred_scores)

            # (Real labels) GT info
            gt_labeled_boxes = targets_dict['ori_unlabeled_boxes'][i]
            sample_gts.append(gt_labeled_boxes)

            # (Pseudo labels) PL info
            pl_labeled_boxes = targets_dict['pl_boxes'][uind]
            pl_scores = targets_dict['pl_scores'][i]
            sample_pls.append(pl_labeled_boxes)
            sample_pl_scores.append(pl_scores)

            pred_weights = targets_dict['rcnn_cls_weights'][uind][mask].detach().clone()
            sample_pred_weights.append(pred_weights)

            batch_loss_cls = F.binary_cross_entropy(pred_scores, target_scores.float(), reduction='none')
            cls_valid_mask = (target_scores >= 0).float()
            # rcnn_loss_cls_norm = (cls_valid_mask * pred_weights).sum(-1)
            rcnn_loss_cls = (batch_loss_cls * pred_weights) 
            sample_objective.append(rcnn_loss_cls)
            sample_valid_mask.append(cls_valid_mask)


            # Teacher refinements (Preds) of student's rois
            if 'ema_gt' in pred_type and self.get('ENABLE_SOFT_TEACHER', False):
                pred_boxes_ema = targets_dict['batch_box_preds_teacher'][uind][mask].detach().clone()
                pred_labeled_boxes_ema = torch.cat([pred_boxes_ema, roi_labels], dim=-1)
                pred_scores_ema = targets_dict['rcnn_cls_score_teacher'][uind][mask].detach().clone()
                ema_preds_of_std_rois.append(pred_labeled_boxes_ema)
                ema_pred_scores_of_std_rois.append(pred_scores_ema)
            if self.model_cfg.get('ENABLE_SOFT_TEACHER', False):
                pred_weights = targets_dict['rcnn_cls_weights'][uind][mask].detach().clone()
                sample_pred_weights.append(pred_weights)

        sample_pred_weights = sample_pred_weights if len(sample_pred_weights) > 0 else None

        if 'pred_gt' in pred_type:
            tag = f'rcnn_pred_gt_metrics_{mask_type}'
            metrics = metric_registry.get(tag)
            metric_inputs = {'preds': sample_preds, 'pred_scores': sample_pred_scores, 'rois': sample_rois,
                             'roi_scores': sample_roi_scores, 'ground_truths': sample_gts,
                             'targets': sample_targets, 'target_scores': sample_target_scores,
                             'pred_weights': sample_pred_weights}
            metrics.update(**metric_inputs)
        if 'ema_gt' in pred_type and self.model_cfg.get('ENABLE_SOFT_TEACHER', False):
            tag = f'rcnn_ema_gt_metrics_{mask_type}'
            metrics_ema = metric_registry.get(tag)
            metric_inputs_ema = {'preds': ema_preds_of_std_rois, 'pred_scores': ema_pred_scores_of_std_rois,
                                 'ground_truths': sample_gts, 'pred_weights': sample_pred_weights}
            metrics_ema.update(**metric_inputs_ema)
        if 'roi_pl' in pred_type:
            tag = f'rcnn_roi_pl_metrics_{mask_type}'
            metrics_roi_pl = metric_registry.get(tag)
            metric_inputs_roi_pl = {'preds': sample_rois, 'pred_scores': sample_roi_scores, 'ground_truths': sample_pls,
                                    'targets': sample_targets, 'target_scores': sample_target_scores,
                                    'pred_weights': sample_pred_weights}
            metrics_roi_pl.update(**metric_inputs_roi_pl)
        if 'pred_pl' in pred_type:
            tag = f'rcnn_pred_pl_metrics_{mask_type}'
            metrics_pred_pl = metric_registry.get(tag)
            metric_inputs_pred_pl = {'preds': sample_preds, 'pred_scores': sample_pred_scores, 'rois': sample_rois,
                                     'roi_scores': sample_roi_scores, 'ground_truths': sample_pls,
                                     'targets': sample_targets, 'target_scores': sample_target_scores,
                                     'pred_weights': sample_pred_weights}
            metrics_pred_pl.update(**metric_inputs_pred_pl)
        if 'roi_pl_gt' in pred_type:
            tag = f'rcnn_roi_pl_gt_metrics_{mask_type}'
            metrics = metric_registry.get(tag)
            metric_inputs = {'preds': sample_rois, 'pred_scores': sample_roi_scores,
                             'ground_truths': sample_gts, 'targets': sample_targets,
                             'pseudo_labels': sample_pls, 'pseudo_label_scores': sample_pl_scores,
                             'target_scores': sample_target_scores, 'pred_weights': sample_pred_weights,
                             'pred_iou_wrt_pl': sample_gt_iou_of_rois,'softmatch_weights':sample_softmatch_weights,
                             'softmatch_thresh':softmatch.st_mean, 'objective':sample_objective, 'loss_mask':sample_valid_mask}
            metrics.update(**metric_inputs)

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)
        targets_dict['points'] = batch_dict['points']

        # Extract number of points in ROIs
        if batch_dict['store_scores_in_pkl']:
            num_point_in_rois = roiaware_pool3d_utils.points_in_boxes_cpu(targets_dict['points'][:, 1:4].cpu(),
                                                                        targets_dict['rois'].view(-1,7).cpu()).squeeze(0).sum(1)
            targets_dict['num_points_in_roi'] = num_point_in_rois.reshape(targets_dict['rois'].shape[0], \
                                                                        targets_dict['rois'].shape[1])


        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict
    def preloss_filtering(self):
        unlabeled_inds = self.forward_ret_dict['unlabeled_inds']
        # unlabeled_rcnn_cls_weights = self.forward_ret_dict['rcnn_cls_weights'][unlabeled_inds]
        ul_interval_mask = self.forward_ret_dict['interval_mask'][unlabeled_inds].clone().detach()
        ul_weights = self.forward_ret_dict['softmatch_weights'][unlabeled_inds].clone().detach()
        ul_targets = self.forward_ret_dict['rcnn_cls_labels'][unlabeled_inds].clone().detach()
        gt_iou_of_rois = self.forward_ret_dict['gt_iou_of_rois'][unlabeled_inds].detach().clone()
        roi_labels = self.forward_ret_dict['roi_labels'][unlabeled_inds].detach().clone() - 1
        softmatch = self.forward_ret_dict['softmatch']
        
        fg_thresh = gt_iou_of_rois.new_tensor(softmatch.st_mean.to(gt_iou_of_rois.device)).reshape(1,1,-1).repeat(*ul_weights.shape[:2],1)
        cls_fg_thresh = torch.gather(fg_thresh, dim=-1, index=roi_labels.unsqueeze(-1)).squeeze(-1)
        
        #NOTE: Interval weighting happens here
        ul_weights[~self.forward_ret_dict['interval_mask'][unlabeled_inds]] = 1
        self.forward_ret_dict['rcnn_cls_weights'] = torch.ones_like(self.forward_ret_dict['rcnn_cls_labels'])
        self.forward_ret_dict['rcnn_cls_weights'][unlabeled_inds] = ul_weights
        # iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt_boxes[:, 0:7])  # (M, N)
        # max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
        # self.forward_ret_dict['gt_iou_of_rois']
    def get_box_reg_layer_loss(self, forward_ret_dict, scalar=True):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size

        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        batch_size = forward_ret_dict['reg_valid_mask'].shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        # if scalar:
        #     fg_sum = fg_mask.long().sum().item()
        # else:
        #     fg_sum = fg_mask.reshape(batch_size, -1).long().sum(-1)

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            if scalar:
                rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() \
                                / max(fg_sum, 1)
            else:
                fg_sum_ = fg_mask.reshape(batch_size, -1).long().sum(-1)
                rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()) \
                                    .reshape(batch_size, -1).sum(-1) / torch.clamp(fg_sum_.float(), min=1.0)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item() if scalar else rcnn_loss_reg

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                split_size = []
                fg_mask_batch = fg_mask.reshape(batch_size, -1)
                for i in range(batch_size):
                    split_size.append(len(torch.nonzero(fg_mask_batch[i])))
                # TODO: need further check
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                if scalar:
                    loss_corner = loss_corner.mean()
                else:
                    loss_corner = torch.split(loss_corner, split_size, dim=0)
                    zero = torch.zeros([1], device=fg_mask.device)
                    loss_corner = [x.mean(dim=0, keepdim=True) if len(x) > 0 else zero for x in loss_corner]
                    loss_corner = torch.cat(loss_corner, dim=0)
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item() if scalar else loss_corner
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict, scalar=True):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            if scalar:
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
                rcnn_acc_cls = (torch.abs(torch.sigmoid(rcnn_cls_flat) - rcnn_cls_labels) * cls_valid_mask).sum() \
                               / torch.clamp(cls_valid_mask.sum(), min=1.0)
            else:
                batch_size = forward_ret_dict['rcnn_cls_labels'].shape[0]
                batch_loss_cls = batch_loss_cls.reshape(batch_size, -1)
                cls_valid_mask = cls_valid_mask.reshape(batch_size, -1)
                if 'rcnn_cls_weights' in forward_ret_dict:
                    rcnn_cls_weights = forward_ret_dict['rcnn_cls_weights']
                    rcnn_loss_cls_norm = (cls_valid_mask * rcnn_cls_weights).sum(-1)
                    rcnn_loss_cls = (batch_loss_cls * cls_valid_mask * rcnn_cls_weights).sum(-1) / torch.clamp(rcnn_loss_cls_norm, min=1.0)
                else:
                    rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum(-1) / torch.clamp(cls_valid_mask.sum(-1), min=1.0)
                rcnn_acc_cls = torch.abs(torch.sigmoid(rcnn_cls_flat) - rcnn_cls_labels).reshape(batch_size, -1)
                rcnn_acc_cls = (rcnn_acc_cls * cls_valid_mask).sum(-1) / torch.clamp(cls_valid_mask.sum(-1), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            if scalar:
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
            else:
                batch_size = forward_ret_dict['rcnn_cls_labels'].shape[0]
                batch_loss_cls = batch_loss_cls.reshape(batch_size, -1)
                cls_valid_mask = cls_valid_mask.reshape(batch_size, -1)
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum(-1) / torch.clamp(cls_valid_mask.sum(-1), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {
            'rcnn_loss_cls': rcnn_loss_cls.item() if scalar else rcnn_loss_cls,
            'rcnn_acc_cls': rcnn_acc_cls.item() if scalar else rcnn_acc_cls
        }
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None, scalar=True):
        tb_dict = {} if tb_dict is None else tb_dict
        self.preloss_filtering()
        if self.model_cfg.get("ENABLE_EVAL", None):
            # self.update_metrics(self.forward_ret_dict, mask_type='reg')
            metrics_pred_types = self.model_cfg.get("METRICS_PRED_TYPES", None)
            if metrics_pred_types is not None:
                self.update_metrics(self.forward_ret_dict, mask_type='cls', pred_type=metrics_pred_types, vis_type='roi_pl')

        # if self.model_cfg.ADAPTIVE_THRESH_CONFIG.get('ENABLE', False):
        #     adaptive_thresh_metric = self.forward_ret_dict['adaptive_thresh_metric']
        #     unlabeled_inds = self.forward_ret_dict['unlabeled_inds']
        #     batch_roi_labels = self.forward_ret_dict['roi_labels'][unlabeled_inds].detach().clone()
        #     batch_roi_ious = self.forward_ret_dict['gt_iou_of_rois'][unlabeled_inds].detach().clone()
        #     metric_inputs = {'batch_roi_labels': batch_roi_labels, 'batch_iou_wrt_pl': batch_roi_ious}
        #     adaptive_thresh_metric.update(**metric_inputs)

        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict, scalar=scalar)
        #rcnn_loss = rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict, scalar=scalar)
        #rcnn_loss += rcnn_loss_reg  #if scalar is False, rcnn_loss_cls is (rcnn_loss_cls + rcnn_loss_reg)
        rcnn_loss = rcnn_loss_cls + rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item() if scalar else rcnn_loss
        if scalar:
            return rcnn_loss, tb_dict
        else:
            return rcnn_loss_cls, rcnn_loss_reg, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds
