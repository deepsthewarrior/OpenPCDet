import copy
import os
import pickle
import numpy as np
import torch
import torch.distributed as dist
from pcdet.datasets.augmentor import augmentor_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from .detector3d_template import Detector3DTemplate
from .pv_rcnn import PVRCNN

from pcdet.utils import common_utils
from pcdet.utils.stats_utils import metrics_registry
from pcdet.utils.prototype_utils import feature_bank_registry
# from tools.visual_utils import open3d_vis_utils as V
from collections import defaultdict
from pcdet.utils.weighting_methods import build_thresholding_method
from pcdet.ops.iou3d_nms import iou3d_nms_utils
class DynamicThreshRegistry(object):
    def __init__(self, **kwargs):
        self._tag_metrics = {}
        self.dataset = kwargs.get('dataset', None)
        self.model_cfg = kwargs.get('model_cfg', None)

    def get(self, tag=None):
        if tag is None:
            tag = 'default'
        if tag in self._tag_metrics.keys():
            metric = self._tag_metrics[tag]
        else:
            metric = build_thresholding_method(tag=tag, dataset=self.dataset, config=self.model_cfg)
            self._tag_metrics[tag] = metric
        return metric

    def tags(self):
        return self._tag_metrics.keys()


class PVRCNN_SSL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # something changes so need deep copy
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)
        self.pv_rcnn = PVRCNN(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.pv_rcnn_ema = PVRCNN(model_cfg=model_cfg_copy, num_class=num_class, dataset=dataset_copy)
        for param in self.pv_rcnn_ema.parameters():
            param.detach_()
        self.add_module('pv_rcnn', self.pv_rcnn)
        self.add_module('pv_rcnn_ema', self.pv_rcnn_ema)
        self.accumulated_itr = 0

        self.thresh = model_cfg.THRESH
        self.sem_thresh = model_cfg.SEM_THRESH
        self.hybrid_thresh = model_cfg.HYBRID_THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.no_nms = model_cfg.NO_NMS
        self.supervise_mode = model_cfg.SUPERVISE_MODE
        self.thresh_registry = DynamicThreshRegistry(dataset=self.dataset, model_cfg=model_cfg)
        for bank_configs in model_cfg.get("FEATURE_BANK_LIST", []):
            feature_bank_registry.register(tag=bank_configs["NAME"], **bank_configs)

        for metrics_configs in model_cfg.get("METRICS_BANK_LIST", []):
            for name in metrics_configs['NAME']:
                metrics_configs['tag'] = name
                metrics_registry.register(dataset=self.dataset, **metrics_configs)

        vals_to_store = ['iou_roi_pl', 'iou_roi_gt', 'pred_scores','roi_scores', 'class_labels', 'iteration','roi_sim_scores','iou_pl_gt',
                         'assigned_gt_pl_labels','pseudo_sem_scores_pl','pseudo_sim_scores_pl','rcnn_scores_pl','pl_iteration','roi_instance_sim_scores','pseudo_instance_sim_scores_pl']
        self.val_dict = {val: [] for val in vals_to_store}
        loss_dict_keys = {'cos_sim_pl_wa','cos_sim_pl_sa','pl_labels','proto_labels'}
        self.loss_dict = {key: [] for key in loss_dict_keys}
        mcont_dict = {'logits','iteration'}
        self.mcont_dict = {key: [] for key in mcont_dict}

    @staticmethod
    def _clone_gt_boxes_and_feats(batch_dict):
        return {
            "batch_size": batch_dict['batch_size'],
            "gt_boxes": batch_dict['gt_boxes'].clone().detach(),
            "point_coords": batch_dict['point_coords'].clone().detach(),
            "point_features": batch_dict['point_features'].clone().detach(),
            "point_cls_scores": batch_dict['point_cls_scores'].clone().detach()
        }

    def _prep_bank_inputs(self, batch_dict, inds, num_points_threshold=20):
        selected_batch_dict = self._clone_gt_boxes_and_feats(batch_dict)
        with torch.no_grad():
            batch_gt_feats = self.pv_rcnn_ema.roi_head.pool_features(selected_batch_dict, use_gtboxes=True)
            batch_size_rcnn = batch_gt_feats.shape[0]
            shared_features = self.pv_rcnn_ema.roi_head.shared_fc_layer(batch_gt_feats.view(batch_size_rcnn, -1, 1))
        batch_gt_feats = shared_features.view(*batch_dict['gt_boxes'].shape[:2], -1)
        bank_inputs = defaultdict(list)
        for ix in inds:
            gt_boxes = selected_batch_dict['gt_boxes'][ix]
            nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
            if nonzero_mask.sum() == 0:
                print(f"no gt instance in frame {batch_dict['frame_id'][ix]}")
                continue
            gt_boxes = gt_boxes[nonzero_mask]
            sample_mask = batch_dict['points'][:, 0].int() == ix
            points = batch_dict['points'][sample_mask, 1:4]
            gt_feat = batch_gt_feats[ix][nonzero_mask]
            gt_labels = gt_boxes[:, -1].int() - 1
            gt_boxes = gt_boxes[:, :7]
            ins_idxs = batch_dict['instance_idx'][ix][nonzero_mask].int()
            smpl_id = torch.from_numpy(batch_dict['frame_id'].astype(np.int32))[ix].to(gt_boxes.device)

            # filter out gt instances with too few points when updating the bank
            num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(points.cpu(), gt_boxes.cpu()).sum(dim=-1)
            valid_gts_mask = (num_points_in_gt >= num_points_threshold)
            # print(f"{(~valid_gts_mask).sum()} gt instance(s) with id(s) {ins_idxs[~valid_gts_mask].tolist()}"
            #       f" and num points {num_points_in_gt[~valid_gts_mask].tolist()} are filtered")
            if valid_gts_mask.sum() == 0:
                print(f"no valid gt instances with enough points in frame {batch_dict['frame_id'][ix]}")
                continue
            bank_inputs['feats'].append(gt_feat[valid_gts_mask])
            bank_inputs['labels'].append(gt_labels[valid_gts_mask])
            bank_inputs['ins_ids'].append(ins_idxs[valid_gts_mask])
            bank_inputs['smpl_ids'].append(smpl_id)

            # valid_boxes = gt_boxes[valid_gts_mask]
            # valid_box_labels = gt_labels[valid_gts_mask]
            # self.vis(valid_boxes, valid_box_labels, points)

        return bank_inputs

    def forward(self, batch_dict):
        if self.training:
            return self._forward_training(batch_dict)

        for cur_module in self.pv_rcnn.module_list:
            batch_dict = cur_module(batch_dict)
        pred_dicts, recall_dicts = self.pv_rcnn.post_processing(batch_dict)

        return pred_dicts, recall_dicts, {}

    def _rectify_pl_scores(self, batch_dict_ema, unlabeled_inds):
        thresh_reg = self.thresh_registry.get(tag='pl_adaptive_thresh')
        pred_wa = torch.sigmoid(batch_dict_ema['batch_cls_preds']).detach().clone()
        # to be used later for updating the EMA (p_model/p_target)
        pred_weak_aug_before_nms_org = pred_wa.clone()
        if thresh_reg.iteration_count > 0:
            pred_wa_ulb = pred_wa[unlabeled_inds, ...]
            pred_wa_ulb_aligned = pred_wa_ulb * thresh_reg.ema_pred_wa_lab / (thresh_reg.ema_pred_wa_ulb + 1e-6)
            pred_wa_ulb_aligned = thresh_reg.normalize_(pred_wa_ulb_aligned)
            pred_wa[unlabeled_inds, ...] = pred_wa_ulb_aligned

        batch_dict_ema['batch_cls_preds_org'] = pred_weak_aug_before_nms_org
        batch_dict_ema['batch_cls_preds'] = pred_wa
        batch_dict_ema['cls_preds_normalized'] = True

    def _gen_pseudo_labels(self, batch_dict_ema, ulb_inds):
        with torch.no_grad():
            # self.pv_rcnn_ema.eval()  # https://github.com/yezhen17/3DIoUMatch-PVRCNN/issues/6
            for cur_module in self.pv_rcnn_ema.module_list:
                try:
                    batch_dict_ema = cur_module(batch_dict_ema, test_only=True)
                except TypeError as e:
                    batch_dict_ema = cur_module(batch_dict_ema)

        if self.model_cfg.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE', False):
            self._rectify_pl_scores(batch_dict_ema, ulb_inds)

        pseudo_labels, _ = self.pv_rcnn_ema.post_processing(batch_dict_ema, no_recall_dict=True)

        return batch_dict_ema,pseudo_labels

    @staticmethod
    def _split_ema_batch(batch_dict):
        batch_dict_ema = {}
        keys = list(batch_dict.keys())
        for k in keys:
            if f'{k}_ema' in keys:
                continue
            if k.endswith('_ema'):
                batch_dict_ema[k[:-4]] = batch_dict[k]
            else:
                batch_dict_ema[k] = batch_dict[k]
        return batch_dict_ema

    @staticmethod
    def _prep_batch_dict(batch_dict):
        labeled_mask = batch_dict['labeled_mask'].view(-1)
        labeled_inds = torch.nonzero(labeled_mask).squeeze(1).long()
        unlabeled_inds = torch.nonzero(1 - labeled_mask).squeeze(1).long()
        batch_dict['unlabeled_inds'] = unlabeled_inds
        batch_dict['labeled_inds'] = labeled_inds
        batch_dict['ori_unlabeled_boxes'] = batch_dict['gt_boxes'][unlabeled_inds, ...].clone().detach()
        batch_dict['ori_unlabeled_boxes_ema'] = batch_dict['gt_boxes_ema'][unlabeled_inds, ...].clone().detach()
        return labeled_inds, unlabeled_inds

    def _forward_training(self, batch_dict):
        lbl_inds, ulb_inds = self._prep_batch_dict(batch_dict)
        batch_dict_ema = self._split_ema_batch(batch_dict)

        batch_dict_ema,pseudo_labels = self._gen_pseudo_labels(batch_dict_ema, ulb_inds)
        batch_dict_prefilter = copy.deepcopy(batch_dict_ema)
        dump_stats_prefilter = self.update_metrics_pred(targets_dict=batch_dict_prefilter,pseudo_labels=pseudo_labels)
        pseudo_boxes, pseudo_scores, pseudo_sem_scores,pseudo_sem_scores_multiclass,pseudo_sim_scores,pseudo_instance_sim_scores = self._filter_pseudo_labels(pseudo_labels, ulb_inds)
        self._fill_with_pseudo_labels(batch_dict, pseudo_boxes, pseudo_scores, pseudo_sem_scores_multiclass, pseudo_sim_scores,pseudo_instance_sim_scores, ulb_inds, lbl_inds)
        
        # apply student's augs on teacher's pseudo-labels (filtered) only (not points)
        batch_dict = self.apply_augmentation(batch_dict, batch_dict, ulb_inds, key='gt_boxes')
        self.update_metrics_pl(targets_dict=batch_dict)

        for cur_module in self.pv_rcnn.module_list:
            batch_dict = cur_module(batch_dict)

        if self.model_cfg.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE', False):
            pred_strong_aug_before_nms_org = torch.sigmoid(batch_dict['batch_cls_preds']).detach().clone()
            pred_dicts_std, recall_dicts_std = self.pv_rcnn_ema.post_processing(batch_dict, no_recall_dict=True)

            metrics_input = defaultdict(list)
            for ind in range(len(pred_dicts_std)):
                batch_type = 'unlab' if ind in ulb_inds else 'lab'
                metrics_input[f'pred_weak_aug_{batch_type}_before_nms'].append(batch_dict_ema['batch_cls_preds_org'][ind])
                metrics_input[f'pred_weak_aug_{batch_type}_after_nms'].append(pseudo_labels[ind]['pred_scores'].clone().detach())
                metrics_input[f'pred_strong_aug_{batch_type}_before_nms'].append(pred_strong_aug_before_nms_org[ind])
                metrics_input[f'pred_strong_aug_{batch_type}_after_nms'].append(pred_dicts_std[ind]['pred_scores'].clone().detach())
            self.thresh_registry.get(tag='pl_adaptive_thresh').update(**metrics_input)

        # Update the bank with student's features from augmented labeled data
        bank = feature_bank_registry.get('gt_aug_lbl_prototypes')
        bank.output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
        sa_gt_lbl_inputs = self._prep_bank_inputs(batch_dict_ema, lbl_inds, bank.num_points_thresh)
        bank.update(**sa_gt_lbl_inputs, iteration=batch_dict['cur_iteration'])

        # For metrics calculation
        self.pv_rcnn.roi_head.forward_ret_dict['unlabeled_inds'] = ulb_inds

        if self.model_cfg['ROI_HEAD'].get('ENABLE_SOFT_TEACHER', False):
            # using teacher to evaluate student's bg/fg proposals through its rcnn head
            with torch.no_grad():
                self._add_teacher_scores(batch_dict, batch_dict_ema, ulb_inds)

        disp_dict = {}
        loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
        loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
        loss_rcnn_cls, loss_rcnn_box, ulb_loss_cls_dist, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)

        loss = 0
        # Use the same reduction method as the baseline model (3diou) by the default
        reduce_loss_fn = getattr(torch, self.model_cfg.REDUCE_LOSS, 'sum')
        loss += reduce_loss_fn(loss_rpn_cls[lbl_inds, ...])
        loss += reduce_loss_fn(loss_rpn_box[lbl_inds, ...]) + reduce_loss_fn(loss_rpn_box[ulb_inds, ...]) * self.unlabeled_weight
        loss += reduce_loss_fn(loss_point[lbl_inds, ...])
        loss += reduce_loss_fn(loss_rcnn_cls[lbl_inds, ...])
        loss += reduce_loss_fn(loss_rcnn_box[lbl_inds, ...])

        if self.unlabeled_supervise_cls:
            loss += reduce_loss_fn(loss_rpn_cls[ulb_inds, ...]) * self.unlabeled_weight
        if self.model_cfg['ROI_HEAD'].get('ENABLE_SOFT_TEACHER', False) or self.model_cfg.get('UNLABELED_SUPERVISE_OBJ', False):
            loss += reduce_loss_fn(loss_rcnn_cls[ulb_inds, ...]) * self.unlabeled_weight
        if self.unlabeled_supervise_refine:
            loss += reduce_loss_fn(loss_rcnn_box[ulb_inds, ...]) * self.unlabeled_weight
        if self.model_cfg['ROI_HEAD'].get('ENABLE_ULB_CLS_DIST_LOSS', False):
            loss += ulb_loss_cls_dist
        if self.model_cfg['ROI_HEAD'].get('ENABLE_INSTANCE_CONTRASTIVE_LOSS', False): 
            instance_cont_loss,classwise_instance_cont_loss = self._get_instance_contrastive_loss(batch_dict,batch_dict_ema, bank, ulb_inds)
            if instance_cont_loss is not None:
                loss += instance_cont_loss * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT']
                tb_dict['instance_cont_loss'] = instance_cont_loss.item() * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT']
                for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                    tb_dict[f'classwise_instance_cont_loss_{class_name}'] = classwise_instance_cont_loss[f'{class_name}_Pl']
        if self.model_cfg['ROI_HEAD'].get('ENABLE_KL_DIV_LOSS', False): 
            instance_cont_loss,classwise_instance_cont_loss = self._get_instance_contrastive_loss(batch_dict,batch_dict_ema, bank, ulb_inds,kl_div_loss=True)
            if instance_cont_loss is not None:
                loss += instance_cont_loss * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT']
                tb_dict['instance_cont_loss'] = instance_cont_loss.item() * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT']
                for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                    tb_dict[f'classwise_instance_cont_loss_{class_name}'] = classwise_instance_cont_loss[f'{class_name}_Pl']
        if self.model_cfg['ROI_HEAD'].get('ENABLE_MEAN_INSTANCE_CONTRASTIVE_LOSS', False): 
            instance_cont_loss,classwise_instance_cont_loss = self._get_instance_contrastive_loss(batch_dict,batch_dict_ema, bank, ulb_inds,mean_instance=True)
            if instance_cont_loss is not None:
                loss += instance_cont_loss * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT']
                tb_dict['instance_cont_loss'] = instance_cont_loss.item() * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT']
                for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                    tb_dict[f'classwise_instance_cont_loss_{class_name}'] = classwise_instance_cont_loss[f'{class_name}_Pl']   

        if self.model_cfg['ROI_HEAD'].get('ENABLE_PROTO_SIM_LOSS', False):
            proto_sim_loss,classwise_proto_sim = self._get_instance_contrastive_loss(batch_dict,batch_dict_ema, bank, ulb_inds,proto_sim=True)
            if proto_sim_loss is not None:
                loss += proto_sim_loss * self.model_cfg['ROI_HEAD']['PROTO_SIM_LOSS_WEIGHT']
                tb_dict['proto_sim_loss'] = proto_sim_loss.item()
                for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                    tb_dict[f'classwise_proto_sim_loss_{class_name}'] = classwise_proto_sim[f'{class_name}_Pl']  

        if self.model_cfg['ROI_HEAD'].get('ENABLE_MCONT_LOSS', False):
            mCont_labeled = bank._get_multi_cont_loss()
            if mCont_labeled is not None:
                loss += mCont_labeled['total_loss'] * self.model_cfg['ROI_HEAD']['MCONT_LOSS_WEIGHT'] 
                tb_dict['mCont_loss'] = mCont_labeled['total_loss'].item()
                for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                    tb_dict[f'mCont_{class_name}_proto'] = mCont_labeled['classwise_loss'][cind].item()
                if self.model_cfg.get('STORE_RAW_SIM_IN_PKL', False):
                    self.mcont_dict['logits'].append(mCont_labeled['raw_logits'].tolist())
                    self.mcont_dict['iteration'].append(batch_dict['cur_iteration'])
                    output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                    file_path = os.path.join(output_dir, 'mcont_raw_logits.pkl')
                    pickle.dump(self.mcont_dict, open(file_path, 'wb'))

        if self.model_cfg['ROI_HEAD'].get('ENABLE_BATCH_MCONT', False):
            selected_batch_dict = self._clone_gt_boxes_and_feats(batch_dict)
            with torch.no_grad():
                batch_gt_feats = self.pv_rcnn_ema.roi_head.pool_features(selected_batch_dict, use_gtboxes=True)
                batch_size_rcnn = batch_gt_feats.shape[0]
                shared_features = self.pv_rcnn_ema.roi_head.shared_fc_layer(batch_gt_feats.view(batch_size_rcnn, -1, 1))
            batch_gt_feats = shared_features.view(*batch_dict['gt_boxes'].shape[:2], -1)
            batch_gt_feats_lb = batch_gt_feats[batch_dict['labeled_mask'].bool()]  
            batch_gt_labels_lb = batch_dict['gt_boxes'][batch_dict['labeled_mask'].bool()][:,:,-1].long()
            batch_gt_feats_lb = torch.cat([batch_gt_feats_lb,batch_gt_labels_lb.unsqueeze(-1)],dim=-1)
            gathered_tensor = self.gather_tensors(batch_gt_feats_lb)
            gathered_labels = gathered_tensor[:,-1].long()
            non_zero_mask = gathered_labels != 0
            gathered_feats = gathered_tensor[:,:-1][non_zero_mask]
            gathered_labels = gathered_labels[non_zero_mask]
            mCont_labeled_features = bank._get_multi_cont_loss_lb_instances(gathered_feats,gathered_labels)
            mCont_labeled_loss =  (mCont_labeled_features['total_loss'] * self.model_cfg['ROI_HEAD']['MCONT_LOSS_WEIGHT'])

            if dist.is_initialized():
                loss+= (mCont_labeled_loss/2)
                tb_dict['mCont_loss_instance'] = (mCont_labeled_features['total_loss'].item())     
            else:
                loss+= mCont_labeled_loss
                tb_dict['mCont_loss_instance'] = (mCont_labeled_features['total_loss'].item()/2) 

            for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                tb_dict[f'mCont_{class_name}_lb_inst'] = mCont_labeled_features['classwise_loss'][cind].item()
                if self.model_cfg.get('STORE_RAW_SIM_IN_PKL', False):
                    self.mcont_dict['logits'].append(mCont_labeled_features['raw_logits'].tolist())
                    self.mcont_dict['iteration'].append(batch_dict['cur_iteration'])
                    output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                    file_path = os.path.join(output_dir, 'mcont_lb_instances_logits.pkl')
                    pickle.dump(self.mcont_dict, open(file_path, 'wb'))

        if self.model_cfg['ROI_HEAD'].get('ENABLE_PROTO_CONTRASTIVE_LOSS', False):
            proto_cont_loss = self._get_proto_contrastive_loss(batch_dict, bank, ulb_inds)
            if proto_cont_loss is not None:
                loss += proto_cont_loss * self.model_cfg['ROI_HEAD']['PROTO_CONTRASTIVE_LOSS_WEIGHT']
                tb_dict['proto_cont_loss'] = proto_cont_loss.item()

        tb_dict_ = self._prep_tb_dict(tb_dict, lbl_inds, ulb_inds, reduce_loss_fn)

        if self.model_cfg.get('STORE_SCORES_IN_PKL', False):
            batch_dict['roi_sim_scores'] = self.pv_rcnn.roi_head.forward_ret_dict['roi_sim_scores']
            batch_dict_ema['prefilter_pls'] = dump_stats_prefilter
            self.dump_statistics(batch_dict, batch_dict_ema, ulb_inds)

        for tag in feature_bank_registry.tags():
            feature_bank_registry.get(tag).compute()

        # update dynamic thresh results
        for tag in self.thresh_registry.tags():
            if results := self.thresh_registry.get(tag).compute():
                tag = f"{tag}/" if tag else ''
                tb_dict_.update({tag + key: val for key, val in results.items()})

        for tag in metrics_registry.tags():
            results = metrics_registry.get(tag).compute()
            if results is not None:
                tb_dict_.update({f"{tag}/{k}": v for k, v in zip(*results)})

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict_, disp_dict

    def _get_proto_contrastive_loss(self, batch_dict, bank, ulb_inds):
        gt_boxes = batch_dict['gt_boxes']
        B, N = gt_boxes.shape[:2]
        sa_pl_feats = self.pv_rcnn.roi_head.pool_features(batch_dict, use_gtboxes=True).view(B * N, -1)
        batch_size_rcnn = sa_pl_feats.shape[0]
        shared_features = self.pv_rcnn.roi_head.shared_fc_layer(sa_pl_feats.view(batch_size_rcnn, -1, 1)) 
        shared_features = shared_features.squeeze(-1)     
        pl_labels = batch_dict['gt_boxes'][..., -1].view(-1).long() - 1
        proto_cont_loss = bank.get_proto_contrastive_loss(shared_features, pl_labels)
        if proto_cont_loss is None:
            return
        nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
        ulb_nonzero_mask = nonzero_mask[ulb_inds]
        filter_thresh = self.model_cfg['ROI_HEAD']['PL_PROTO_CONTRASTIVE_THRESH']
        valid_pl = gt_boxes[...,-1][ulb_inds][ulb_nonzero_mask].long().unsqueeze(-1) - 1
        clswise_filter_thresh = torch.tensor(filter_thresh,device=valid_pl.device).unsqueeze(0).repeat(valid_pl.shape[0],1).gather(index=(valid_pl),dim=1).squeeze(-1)
        valid_filtered_pls = (batch_dict['pseudo_scores'][ulb_inds][ulb_nonzero_mask] >= clswise_filter_thresh)
                
        if ulb_nonzero_mask.sum() == 0 or valid_filtered_pls.sum() == 0:
            print(f"No pl instances predicted for strongly augmented frame(s) {batch_dict['frame_id'][ulb_inds.cpu().numpy()]}")
            return
        return proto_cont_loss.view(B, N)[ulb_inds][ulb_nonzero_mask][valid_filtered_pls].mean()

    def _get_instance_contrastive_loss(self,batch_dict,batch_dict_ema,bank,ulb_inds,mean_instance=False,proto_sim=False,kl_div_loss=False): #TODO: Deepika: Refactor this function
        batch_dict_wa_gt = {'unlabeled_inds': batch_dict['unlabeled_inds'],
                          'labeled_inds': batch_dict['labeled_inds'],
                          'rois': batch_dict['rois'].data.clone(),
                          'roi_scores': batch_dict['roi_scores'].data.clone(),
                          'roi_labels': batch_dict['roi_labels'].data.clone(),
                          'has_class_labels': batch_dict['has_class_labels'],
                          'batch_size': batch_dict['batch_size'],
                          'gt_boxes': batch_dict['gt_boxes'].data.clone(),
                          # using teacher features
                          'point_features': batch_dict_ema['point_features'].data.clone(),
                          'point_coords': batch_dict_ema['point_coords'].data.clone(),
                          'point_cls_scores': batch_dict_ema['point_cls_scores'].data.clone(),
                          
        }
        batch_dict_wa_gt = self.reverse_augmentation(batch_dict_wa_gt, batch_dict, ulb_inds,key='gt_boxes')
        gt_boxes = batch_dict['gt_boxes']
        B, N = gt_boxes.shape[:2]
        gt_labels = gt_boxes[..., -1].view(B,N).long() - 1
        with torch.no_grad():
            batch_gt_feats_wa = self.pv_rcnn_ema.roi_head.pool_features(batch_dict_wa_gt, use_gtboxes=True)
            batch_size_rcnn = batch_gt_feats_wa.shape[0]
            shared_features_wa = self.pv_rcnn_ema.roi_head.shared_fc_layer(batch_gt_feats_wa.view(batch_size_rcnn, -1, 1)).squeeze(-1)
            shared_features_wa = shared_features_wa.view(*batch_dict['gt_boxes'].shape[:2], -1)

        batch_gt_feats_sa = self.pv_rcnn.roi_head.pool_features(batch_dict, use_gtboxes=True)
        batch_size_rcnn = batch_gt_feats_sa.shape[0]
        shared_features_sa = self.pv_rcnn.roi_head.shared_fc_layer(batch_gt_feats_sa.view(batch_size_rcnn, -1, 1)).squeeze(-1)
        shared_features_sa = shared_features_sa.view(*batch_dict['gt_boxes'].shape[:2], -1)

        assert batch_gt_feats_sa.shape[0] == batch_gt_feats_wa.shape[0], "batch_dict  mismatch"

        if proto_sim == True:
            shared_features_wa_ulb = shared_features_wa[ulb_inds]
            shared_features_sa_ulb = shared_features_sa[ulb_inds]
            proto_sim_loss = bank.get_proto_sim_loss(shared_features_wa_ulb,shared_features_sa_ulb)
            if proto_sim_loss is None:
                return None,None
            ulb_nonzero_mask = torch.logical_not(torch.eq(gt_boxes[ulb_inds], 0).all(dim=-1))

            pseudo_scores = batch_dict['pseudo_scores'][ulb_inds][ulb_nonzero_mask]
            pseudo_conf_thresh = self.model_cfg['ROI_HEAD']['PL_PROTO_SIM_THRESH']
            valid_pl = gt_boxes[ulb_inds][ulb_nonzero_mask][:,-1].long().unsqueeze(-1) 
            clswise_pseudo_thresh = torch.tensor(pseudo_conf_thresh,device=valid_pl.device).unsqueeze(0).repeat(valid_pl.shape[0],1).gather(index=(valid_pl-1),dim=1).squeeze(-1)
            valid_pls = (pseudo_scores >= clswise_pseudo_thresh)
            if ulb_nonzero_mask.sum() == 0 or valid_pls.sum() == 0:
                print(f"No pl instances predicted for strongly augmented frame(s) {batch_dict['frame_id'][ulb_inds.cpu().numpy()]}")
                return None,None
            proto_sim_loss_total = proto_sim_loss['total_loss'].view(shared_features_sa_ulb.shape[0], N,3)
            proto_loss = proto_sim_loss_total[ulb_nonzero_mask][valid_pls].sum(-1).mean()
            
            Car_instance_proto_loss =  (proto_sim_loss_total[:,:,0])[ulb_nonzero_mask] 
            Ped_instance_proto_loss =  (proto_sim_loss_total[:,:,1])[ulb_nonzero_mask]
            Cyc_instance_proto_loss =  (proto_sim_loss_total[:,:,2])[ulb_nonzero_mask]
            classwise_loss = {'Car_Pl':{},'Pedestrian_Pl':{},'Cyclist_Pl':{}}
            for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                classwise_loss[f'{class_name}_Pl'] = { #TODO: Deepika: Bugfix the weights
                        'Car_proto': Car_instance_proto_loss[gt_labels[ulb_inds][ulb_nonzero_mask]==cind].mean().item() * self.model_cfg['ROI_HEAD']['PROTO_SIM_LOSS_WEIGHT'],
                        'Ped_proto': Ped_instance_proto_loss[gt_labels[ulb_inds][ulb_nonzero_mask]==cind].mean().item() * self.model_cfg['ROI_HEAD']['PROTO_SIM_LOSS_WEIGHT'],
                        'Cyc_proto': Cyc_instance_proto_loss[gt_labels[ulb_inds][ulb_nonzero_mask]==cind].mean().item() * self.model_cfg['ROI_HEAD']['PROTO_SIM_LOSS_WEIGHT'],
                }
            self.loss_dict['cos_sim_pl_wa'].append(proto_sim_loss['cos_sim_wa'].tolist())
            self.loss_dict['cos_sim_pl_sa'].append(proto_sim_loss['cos_sim_sa'].tolist())
            self.loss_dict['pl_labels'].append(gt_labels[ulb_inds][ulb_nonzero_mask].tolist())
            if self.model_cfg.get('STORE_RAW_SIM_IN_PKL', False):
                output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                file_path = os.path.join(output_dir, 'cos_sim.pkl')
                pickle.dump(self.loss_dict, open(file_path, 'wb'))

            return proto_loss,classwise_loss


        if mean_instance == False or kl_div_loss == True:
            if kl_div_loss == True:
                instance_cont_tuple = bank.get_kl_divergence_loss(shared_features_wa,shared_features_sa)
            else:
                instance_cont_tuple = bank.get_simmatch_loss(shared_features_wa,shared_features_sa) # normal_simmatch_loss
            if instance_cont_tuple is None:
                return None,None
            nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
            ulb_nonzero_mask = nonzero_mask[ulb_inds]
            filter_thresh = self.model_cfg['ROI_HEAD']['PL_PROTO_CONTRASTIVE_THRESH']
            valid_pl = gt_boxes[...,-1][ulb_inds][ulb_nonzero_mask].long().unsqueeze(-1) - 1
            clswise_filter_thresh = torch.tensor(filter_thresh,device=valid_pl.device).unsqueeze(0).repeat(valid_pl.shape[0],1).gather(index=(valid_pl),dim=1).squeeze(-1)
            valid_filtered_pls = (batch_dict['pseudo_scores'][ulb_inds][ulb_nonzero_mask] >= clswise_filter_thresh)
            if ulb_nonzero_mask.sum() == 0 or valid_filtered_pls.sum() == 0:
                print(f"No pl instances predicted for strongly augmented frame(s) {batch_dict['frame_id'][ulb_inds.cpu().numpy()]}")
                return None,None
            loss_labels = instance_cont_tuple[1]
            instance_cont_tuple[0] = instance_cont_tuple[0].view(B, N, -1)
            instance_cont_sum = instance_cont_tuple[0].sum(-1)# calculates sum of all terms of CE for a particular instance
            instance_cont_loss = instance_cont_sum[ulb_inds][ulb_nonzero_mask][valid_filtered_pls].mean()# mean of all instances           
            cos_sim_wa = instance_cont_tuple[2].view(B, N, -1)
            cos_sim_sa = instance_cont_tuple[3].view(B, N, -1)
             # metrics update
            Car_instance_proto_loss = instance_cont_tuple[0][:,:,loss_labels==0][ulb_inds][ulb_nonzero_mask].sum(-1)
            Ped_instance_proto_loss =  instance_cont_tuple[0][:,:,loss_labels==1][ulb_inds][ulb_nonzero_mask].sum(-1)
            Cyc_instance_proto_loss = instance_cont_tuple[0][:,:,loss_labels==2][ulb_inds][ulb_nonzero_mask].sum(-1)
            classwise_loss = {'Car_Pl':{},'Pedestrian_Pl':{},'Cyclist_Pl':{}}
            for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                classwise_loss[f'{class_name}_Pl'] = {
                        'Car_proto': Car_instance_proto_loss[gt_labels[ulb_inds][ulb_nonzero_mask]==cind].mean().item() * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT'],
                        'Ped_proto': Ped_instance_proto_loss[gt_labels[ulb_inds][ulb_nonzero_mask]==cind].mean().item() * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT'],
                        'Cyc_proto': Cyc_instance_proto_loss[gt_labels[ulb_inds][ulb_nonzero_mask]==cind].mean().item() * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT'],
                }

            if self.model_cfg.get('STORE_RAW_SIM_IN_PKL', False):
                self.loss_dict['proto_labels'] = loss_labels.tolist()
                self.loss_dict['pl_labels'].append(gt_labels[ulb_inds][ulb_nonzero_mask].tolist())
                self.loss_dict['cos_sim_pl_wa'].append(cos_sim_wa[ulb_inds][ulb_nonzero_mask].tolist())
                self.loss_dict['cos_sim_pl_sa'].append(cos_sim_sa[ulb_inds][ulb_nonzero_mask].tolist())
                output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                file_path = os.path.join(output_dir, 'cos_sim.pkl')
                pickle.dump(self.loss_dict, open(file_path, 'wb'))
            return instance_cont_loss,classwise_loss
        else:
            instance_cont_tuple = bank.get_simmatch_mean_loss(shared_features_wa,shared_features_sa,ulb_inds) # mean_simmatch_loss, instead of logits for each instance, classwise logits are used. 
            if instance_cont_tuple is None:
                return None,None
            nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
            ulb_nonzero_mask = nonzero_mask[ulb_inds]
            if ulb_nonzero_mask.sum() == 0:
                print(f"No pl instances predicted for strongly augmented frame(s) {batch_dict['frame_id'][ulb_inds.cpu().numpy()]}")
                return None,None
            
            loss_labels = instance_cont_tuple[1]
            instance_cont_tuple[0] = instance_cont_tuple[0].view(B, N, -1)
            instance_cont_sum = instance_cont_tuple[0].sum(-1) # calculates sum of all terms of CE for a particular instance
            instance_cont_loss = instance_cont_sum[ulb_inds][ulb_nonzero_mask].mean() # mean of all instances
            
            # metrics update
            Car_instance_proto_loss = instance_cont_tuple[0][:,:,0][ulb_inds][ulb_nonzero_mask].mean(-1) 
            Ped_instance_proto_loss =  instance_cont_tuple[0][:,:,1][ulb_inds][ulb_nonzero_mask].mean(-1)
            Cyc_instance_proto_loss = instance_cont_tuple[0][:,:,2][ulb_inds][ulb_nonzero_mask].mean(-1)
            classwise_loss = {'Car_Pl':{},'Pedestrian_Pl':{},'Cyclist_Pl':{}}
            for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                classwise_loss[f'{class_name}_Pl'] = {
                        'Car_proto': Car_instance_proto_loss * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT'],
                        'Ped_proto': Ped_instance_proto_loss * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT'],
                        'Cyc_proto': Cyc_instance_proto_loss * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT'],
                }
            self.loss_dict['cos_sim_pl_wa'].append(instance_cont_tuple[2].tolist())
            self.loss_dict['cos_sim_pl_sa'].append(instance_cont_tuple[3].tolist())
            if self.model_cfg.get('STORE_RAW_SIM_IN_PKL', False):
                output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                file_path = os.path.join(output_dir, 'cos_sim.pkl')
                pickle.dump(self.loss_dict, open(file_path, 'wb'))
            
            return instance_cont_loss,classwise_loss

    def gather_tensors(self,tensor,labels=False):
            """
            Returns the gathered tensor to all GPUs in DDP else returns the tensor as such
            dist.gather_all needs the gathered tensors to be of same size.
            We get the sizes of the tensors first, zero pad them to match the size
            Then gather and filter the padding

            Args:
                tensor: tensor to be gathered
                labels: bool True if the tensor represents label information TODO:Deepika Remove this arg and make function tensor agnostic 
            """

            assert tensor.ndim == 3,"features should be of shape N,1,256"
            tensor = tensor.view(-1,257)
            
            if not dist.is_initialized():
                return tensor
                # Determine sizes first
            WORLD_SIZE = dist.get_world_size()
            local_size = torch.tensor(tensor.size(), device=tensor.device)
            all_sizes = [torch.zeros_like(local_size) for _ in range(WORLD_SIZE)]
            
            dist.barrier() 
            dist.all_gather(all_sizes,local_size)
            dist.barrier()

            print(f'all_sizes {all_sizes}')
            # make zero-padded version https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of
            max_length = max([size[0] for size in all_sizes])
            # print(f'max_length {max_length}')
            
            diff = max_length - local_size[0].item()
            if diff:
                pad_size =[diff.item()] #pad with zeros 
                if local_size.ndim >= 1:
                    pad_size.extend(dimension.item() for dimension in local_size[1:])
                padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
                # print(f'size of padding {padding.shape} in device {padding.device}')
                tensor = torch.cat((tensor,padding),)
            # print(f'all tensors after padding {tensor.shape} in devce {tensor.device}')
            all_tensors_padded = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]

            dist.barrier()
            dist.all_gather(all_tensors_padded,tensor)
            dist.barrier()

            gathered_tensor = torch.cat(all_tensors_padded)
            print(f'gathered tensor {gathered_tensor.shape} in devce {gathered_tensor.device}')
            non_zero_mask = torch.any(gathered_tensor!=0,dim=-1).squeeze()
            gathered_tensor = gathered_tensor[non_zero_mask]
            return gathered_tensor

            
    # def evenly_divisible_all_gather(self,data: torch.Tensor):
    #     """
    #     Utility function for distributed data parallel to pad tensor to make it evenly divisible for all_gather.
    #     Args:
    #         data: source tensor to pad and execute all_gather in distributed data parallel.

    #     """
    #     if idist.get_world_size() <= 1:
    #         return data
    #     # make sure the data is evenly-divisible on multi-GPUs
    #     length = data.shape[0]
    #     all_lens = idist.all_gather(length)
    #     max_len = max(all_lens).item()
    #     if length < max_len:
    #         size = [max_len - length] + list(data.shape[1:])
    #         data = torch.cat([data, data.new_full(size, float("NaN"))], dim=0)
    #     # all gather across all processes
    #     data = idist.all_gather(data)
    #     # delete the padding NaN items
    #     return torch.cat([data[i * max_len : i * max_len + l, ...] for i, l in enumerate(all_lens)], dim=0)

    @staticmethod
    def _prep_tb_dict(tb_dict, lbl_inds, ulb_inds, reduce_loss_fn):
        tb_dict_ = {}
        ignore_keys = ['proto_cont_loss','instance_cont_loss','classwise_instance_cont_loss_Car','classwise_instance_cont_loss_Pedestrian','classwise_instance_cont_loss_Cyclist',
                        'mCont_loss','mCont_Car_lb','mCont_Pedestrian_lb','mCont_Cyclist_lb','mCont_loss_instance','mCont_Car_lb_inst','mCont_Pedestrian_lb_inst','mCont_Cyclist_lb_inst',
                        'classwise_proto_sim_loss_Car','classwise_proto_sim_loss_Pedestrian','classwise_proto_sim_loss_Cyclist','proto_sim_loss']
        for key in tb_dict.keys():
            if key in ignore_keys:
                tb_dict_[key] = tb_dict[key]
            elif 'loss' in key or 'acc' in key or 'point_pos_num' in key:
                tb_dict_[f"{key}_labeled"] = reduce_loss_fn(tb_dict[key][lbl_inds, ...])
                tb_dict_[f"{key}_unlabeled"] = reduce_loss_fn(tb_dict[key][ulb_inds, ...])
            else:
                tb_dict_[key] = tb_dict[key]

        return tb_dict_

    def _add_teacher_scores(self, batch_dict, batch_dict_ema, ulb_inds):
        batch_dict_std = {'unlabeled_inds': batch_dict['unlabeled_inds'],
                          'labeled_inds': batch_dict['labeled_inds'],
                          'rois': batch_dict['rois'].data.clone(),
                          'roi_scores': batch_dict['roi_scores'].data.clone(),
                          'roi_labels': batch_dict['roi_labels'].data.clone(),
                          'has_class_labels': batch_dict['has_class_labels'],
                          'batch_size': batch_dict['batch_size'],
                          # using teacher features
                          'point_features': batch_dict_ema['point_features'].data.clone(),
                          'point_coords': batch_dict_ema['point_coords'].data.clone(),
                          'point_cls_scores': batch_dict_ema['point_cls_scores'].data.clone()
        }

        batch_dict_std = self.reverse_augmentation(batch_dict_std, batch_dict, ulb_inds)

        # Perturb Student's ROIs before using them for Teacher's ROI head
        if self.model_cfg.ROI_HEAD.ROI_AUG.get('ENABLE', False):
            augment_rois = getattr(augmentor_utils, self.model_cfg.ROI_HEAD.ROI_AUG.AUG_TYPE, augmentor_utils.roi_aug_ros)
            # rois_before_aug is used only for debugging, can be removed later
            batch_dict_std['rois_before_aug'] = batch_dict_std['rois'].clone().detach()
            batch_dict_std['rois'][ulb_inds] = augment_rois(batch_dict_std['rois'][ulb_inds], self.model_cfg.ROI_HEAD)

        self.pv_rcnn_ema.roi_head.forward(batch_dict_std, test_only=True)
        batch_dict_std = self.apply_augmentation(batch_dict_std, batch_dict, ulb_inds, key='batch_box_preds')
        pred_dicts_std, recall_dicts_std = self.pv_rcnn_ema.post_processing(batch_dict_std,
                                                                            no_recall_dict=True,
                                                                            no_nms_for_unlabeled=True)
        rcnn_cls_score_teacher = -torch.ones_like(self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_labels'])
        batch_box_preds_teacher = torch.zeros_like(self.pv_rcnn.roi_head.forward_ret_dict['batch_box_preds'])
        for uind in ulb_inds:
            rcnn_cls_score_teacher[uind] = pred_dicts_std[uind]['pred_scores']
            batch_box_preds_teacher[uind] = pred_dicts_std[uind]['pred_boxes']

        self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_score_teacher'] = rcnn_cls_score_teacher
        self.pv_rcnn.roi_head.forward_ret_dict['batch_box_preds_teacher'] = batch_box_preds_teacher # for metrics

    @staticmethod
    def vis(boxes, box_labels, points):
        boxes = boxes.cpu().numpy()
        points = points.cpu().numpy()
        box_labels = box_labels.cpu().numpy()
        V.draw_scenes(points=points, gt_boxes=boxes, gt_labels=box_labels)

    def dump_statistics(self, batch_dict, batch_dict_ema, unlabeled_inds):
        # Store different types of scores over all itrs and epochs and dump them in a pickle for offline modeling
        # TODO (shashank) : Can be optimized later to save computational time, currently takes about 0.002sec
        batch_roi_labels = self.pv_rcnn.roi_head.forward_ret_dict['roi_labels'][unlabeled_inds]
        batch_roi_labels = [roi_labels.clone().detach() for roi_labels in batch_roi_labels]

        batch_rois = self.pv_rcnn.roi_head.forward_ret_dict['rois'][unlabeled_inds]
        batch_rois = [rois.clone().detach() for rois in batch_rois]

        batch_ori_gt_boxes = self.pv_rcnn.roi_head.forward_ret_dict['ori_unlabeled_boxes']
        batch_ori_gt_boxes = [ori_gt_boxes.clone().detach() for ori_gt_boxes in batch_ori_gt_boxes]

        batch_ori_gt_boxes_ema = batch_dict['ori_unlabeled_boxes_ema']
        batch_ori_gt_boxes_ema = [ori_gt_boxes_ema.clone().detach() for ori_gt_boxes_ema in batch_ori_gt_boxes_ema]

        batch_pls = batch_dict_ema['prefilter_pls']['gt_boxes_ema'][unlabeled_inds]
        batch_pls = [pls.clone().detach() for pls in batch_pls]
        for i in range(len(batch_pls)):
            valid_pl_boxes_mask = torch.logical_not(torch.all(batch_pls[i] == 0, dim=-1))
            valid_pls = batch_pls[i][valid_pl_boxes_mask]
            valid_pl_labels = batch_pls[i][valid_pl_boxes_mask][:, -1].int() 

            valid_gt_boxes_pl_mask = torch.logical_not(torch.all(batch_ori_gt_boxes_ema[i] == 0, dim=-1))
            valid_gt_boxes_pl = batch_ori_gt_boxes_ema[i][valid_gt_boxes_pl_mask]
            valid_gt_pl_labels = batch_ori_gt_boxes_ema[i][valid_gt_boxes_pl_mask][:, -1].int()
            num_pls = valid_pl_boxes_mask.sum()
            num_gt_pls = valid_gt_boxes_pl_mask.sum()
            cur_unlabeled_ind = unlabeled_inds[i]

            if num_pls > 0 and num_gt_pls > 0:
                # Find IoU between Student's PL v/s Teacher's GTs
                overlap = iou3d_nms_utils.boxes_iou3d_gpu(valid_pls[:, 0:7], valid_gt_boxes_pl[:, 0:7])
                pls_iou_max, assigned_gt_inds = overlap.max(dim=1)
                self.val_dict['iou_pl_gt'].extend(pls_iou_max.tolist())
                self.val_dict['assigned_gt_pl_labels'].extend(valid_gt_pl_labels[assigned_gt_inds].tolist())

                assert batch_dict_ema['prefilter_pls']['rcnn_scores_ema_prefilter'][cur_unlabeled_ind].shape[0] == valid_pl_boxes_mask.shape[0]
                assert batch_dict_ema['prefilter_pls']['pseudo_sem_scores_multiclass'][cur_unlabeled_ind].shape[0] == valid_pl_boxes_mask.shape[0]
                assert batch_dict_ema['prefilter_pls']['pseudo_sim_scores_emas_prefilter'][cur_unlabeled_ind].shape[0] == valid_pl_boxes_mask.shape[0]
                
                self.val_dict['rcnn_scores_pl'].extend((batch_dict_ema['prefilter_pls']['rcnn_scores_ema_prefilter'][cur_unlabeled_ind][valid_pl_boxes_mask]).tolist())
                self.val_dict['pseudo_sem_scores_pl'].extend((batch_dict_ema['prefilter_pls']['pseudo_sem_scores_multiclass'][cur_unlabeled_ind][valid_pl_boxes_mask]).tolist())
                self.val_dict['pseudo_sim_scores_pl'].extend((batch_dict_ema['prefilter_pls']['pseudo_sim_scores_emas_prefilter'][cur_unlabeled_ind][valid_pl_boxes_mask]).tolist())
                self.val_dict['pl_iteration'].extend((torch.ones_like(pls_iou_max) * batch_dict['cur_iteration']).tolist())
                self.val_dict['pseudo_instance_sim_scores_pl'].extend((batch_dict_ema['prefilter_pls']['pseudo_instance_sim_scores_emas_prefilter'][cur_unlabeled_ind][valid_pl_boxes_mask]).tolist())  

        for i in range(len(batch_rois)):
            valid_rois_mask = torch.logical_not(torch.all(batch_rois[i] == 0, dim=-1))
            valid_rois = batch_rois[i][valid_rois_mask]
            valid_roi_labels = batch_roi_labels[i][valid_rois_mask]
            valid_roi_labels -= 1  # Starting class indices from zero

            valid_gt_boxes_mask = torch.logical_not(torch.all(batch_ori_gt_boxes[i] == 0, dim=-1))
            valid_gt_boxes = batch_ori_gt_boxes[i][valid_gt_boxes_mask]
            valid_gt_boxes[:, -1] -= 1  # Starting class indices from zero
            num_gts = valid_gt_boxes_mask.sum()
            num_preds = valid_rois_mask.sum()

            cur_unlabeled_ind = unlabeled_inds[i]
            if num_gts > 0 and num_preds > 0:
                # Find IoU between Student's ROI v/s Original GTs
                overlap = iou3d_nms_utils.boxes_iou3d_gpu(valid_rois[:, 0:7], valid_gt_boxes[:, 0:7])
                preds_iou_max, assigned_gt_inds = overlap.max(dim=1)
                self.val_dict['iou_roi_gt'].extend(preds_iou_max.tolist())

                cur_iou_roi_pl = self.pv_rcnn.roi_head.forward_ret_dict['gt_iou_of_rois'][cur_unlabeled_ind]
                self.val_dict['iou_roi_pl'].extend(cur_iou_roi_pl.tolist())

                cur_pred_score = torch.sigmoid(batch_dict['batch_cls_preds'][cur_unlabeled_ind]).squeeze()
                self.val_dict['pred_scores'].extend(cur_pred_score.tolist())

                # if 'rcnn_cls_score_teacher' in self.pv_rcnn.roi_head.forward_ret_dict:
                #     cur_teacher_pred_score = self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_score_teacher'][
                #         cur_unlabeled_ind]
                #     self.val_dict['teacher_pred_scores'].extend(cur_teacher_pred_score.tolist())

                #     cur_weight = self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_weights'][cur_unlabeled_ind]
                #     self.val_dict['weights'].extend(cur_weight.tolist())

                cur_roi_score = torch.sigmoid(self.pv_rcnn.roi_head.forward_ret_dict['roi_scores'][cur_unlabeled_ind])
                self.val_dict['roi_scores'].extend(cur_roi_score.tolist())

                cur_roi_sim_score = self.pv_rcnn.roi_head.forward_ret_dict['roi_sim_scores'][cur_unlabeled_ind]
                self.val_dict['roi_sim_scores'].extend(cur_roi_sim_score.tolist())

                # cur_pcv_score = self.pv_rcnn.roi_head.forward_ret_dict['pcv_scores'][cur_unlabeled_ind]
                # self.val_dict['pcv_scores'].extend(cur_pcv_score.tolist())

                # cur_num_points_roi = self.pv_rcnn.roi_head.forward_ret_dict['num_points_in_roi'][cur_unlabeled_ind]
                # self.val_dict['num_points_in_roi'].extend(cur_num_points_roi.tolist())

                cur_roi_label = self.pv_rcnn.roi_head.forward_ret_dict['roi_labels'][cur_unlabeled_ind].squeeze()
                self.val_dict['class_labels'].extend(cur_roi_label.tolist())

                cur_iteration = torch.ones_like(preds_iou_max) * (batch_dict['cur_iteration'])
                self.val_dict['iteration'].extend(cur_iteration.tolist())

                cur_instance_sim_score = self.pv_rcnn.roi_head.forward_ret_dict['roi_instance_sim_scores'][cur_unlabeled_ind]
                self.val_dict['roi_instance_sim_scores'].extend(cur_instance_sim_score.tolist())
        # replace old pickle data (if exists) with updated one
        # if (batch_dict['cur_epoch']) == batch_dict['total_epochs']:
        output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
        file_path = os.path.join(output_dir, 'scores.pkl')
        pickle.dump(self.val_dict, open(file_path, 'wb'))

    # def update_metrics(self, input_dict, pred_dict, unlabeled_inds, labeled_inds):
    #     """
    #     Recording PL vs GT statistics BEFORE filtering
    #     """
    #     if 'pl_gt_metrics_before_filtering' in self.model_cfg.ROI_HEAD.METRICS_PRED_TYPES:
    #         pseudo_boxes, pseudo_labels, pseudo_scores, pseudo_sem_scores, _, _ = self._unpack_predictions(
    #             pred_dict, unlabeled_inds)
    #         pseudo_boxes = [torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1) \
    #                         for (pseudo_box, pseudo_label) in zip(pseudo_boxes, pseudo_labels)]
    #
    #         # Making consistent # of pseudo boxes in each batch
    #         # NOTE: Need to store them in batch_dict in a new key, which can be removed later
    #         input_dict['pseudo_boxes_prefilter'] = torch.zeros_like(input_dict['gt_boxes'])
    #         self._fill_with_pseudo_labels(input_dict, pseudo_boxes, unlabeled_inds, labeled_inds,
    #                                       key='pseudo_boxes_prefilter')
    #
    #         # apply student's augs on teacher's pseudo-boxes (w/o filtered)
    #         batch_dict = self.apply_augmentation(input_dict, input_dict, unlabeled_inds, key='pseudo_boxes_prefilter')
    #
    #         tag = f'pl_gt_metrics_before_filtering'
    #         metrics = metrics_registry.get(tag)
    #
    #         preds_prefilter = [batch_dict['pseudo_boxes_prefilter'][uind] for uind in unlabeled_inds]
    #         gts_prefilter = [batch_dict['gt_boxes'][uind] for uind in unlabeled_inds]
    #         metric_inputs = {'preds': preds_prefilter, 'pred_scores': pseudo_scores, 'roi_scores': pseudo_sem_scores,
    #                          'ground_truths': gts_prefilter}
    #         metrics.update(**metric_inputs)
    #         batch_dict.pop('pseudo_boxes_prefilter')

    # TODO(farzad) refactor and remove this!
    def _unpack_predictions(self, pred_dicts, unlabeled_inds):
        pseudo_boxes = []
        pseudo_scores = []
        pseudo_sem_scores = []
        pseudo_labels = []
        pseudo_boxes_var = []
        pseudo_scores_var = []
        pseudo_sem_scores_multiclass = []
        pseudo_sim_scores = []
        pseudo_instance_sim_scores = []
        for ind in unlabeled_inds:
            pseudo_score = pred_dicts[ind]['pred_scores']
            pseudo_box = pred_dicts[ind]['pred_boxes']
            pseudo_label = pred_dicts[ind]['pred_labels']
            pseudo_sem_score = pred_dicts[ind]['pred_sem_scores']
            pseudo_sem_score_multiclass = pred_dicts[ind]['pred_sem_scores_multiclass']
            pseudo_sim_score = pred_dicts[ind]['pred_sim_scores']
            pseudo_instance_sim_score = pred_dicts[ind]['pred_instance_sim_scores']
            # TODO(farzad) REFACTOR LATER!
            pseudo_box_var = -1 * torch.ones_like(pseudo_box)
            if "pred_boxes_var" in pred_dicts[ind].keys():
                pseudo_box_var = pred_dicts[ind]['pred_boxes_var']
            pseudo_score_var = -1 * torch.ones_like(pseudo_score)
            if "pred_scores_var" in pred_dicts[ind].keys():
                pseudo_score_var = pred_dicts[ind]['pred_scores_var']
            if len(pseudo_label) == 0:
                pseudo_boxes.append(pseudo_label.new_zeros((1, 7)).float())
                pseudo_boxes_var.append(pseudo_label.new_zeros((1, 7)).float())
                pseudo_sem_scores.append(pseudo_label.new_zeros((1,)).float())
                pseudo_scores.append(pseudo_label.new_zeros((1,)).float())
                pseudo_scores_var.append(pseudo_label.new_zeros((1,)).float())
                pseudo_labels.append(pseudo_label.new_zeros((1,)).float())
                pseudo_sem_scores_multiclass.append(pseudo_label.new_zeros((1,3)).float())
                pseudo_sim_scores.append(pseudo_label.new_zeros((1,3)).float())
                pseudo_instance_sim_scores.append(pseudo_label.new_zeros((1,3)).float())
                continue

            pseudo_boxes.append(pseudo_box)
            pseudo_boxes_var.append(pseudo_box_var)
            pseudo_sem_scores.append(pseudo_sem_score)
            pseudo_scores.append(pseudo_score)
            pseudo_scores_var.append(pseudo_score_var)
            pseudo_labels.append(pseudo_label)
            pseudo_sem_scores_multiclass.append(pseudo_sem_score_multiclass)
            pseudo_sim_scores.append(pseudo_sim_score)
            pseudo_instance_sim_scores.append(pseudo_instance_sim_score)

        return pseudo_boxes, pseudo_labels, pseudo_scores, pseudo_sem_scores, pseudo_boxes_var, pseudo_scores_var, pseudo_sem_scores_multiclass, pseudo_sim_scores,pseudo_instance_sim_scores

    # TODO(farzad) refactor and remove this!
    def _filter_pseudo_labels(self, pred_dicts, unlabeled_inds):
        pseudo_boxes = []
        pseudo_scores = []
        pseudo_sem_scores = []
        pseudo_sem_scores_multiclass = []
        pseudo_sim_scores = []
        pseudo_instance_sim_scores = []
        for pseudo_box, pseudo_label, pseudo_score, pseudo_sem_score, pseudo_box_var, pseudo_score_var,pseudo_sem_score_multiclass,pseudo_sim_score,pseudo_instance_sim_score in zip(
                *self._unpack_predictions(pred_dicts, unlabeled_inds)):
            assert isinstance(pseudo_sem_score_multiclass, torch.Tensor), "stupid assert"
            if pseudo_box.sum() == 0:
                pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
                pseudo_sem_scores.append(pseudo_sem_score)
                pseudo_scores.append(pseudo_score)
                assert isinstance(pseudo_sem_score_multiclass, torch.Tensor), "stupid assert"
                pseudo_sem_score_multiclass = torch.zeros((pseudo_box.shape[0], 3), device=pseudo_box.device)
                pseudo_sim_score = torch.zeros((pseudo_box.shape[0], 3), device=pseudo_box.device)
                assert pseudo_sem_score_multiclass.shape[-1] == 3, "stupid assert"
                pseudo_sem_scores_multiclass.append(pseudo_sem_score_multiclass)
                pseudo_sim_scores.append(pseudo_sim_score)
                pseudo_instance_sim_scores.append(pseudo_instance_sim_score)
                continue

            pl_thresh = self.thresh
            if self.model_cfg.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE', False):
                thresh_reg = self.thresh_registry.get(tag='pl_adaptive_thresh')
                if thresh_reg.relative_ema_threshold is not None:
                   pl_thresh = [thresh_reg.relative_ema_threshold.item()] * len(self.thresh)

            conf_thresh = torch.tensor(pl_thresh, device=pseudo_label.device).unsqueeze(
                0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label - 1).unsqueeze(-1))

            sem_conf_thresh = torch.tensor(self.sem_thresh, device=pseudo_label.device).unsqueeze(
                0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label - 1).unsqueeze(-1))

            if self.model_cfg.ROI_HEAD.HYBRID_THRESHOLDING:
                if (pseudo_sim_score.sum(-1) <= 0).any(): # until the time we have sim scores for all classes
                    valid_inds = pseudo_score > conf_thresh.squeeze()
                   
                else:
                    alpha = self.model_cfg.ROI_HEAD.HYBRID_ALPHA
                    sim_scores = torch.gather(pseudo_sim_score, dim=-1, index=(pseudo_label - 1).unsqueeze(-1)).squeeze(-1)
                    hybrid_scores = (alpha * sim_scores) + ((1 - alpha) * pseudo_score)  
                    hybrid_conf_thresh = torch.tensor(self.hybrid_thresh, device=pseudo_label.device).unsqueeze(
                        0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label - 1).unsqueeze(-1))

                    valid_inds = hybrid_scores > hybrid_conf_thresh.squeeze()
            else:
                valid_inds = pseudo_score > conf_thresh.squeeze()
            
            valid_inds = valid_inds & (pseudo_sem_score > sem_conf_thresh.squeeze())

            pseudo_sem_score = pseudo_sem_score[valid_inds]
            pseudo_box = pseudo_box[valid_inds]
            pseudo_label = pseudo_label[valid_inds]
            pseudo_score = pseudo_score[valid_inds]
            pseudo_sem_score_multiclass = pseudo_sem_score_multiclass[valid_inds]
            assert isinstance(pseudo_sem_score_multiclass, torch.Tensor), "stupid assert"
            pseudo_sim_score = pseudo_sim_score[valid_inds] if pseudo_sim_score is not None else None
            assert isinstance(pseudo_sim_score, torch.Tensor), "stupid assert"
            pseudo_instance_sim_score = pseudo_instance_sim_score[valid_inds] if pseudo_instance_sim_score is not None else None
            pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
            pseudo_sem_scores.append(pseudo_sem_score)
            pseudo_scores.append(pseudo_score)
            pseudo_sem_scores_multiclass.append(pseudo_sem_score_multiclass)
            pseudo_sim_scores.append(pseudo_sim_score)
            pseudo_instance_sim_scores.append(pseudo_instance_sim_score)

        return pseudo_boxes, pseudo_scores, pseudo_sem_scores, pseudo_sem_scores_multiclass, pseudo_sim_scores,pseudo_instance_sim_scores

    def _fill_with_pseudo_labels(self, batch_dict, pseudo_boxes, pseudo_scores, pseudo_sem_scores_multiclass, pseudo_sim_scores,pseudo_instance_sim_scores, unlabeled_inds, labeled_inds, key=None):
        key = 'gt_boxes' if key is None else key
        max_box_num = batch_dict['gt_boxes'].shape[1]
        batch_dict['pseudo_sem_scores_multiclass'] = torch.zeros((batch_dict['gt_boxes'].shape[0], max_box_num, 3), device=batch_dict['gt_boxes'].device)
        batch_dict['pseudo_sim_scores'] = torch.zeros((batch_dict['gt_boxes'].shape[0], max_box_num, 3), device=batch_dict['gt_boxes'].device)
        batch_dict['pseudo_instance_sim_scores'] = torch.zeros((batch_dict['gt_boxes'].shape[0], max_box_num, 3), device=batch_dict['gt_boxes'].device)
        batch_dict['pseudo_scores'] = torch.zeros((batch_dict['gt_boxes'].shape[0], max_box_num), device=batch_dict['gt_boxes'].device)
        # Ignore the count of pseudo boxes if filled with default values(zeros) when no preds are made
        max_pseudo_box_num = max(
            [torch.logical_not(torch.all(ps_box == 0, dim=-1)).sum().item() for ps_box in pseudo_boxes])

        if max_box_num >= max_pseudo_box_num:
            for i, pseudo_box in enumerate(pseudo_boxes):
                diff = max_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                    assert isinstance(pseudo_sem_scores_multiclass[i],torch.Tensor), "stupid assert"
                    pseudo_sem_score_multiclass = torch.cat([pseudo_sem_scores_multiclass[i], torch.zeros((diff, 3), device=pseudo_box.device)], dim=0)
                    pseudo_sim_score = torch.cat([pseudo_sim_scores[i], torch.zeros((diff, 3), device=pseudo_box.device)], dim=0) 
                    pseudo_instance_sim_score = torch.cat([pseudo_instance_sim_scores[i], torch.zeros((diff, 3), device=pseudo_box.device)], dim=0) 
                    pseudo_score = torch.cat([pseudo_scores[i], torch.zeros((diff,), device=pseudo_box.device)], dim=0)
                else:
                    pseudo_sem_score_multiclass = pseudo_sem_scores_multiclass[i]
                    pseudo_sim_score = pseudo_sim_scores[i]
                    pseudo_instance_sim_score = pseudo_instance_sim_scores[i]
                    pseudo_score = pseudo_scores[i]
                batch_dict[key][unlabeled_inds[i]] = pseudo_box
                batch_dict['pseudo_sem_scores_multiclass'][unlabeled_inds[i]] = pseudo_sem_score_multiclass
                batch_dict['pseudo_sim_scores'][unlabeled_inds[i]] = pseudo_sim_score 
                batch_dict['pseudo_instance_sim_scores'][unlabeled_inds[i]] = pseudo_instance_sim_score
                batch_dict['pseudo_scores'][unlabeled_inds[i]] = pseudo_score
        else:
            ori_boxes = batch_dict['gt_boxes']
            ori_ins_ids = batch_dict['instance_idx']
            new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[2]),
                                    device=ori_boxes.device)
            new_ins_idx = torch.full((ori_boxes.shape[0], max_pseudo_box_num), fill_value=-1, device=ori_boxes.device)
            new_sem_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, 3),device=ori_boxes.device)
            new_sim_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, 3),device=ori_boxes.device)
            new_scores = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num),device=ori_boxes.device)
            new_instance_sim_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, 3),device=ori_boxes.device)
            for idx in labeled_inds:
                diff = max_pseudo_box_num - ori_boxes[idx].shape[0]
                new_box = torch.cat([ori_boxes[idx], torch.zeros((diff, 8), device=ori_boxes[idx].device)], dim=0)
                new_boxes[idx] = new_box
                new_ins_idx[idx] = torch.cat([ori_ins_ids[idx], -torch.ones((diff,), device=ori_boxes[idx].device)], dim=0)
                new_sem_boxes[idx] = torch.cat([batch_dict['pseudo_sem_scores_multiclass'][idx], torch.zeros((diff, 3), device=ori_boxes[idx].device)], dim=0)
                new_sim_boxes[idx] = torch.cat([batch_dict['pseudo_sim_scores'][idx], torch.zeros((diff, 3), device=ori_boxes[idx].device)], dim=0) 
                new_instance_sim_boxes[idx] = torch.cat([batch_dict['pseudo_instance_sim_scores'][idx], torch.zeros((diff, 3), device=ori_boxes[idx].device)], dim=0)
                new_scores[idx] = torch.cat([batch_dict['pseudo_scores'][idx], torch.zeros((diff,), device=ori_boxes[idx].device)], dim=0)
            for i, pseudo_box in enumerate(pseudo_boxes):    
                diff = max_pseudo_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                    pseudo_sem_score_multiclass = torch.cat([pseudo_sem_scores_multiclass[i], torch.zeros((diff, 3), device=pseudo_box.device)], dim=0)
                    pseudo_sim_score = torch.cat([pseudo_sim_scores[i], torch.zeros((diff, 3), device=pseudo_box.device)], dim=0) if batch_dict['pseudo_sim_scores'][idx] is not None else None
                    pseudo_instance_sim_score = torch.cat([pseudo_instance_sim_scores[i], torch.zeros((diff, 3), device=pseudo_box.device)], dim=0) 
                    pseudo_score = torch.cat([pseudo_scores[i], torch.zeros((diff,), device=pseudo_box.device)], dim=0)
                else:
                    pseudo_sem_score_multiclass = pseudo_sem_scores_multiclass[i]
                    pseudo_sim_score = pseudo_sim_scores[i]
                    pseudo_instance_sim_score = pseudo_instance_sim_scores[i]
                    pseudo_score = pseudo_scores[i]
                new_boxes[unlabeled_inds[i]] = pseudo_box
                new_sem_boxes[unlabeled_inds[i]] = pseudo_sem_score_multiclass
                new_sim_boxes[unlabeled_inds[i]] = pseudo_sim_score
                new_instance_sim_boxes[unlabeled_inds[i]] = pseudo_instance_sim_score
                new_scores[unlabeled_inds[i]] = pseudo_score

            batch_dict[key] = new_boxes
            batch_dict['instance_idx'] = new_ins_idx
            batch_dict['pseudo_sem_scores_multiclass'] = new_sem_boxes
            batch_dict['pseudo_sim_scores'] = new_sim_boxes
            batch_dict['pseudo_instance_sim_scores'] = new_instance_sim_boxes
            batch_dict['pseudo_scores'] = new_scores

    def _fill_with_pseudo_labels_prefilter(self, batch_dict, pseudo_boxes, pseudo_sem_scores_multiclass, pseudo_sim_scores, pseudo_scores,pseudo_instance_sim_scores,unlabeled_inds, labeled_inds, key=None):
            key = 'gt_boxes' if key is None else key
            max_box_num = batch_dict['gt_boxes'].shape[1]
            batch_dict['pseudo_sem_scores_multiclass'] = torch.zeros((batch_dict['gt_boxes'].shape[0], max_box_num, 3), device=batch_dict['gt_boxes'].device) #[N,3]
            batch_dict['pseudo_sim_scores'] = torch.zeros((batch_dict['gt_boxes'].shape[0], max_box_num, 3), device=batch_dict['gt_boxes'].device)
            batch_dict['pseudo_scores'] = torch.zeros((batch_dict['gt_boxes'].shape[0], max_box_num), device=batch_dict['gt_boxes'].device)
            batch_dict['pseudo_instance_sim_scores'] = torch.zeros((batch_dict['gt_boxes'].shape[0], max_box_num, 3), device=batch_dict['gt_boxes'].device)

            # Ignore the count of pseudo boxes if filled with default values(zeros) when no preds are made
            max_pseudo_box_num = max(
                [torch.logical_not(torch.all(ps_box == 0, dim=-1)).sum().item() for ps_box in pseudo_boxes])

            if max_box_num >= max_pseudo_box_num:
                for i, pseudo_box in enumerate(pseudo_boxes):
                    diff = max_box_num - pseudo_box.shape[0]
                    if diff > 0:
                        pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        assert isinstance(pseudo_sem_scores_multiclass[i],torch.Tensor), "stupid assert"
                        pseudo_sem_score_multiclass = torch.cat([pseudo_sem_scores_multiclass[i], torch.zeros((diff, 3), device=pseudo_box.device)], dim=0)
                        pseudo_sim_score = torch.cat([pseudo_sim_scores[i], torch.zeros((diff, 3), device=pseudo_box.device)], dim=0) 
                        pseudo_score = torch.cat([pseudo_scores[i], torch.zeros((diff,), device=pseudo_box.device)], dim=0)
                        pseudo_instance_sim_score = torch.cat([pseudo_instance_sim_scores[i], torch.zeros((diff, 3), device=pseudo_box.device)], dim=0) 

                    else:
                        pseudo_sem_score_multiclass = pseudo_sem_scores_multiclass[i]
                        pseudo_sim_score = pseudo_sim_scores[i]
                        pseudo_score = pseudo_scores[i]
                        pseudo_instance_sim_score = pseudo_instance_sim_scores[i]
                    batch_dict[key][unlabeled_inds[i]] = pseudo_box
                    batch_dict['pseudo_sem_scores_multiclass'][unlabeled_inds[i]] = pseudo_sem_score_multiclass
                    batch_dict['pseudo_sim_scores'][unlabeled_inds[i]] = pseudo_sim_score 
                    batch_dict['pseudo_scores'][unlabeled_inds[i]] = pseudo_score
                    batch_dict['pseudo_instance_sim_scores'][unlabeled_inds[i]] = pseudo_instance_sim_score
                    
            else:
                ori_boxes = batch_dict['gt_boxes']
                ori_ins_ids = batch_dict['instance_idx']
                new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[2]),
                                        device=ori_boxes.device)
                new_ins_idx = torch.full((ori_boxes.shape[0], max_pseudo_box_num), fill_value=-1, device=ori_boxes.device)
                new_sem_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, 3),device=ori_boxes.device)
                new_sim_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, 3),device=ori_boxes.device)
                new_instance_sim_scores = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, 3),device=ori_boxes.device)
                new_scores = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num),device=ori_boxes.device)
                for idx in labeled_inds:
                    diff = max_pseudo_box_num - ori_boxes[idx].shape[0]
                    new_box = torch.cat([ori_boxes[idx], torch.zeros((diff, 8), device=ori_boxes[idx].device)], dim=0)
                    new_boxes[idx] = new_box
                    new_ins_idx[idx] = torch.cat([ori_ins_ids[idx], -torch.ones((diff,), device=ori_boxes[idx].device)], dim=0)
                    new_sem_boxes[idx] = torch.cat([batch_dict['pseudo_sem_scores_multiclass'][idx], torch.zeros((diff, 3), device=ori_boxes[idx].device)], dim=0)
                    new_sim_boxes[idx] = torch.cat([batch_dict['pseudo_sim_scores'][idx], torch.zeros((diff, 3), device=ori_boxes[idx].device)], dim=0) 
                    new_instance_sim_scores[idx] = torch.cat([batch_dict['pseudo_instance_sim_scores'][idx], torch.zeros((diff, 3), device=ori_boxes[idx].device)], dim=0)               
                for i, pseudo_box in enumerate(pseudo_boxes):
                    diff = max_pseudo_box_num - pseudo_box.shape[0]
                    if diff > 0:
                        pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        pseudo_sem_score_multiclass = torch.cat([pseudo_sem_scores_multiclass[i], torch.zeros((diff, 3), device=pseudo_box.device)], dim=0)
                        pseudo_sim_score = torch.cat([pseudo_sim_scores[i], torch.zeros((diff, 3), device=pseudo_box.device)], dim=0) 
                        pseudo_score = torch.cat([pseudo_scores[i], torch.zeros((diff,), device=pseudo_box.device)], dim=0)
                        pseudo_instance_sim_score = torch.cat([pseudo_instance_sim_scores[i], torch.zeros((diff, 3), device=pseudo_box.device)], dim=0)
                    else:
                        pseudo_sem_score_multiclass = pseudo_sem_scores_multiclass[i]
                        pseudo_sim_score = pseudo_sim_scores[i]
                        pseudo_score = pseudo_scores[i]
                        pseudo_instance_sim_score = pseudo_instance_sim_scores[i]
                    new_boxes[unlabeled_inds[i]] = pseudo_box
                    new_sem_boxes[unlabeled_inds[i]] = pseudo_sem_score_multiclass
                    new_sim_boxes[unlabeled_inds[i]] = pseudo_sim_score
                    new_scores[unlabeled_inds[i]] = pseudo_score
                    new_instance_sim_scores[unlabeled_inds[i]] = pseudo_instance_sim_score
                batch_dict[key] = new_boxes
                batch_dict['instance_idx'] = new_ins_idx
                batch_dict['pseudo_sem_scores_multiclass'] = new_sem_boxes
                batch_dict['pseudo_sim_scores'] = new_sim_boxes
                batch_dict['pseudo_scores'] = new_scores
                batch_dict['pseudo_instance_sim_scores'] = new_instance_sim_scores
    @staticmethod
    def apply_augmentation(batch_dict, batch_dict_org, unlabeled_inds, key='rois'):
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_x_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_x'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_y_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_y'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_rotation_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['rot_angle'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_scaling_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['scale'][unlabeled_inds])

        batch_dict[key][unlabeled_inds, :, 6] = common_utils.limit_period(
            batch_dict[key][unlabeled_inds, :, 6], offset=0.5, period=2 * np.pi
        )

        return batch_dict

    @staticmethod
    def reverse_augmentation(batch_dict, batch_dict_org, unlabeled_inds, key='rois'):
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_scaling_bbox(
            batch_dict[key][unlabeled_inds], 1.0 / batch_dict_org['scale'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_rotation_bbox(
            batch_dict[key][unlabeled_inds], - batch_dict_org['rot_angle'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_y_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_y'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_x_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_x'][unlabeled_inds])

        batch_dict[key][unlabeled_inds, :, 6] = common_utils.limit_period(
            batch_dict[key][unlabeled_inds, :, 6], offset=0.5, period=2 * np.pi
        )

        return batch_dict

    def update_global_step(self):
        self.global_step += 1
        self.accumulated_itr += 1
        if self.accumulated_itr % self.model_cfg.EMA_UPDATE_INTERVAL != 0:
            return
        alpha = self.model_cfg.EMA_ALPHA
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.pv_rcnn_ema.parameters(), self.pv_rcnn.parameters()):
            # TODO(farzad) check this
            ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)
        self.accumulated_itr = 0

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            new_key = 'pv_rcnn.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
            new_key = 'pv_rcnn_ema.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def update_adaptive_thresholding_metrics(self, pred_dicts, unlabeled_inds, tag = 'pl_adaptive_thresh'):
        metrics_input = defaultdict(list)
        for ind in unlabeled_inds:
            pseudo_score = pred_dicts[ind]['pred_scores']
            pseudo_label = pred_dicts[ind]['pred_labels']
            pseudo_sem_score = pred_dicts[ind]['pred_sem_scores']
            if len(pseudo_label):
                metrics_input['pred_labels'].append(pseudo_label)
                metrics_input['pseudo_score'].append(pseudo_score)
                metrics_input['pseudo_sem_score'].append(pseudo_sem_score)
        self.thresh_registry.get(tag).update(**metrics_input)

    def update_metrics_pred(self, targets_dict,pseudo_labels,mask_type='cls',bank=None):
        pseudo_boxes, pseudo_labels, pseudo_score, pseudo_sem_score, pseudo_box_var, pseudo_score_var,pseudo_sem_score_multiclass,pseudo_sim_score,pseudo_instance_sim_score = self._unpack_predictions(pseudo_labels, targets_dict['unlabeled_inds'])
        pseudo_boxes = [torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1) \
                            for (pseudo_box, pseudo_label) in zip(pseudo_boxes, pseudo_labels)] # add label to boxes
        # pseudo_sem_scores_multiclass = [pseudo_sem_score_multiclass]
        # pseudo_sim_scores = torch.cat(pseudo_sim_score, dim=0).unsqueeze(0)
        self._fill_with_pseudo_labels_prefilter(targets_dict, pseudo_boxes, pseudo_sem_score_multiclass, pseudo_sim_score, pseudo_score,pseudo_instance_sim_score,targets_dict['unlabeled_inds'], targets_dict['labeled_inds']) #TODO: check if this is correct
        targets_dict['gt_boxes_emas_prefilter'] = targets_dict['gt_boxes'].clone()
        targets_dict['pseudo_sem_scores_multiclass_emas_prefilter'] = targets_dict['pseudo_sem_scores_multiclass'].clone()
        targets_dict['pseudo_sim_scores_emas_prefilter'] = targets_dict['pseudo_sim_scores'].clone()
        targets_dict['rcnn_scores_ema_prefilter'] = targets_dict['pseudo_scores'].clone()
        targets_dict['pseudo_instance_sim_scores_emas_prefilter'] = targets_dict['pseudo_instance_sim_scores'].clone()

        self.apply_augmentation(targets_dict, targets_dict, targets_dict['unlabeled_inds'], key='gt_boxes')
        metrics_input = defaultdict(list)
        for i, uind in enumerate(targets_dict['unlabeled_inds']):
            # mask = (targets_dict['reg_valid_mask'][uind] > 0) if mask_type == 'reg' else (
            #             targets_dict['rcnn_cls_labels'][uind] >= 0)
            # if mask.sum() == 0:
            #     # print(f'Warning: No {mask_type} rois for unlabeled index {uind}')
            #     continue

            # (Proposals) PLs are passed in as ROIs
            rois = targets_dict['gt_boxes'][uind].detach().clone()
            roi_labels = targets_dict['gt_boxes'][...,-1][uind].unsqueeze(-1).clone().detach()
            roi_scores_multiclass = targets_dict['pseudo_sem_scores_multiclass'][uind].clone().detach()
            roi_sim_scores_multiclass = targets_dict['pseudo_sim_scores'][uind].clone().detach()
            roi_instance_sim_scores_multiclass = targets_dict['pseudo_instance_sim_scores'][uind].detach().clone()
            metrics_input['roi_instance_sim_scores'].append(roi_instance_sim_scores_multiclass)
            metrics_input['rois'].append(rois)
            metrics_input['roi_scores'].append(roi_scores_multiclass)
            metrics_input['roi_sim_scores'].append(roi_sim_scores_multiclass)

            # (Real labels) GT info: Original GTs are passed in as GTs
            gt_labeled_boxes = targets_dict['ori_unlabeled_boxes'][i]
            metrics_input['ground_truths'].append(gt_labeled_boxes)
            metrics_input['roi_weights'] = None
            metrics_input['roi_iou_wrt_pl'] = None
            metrics_input['roi_target_scores'] = None

            bs_id = targets_dict['points'][:, 0] == uind
            points = targets_dict['points'][bs_id, 1:].detach().clone()
            metrics_input['points'].append(points)
        if len(metrics_input['rois']) == 0:
            # print(f'Warning: No {mask_type} rois for any unlabeled index')
            return
        tag = f'pl_gt_metrics_before_filtering_{mask_type}'
        metrics_registry.get(tag).update(**metrics_input)
        return {
            'gt_boxes_ema': targets_dict['gt_boxes_emas_prefilter'],
            'pseudo_sem_scores_multiclass': targets_dict['pseudo_sem_scores_multiclass_emas_prefilter'],
            'pseudo_sim_scores_emas_prefilter': targets_dict['pseudo_sim_scores_emas_prefilter'],
            'pseudo_instance_sim_scores_emas_prefilter': targets_dict['pseudo_instance_sim_scores_emas_prefilter'],
            'rcnn_scores_ema_prefilter' : targets_dict['rcnn_scores_ema_prefilter']
        }

    def update_metrics_pl(self,targets_dict, mask_type='cls'):
        metrics_input = defaultdict(list)
        for i, uind in enumerate(targets_dict['unlabeled_inds']):
            # mask = (targets_dict['reg_valid_mask'][uind] > 0) if mask_type == 'reg' else (
            #             targets_dict['rcnn_cls_labels'][uind] >= 0)
            # if mask.sum() == 0:
            #     # print(f'Warning: No {mask_type} rois for unlabeled index {uind}')
            #     continue

            # (Proposals) ROI info
            rois = targets_dict['gt_boxes'][uind].detach().clone()
            roi_labels = targets_dict['gt_boxes'][...,-1][uind].unsqueeze(-1).detach().clone()
            roi_scores_multiclass = targets_dict['pseudo_sem_scores_multiclass'][uind].detach().clone()
            roi_instance_sim_scores_multiclass = targets_dict['pseudo_instance_sim_scores'][uind].detach().clone()
            roi_sim_scores_multiclass = targets_dict['pseudo_sim_scores'][uind].detach().clone()
            roi_instance_sim_scores_multiclass = targets_dict['pseudo_instance_sim_scores'][uind].detach().clone()
            metrics_input['rois'].append(rois)
            metrics_input['roi_scores'].append(roi_scores_multiclass)
            metrics_input['roi_sim_scores'].append(roi_sim_scores_multiclass)
            metrics_input['roi_instance_sim_scores'].append(roi_instance_sim_scores_multiclass)

            # (Real labels) GT info
            gt_labeled_boxes = targets_dict['ori_unlabeled_boxes'][i]
            metrics_input['ground_truths'].append(gt_labeled_boxes)
            metrics_input['roi_weights'] = None
            metrics_input['roi_iou_wrt_pl'] = None
            metrics_input['roi_target_scores'] = None

            bs_id = targets_dict['points'][:, 0] == uind
            points = targets_dict['points'][bs_id, 1:].clone().detach()
            metrics_input['points'].append(points)
        if len(metrics_input['rois']) == 0:
            # print(f'Warning: No {mask_type} rois for any unlabeled index')
            return
        tag = f'pl_gt_metrics_after_filtering_{mask_type}'
        metrics_registry.get(tag).update(**metrics_input)
        