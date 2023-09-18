from .detector3d_template import Detector3DTemplate
from pcdet.utils.prototype_utils import feature_bank_registry
import torch
from collections import defaultdict
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        feature_bank_registry.register('gt_aug_lbl_prototypes')
        self.model_cfg = model_cfg
    @staticmethod
    def _clone_gt_boxes_and_feats(batch_dict):
        return {
            "batch_size": batch_dict['batch_size'],
            "gt_boxes": batch_dict['gt_boxes'].clone().detach(),
            "point_coords": batch_dict['point_coords'].clone().detach(),
            "point_features": batch_dict['point_features'].clone().detach(),
            "point_cls_scores": batch_dict['point_cls_scores'].clone().detach()
        }        
    def _prep_bank_inputs(self, batch_dict, num_points_threshold=20):
        selected_batch_dict = self._clone_gt_boxes_and_feats(batch_dict)
        with torch.no_grad():
            batch_gt_pool_feats = self.roi_head.pool_features(selected_batch_dict, use_gtboxes=True)
            batch_gt_feats = self.roi_head.projected_layer(batch_gt_pool_feats.view(batch_gt_pool_feats.shape[0],-1,1))
        batch_gt_feats = batch_gt_feats.view(*batch_dict['gt_boxes'].shape[:2], -1)
        bank_inputs = defaultdict(list)
        for ix in range(batch_dict['gt_boxes'].shape[0]):
            gt_boxes = selected_batch_dict['gt_boxes'][ix]
            nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
            if nonzero_mask.sum() == 0:
                # print(f"no gt instance in frame {batch_dict['frame_id'][ix]}")
                continue
            gt_boxes = gt_boxes[nonzero_mask]
            sample_mask = batch_dict['points'][:, 0].int() == ix
            points = batch_dict['points'][sample_mask, 1:4]
            gt_feat = batch_gt_feats[ix][nonzero_mask]
            gt_labels = gt_boxes[:, -1].int() - 1
            gt_boxes = gt_boxes[:, :7]
            # ins_idxs = batch_dict['instance_idx'][ix][nonzero_mask].int()
            # smpl_id = torch.from_numpy(batch_dict['frame_id'].astype(np.int32))[ix].to(gt_boxes.device)

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
            # bank_inputs['ins_ids'].append(ins_idxs[valid_gts_mask])
            # bank_inputs['smpl_ids'].append(smpl_id)

            # valid_boxes = gt_boxes[valid_gts_mask]
            # valid_box_labels = gt_labels[valid_gts_mask]
            # self.vis(valid_boxes, valid_box_labels, points)

        return bank_inputs
    
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            bank = feature_bank_registry.get('gt_aug_lbl_prototypes')
            sa_gt_lbl_inputs = self._prep_bank_inputs(batch_dict,bank.num_points_thresh)
            bank.update(**sa_gt_lbl_inputs, iteration=batch_dict['cur_iteration'])
            feature_bank_registry.get('gt_aug_lbl_prototypes').compute()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, {}

    def get_training_loss(self,batch_dict):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        protocon_loss = self._get_proto_contrastive_loss(batch_dict, feature_bank_registry.get('gt_aug_lbl_prototypes'))
        loss = loss_rpn + loss_point + loss_rcnn
        if protocon_loss is not None:
            protocon_loss *= self.model_cfg.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.protocon_weight
            loss += protocon_loss
            tb_dict['protocon_loss'] = protocon_loss
        return loss, tb_dict, disp_dict
    
    def _get_proto_contrastive_loss(self, batch_dict, bank):
        gt_boxes = batch_dict['gt_boxes']
        B, N = gt_boxes.shape[:2]
        sa_pl_pool_feats = self.roi_head.pool_features(batch_dict, use_gtboxes=True).view(B * N, -1, 1)
        sa_pl_feats = self.roi_head.projected_layer(sa_pl_pool_feats).view(B * N, -1)
        pl_labels = batch_dict['gt_boxes'][..., -1].view(-1).long() - 1
        proto_cont_loss = bank.get_proto_contrastive_loss(sa_pl_feats, pl_labels)
        if proto_cont_loss is None:
            return
        nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
        if nonzero_mask.sum() == 0:
            print(f"No pl instances predicted for strongly augmented frame(s) {batch_dict['frame_id']}")
            return
        return proto_cont_loss.view(B, N).mean()