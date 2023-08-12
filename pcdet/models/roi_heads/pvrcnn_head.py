import torch.nn as nn
import torch
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
import torch.nn.functional as F
import pickle
class PVRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg,
                         predict_boxes_when_training=predict_boxes_when_training)
        self.model_cfg = model_cfg

        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]
            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.protos = None
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        self.projector_fc_layer = nn.Sequential(*shared_fc_list)
        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

        self.print_loss_when_eval = False
        self.pooled_prototype_proj_lb_ = []
        self.pooled_prototype_sh_lb_ = []
        pkl_file = self.model_cfg.BASE_PROTOTYPE
        with open(pkl_file, 'rb') as file:
        # Load the data from the .pkl file
            data = pickle.load(file) # 0->Car, 1->Ped, 2->Cyc
        self.pooled_prototype_proj_lb_.append((data['Car']['mean']['proj'].cuda()).contiguous().view(-1)) # (256)
        self.pooled_prototype_proj_lb_.append((data['Ped']['mean']['proj'].cuda()).contiguous().view(-1))
        self.pooled_prototype_proj_lb_.append((data['Cyc']['mean']['proj'].cuda()).contiguous().view(-1))
        self.pooled_prototype_proj_lb = torch.stack(self.pooled_prototype_proj_lb_,dim=0).cuda() # [3, 27648] == [num_protos, 256]
        with open(pkl_file, 'rb') as file:
        # Load the data from the .pkl file
            data = pickle.load(file) # 0->Car, 1->Ped, 2->Cyc
        self.pooled_prototype_sh_lb_.append((data['Car']['mean']['sh'].cuda()).contiguous().view(-1)) # (256)
        self.pooled_prototype_sh_lb_.append((data['Ped']['mean']['sh'].cuda()).contiguous().view(-1))
        self.pooled_prototype_sh_lb_.append((data['Cyc']['mean']['sh'].cuda()).contiguous().view(-1))
        self.pooled_prototype_sh_lb = torch.stack(self.pooled_prototype_sh_lb_,dim=0).cuda() # [3, 27648] == [num_protos, 256]
        
    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict,pool_gtboxes=False):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict["point_coords"]
        point_features = batch_dict["point_features"]
        point_cls_scores = batch_dict["point_cls_scores"]

        point_features = point_features * point_cls_scores.view(-1, 1)

        if pool_gtboxes:
            rois = batch_dict['gt_boxes'][...,0:7]

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict, disable_gt_roi_when_pseudo_labeling=False):
        """
        :param input_data: input dict
        :return:
        """

        # use test-time nms for pseudo label generation
        nms_mode = self.model_cfg.NMS_CONFIG['TRAIN' if self.training and not disable_gt_roi_when_pseudo_labeling else 'TEST']

        # proposal_layer doesn't continue if the rois are already in the batch_dict.
        # However, for labeled data proposal layer should continue!
        targets_dict = self.proposal_layer(batch_dict, nms_config=nms_mode)
        # should not use gt_roi for pseudo label generation
        if (self.training or self.print_loss_when_eval) and not disable_gt_roi_when_pseudo_labeling:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_scores'] = targets_dict['roi_scores']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            # Temporarily add infos to targets_dict for metrics
            targets_dict['unlabeled_inds'] = batch_dict['unlabeled_inds']
            targets_dict['ori_unlabeled_boxes'] = batch_dict['ori_unlabeled_boxes']
            # TODO(farzad) refactor this with global registry,
            #  accessible in different places, not via passing through batch_dict
            targets_dict['metric_registry'] = batch_dict['metric_registry']

        if 'cal_gt_proto' in batch_dict.keys(): # do only during the batch_dict_pl
            with torch.no_grad():
                batch_dict['gt_boxes']
                pooled_features_gt = self.roi_grid_pool(batch_dict,pool_gtboxes=True)  # (BxNum_GT, 6x6x6, C)
                grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
                batch_size_rcnn = pooled_features_gt.shape[0]
                pooled_features_gt = pooled_features_gt.permute(0, 2, 1).\
                    contiguous().view(batch_dict['batch_size'],-1,128, grid_size, grid_size, grid_size)  # (BxNum_GT, C, 6, 6, 6)
                shared_features_gt = self.shared_fc_layer(pooled_features_gt.view(batch_size_rcnn, -1, 1))
                batch_dict['pooled_features_gt'] = pooled_features_gt
                                            
        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
        pooled_features_permute = pooled_features.view(batch_size_rcnn, -1, 1)
        shared_features = self.shared_fc_layer(pooled_features_permute)
        projected_features = self.projector_fc_layer(pooled_features_permute.clone().detach())
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        
        # Cosine similarity between shared features and prototype
        if 'labeled_prototype' in batch_dict.keys(): # cannot do this during obtaining batch_dict_std and batch_dict_pl
            labels = (targets_dict['roi_labels'].clone().view(-1)) - 1 

            # Cosine similarity between pooled features and prototype
            pooled_features_clone = pooled_features.view(batch_size_rcnn,-1).clone().detach() #(N,128,6,6,6) => (N,27648)
            pool_protos_lb = batch_dict['labeled_prototype'].to(labels.device) #[3, 27648]
            pool_proposals = pooled_features_clone.squeeze()
            projected_features_permute = projected_features.permute(0,1,2)
            shared_feat_proposals = shared_features.clone().detach().squeeze(dim=-1)
            shared_cos_sim = F.normalize(shared_feat_proposals) @ F.normalize(self.pooled_prototype_sh_lb).t()
            proj_cos_sim = F.normalize(projected_features_permute.squeeze()) @ F.normalize(self.pooled_prototype_proj_lb.to(labels.device)).t()

            targets_dict['cos_scores_car_sh'] = shared_cos_sim[:,0].view(batch_dict['roi_scores'].shape[0],-1)
            targets_dict['cos_scores_ped_sh'] = shared_cos_sim[:,1].view(batch_dict['roi_scores'].shape[0],-1)
            targets_dict['cos_scores_cyc_sh'] = shared_cos_sim[:,2].view(batch_dict['roi_scores'].shape[0],-1)
            batched_shared_cos_sim = shared_cos_sim.view(batch_dict['roi_scores'].shape[0],-1,3)
            targets_dict['cos_scores_sh_norm'] = F.softmax(batched_shared_cos_sim,dim=-1) # (N,128,3)
            

            targets_dict['cos_scores_car_proj'] = proj_cos_sim[:,0].view(batch_dict['roi_scores'].shape[0],-1)
            targets_dict['cos_scores_ped_proj'] = proj_cos_sim[:,1].view(batch_dict['roi_scores'].shape[0],-1)
            targets_dict['cos_scores_cyc_proj'] = proj_cos_sim[:,2].view(batch_dict['roi_scores'].shape[0],-1)
            batched_proj_cos_sim = proj_cos_sim.view(batch_dict['roi_scores'].shape[0],-1,3)
            targets_dict['cos_scores_proj_norm'] = F.softmax(batched_proj_cos_sim,dim=-1) # (N,128,3)
            targets_dict['cos_sim'] = F.softmax(proj_cos_sim,dim=-1) 
            
        # if self.protos is not None:
        #     cos_sim = F.normalize(projected_features.permute(0,2,1).squeeze(1)) @ F.normalize(self.protos).t()
        #     targets_dict['cos_sim'] = F.softmax(cos_sim,dim=1)
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            # note that the rpn batch_cls_preds and batch_box_preds are being overridden here by rcnn preds
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
            # Temporarily add infos to targets_dict for metrics
            targets_dict['batch_box_preds'] = batch_box_preds

        if self.training or self.print_loss_when_eval:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['projected_features'] = (projected_features.clone()).detach().permute(0,2,1).contiguous()
            self.forward_ret_dict = targets_dict
        return batch_dict
