import torch
import numpy as np
from .detector3d_template import Detector3DTemplate
from pcdet.datasets.augmentor import augmentor_utils
from ...utils import common_utils

class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict = self.apply_augmentation(batch_dict, batch_dict, key='gt_boxes')
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, {}

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def apply_augmentation(self, batch_dict, batch_dict_org, key='rois'):
        batch_dict[key] = augmentor_utils.random_flip_along_x_bbox(
            batch_dict[key], batch_dict_org['flip_x'])
        batch_dict[key] = augmentor_utils.random_flip_along_y_bbox(
            batch_dict[key], batch_dict_org['flip_y'])
        batch_dict[key] = augmentor_utils.global_rotation_bbox(
            batch_dict[key], batch_dict_org['rot_angle'])
        batch_dict[key] = augmentor_utils.global_scaling_bbox(
            batch_dict[key], batch_dict_org['scale'])

        batch_dict[key] = common_utils.limit_period(
            batch_dict[key], offset=0.5, period=2 * np.pi
        )
        return batch_dict