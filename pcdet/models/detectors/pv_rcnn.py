from .detector3d_template import Detector3DTemplate
import torch

class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.proto_count = 0

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            if self.roi_head.protos is None or self.proto_count<=30: # first pass in which all classes' ROIs are generated 
                    cur_features = []
                    for cls in range(0,3):
                        cls_mask = self.roi_head.forward_ret_dict['roi_labels'] == cls + 1
                        fg_mask = self.roi_head.forward_ret_dict['rcnn_cls_labels'] >= 0.7 
                        cls_fg_mask = cls_mask & fg_mask
                        # if one of the classes is not predicted the protos will be None, waiting for the model to learn all classes
                        if not torch.any(cls_fg_mask):
                            cur_features = []
                            break
                        else:
                            cls_features = self.roi_head.forward_ret_dict['projected_features'].view(batch_dict['batch_size'],-1, 256)[cls_fg_mask]
                            cur_features.append(cls_features.mean(dim=0))
                            if cls == 2:
                                self.roi_head.protos = torch.stack(cur_features, dim=0)
                                self.proto_count += 1    
            else: # all other passes where EMA is done
                    for cls in range(0,3):
                        cls_mask = self.roi_head.forward_ret_dict['roi_labels'] == cls + 1
                        fg_mask = self.roi_head.forward_ret_dict['rcnn_cls_labels'] >= 0.6
                        cls_fg_mask = cls_mask & fg_mask
                        if torch.any(cls_fg_mask):
                            cls_features = self.roi_head.forward_ret_dict['projected_features'].view(batch_dict['batch_size'],-1, 256)[cls_fg_mask]
                            self.roi_head.protos[cls] = (self.roi_head.protos[cls] * 0.9) + (cls_features.mean(dim=0) * 0.1)        
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
