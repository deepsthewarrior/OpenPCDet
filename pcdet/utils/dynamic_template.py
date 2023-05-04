import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import norm
import pickle
from torchmetrics import Metric
from pcdet.config import cfg

class Prototype(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.tag = kwargs.get('tag', None)
        self.dataset = kwargs.get('dataset', None)
        self.quantile= kwargs.get('quantile', False)
        self.momentum= kwargs.get('momentum', 0.90)
        # self.template_momentum = kwargs.get('template_momentum',0.8)
        self.enable_clipping = kwargs.get('enable_clipping', True)
        self.metrics_name = ['batchwise_mean','batchwise_variance','ema_mean','ema_variance']
        self.reset_state_interval = kwargs.get('reset_state_interval', 5)

        if self.dataset is not None:
            self.class_names  = self.dataset.class_names
            if isinstance(self.class_names, list):
                self.class_names = {i: self.dataset.class_names[i] for i in range(len(self.dataset.class_names))}
            self.num_classes = len(self.dataset.class_names)
        else:
            self.class_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
            self.num_classes = 3
        self.state_list = ['car_template','ped_template','cyc_template']
        for cls in self.state_list:
            self.add_state(cls, default=[], dist_reduce_fx='cat')
        # self.add_state("count",default=[],dist_reduce_fx='sum')
        self.st_mean = torch.ones((self.num_classes)) / self.num_classes     
        self.st_var = torch.ones(self.num_classes)
        self.batch_mean = self.st_mean
        self.batch_var = self.st_var
        self.classes = ['Car','Ped','Cyc']
        with open('ema_cls_sh.pkl','rb') as f:
            self.rcnn_features = pickle.loads(f.read())
        rcnn_sh_mean = []
        rcnn_cls_mean = []
        avg = "mean"
        param = "sh"
        for cls in self.classes:
            rcnn_sh_mean.append(self.rcnn_features[cls][avg][param].unsqueeze(dim=0).detach().cpu())
        self.rcnn_sh_mean_ = torch.stack(rcnn_sh_mean)
        self.rcnn_sh_mean = self.rcnn_sh_mean_.detach().cuda()
        # avg = "mean"
        # param = "cls"
        # for cls in self.class_names.values():
        #     rcnn_cls_mean.append(self.rcnn_features[cls][avg][param].unsqueeze(dim=0).detach().cpu())
        # self.rcnn_cls_mean_ = torch.stack(rcnn_cls_mean)
        # self.rcnn_cls_mean = self.rcnn_cls_mean_.detach().cuda()
        
        
    def update(self, car_template, ped_template, cyc_template) -> None:
        # if car_template == 1: # Unsqueeze for DDP
        #     roi_labels=roi_labels.unsqueeze(dim=0)
        # if iou_wrt_pl.ndim == 1: # Unsqueeze for DDP
        #     iou_wrt_pl=iou_wrt_pl.unsqueeze(dim=0)
        # for temps in self.car_template:

        for i,feature in enumerate(car_template):
            self.car_template.append(feature.unsqueeze(dim=0))
        for i,feature in enumerate(ped_template):
            self.ped_template.append(feature.unsqueeze(dim=0))
        for i,feature in enumerate(cyc_template):
            self.cyc_template.append(feature.unsqueeze(dim=0))
        # self.count.append(1.0)


    def compute(self):

        template_state = [self.car_template,self.ped_template,self.cyc_template]
        template = {cls:[] for cls in self.classes}
        
        for i,templ in enumerate(template_state):
            for temps in templ:
                if temps != None:
                        template[self.classes[i]].append(temps)

        for i,final_template in enumerate(template.values()):
            cls_template  = self.rcnn_sh_mean[0].new_zeros(self.rcnn_sh_mean[0].shape).fill_(float('nan'))
            if len(final_template):
                cls_template = torch.mean(torch.stack(final_template),dim=0)
            if torch.all(~torch.isnan(cls_template)) and len(final_template):
                self.rcnn_sh_mean[i] = self.momentum*self.rcnn_sh_mean[i] + (1-self.momentum)*cls_template

        self.reset()     


        return self.rcnn_sh_mean
        
        
        
