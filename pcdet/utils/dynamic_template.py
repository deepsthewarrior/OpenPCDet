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
        self.reset_state_interval = kwargs.get('reset_state_interval', 20)

        if self.dataset is not None:
            self.class_names  = self.dataset.class_names
            if isinstance(self.class_names, list):
                self.class_names = {i: self.dataset.class_names[i] for i in range(len(self.dataset.class_names))}
            self.num_classes = len(self.dataset.class_names)
        else:
            self.class_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
            self.num_classes = 3
        self.state_list = ['car_template','ped_template','cyc_template','iteration','templates','labels']
        for cls in self.state_list:
            self.add_state(cls, default=[],dist_reduce_fx='cat')
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
        self.rcnn_sh_mean = self.rcnn_sh_mean_.detach().clone().cuda()
        # avg = "mean"
        # param = "cls"
        # for cls in self.class_names.values():
        #     rcnn_cls_mean.append(self.rcnn_features[cls][avg][param].unsqueeze(dim=0).detach().cpu())
        # self.rcnn_cls_mean_ = torch.stack(rcnn_cls_mean)
        # self.rcnn_cls_mean = self.rcnn_cls_mean_.detach().cuda()
        
        
    def update(self,templates=None,labels=None) -> None:
        # if car_template == 1: # Unsqueeze for DDP
        #     roi_labels=roi_labels.unsqueeze(dim=0)
        # if iou_wrt_pl.ndim == 1: # Unsqueeze for DDP
        #     iou_wrt_pl=iou_wrt_pl.unsqueeze(dim=0)
        # for temps in self.car_template:
        if len(templates) != 0: 
            templates = torch.stack(templates)
            if templates.ndim == 1: # Unsqueeze for DDP
                templates = templates.unsqueeze(dim=0)
            self.templates.append(templates)                
        if len(labels) != 0:
            labels = torch.stack(labels)
            if labels.ndim == 1:
                labels = labels.unsqueeze(dim=0)        
            self.labels.append(labels)  
        # if len(car_template) != 0: 
        #     if car_template.ndim == 1: # Unsqueeze for DDP
        #         car_template = car_template.unsqueeze(dim=0)
        #     self.car_template.append(car_template)                
        # if len(ped_template) != 0:
        #     if ped_template.ndim == 1:
        #         ped_template = ped_template.unsqueeze(dim=0)
        #     self.ped_template.append(ped_template)
        # if len(cyc_template) != 0:
        #     if cyc_template.ndim == 1:
        #         cyc_template = cyc_template.unsqueeze(dim=0)
        #     self.cyc_template.append(cyc_template)
        # if iteration.ndim == 1:
        #     iteration = iteration.unsqueeze(dim=0)
        # self.iteration.append(iteration)


    def compute(self):

        template = {cls:None for cls in self.classes}
        if len(self.templates) == 0:
            return self.rcnn_sh_mean
        templates = torch.cat(self.templates)
        labels = torch.cat(self.labels)
        labels = labels.squeeze(dim=1)
        for cls in self.class_names.keys():
            mask = labels == cls+1
            if torch.any(mask):
                template[self.class_names[cls]] = templates[mask]
                self.rcnn_sh_mean[cls] = self.momentum*self.rcnn_sh_mean[cls] + (1-self.momentum)*torch.mean(templates[mask],dim=0).unsqueeze(0)


        self.reset()     
        return self.rcnn_sh_mean
        
        
        
