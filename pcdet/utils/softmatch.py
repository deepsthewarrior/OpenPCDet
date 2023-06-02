
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import norm
import copy
from torchmetrics import Metric
from pcdet.config import cfg
import seaborn as sns

class AdaptiveThresholding(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.percent = kwargs.get('PERCENT', 0.1)
        self.pre_filter_thresh = kwargs.get('PRE_FILTERING_THRESH', 0.25)
        self.tag = kwargs.get('tag', None)
        self.dataset = kwargs.get('dataset', None)
        self.quantile= kwargs.get('quantile', False)
        self.momentum= kwargs.get('momentum', 0.9)
        self.enable_clipping = kwargs.get('enable_clipping', True)
        self.metrics_name = ['batchwise_mean','batchwise_variance','ema_mean','ema_variance','non_linear_mean']
        self.config = kwargs['config']
        self.bg_thresh = self.config.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH
        self.reset_state_interval = self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.RESET_STATE_INTERVAL
        if self.dataset is not None:
            self.class_names  = self.dataset.class_names
            if isinstance(self.class_names, list):
                self.class_names = {i: self.dataset.class_names[i] for i in range(len(self.dataset.class_names))}
            self.num_classes = len(self.dataset.class_names)
        else:
            self.class_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
            self.num_classes = 3

        self.add_state("iou_scores", default=[], dist_reduce_fx='cat')
        self.add_state("labels", default=[], dist_reduce_fx='cat')
        self.add_state("softmatch_weights",default=[], dist_reduce_fx='cat')
        self.add_state("sampled_ious",default=[], dist_reduce_fx='cat')
        self.add_state("sampled_labels",default=[], dist_reduce_fx='cat')
        self.raw_mean = torch.ones(self.num_classes) / self.num_classes  
        self.st_var = torch.zeros(self.num_classes)
        self.st_mean = self.raw_mean*(2 - self.raw_mean)
        self.batch_mean = torch.zeros(self.num_classes) 
        self.batch_var = torch.ones(self.num_classes)
        self.enable_plots =  self.config.ROI_HEAD.TARGET_CONFIG.ENABLE_PLOTS

    def update(self, roi_labels: torch.Tensor, iou_wrt_pl: torch.Tensor,weights:torch.Tensor,sampled_labels:torch.Tensor,sampled_ious:torch.Tensor) -> None:
        if roi_labels.ndim == 1: # Unsqueeze for DDP
            roi_labels=roi_labels.unsqueeze(dim=0)
        if iou_wrt_pl.ndim == 1: # Unsqueeze for DDP
            iou_wrt_pl=iou_wrt_pl.unsqueeze(dim=0)   
        if weights.ndim == 1: # Unsqueeze for DDP
            weights=weights.unsqueeze(dim=0)  
        if sampled_labels.ndim == 1: # Unsqueeze for DDP
            sampled_labels=sampled_labels.unsqueeze(dim=0)     
        if sampled_ious.ndim == 1: # Unsqueeze for DDP
            sampled_ious=sampled_ious.unsqueeze(dim=0)                    
        self.iou_scores.append(iou_wrt_pl)
        self.labels.append(roi_labels)    
        self.softmatch_weights.append(weights)
        self.sampled_ious.append(sampled_ious)
        self.sampled_labels.append(sampled_labels)

    def compute(self):
        results = {}
        classwise_metrics = {}
        num_classes = len(self.dataset.class_names)
        # if not cfg.MODEL.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.get('ENABLE', False): return
        if len(self.iou_scores) >= self.reset_state_interval:
        # cls_wise_ious = get_cls_wise_ious(self.iou_scores, self.labels, fill_value=0.0, num_classes=self.num_classes)
            cls_wise_iou_mean_ = []
            cls_wise_iou_var_ = []
            cls_wise_thresholded = []
            all_iou = [i.detach().cpu() for i in self.iou_scores]
            all_label = [i.detach().cpu() for i in self.labels]
            all_weights = [i.detach().cpu() for i in self.softmatch_weights]
            all_sampled_labels = [i.detach().cpu() for i in self.sampled_labels]
            all_sampled_ious = [i.detach().cpu() for i in self.sampled_ious]

            ious = torch.cat(all_iou, dim=0)
            labels = torch.cat(all_label, dim=0)
            weights = torch.cat(all_weights,dim=0)
            sampled_labels = torch.cat(all_sampled_labels,dim=0)
            sampled_ious = torch.cat(all_sampled_ious,dim=0)
            valid_sampled = sampled_labels != 0
            valid_mask = ious != 0
            if not valid_mask.any():
                return None
            ious = ious[valid_mask]
            labels = labels[valid_mask]
            labels -= 1
            sampled_labels = sampled_labels[valid_sampled]
            weights = weights[valid_sampled]
            sampled_ious = sampled_ious[valid_sampled]
            cls_wise_ious = [ious[labels == cind] for cind in range(self.num_classes)]
            cls_wise_weights = [weights[sampled_labels == cind] for cind in range(self.num_classes)]
            cls_wise_thresholded = [cls_wise_ious[cind][cls_wise_ious[cind] >= self.bg_thresh] for cind in range(self.num_classes)]
            # weights_thresholded = [cls_wise_weights[cind][cls_wise_ious[cind] >= self.bg_thresh] for cind in range(self.num_classes)] 

            num_bins = 20
            bin_range = (0,1)           
            for i in  range(len(cls_wise_ious)):
                    cls_wise_iou_mean_.append(cls_wise_thresholded[i].mean().clone())
                    cls_wise_iou_var_.append(cls_wise_thresholded[i].var(unbiased=True).clone())

            # gaussian with previous weights
            if self.enable_plots:
                fig,axs = plt.subplots(1, 3, figsize=(12, 8))
                for cind in range(self.num_classes):
                    iou_range = np.linspace(0.0,1.0,100)
                    iou_range = torch.tensor(iou_range).to(device=ious.device)
                    range_weights = self.gaussian_weights_classwise(roi_ious = iou_range,labels = sampled_labels,cind = cind)
                    axs[cind].hist(cls_wise_thresholded[cind].numpy(),bins='auto',density=True)
                    axs[cind].plot(iou_range.numpy(),range_weights.numpy())
                    # axs[cind].plot(cls_wise_thresholded[cind],label='Mean')
                    axs[cind].axvline(x=self.st_mean[cind].item(),color='red',linestyle='--',label='Mean')
                    axs[cind].set_title('{}'.format(self.class_names[cind]))
                    axs[cind].legend()
                    plt.tight_layout()
                results['threshold_weight_trend'] = fig.get_figure()
                plt.close()

            #NOTE: mean of empty tensor is nan,common among tail classes
            self.batch_mean = torch.stack(cls_wise_iou_mean_).clone()
            self.batch_var = torch.stack(cls_wise_iou_var_).clone()
            #NOTE: replace nan with previous dynamic threshold. replacing nan with zero reduces the dyn threshold value
            for cind in range(num_classes):
                self.batch_var[cind] = self.batch_var[cind].nan_to_num(nan=self.st_var[cind])
                self.batch_mean[cind] = self.batch_mean[cind].nan_to_num(nan=self.raw_mean[cind])

            # *** ema update happens here ***
            self.raw_mean = self.momentum*(self.raw_mean) + (1-self.momentum)*self.batch_mean
            self.st_var = self.momentum*(self.st_var) + ((self.reset_state_interval/(self.reset_state_interval-1))*(1-self.momentum)*self.batch_var)
            self.raw_mean = torch.clamp(self.raw_mean, min=0.25,max=0.90).clone()
            self.st_var = torch.clamp(self.st_var,min=0.0).clone()

            if self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.NON_LINEARITY:
                self.st_mean = self.raw_mean*(2 - self.raw_mean)
            else:
                self.st_mean = self.raw_mean

            classwise_metrics={}
            for metric_name in self.metrics_name:
                classwise_metrics[metric_name] = all_iou[0].new_zeros(self.num_classes).fill_(float('nan'))
            for cind in range(num_classes):
                if self.config.ROI_HEAD.ADAPTIVE_THRESH_CONFIG.NON_LINEARITY:
                    classwise_metrics['non_linear_mean'][cind] = self.st_mean[cind]
                else:
                    classwise_metrics['non_linear_mean'][cind] = self.st_mean[cind]*(2 - self.st_mean[cind])
                classwise_metrics['batchwise_mean'][cind] = self.batch_mean[cind]
                classwise_metrics['batchwise_variance'][cind] = self.batch_var[cind]
                classwise_metrics['ema_mean'][cind] = self.raw_mean[cind]
                classwise_metrics['ema_variance'][cind] = self.st_var[cind]
            self.reset()

        else:
            classwise_metrics = {}
            for metric_name in self.metrics_name:
               classwise_metrics[metric_name] = self.iou_scores[0].new_zeros(num_classes).fill_(float('nan'))

        for key in self.metrics_name: 
            classwise_results = {}
            #if all values are nan, then return a list with nan values(gets filtered in train_utils)
            if torch.all(classwise_metrics[key].isnan() == True):
                results[key] = self.iou_scores[0].new_zeros(1).fill_(float('nan'))
            else:
                for cind,cls in enumerate(self.dataset.class_names):
                    classwise_results[cls] = classwise_metrics[key][cind].item()
                results[key] = copy.deepcopy(classwise_results)                   
        return results

    def gaussian_weights_classwise(self,roi_ious,labels,cind):
        diff = torch.square(roi_ious - self.st_mean[cind])
        scaler = self.config.ROI_HEAD.TARGET_CONFIG.SOFTMATCH_SCALER
        scaled_var = scaler*torch.square(self.st_var[cind])
        gaussian = torch.exp(-diff/scaled_var)
        return gaussian


def compute_softmatch_weights(max_ious, mu_t, var_t, n_sigma=2):
    """
    Compute SoftMatch weights based on maximum IoU values and mean/variance parameters.
    Args:
        max_ious (torch.Tensor): Maximum IoU values, shape (N,)
        mu_t (torch.Tensor): Mean parameter, shape (N,)
        var_t (torch.Tensor): Variance parameter, shape (N,)
        n_sigma (float): Scaling factor for variance, default is 2.
    Returns:
        weights (torch.Tensor): SoftMatch weights, shape (N,)
    """
    diff = torch.clamp(max_ious - mu_t, max=0.0) ** 2
    scaled_var_t = var_t / (n_sigma ** 2)
    weights = torch.exp(-diff / (2 * scaled_var_t))

    return weights

def nanmean(v: torch.Tensor, *args, allnan=np.nan, **kwargs) -> torch.Tensor:
    """
    :param v: tensor to take mean
    :param dim: dimension(s) over which to take the mean
    :param allnan: value to use in case all values averaged are NaN.
        Defaults to np.nan, consistent with np.nanmean.
    :return: mean.
    """
    def isnan(v):
        if v.dtype is torch.long:
            return v == torch.tensor(np.nan).long()
        else:
            return torch.isnan(v)
    v = v.clone()
    is_nan = isnan(v)
    v[is_nan] = 0

    if np.isnan(allnan):
        return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)
    else:
        sum_nonnan = v.sum(*args, **kwargs)
        n_nonnan = float(~is_nan).sum(*args, **kwargs)
        mean_nonnan = torch.zeros_like(sum_nonnan) + allnan
        any_nonnan = n_nonnan > 1
        mean_nonnan[any_nonnan] = (
                sum_nonnan[any_nonnan] / n_nonnan[any_nonnan])
        return mean_nonnan
    
def gaussian_weights(self,roi_ious,labels):
    diff = torch.square(roi_ious - self.st_mean[labels - 1])
    scaler = self.config.ROI_HEAD.TARGET_CONFIG.SOFTMATCH_SCALER
    scaled_var = scaler*torch.square(self.st_var[labels - 1])
    gaussian = torch.exp(-diff/scaled_var)
    return gaussian

    
