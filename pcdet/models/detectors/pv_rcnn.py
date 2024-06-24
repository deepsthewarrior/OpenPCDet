from .detector3d_template import Detector3DTemplate
from pcdet.utils.prototype_utils import feature_bank_registry
import torch
from collections import defaultdict
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from torch.nn import functional as F
import numpy as np
import torch.distributed as dist
from pcdet.datasets.augmentor import augmentor_utils
import copy
from matplotlib import pyplot as plt

class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        bank_configs = {
            'TEMPERATURE': model_cfg.ROI_HEAD.LOSS_CONFIG.TEMPERATURE,
            'MOMENTUM': model_cfg.ROI_HEAD.LOSS_CONFIG.MOMENTUM,
            'FEATURE_SIZE': model_cfg.ROI_HEAD.LOSS_CONFIG.FEAT_SIZE,
        }
        feature_bank_registry.register('gt_aug_lbl_prototypes',**bank_configs)
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

    def _prep_wa_bank_inputs(self, batch_dict_ema, inds, bank, iteration, num_points_threshold=20):
        projections_gt = batch_dict_ema['projected_features_gt']
        projections_gt = projections_gt.view(*batch_dict_ema['gt_boxes'].shape[:2], -1)

        bank_inputs = defaultdict(list)
        for ix in inds:
            gt_boxes = batch_dict_ema['gt_boxes'][ix]
            gt_conf_preds = batch_dict_ema['gt_conf_scores'][ix]
            nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
            if nonzero_mask.sum() == 0:
                print(f"no gt instance in frame {batch_dict_ema['frame_id'][ix]}")
                continue
            gt_boxes = gt_boxes[nonzero_mask]
            sample_mask = batch_dict_ema['points'][:, 0].int() == ix
            points = batch_dict_ema['points'][sample_mask, 1:4]
            gt_feat = projections_gt[ix][nonzero_mask] # Store labeled projections into bank
            gt_labels = gt_boxes[:, 7].int()
            gt_boxes = gt_boxes[:, :7]
            ins_idxs = batch_dict_ema['instance_idx'][ix][nonzero_mask].int()
            smpl_id = torch.from_numpy(batch_dict_ema['frame_id'].astype(np.int32))[ix].to(gt_boxes.device)
            num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(points.cpu(), gt_boxes.cpu()).sum(dim=-1)
            valid_gts_mask = (num_points_in_gt >= num_points_threshold)
            if valid_gts_mask.sum() == 0:
                print(f"no valid gt instances with enough points in frame {batch_dict_ema['frame_id'][ix]}")
                continue
            bank_inputs['feats'].append(gt_feat[valid_gts_mask])
            bank_inputs['labels'].append(gt_labels[valid_gts_mask])
            bank_inputs['ins_ids'].append(ins_idxs[valid_gts_mask])
            bank_inputs['smpl_ids'].append(smpl_id)
            bank_inputs['conf_scores'].append(gt_conf_preds[nonzero_mask][valid_gts_mask])
        return bank_inputs    

    @staticmethod
    def _split_batch(batch_dict, tag='ema'):
        assert tag in ['ema'], f'{tag} not in list [ema]'
        batch_dict_out = {}
        keys = list(batch_dict.keys())
        for k in keys:
            if f'{k}_{tag}' in keys:
                continue
            if k.endswith(f'_{tag}'):
                batch_dict_out[k[:-(len(tag)+1)]] = batch_dict[k]
                batch_dict.pop(k)
            if k in ['batch_size']:
                batch_dict_out[k] = batch_dict[k]
        return batch_dict_out


    def forward(self, batch_dict):
        batch_dict_ema = self._split_batch(batch_dict, tag='ema')
        batch_dict_ema['batch_size'] = batch_dict['batch_size']
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            #Now pass with second view to compute projected_features_augmented
            with torch.no_grad():
                for cur_module in self.module_list:
                    try: # disallow saving view2 predictions @ anchor_head, point_head, roi_head
                        batch_dict_ema = cur_module(batch_dict_ema, test_only=True)
                    except TypeError as e:
                        batch_dict_ema = cur_module(batch_dict_ema)
                exclude_keys = ['gt_boxes', 'instance_idx', 'projected_features_gt']
                for key in list(batch_dict_ema.keys()):
                    if key not in exclude_keys:
                        batch_dict_ema.pop(key)
            if self.model_cfg.ROI_HEAD.get('ENABLE_SUPCON', False):
                lbl_inst_cont_loss, supcon_classwise_loss = self._get_instance_contrastive_loss(tb_dict, batch_dict, batch_dict_ema)
                if lbl_inst_cont_loss is not None:
                    if dist.is_initialized():
                        loss+= (lbl_inst_cont_loss/2)
                    else:
                        loss += lbl_inst_cont_loss
                    tb_dict['instloss_car'] = supcon_classwise_loss['instloss_car']
                    tb_dict['instloss_ped'] = supcon_classwise_loss['instloss_ped']
                    tb_dict['instloss_cyc'] = supcon_classwise_loss['instloss_cyc']
                    tb_dict['instloss_all'] = supcon_classwise_loss['instloss_all']

            if self.model_cfg.ROI_HEAD.get('ENABLE_LPCONT', False):
                lpcont_loss_pretrain, lpcont_matrix = self.get_lpcont_loss_pretrain(batch_dict, batch_dict_ema)
                if lpcont_loss_pretrain is not None:
                    if dist.is_initialized():
                        loss+= (lpcont_loss_pretrain/2)
                    else:
                        loss += lpcont_loss_pretrain
                    tb_dict['lpcont_loss_pretrain'] = lpcont_loss_pretrain
                    sim_matrix, labels, pseudo_labels = lpcont_matrix
                    sim_matrix = sim_matrix.detach().cpu().numpy()
                    labels = labels.cpu().numpy()
                    pseudo_labels = pseudo_labels.cpu().numpy()
                    fig, ax = plt.subplots(figsize=(max(8, len(pseudo_labels) * 0.2), max(8, len(labels) * 0.2))) 
                    ax.imshow(sim_matrix, interpolation='nearest', cmap=plt.cm.Blues) #(sim_matrix, interpolation='nearest', cmap=plt.cm.Blues,vmin=0, vmax=1)
                    ax.set_title(" LPCont Similarity matrix")
                    fig.colorbar(ax.imshow(sim_matrix, interpolation='nearest', cmap=plt.cm.Blues))
                    x_tick_marks = np.arange(len(pseudo_labels))
                    ax.set_xticks(x_tick_marks)
                    ax.set_xticklabels(pseudo_labels, rotation=45)
                    y_tick_marks = np.arange(len(labels))   
                    ax.set_yticks(y_tick_marks)
                    ax.set_yticklabels(labels)
                    ax.set_ylabel('Labeled features')
                    ax.set_xlabel('Pseudo features')
                    fig.tight_layout()
                    tb_dict['sim_matrix'] = fig
            # bank = feature_bank_registry.get('gt_aug_lbl_prototypes')
            # wa_gt_lbl_inputs = self._prep_wa_bank_inputs(batch_dict, range(len(batch_dict['gt_boxes'])), bank, batch_dict['cur_iteration'], bank.num_points_thresh)
            # bank.update(**wa_gt_lbl_inputs, iteration=batch_dict['cur_iteration'])
            # feature_bank_registry.get('gt_aug_lbl_prototypes').compute()
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

    def _sort_instance_pairs(self, batch_dict, batch_dict_wa):
        
        embed_size = batch_dict['projected_features_gt'].squeeze().shape[-1]
        shared_ft_sa = batch_dict['projected_features_gt'].view(batch_dict['batch_size'],-1,embed_size)         # shared_ft_sa = batch_dict['projected_features_gt'].view(batch_dict['batch_size'],-1,embed_size)
        shared_ft_sa = shared_ft_sa.view(-1,256)
        shared_ft_wa = batch_dict_wa['projected_features_gt'].view(batch_dict['batch_size'],-1,embed_size)         # shared_ft_wa = batch_dict_wa['projected_features_gt'].view(batch_dict['batch_size'],-1,embed_size)
        shared_ft_wa = shared_ft_wa.view(-1,256)
        labels_sa = batch_dict['gt_boxes'][:,:,-1].view(-1)
        labels_wa = batch_dict_wa['gt_boxes'][:,:,-1].view(-1)
        instance_idx_sa = batch_dict['instance_idx'].view(-1)
        instance_idx_wa = batch_dict_wa['instance_idx'].view(-1)
        nonzero_mask_sa = torch.logical_not(torch.eq(instance_idx_sa, 0))
        nonzero_mask_wa = torch.logical_not(torch.eq(instance_idx_wa, 0))

        # Strip off zero masking
        instance_idx_sa = instance_idx_sa.masked_select(nonzero_mask_sa)
        labels_sa = labels_sa.masked_select(nonzero_mask_sa)
        assert instance_idx_sa.size(0)==labels_sa.size(0)
        instance_idx_wa = instance_idx_wa.masked_select(nonzero_mask_wa)
        labels_wa = labels_wa.masked_select(nonzero_mask_wa)
        assert instance_idx_wa.size(0)==labels_wa.size(0)
        shared_ft_sa = shared_ft_sa.masked_select(nonzero_mask_sa.unsqueeze(-1).expand(-1,embed_size))
        shared_ft_sa = shared_ft_sa.view(-1,embed_size)
        shared_ft_wa = shared_ft_wa.masked_select(nonzero_mask_wa.unsqueeze(-1).expand(-1,embed_size))
        shared_ft_wa = shared_ft_wa.view(-1,embed_size)

        # Finds correspondences of common instances between SA and WA, return corresponding labels and features

        common_instnaces, ids_sa_common, ids_wa_common = np.intersect1d(instance_idx_sa.cpu().numpy(),instance_idx_wa.cpu().numpy(), return_indices = True) 
        valid_instances = torch.from_numpy(common_instnaces).to(labels_sa.device)
        ids_sa_common = torch.from_numpy(ids_sa_common).to(labels_sa.device)
        ids_wa_common = torch.from_numpy(ids_wa_common).to(labels_sa.device)

        instance_idx_sa_common = valid_instances.clone()
        instance_idx_wa_common = valid_instances.clone()

        labels_sa_common = torch.index_select(labels_sa, 0,ids_sa_common)
        labels_wa_common = torch.index_select(labels_wa, 0, ids_wa_common)
        assert torch.equal(labels_sa_common, labels_wa_common)
        
        print("shared_ft_wa_shape:",shared_ft_wa.shape)
        shared_ft_sa_common= torch.index_select(shared_ft_sa, 0, ids_sa_common)
        shared_ft_wa_common = torch.index_select(shared_ft_wa, 0, ids_wa_common)

        print("Labels_wa_shape:", labels_wa_common.shape)
        print("Shared_ft_wa_masked_shape:", shared_ft_wa_common.shape)

        aligned_wa_info = torch.cat([shared_ft_wa_common, labels_wa_common.unsqueeze(-1)], dim=-1)
        aligned_sa_info = torch.cat([shared_ft_sa_common, labels_sa_common.unsqueeze(-1)], dim=-1)
        return aligned_wa_info, aligned_sa_info

    def _get_instance_contrastive_loss(self, tb_dict, batch_dict, batch_dict_wa, temperature=1.0, base_temperature=1.0):
        '''
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: roi_labels[B,N].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        '''
        aligned_wa_info, aligned_sa_info = self._sort_instance_pairs(batch_dict, batch_dict_wa)

        gathered_sa_tensor = self.gather_tensors(aligned_sa_info)
        gathered_sa_labels = gathered_sa_tensor[:,-1].long()
        non_zero_mask = gathered_sa_labels != 0
        gathered_sa_feats = gathered_sa_tensor[:,:-1][non_zero_mask]
        gathered_labels_sa = gathered_sa_labels[non_zero_mask]

        gathered_wa_tensor = self.gather_tensors(aligned_wa_info)
        gathered_wa_labels = gathered_wa_tensor[:,-1].long()
        non_zero_mask2 = gathered_wa_labels != 0
        gathered_wa_feats = gathered_wa_tensor[:,:-1][non_zero_mask2]
        gathered_labels_wa = gathered_wa_labels[non_zero_mask2]

        batch_size_labeled = gathered_labels_sa.shape[0]
        device = gathered_sa_feats.device
        tb_dict = {} if tb_dict is None else tb_dict
        supcon_classwise_loss = {'instloss_car':{},'instloss_ped':{},'instloss_cyc':{},'instloss_all':{}}
        temperature = self.model_cfg['ROI_HEAD'].get('TEMPERATURE', 1.0)

        labels = torch.cat((gathered_labels_sa, gathered_labels_wa), dim=0)
        combined_embed_features = torch.cat([gathered_sa_feats.unsqueeze(1), gathered_wa_feats.unsqueeze(1)], dim=1) # B*N,num_pairs,channel_dim
        num_pairs = combined_embed_features.shape[1]
        assert num_pairs == 2  # contrast_count = 2

        '''Create Contrastive Mask'''
        labels_sa = gathered_labels_sa.contiguous().view(-1, 1)
        mask = torch.eq(labels_sa, labels_sa.T).float().to(device) # (B*N, B*N)
        mask = mask.repeat(num_pairs, num_pairs)        # Tiling mask from N,N -> 2N, 2N)
        logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size_labeled * num_pairs).view(-1, 1).to(device),0)    # mask-out self-contrast cases
        mask = mask * logits_mask

        contrast_feature = torch.cat(torch.unbind(combined_embed_features, dim=1), dim=0)   # B,2,256 -> Tuple(B,256) with len(2) -> (2*B,256)
        contrast_feature = F.normalize(contrast_feature.view(-1,combined_embed_features.shape[-1]),dim=-1) # normalized features for cosine similarity
        sim_supcon_matrix = torch.div(torch.matmul(contrast_feature, contrast_feature.T),temperature)  # compute similarity matrix

        # for numerical stability
        logits_max, _ = torch.max(sim_supcon_matrix, dim=1, keepdim=True)
        logits = sim_supcon_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask # (NOTE:exponent over mostly negative values)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)         # compute mean of log-likelihood over positive        
        
        instance_loss = - ( temperature / base_temperature) * mean_log_prob_pos 
        
        if instance_loss is None:
            return

        instloss_car = instance_loss[labels==1].mean() * 0.2
        instloss_ped = instance_loss[labels==2].mean() * 0.2
        instloss_cyc = instance_loss[labels==3].mean() * 0.2
        instloss_all = instance_loss.mean() * 0.2

        supcon_classwise_loss= {
                        'instloss_car': instloss_car.item() ,
                        'instloss_ped': instloss_ped.item() ,
                        'instloss_cyc': instloss_cyc.item() ,
                        'instloss_all': instloss_all.item()
                }

        return instloss_all, supcon_classwise_loss


    def gather_tensors(self,tensor):
            """
            Returns the gathered tensor to all GPUs in DDP else returns the tensor as such
            dist.gather_all needs the gathered tensors to be of same size.
            We get the sizes of the tensors first, zero pad them to match the size
            Then gather and filter the padding

            Args:
                tensor: tensor to be gathered
                
            """

            assert tensor.size(-1) == 257 , "features should be of size common_instances,(ft_size+ lbl_size)"
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
            # make zero-padded version https://sftackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of
            max_length = max([size[0] for size in all_sizes])
            
            diff = max_length - local_size[0].item()
            if diff:
                pad_size =[diff.item()] #pad with zeros 
                if local_size.ndim >= 1:
                    pad_size.extend(dimension.item() for dimension in local_size[1:])
                padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
                tensor = torch.cat((tensor,padding),)

            all_tensors_padded = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]

            dist.barrier()
            dist.all_gather(all_tensors_padded,tensor)
            dist.barrier()

            gathered_tensor = torch.cat(all_tensors_padded)
            non_zero_mask = torch.any(gathered_tensor!=0,dim=-1).squeeze()
            gathered_tensor = gathered_tensor[non_zero_mask]
            return gathered_tensor


    def get_lpcont_loss_pretrain(self, batch_dict, batch_dict_wa, CLIP_CE=False):
    #     """
    #     param pseudo_positives: Pseudo positive student features(Mk, Channel=256)
    #     param topk_labels: Labels for pseudo positive student features
    #     return:
    #     """
        print("lpcont loss!")
        contrastive_loss = torch.tensor(0.0, device = batch_dict['projected_features_gt'].device)
        print(batch_dict.keys())
        print(batch_dict_wa.keys())
        aligned_wa_info, aligned_sa_info = self._sort_instance_pairs(batch_dict, batch_dict_wa)
        gathered_sa_tensor = self.gather_tensors(aligned_sa_info)
        gathered_sa_labels = gathered_sa_tensor[:,-1].long()
        non_zero_mask = gathered_sa_labels != 0
        gathered_sa_feats = gathered_sa_tensor[:,:-1][non_zero_mask]
        gathered_labels_sa = gathered_sa_labels[non_zero_mask]

        gathered_wa_tensor = self.gather_tensors(aligned_wa_info)
        gathered_wa_labels = gathered_wa_tensor[:,-1].long()
        non_zero_mask2 = gathered_wa_labels != 0
        gathered_wa_feats = gathered_wa_tensor[:,:-1][non_zero_mask2]
        gathered_labels_wa = gathered_wa_labels[non_zero_mask2]
        label_mask = gathered_labels_sa.unsqueeze(1)== gathered_labels_wa.unsqueeze(0)
        positive_mask = label_mask
        negative_mask = ~label_mask           
        ## Product of conf_scores and pseudo_conf_scores as weights for negative_mask
        # confidence_weights = sorted_conf_scores.unsqueeze(1) * sorted_pseudo_conf_scores.unsqueeze(0)

        norm_sorted_prototypes = F.normalize(gathered_sa_feats, dim=-1)
        norm_pseudo_topk_features = F.normalize(gathered_wa_feats, dim=-1)
        sim_matrix = norm_sorted_prototypes @ norm_pseudo_topk_features.t()
        temperature = 1.0
        exp_sim_pos_matrix = torch.exp(sim_matrix * positive_mask /temperature)
        # sim_pos_matrix_row = exp_sim_pos_matrix.clone()
        positive_sum = torch.log(exp_sim_pos_matrix).sum() # Sum of log(exp(positives))
        # number_negative_pairs = negative_mask.sum(dim=0, keepdims=True) # Number of negative pairs for each positive
        negative_sum = (torch.exp(sim_matrix) * negative_mask).sum(dim=0, keepdims=True) #/((unique_labels.size(0)-1) * counts.min())
        pairwise_negatives_sum =  torch.log(negative_sum).sum()
        unscaled_contrastive_loss = ((positive_sum - pairwise_negatives_sum)) 
        contrastive_loss = contrastive_loss +  -1 * (unscaled_contrastive_loss / (norm_sorted_prototypes.size(0) * norm_pseudo_topk_features.size(0) * 3))            
        contrastive_loss = contrastive_loss * 0.2
        return contrastive_loss, (sim_matrix, gathered_labels_sa, gathered_labels_wa)


