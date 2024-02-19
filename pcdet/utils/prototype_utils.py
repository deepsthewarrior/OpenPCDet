import torch
import pickle
import os
import torch.nn as nn
from torch.functional import F
from torchmetrics import Metric
import numpy as np
from torch.distributions import Categorical


class FeatureBank(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):

        super().__init__()
        self.tag = kwargs.get('NAME', None)

        self.temperature = kwargs.get('TEMPERATURE')
        self.feat_size = kwargs.get('FEATURE_SIZE')
        self.bank_size = kwargs.get('BANK_SIZE')  # e.g., num. of classes or labeled instances
        self.momentum = kwargs.get('MOMENTUM')
        self.direct_update = kwargs.get('DIRECT_UPDATE')
        self.reset_state_interval = kwargs.get('RESET_STATE_INTERVAL')  # reset the state when N unique samples are seen
        self.num_points_thresh = kwargs.get('FILTER_MIN_POINTS_IN_GT', 0)
        self.record_pkl = kwargs.get('RECORD_PKL', False)
        self.initialized = False
        self.insId_protoId_mapping = None  # mapping from instance index to prototype index
        self.tt = kwargs.get('TT', 0.1)
        self.st = kwargs.get('ST', 0.1)
        proto_dict_keys = ['instance_prototypes','proto_labels']
        self.proto_dict = {key: [] for key in proto_dict_keys}
        # Globally synchronized prototypes used in each process
        self.prototypes = None
        self.classwise_prototypes = None
        self.proto_labels = None
        self.num_updates = None
        self.output_dir = None
        # Local feature/label which are used to update the global ones
        self.add_state('feats', default=[], dist_reduce_fx='cat')
        self.add_state('labels', default=[], dist_reduce_fx='cat')
        self.add_state('ins_ids', default=[], dist_reduce_fx='cat')
        self.add_state('smpl_ids', default=[], dist_reduce_fx='cat')
        self.add_state('iterations', default=[], dist_reduce_fx='cat')
        
    def _init(self, unique_ins_ids, labels):
        self.bank_size = len(unique_ins_ids)
        print(f"Initializing the feature bank with size {self.bank_size} and feature size {self.feat_size}")
        self.prototypes = torch.zeros((self.bank_size, self.feat_size)).cuda()
        self.classwise_prototypes = torch.zeros((3, self.feat_size)).cuda()
        self.proto_labels = labels
        self.num_updates = torch.zeros(self.bank_size).cuda()
        self.insId_protoId_mapping = {unique_ins_ids[i]: i for i in range(len(unique_ins_ids))}



    def update(self, feats: [torch.Tensor], labels: [torch.Tensor], ins_ids: [torch.Tensor], smpl_ids: torch.Tensor,
               iteration: int) -> None:
        for i in range(len(feats)):
            self.feats.append(feats[i])                 # (N, C)
            self.labels.append(labels[i].view(-1))      # (N,)
            self.ins_ids.append(ins_ids[i].view(-1))    # (N,)
            self.smpl_ids.append(smpl_ids[i].view(-1))  # (1,)
            rois_iter = torch.tensor(iteration, device=feats[0].device).expand_as(ins_ids[i].view(-1))
            self.iterations.append(rois_iter)           # (N,)
    def compute(self):
        try:
            unique_smpl_ids = torch.unique(torch.cat((self.smpl_ids,), dim=0))
        except:
            unique_smpl_ids = torch.unique(torch.cat((self.smpl_ids), dim=0))
        if len(unique_smpl_ids) < self.reset_state_interval:
            return None
        try:
            features = torch.cat((self.feats,), dim=0)
            ins_ids = torch.cat((self.ins_ids,),dim=0).int().cpu().numpy()
            labels = torch.cat((self.labels,), dim=0).int()
            iterations = torch.cat((self.iterations,),dim=0).int().cpu().numpy()
            ins_ids = torch.cat((self.ins_ids,), dim=0).int().cpu().numpy()
            iterations = torch.cat((self.iterations,), dim=0).int().cpu().numpy()
        except:
            features = torch.cat((self.feats), dim=0)
            ins_ids = torch.cat(self.ins_ids).int().cpu().numpy()
            labels = torch.cat((self.labels), dim=0).int()
            iterations = torch.cat(self.iterations).int().cpu().numpy()
            ins_ids = torch.cat((self.ins_ids), dim=0).int().cpu().numpy()
            iterations = torch.cat((self.iterations), dim=0).int().cpu().numpy()            
    
        assert len(features) == len(labels) == len(ins_ids) == len(iterations), \
            "length of features, labels, ins_ids, and iterations should be the same"
        sorted_ins_ids, arg_sorted_ins_ids = np.sort(ins_ids), np.argsort(ins_ids)
        unique_ins_ids, split_indices = np.unique(sorted_ins_ids, return_index=True)

        if not self.initialized:
            self._init(unique_ins_ids, labels[arg_sorted_ins_ids[split_indices]])

        # Group by ins_ids
        inds_groupby_ins_ids = np.split(arg_sorted_ins_ids, split_indices[1:])
        # For each group sort instance ids by iterations in ascending order and apply reduction operation
        for grouped_inds in inds_groupby_ins_ids:
            grouped_inds = grouped_inds[np.argsort(iterations[grouped_inds])]
            ins_id = ins_ids[grouped_inds[0]]
            proto_id = self.insId_protoId_mapping[ins_id]
            assert torch.allclose(labels[grouped_inds[0]], labels[grouped_inds]), "labels should be the same for the same instance id"

            if not self.initialized or self.direct_update:
                self.num_updates[proto_id] += len(grouped_inds)
                new_prototype = torch.mean(features[grouped_inds], dim=0, keepdim=True)  # TODO: maybe it'd be better to replaced it by the EMA
                self.prototypes[proto_id] = new_prototype
            else:
                for ind in grouped_inds:
                    new_prototype = self.momentum * self.prototypes[proto_id] + (1 - self.momentum) * features[ind]
                    self.prototypes[proto_id] = new_prototype
            if self.record_pkl:
                self._record_pkl(iterations)
        self._update_classwise_prototypes()
        self.initialized = True
        self.reset()
        return self.prototypes, self.proto_labels, self.num_updates, unique_smpl_ids

    def _update_classwise_prototypes(self):
        classwise_prototypes = torch.zeros((3, self.feat_size)).cuda()
        for i in range(3):  # TODO: refactor it
            inds = torch.where(self.proto_labels == i)[0]
            print(f"Update classwise prototypes for class {i} with {len(inds)} instances.")
            classwise_prototypes[i] = torch.mean(self.prototypes[inds], dim=0)
        self.classwise_prototypes = self.momentum * self.classwise_prototypes + (1 - self.momentum) * classwise_prototypes

    def _record_pkl(self,iterations):
        iteration = iterations.max().item()
        self.proto_dict['instance_prototypes'] = self.prototypes
        self.proto_dict['proto_labels'] = self.proto_labels
        file_path = os.path.join(self.output_dir, f'prototypes_{iteration}_iter.pkl')
        pickle.dump(self.proto_dict, open(file_path, 'wb'))        
        
    @torch.no_grad()
    def get_sim_scores(self, input_features, use_classwise_prototypes=True,return_raw_scores=False):
        assert input_features.shape[1] == self.feat_size, "input feature size is not equal to the bank feature size"
        if not self.initialized:
            return input_features.new_zeros(input_features.shape[0], 3)
        if return_raw_scores and use_classwise_prototypes: 
            return F.normalize(input_features) @ F.normalize(self.classwise_prototypes).t() #without temperature and softmax
        if use_classwise_prototypes: 
            cos_sim = F.normalize(input_features) @ F.normalize(self.classwise_prototypes).t()
            return F.softmax(cos_sim / self.temperature, dim=-1) #with temperature and softmax
        else:
            return self._get_sim_scores_with_instance_prototypes(input_features)

    def _get_sim_scores_with_instance_prototypes(self, input_features):
        cos_sim = F.normalize(input_features) @ F.normalize(self.prototypes).t()
        norm_cos_sim = F.softmax(cos_sim / self.temperature, dim=-1)
        classwise_sim = cos_sim.new_zeros(input_features.shape[0], 3)
        lbs = self.proto_labels.expand_as(cos_sim).long()
        classwise_sim.scatter_add_(1, lbs, (cos_sim/self.temperature))
        # classwise_sim.scatter_add_(1, lbs, cos_sim)
        # protos_cls_counts = torch.bincount(self.proto_labels).view(1, -1)
        # classwise_sim /= protos_cls_counts  # Note: not probability
        classwise_sim /= classwise_sim.mean(dim=0)
        return classwise_sim

    def get_pairwise_protos_sim_matrix(self):
        sorted_lbs, arg_sorted_lbs = torch.sort(self.proto_labels)
        protos = self.prototypes[arg_sorted_lbs]
        sim_matrix = F.normalize(protos) @ F.normalize(protos).t()

        return sim_matrix.cpu().numpy(), sorted_lbs.cpu().numpy()

    def get_proto_contrastive_loss(self, feats, labels):
        """
        :param feats: pseudo-box features of the strongly augmented unlabeled samples (N, C)
        :param labels: pseudo-labels of the strongly augmented unlabeled samples (N,)
        :return:
        """
        if not self.initialized:
            return None
        sim_scores = F.normalize(feats) @ F.normalize(self.classwise_prototypes).t()
        log_probs = F.log_softmax(sim_scores / self.temperature, dim=-1)
        return -log_probs[torch.arange(len(labels)), labels]

    def get_simmatch_loss(self, feats_wa, feats_sa, labels):
        """
        :param feats_w: pseudo-box features of the weakly augmented unlabeled samples (N, C)
        :param feats: pseudo-labels of the strongly augmented unlabeled samples (N,)
        :return:
        """

        if not self.initialized:
            return None
        cos_sim_wa = F.normalize(feats_wa) @ F.normalize(self.prototypes).t()
        norm_cos_sim_wa = F.softmax(cos_sim_wa / self.temperature, dim=-1)

        cos_sim_sa = F.normalize(feats_sa) @ F.normalize(self.prototypes).t() #160 instance protos => 134 11 8  1,1,1
        log_norm_cos_sim_sa = -1 * F.log_softmax(cos_sim_sa / self.temperature, dim=-1)


        return [(norm_cos_sim_wa * log_norm_cos_sim_sa), self.proto_labels, cos_sim_wa, cos_sim_sa]

    def get_simmatch_mean_loss(self, feats_wa, feats_sa, labels):
        """
        :param feats_w: pseudo-box features of the weakly augmented unlabeled samples (N, C)
        :param feats: pseudo-labels of the strongly augmented unlabeled samples (N,)
        :return:
        """

        if not self.initialized:
            return None
        cos_sim_wa = F.normalize(feats_wa) @ F.normalize(self.prototypes).t()
        norm_cos_sim_wa = F.softmax(cos_sim_wa / self.temperature, dim=-1)
        
        cos_sim_sa = F.normalize(feats_sa) @ F.normalize(self.prototypes).t()
        norm_cos_sim_sa = F.softmax(cos_sim_sa / self.temperature, dim=-1)

        masks = [self.proto_labels == cind for cind in range(0,3)]
        masks = torch.stack(masks, dim=0)

        classwise_sim_wa = [norm_cos_sim_wa[:,masks[0]].mean(-1),norm_cos_sim_wa[:,masks[1]].mean(-1),norm_cos_sim_wa[:,masks[2]].mean(-1)]
        classwise_sim_wa = torch.stack(classwise_sim_wa,dim=1)

        classwise_sim_sa = [norm_cos_sim_sa[:,masks[0]].mean(-1),norm_cos_sim_sa[:,masks[1]].mean(-1),norm_cos_sim_sa[:,masks[2]].mean(-1)]
        classwise_sim_sa = torch.stack(classwise_sim_sa,dim=1)

        log_classwise_cos_sim_sa = -1 * torch.log(classwise_sim_sa / self.temperature)

        # log_norm_cos_sim_sa = -1 * F.log_softmax(cos_sim_sa / self.temperature, dim=-1)
        

        return [(classwise_sim_wa * log_classwise_cos_sim_sa), self.proto_labels, cos_sim_wa, cos_sim_sa]


    def _get_multi_cont_loss(self,):
        """
        k: number of classes
        n: number of instances in each class

        :return:
        MCont Loss for each class as per the paper
        """
        if not self.initialized:
            return None
        else:
            # sort the prototypes according to the labels and filter the prototypes according to the labels
            # sorted_protos,sorted_indices = torch.sort(self.proto_labels)
            # sorted_prototypes = self.prototypes[sorted_indices,:]

            # car_protos = torch.zeros_like(self.prototypes[self.proto_labels == 0])
            # ped_protos = torch.zeros_like(self.prototypes[self.proto_labels == 1])
            # cyc_protos = torch.zeros_like(self.prototypes[self.proto_labels == 2])
            
            car_mask = self.proto_labels == 0
            ped_mask = self.proto_labels == 1
            cyc_mask = self.proto_labels == 2

            car_protos =  torch.gather(self.prototypes,0,car_mask.nonzero().expand(-1,256))
            ped_protos =  torch.gather(self.prototypes,0,ped_mask.nonzero().expand(-1,256))
            cyc_protos =  torch.gather(self.prototypes,0,cyc_mask.nonzero().expand(-1,256))


            # get the positive and negative samples for each class
            # positive samples are the [1 2 3 1 2 3 1 2 3] (repeated n times, n is the number of instances in each class)
            # negative samples are the [3 2 1 1 3 2 2 1 3] (flipped and rolled n times)
            car_protos_pos = car_protos.repeat(car_protos.shape[0],1)
            car_protos_neg = torch.flip(car_protos.clone(),dims=[0])
            car_protos_neg_sub = car_protos_neg.clone()
            for i in range(car_protos.shape[0]-1):
                car_protos_neg_sub = torch.roll(car_protos_neg_sub,1,dims=0)
                car_protos_neg = torch.cat((car_protos_neg,car_protos_neg_sub),dim=0) # (134*134), 256 => (17956,256)
            
            ped_protos_pos = ped_protos.repeat(ped_protos.shape[0],1)
            ped_protos_neg =  torch.flip(ped_protos.clone(),dims=[0])
            ped_protos_neg_sub = ped_protos_neg.clone()
            for i in range(ped_protos.shape[0]-1):
                ped_protos_neg_sub = torch.roll(ped_protos_neg_sub,1,dims=0)
                ped_protos_neg = torch.cat((ped_protos_neg,ped_protos_neg_sub),dim=0) # (18*18), 256 => (324,256)

            cyc_protos_pos = cyc_protos.repeat(cyc_protos.shape[0],1)
            cyc_protos_neg = torch.flip(cyc_protos.clone(),dims=[0])
            cyc_protos_neg_sub = cyc_protos.clone()
            for i in range(cyc_protos.shape[0]-1):
                cyc_protos_neg_sub = torch.roll(cyc_protos_neg_sub,1,dims=0)
                cyc_protos_neg = torch.cat((cyc_protos_neg,cyc_protos_neg_sub),dim=0) # (9*9), 256 => (81,256)
            
            dims_max = car_protos_pos.shape[0] if car_protos_pos.shape[0] > ped_protos_pos.shape[0] else ped_protos_pos.shape[0]
            dims_max = cyc_protos_pos.shape[0] if cyc_protos_pos.shape[0] > dims_max else dims_max

            feat_car_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - car_protos_pos.shape[0]))(car_protos_pos),1).reshape(1,-1) #(17956,256) => (17956,256) = (1,4596736)
            feat_ped_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - ped_protos_pos.shape[0]))(ped_protos_pos),1).reshape(1,-1) #(324,256) => (17956,256) = (1,4596736)
            feat_cyc_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - cyc_protos_pos.shape[0]))(cyc_protos_pos),1).reshape(1,-1) #(81,256) => (17956,256) = (1,4596736)
            
            feat_car_rev_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - car_protos_neg.shape[0]))(car_protos_neg),1).reshape(1,-1) 
            feat_ped_rev_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - ped_protos_neg.shape[0]))(ped_protos_neg),1).reshape(1,-1)
            feat_cyc_rev_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - cyc_protos_neg.shape[0]))(cyc_protos_neg),1).reshape(1,-1)

            feat_pos = torch.cat((feat_car_flatten,feat_ped_flatten,feat_cyc_flatten),dim=0) # (3,17956,256)
            feat_neg = torch.cat((feat_car_rev_flatten,feat_ped_rev_flatten,feat_cyc_rev_flatten),dim=0) # (3,17956,256) 17k [1] [0.4] 81/17K

            car_protos_expanded = car_protos_pos.shape[0]
            ped_protos_expanded = ped_protos_pos.shape[0]
            cyc_protos_expanded = cyc_protos_pos.shape[0]
            # sum in loss as such does not make sense, in imbalanced case
            car_norm = [min(car_protos_expanded,car_protos_expanded),min(car_protos_expanded,ped_protos_expanded),min(car_protos_expanded,cyc_protos_expanded)]
            ped_norm = [min(ped_protos_expanded,car_protos_expanded),min(ped_protos_expanded,ped_protos_expanded),min(ped_protos_expanded,cyc_protos_expanded)]
            cyc_norm = [min(cyc_protos_expanded,car_protos_expanded),min(cyc_protos_expanded,ped_protos_expanded),min(cyc_protos_expanded,cyc_protos_expanded)]

            total_norm = torch.tensor([car_norm,ped_norm,cyc_norm],dtype=torch.float32).cuda() # divide it by norm to scale the values between 0 and 1, also reducing the effect of extra zeroes added


            logits = feat_pos @ feat_neg.t()
            sharpened_logits = (logits / total_norm) / 0.2 #(0.005 is the temperature making the resonably sharp. need to be tuned)
            # MultiCrossEntropyLoss
            labels = torch.diag(torch.ones_like(logits[0]))
            loss = -1 * labels * F.log_softmax(sharpened_logits,dim=-1)
            return {
                    'total_loss': loss.sum(),
                    'classwise_loss': loss.sum(dim=-1),
                    'raw_logits': logits
                    }

    def _get_multi_cont_loss_lb_instances(self,features, labels):

        """
        Calculates the multi contrastive loss for the labeled instances

        Args:
        features: (N,256)
        labels: (N,)

        Returns:
        {
                'total_loss': actual loss
                'classwise_loss': claswise loss for metrics
                'raw_logits': raw logits for pkl visualization
                } 

        """

        # sort the prototypes according to the labels and filter the prototypes according to the labels
        # sorted_protos,sorted_indices = torch.sort(self.proto_labels)
        # sorted_prototypes = self.prototypes[sorted_indices,:]

        labels = labels - 1

        assert labels.min() >= 0 and labels.max() <= 2, "labels should be in the range [0, 2]"

        car_mask = labels == 0
        ped_mask = labels == 1
        cyc_mask = labels == 2

        car_protos =  torch.gather(features,0,car_mask.nonzero().expand(-1,256))
        ped_protos =  torch.gather(features,0,ped_mask.nonzero().expand(-1,256))
        cyc_protos =  torch.gather(features,0,cyc_mask.nonzero().expand(-1,256))


        # get the positive and negative samples for each class
        # positive samples are the [1 2 3 1 2 3 1 2 3] (repeated n times, n is the number of instances in each class)
        # negative samples are the [3 2 1 1 3 2 2 1 3] (flipped and rolled n times)
        car_protos_pos = car_protos.repeat(car_protos.shape[0],1)
        car_protos_neg = torch.flip(car_protos.clone(),dims=[0])
        car_protos_neg_sub = car_protos_neg.clone()
        for i in range(car_protos.shape[0]-1):
            car_protos_neg_sub = torch.roll(car_protos_neg_sub,1,dims=0)
            car_protos_neg = torch.cat((car_protos_neg,car_protos_neg_sub),dim=0) # (134*134), 256 => (17956,256)
        
        ped_protos_pos = ped_protos.repeat(ped_protos.shape[0],1)
        ped_protos_neg =  torch.flip(ped_protos.clone(),dims=[0])
        ped_protos_neg_sub = ped_protos_neg.clone()
        for i in range(ped_protos.shape[0]-1):
            ped_protos_neg_sub = torch.roll(ped_protos_neg_sub,1,dims=0)
            ped_protos_neg = torch.cat((ped_protos_neg,ped_protos_neg_sub),dim=0) # (18*18), 256 => (324,256)

        cyc_protos_pos = cyc_protos.repeat(cyc_protos.shape[0],1)
        cyc_protos_neg = torch.flip(cyc_protos.clone(),dims=[0])
        cyc_protos_neg_sub = cyc_protos.clone()
        for i in range(cyc_protos.shape[0]-1):
            cyc_protos_neg_sub = torch.roll(cyc_protos_neg_sub,1,dims=0)
            cyc_protos_neg = torch.cat((cyc_protos_neg,cyc_protos_neg_sub),dim=0) # (9*9), 256 => (81,256)
        
        dims_max = car_protos_pos.shape[0] if car_protos_pos.shape[0] > ped_protos_pos.shape[0] else ped_protos_pos.shape[0]
        dims_max = cyc_protos_pos.shape[0] if cyc_protos_pos.shape[0] > dims_max else dims_max

        feat_car_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - car_protos_pos.shape[0]))(car_protos_pos),1).reshape(1,-1) #(17956,256) => (17956,256) = (1,4596736)
        feat_ped_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - ped_protos_pos.shape[0]))(ped_protos_pos),1).reshape(1,-1) #(324,256) => (17956,256) = (1,4596736)
        feat_cyc_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - cyc_protos_pos.shape[0]))(cyc_protos_pos),1).reshape(1,-1) #(81,256) => (17956,256) = (1,4596736)
        
        feat_car_rev_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - car_protos_neg.shape[0]))(car_protos_neg),1).reshape(1,-1) 
        feat_ped_rev_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - ped_protos_neg.shape[0]))(ped_protos_neg),1).reshape(1,-1)
        feat_cyc_rev_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - cyc_protos_neg.shape[0]))(cyc_protos_neg),1).reshape(1,-1)

        feat_pos = torch.cat((feat_car_flatten,feat_ped_flatten,feat_cyc_flatten),dim=0) # (3,17956,256)
        feat_neg = torch.cat((feat_car_rev_flatten,feat_ped_rev_flatten,feat_cyc_rev_flatten),dim=0) # (3,17956,256) 17k [1] [0.4] 81/17K

        car_protos_expanded = car_protos_pos.shape[0]
        ped_protos_expanded = ped_protos_pos.shape[0]
        cyc_protos_expanded = cyc_protos_pos.shape[0]
        # sum in loss as such does not make sense, in imbalanced case
        car_norm = [min(car_protos_expanded,car_protos_expanded),min(car_protos_expanded,ped_protos_expanded),min(car_protos_expanded,cyc_protos_expanded)]
        ped_norm = [min(ped_protos_expanded,car_protos_expanded),min(ped_protos_expanded,ped_protos_expanded),min(ped_protos_expanded,cyc_protos_expanded)]
        cyc_norm = [min(cyc_protos_expanded,car_protos_expanded),min(cyc_protos_expanded,ped_protos_expanded),min(cyc_protos_expanded,cyc_protos_expanded)]

        total_norm = torch.tensor([car_norm,ped_norm,cyc_norm],dtype=torch.float32).cuda() # divide it by norm to scale the values between 0 and 1, also reducing the effect of extra zeroes added


        logits = feat_pos @ feat_neg.t()
        sharpened_logits = (logits / total_norm) / 0.2 #(0.005 is the temperature making the resonably sharp. need to be tuned)
        # MultiCrossEntropyLoss
        labels = torch.diag(torch.ones_like(logits[0]))
        loss = -1 * labels * F.log_softmax(sharpened_logits,dim=-1)
        return {
                'total_loss': loss.sum(),
                'classwise_loss': loss.sum(dim=-1),
                'raw_logits': logits
                } 

    def get_proto_sim_loss(self,feats_wa,feats_sa):        
        if not self.initialized:
            return None
        feats_sa = feats_sa.view(-1,self.feat_size)
        feats_wa = feats_wa.view(-1,self.feat_size)
        cos_sim_wa = F.normalize(feats_wa) @ F.normalize(self.classwise_prototypes).t()
        norm_cos_sim_wa = F.softmax(cos_sim_wa / self.tt, dim=-1)
        cos_sim_sa = F.normalize(feats_sa) @ F.normalize(self.classwise_prototypes).t()
        log_norm_cos_sim_sa = F.log_softmax(cos_sim_sa / self.st, dim=-1)

        loss = -1 * (norm_cos_sim_wa * log_norm_cos_sim_sa)
        
        return {
                'total_loss': loss,
                'cos_sim_wa': cos_sim_wa,
                'cos_sim_sa': cos_sim_sa,
                } 

class FeatureBankRegistry(object):
    def __init__(self, **kwargs):
        self._banks = {}

    def register(self, tag=None, **bank_configs):
        if tag is None:
            tag = 'default'
        if tag in self.tags():
            raise ValueError(f'Feature bank with tag {tag} already exists')
        bank = FeatureBank(**bank_configs)
        self._banks[tag] = bank
        return self._banks[tag]

    def get(self, tag=None):
        if tag is None:
            tag = 'default'
        if tag not in self.tags():
            raise ValueError(f'Feature bank with tag {tag} does not exist')
        return self._banks[tag]

    def tags(self):
        return self._banks.keys()


feature_bank_registry = FeatureBankRegistry()
