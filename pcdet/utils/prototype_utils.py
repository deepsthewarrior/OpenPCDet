import torch
from torch.functional import F
from torchmetrics import Metric
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.distributed as dist
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics import precision_score


def _arr2dict(array, ignore_zeros=False, ignore_nan=False):
    def should_include(value):
        return not ((ignore_zeros and value == 0) or (ignore_nan and np.isnan(value)))

    classes = ['Bg', 'Fg'] if array.shape[-1] == 2 else ['Car', 'Pedestrian', 'Cyclist']
    return {cls: array[cind] for cind, cls in enumerate(classes) if should_include(array[cind])}

class FeatureBank(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):

        super().__init__()
        self.tag = kwargs.get('NAME', None)
        self.temperature = kwargs.get('TEMPERATURE')
        self.num_classes = 3
        self.feat_size = kwargs.get('FEATURE_SIZE')
        self.bank_size = kwargs.get('BANK_SIZE')  # e.g., num. of classes or labeled instances
        self.momentum = kwargs.get('MOMENTUM')
        self.direct_update = kwargs.get('DIRECT_UPDATE')
        self.reset_state_interval = 1 #kwargs.get('RESET_STATE_INTERVAL')  # reset the state when N unique samples are seen
        self.num_points_thresh = kwargs.get('FILTER_MIN_POINTS_IN_GT', 0)

        self.initialized = False
        self.insId_protoId_mapping = None  # mapping from instance index to prototype index

        # Globally synchronized prototypes used in each process
        self.prototypes = None
        self.classwise_prototypes = None
        self.classwise_meta_instance = None
        self.proto_labels = None
        self.num_updates = None
        self.pp_feats = None
        self.pp_labels = None
        self.proto_conf_scores = None

        # Local feature/label which are used to update the global ones
        self.add_state('feats', default=[], dist_reduce_fx='cat')
        self.add_state('labels', default=[], dist_reduce_fx='cat')
        self.add_state('ins_ids', default=[], dist_reduce_fx='cat')
        self.add_state('smpl_ids', default=[], dist_reduce_fx='cat')
        self.add_state('iterations', default=[], dist_reduce_fx='cat')
        self.add_state('conf_scores', default=[], dist_reduce_fx='cat')

    def _init(self, unique_ins_ids, labels):
        self.bank_size = len(unique_ins_ids)
        print(f"Initializing the feature bank with size {self.bank_size} and feature size {self.feat_size}")
        self.prototypes = torch.zeros((self.bank_size, self.feat_size)).cuda()
        self.classwise_prototypes = torch.zeros((3, self.feat_size)).cuda()
        self.proto_labels = labels
        self.num_updates = torch.zeros(self.bank_size).cuda()
        self.insId_protoId_mapping = {unique_ins_ids[i]: i for i in range(len(unique_ins_ids))}

    def update(self, feats: [torch.Tensor], labels: [torch.Tensor], ins_ids: [torch.Tensor], smpl_ids: torch.Tensor,
               conf_scores: [torch.Tensor], iteration: int) -> None:
        for i in range(len(feats)):
            self.feats.append(feats[i])                 # (N, C)
            self.labels.append(labels[i].view(-1))      # (N,)
            self.ins_ids.append(ins_ids[i].view(-1))    # (N,)
            self.smpl_ids.append(smpl_ids[i].view(-1))  # (1,)     
            self.conf_scores.append(conf_scores[i].view(-1))  # (N,)
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
            conf_scores = torch.cat((self.conf_scores,),dim=0)
        except:
            features = torch.cat((self.feats), dim=0)
            ins_ids = torch.cat(self.ins_ids).int().cpu().numpy()
            labels = torch.cat((self.labels), dim=0).int()
            iterations = torch.cat(self.iterations).int().cpu().numpy()
            ins_ids = torch.cat((self.ins_ids), dim=0).int().cpu().numpy()
            iterations = torch.cat((self.iterations), dim=0).int().cpu().numpy()            
            conf_scores = torch.cat((self.conf_scores),dim=0)

        assert len(features) == len(labels) == len(ins_ids) == len(iterations) == len(conf_scores), \
            "length of features, labels, ins_ids, conf_scores and iterations should be the same"
        sorted_ins_ids, arg_sorted_ins_ids = np.sort(ins_ids), np.argsort(ins_ids)
        unique_ins_ids, split_indices = np.unique(sorted_ins_ids, return_index=True)
        self.proto_conf_scores = conf_scores[arg_sorted_ins_ids[split_indices]] # Update proto_conf_scores at every compute() call -> with fresh batch's conf_scores

        if not self.initialized:
            self._init(unique_ins_ids, labels[arg_sorted_ins_ids[split_indices]])

        # Group by ins_ids
        inds_groupby_ins_ids = np.split(arg_sorted_ins_ids, split_indices[1:])
        # For each group sort instance ids by iterations in ascending order and apply reduction operation
        for grouped_inds in inds_groupby_ins_ids:
            grouped_inds = grouped_inds[np.argsort(iterations[grouped_inds])]
            if ins_ids[grouped_inds[0]] in self.insId_protoId_mapping:
                ins_id = ins_ids[grouped_inds[0]]
            try:
                proto_id = self.insId_protoId_mapping[ins_id]
            except KeyError:
                continue
            assert torch.allclose(labels[grouped_inds[0]], labels[grouped_inds]), "labels should be the same for the same instance id"

            if not self.initialized or self.direct_update:
                self.num_updates[proto_id] += len(grouped_inds)
                new_prototype = torch.mean(features[grouped_inds], dim=0, keepdim=True)  # Inconsitent- some samples get updated only once, some get 40 updates. Better to use features[grouped_inds][-1] than mean()
                self.prototypes[proto_id] = new_prototype
            else:
                for ind in grouped_inds:
                    new_prototype = self.momentum * self.prototypes[proto_id] + (1 - self.momentum) * features[ind]
                    self.prototypes[proto_id] = new_prototype
        self._update_classwise_prototypes()
        self.initialized = True
        self.reset()
        return self.prototypes, self.proto_labels, self.num_updates

    def _update_classwise_prototypes(self):
        classwise_prototypes = torch.zeros((3, self.feat_size)).cuda()
        for i in range(self.num_classes):
            inds = torch.where(self.proto_labels == (i+1))[0]
            print(f"Update classwise prototypes for class {(i+1)} with {len(inds)} instances.")
            classwise_prototypes[i] = torch.mean(self.prototypes[inds], dim=0)
        self.classwise_prototypes = self.momentum * self.classwise_prototypes + (1 - self.momentum) * classwise_prototypes


    # def get_proto_contrastive_loss(self, feats, labels):
    #     """
    #     :param feats: pseudo-box features of the strongly augmented unlabeled samples (N, C)
    #     :param labels: pseudo-labels of the strongly augmented unlabeled samples (N,)
    #     :return:
    #     """
    #     if not self.initialized:
    #         return None
    #     sim_scores = F.normalize(feats) @ F.normalize(self.classwise_prototypes).t()
    #     log_probs = F.log_softmax(sim_scores / self.temperature, dim=-1)
    #     return -log_probs[torch.arange(len(labels)), labels]

    def is_initialized(self):
        return self.initialized 
    
    def get_computed_dict(self):
        return self.prototypes, self.proto_labels, self.num_updates

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