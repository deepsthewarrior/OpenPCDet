import torch
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
        self.reset_state_interval = kwargs.get('RESET_STATE_INTERVAL',1)  # reset the state when N unique samples are seen
        self.num_points_thresh = kwargs.get('FILTER_MIN_POINTS_IN_GT', 0)

        self.initialized = False
        self.insId_protoId_mapping = None  # mapping from instance index to prototype index

        # Globally synchronized prototypes used in each process
        self.prototypes = None
        self.classwise_prototypes = None
        self.proto_labels = None
        self.num_updates = None

        # Local feature/label which are used to update the global ones
        self.add_state('feats', default=[], dist_reduce_fx='cat')
        self.add_state('labels', default=[], dist_reduce_fx='cat')
        # self.add_state('ins_ids', default=[], dist_reduce_fx='cat')
        # self.add_state('smpl_ids', default=[], dist_reduce_fx='cat')
        self.add_state('iterations', default=[], dist_reduce_fx='cat')


    def update(self, feats: [torch.Tensor], labels: [torch.Tensor],
               iteration: int) -> None:
        for i in range(len(feats)):
            self.feats.append(feats[i])                 # (N, C)
            self.labels.append(labels[i].view(-1))      # (N,)
            # self.ins_ids.append(ins_ids[i].view(-1))    # (N,)
            # self.smpl_ids.append(smpl_ids[i].view(-1))  # (1,)
            rois_iter = torch.tensor(iteration, device=feats[0].device).expand_as(labels[i].view(-1))
            self.iterations.append(rois_iter)           # (N,)

    def compute(self):
        # unique_smpl_ids = torch.unique(torch.cat((self.smpl_ids,), dim=0))
        try:
            features = torch.cat((self.feats,), dim=0)
            # ins_ids = torch.cat(self.ins_ids).int().cpu().numpy()
            labels = torch.cat((self.labels,), dim=0).int()
            iterations = torch.cat((self.iterations,), dim=0).int().cpu().numpy()
        except:
            features = torch.cat((self.feats), dim=0)
            # ins_ids = torch.cat(self.ins_ids).int().cpu().numpy()
            labels = torch.cat((self.labels), dim=0).int()
            iterations = torch.cat(self.iterations).int().cpu().numpy()
        
        assert len(features) == len(labels) == len(iterations), \
            "length of features, labels, ins_ids, and iterations should be the same"
        
        self._update_classwise_prototypes(labels,features,iterations)
        self.initialized = True
        self.reset()

    def _update_classwise_prototypes(self,labels,features,iterations):
        classwise_prototypes = torch.zeros((3, self.feat_size)).cuda()
        for i in range(3):  # TODO: refactor it
            inds = torch.where( labels == i)[0]
            print(f"Update classwise prototypes for class {i} with {len(inds)} instances.")
            classwise_prototypes[i] = torch.mean(features[inds], dim=0)
        if iterations.max() < 20:
            self.classwise_prototypes = classwise_prototypes
        else:
            self.classwise_prototypes = self.momentum * self.classwise_prototypes + (1 - self.momentum) * classwise_prototypes

    @torch.no_grad()
    def get_sim_scores(self, input_features, use_classwise_prototypes=True):
        assert input_features.shape[1] == self.feat_size, "input feature size is not equal to the bank feature size"
        if not self.initialized:
            return input_features.new_zeros(input_features.shape[0], 3)
        cos_sim = F.normalize(input_features) @ F.normalize(self.classwise_prototypes).t()
        return F.softmax(cos_sim / self.temperature, dim=-1)



    def get_proto_contrastive_loss(self, feats, labels):
        """
        :param feats: pseudo-box features of the strongly augmented unlabeled samples (N, C)
        :param labels: pseudo-labels of the strongly augmented unlabeled samples (N,)
        :return:
        """
        if not self.initialized: 
            return F.normalize(feats) @ torch.zeros(1,256).t().to(feats.device)
        sim_scores = F.normalize(feats) @ F.normalize(self.classwise_prototypes).t()
        log_probs = F.log_softmax(sim_scores / self.temperature, dim=-1)
        return -log_probs[torch.arange(len(labels)), labels]


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
