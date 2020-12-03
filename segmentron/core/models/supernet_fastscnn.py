import copy
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from ..models.common_layers import get_nddr
from ..utils import AttrDict
from ..tasks import get_tasks
from ..utils.losses import poly, entropy_loss, l1_loss
sys.path.append('../..')
from ...models.fast_scnn import FeatureFusionModule, Classifer
from ...solver.loss import get_segmentation_loss

class GeneralizedFastSCNNNet(nn.Module):
    def __init__(self, cfg, net_branch, net_fusion,
                 net1_connectivity_matrix,
                 net2_connectivity_matrix,
                 net1_net2_factor=4,
                 criterion=nn.CrossEntropyLoss
                ):
        """
        :param net1: task one network
        :param net2: task two network
        :param net1_connectivity_matrix: Adjacency list for the Single sided NDDR connections
        :param net2_connectivity_matrix: Adjacency list for the Single sided NDDR connections
        """
        super(GeneralizedFastSCNNNet, self).__init__()
        self.cfg = cfg
        self.nclass = cfg.DATASET.NUM_CLASS
        self.aux = cfg.SOLVER.AUX
        self.net_branch = net_branch
        self.net1 = net_fusion[0]
        self.net2 = net_fusion[1]
        self.net1_net2_factor = net1_net2_factor
        self.criterion = criterion

        assert len(self.net1.stages) == len(self.net2.stages)
        print("Model has %d stages" % len(self.net2.stages))
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, self.nclass)

        # self.task1, self.task2 = get_tasks(cfg)
        self.num_stages = len(self.net1.stages)
        self.net1_connectivity_matrix = net1_connectivity_matrix
        self.net2_connectivity_matrix = net2_connectivity_matrix
        net1_shared_index = net1_connectivity_matrix.sum(axis=1)
        net2_shared_index = net2_connectivity_matrix.sum(axis=1)
        net1_fusion_ops = []  # used for incoming feature fusion
        net2_fusion_ops = []  # used for incoming feature fusion

        if is not cfg.ARCH.SKIP_CONNECTION:
            for stage_id in range(self.num_stages):
                n_channel = self.net1.stages[stage_id].out_channels
                net1_op = get_nddr(cfg,
                    128,  # +1 for original upstream input
                    64, self.net1_net2_factor)
                net2_op = get_nddr(cfg,
                    64,  # +1 for original upstream input
                    128, 1/self.net1_net2_factor)
                net1_fusion_ops.append(net1_op)
                net2_fusion_ops.append(net2_op)
        else:
            for stage_id in range(self.num_stages):
                n_channel = self.net1.stages[stage_id].out_channels
                net1_op = get_nddr(cfg,
                    128,  # +1 for original upstream input
                    64, self.net1_net2_factor)
                net2_op = get_nddr(cfg,
                    64,  # +1 for original upstream input
                    128, 1/self.net1_net2_factor)
                net1_fusion_ops.append(net1_op)
                net2_fusion_ops.append(net2_op)
                

        net1_fusion_ops = nn.ModuleList(net1_fusion_ops)
        net2_fusion_ops = nn.ModuleList(net2_fusion_ops)

        self.net1_alphas = nn.Parameter(torch.zeros(net1_connectivity_matrix.shape))
        self.net2_alphas = nn.Parameter(torch.zeros(net2_connectivity_matrix.shape))

        if cfg.ARCH.SKIP_CONNECTION:
            self.net1_skip_alphas = nn.Parameter(torch.zeros(net1_connectivity_matrix.shape))
            self.net2_skip_alphas = nn.Parameter(torch.zeros(net2_connectivity_matrix.shape))

        self.paths = nn.ModuleDict({
            'net1_paths': net1_fusion_ops,
            'net2_paths': net2_fusion_ops,
        })
        
        path_cost = np.array([stage.out_channels for stage in self.net1.stages])
        path_cost = path_cost[:, None] * net1_connectivity_matrix
        path_cost = path_cost * net1_connectivity_matrix.sum() / path_cost.sum()
        path_cost = path_cost[np.nonzero(path_cost)]
        
        self.register_buffer("path_costs", torch.tensor(path_cost).float())

        self._step = 0

        self._arch_parameters = dict()
        self._net_parameters = dict()
        for k, v in self.named_parameters():
            if 'alpha' in k:
                self._arch_parameters[k] = v
            else:
                self._net_parameters[k] = v

        self.hard_weight_training = cfg.ARCH.HARD_WEIGHT_TRAINING
        self.hard_arch_training = cfg.ARCH.HARD_ARCH_TRAINING
        self.hard_evaluation = cfg.ARCH.HARD_EVAL
        self.stochastic_evaluation = cfg.ARCH.STOCHASTIC_EVAL

        self.arch_training = False
        self.retraining = False
        
        self.supernet = False
        if cfg.MODEL.SUPERNET:
            print("Running Supernet Baseline")
            self.supernet = True

    def loss(self, image, labels):
        result = self.forward(image)
        loss = self.criterion(result, labels)

        if self.arch_training:
            arch_parameters = [
                self.net1_alphas[np.nonzero(self.net1_connectivity_matrix)],
                self.net2_alphas[np.nonzero(self.net2_connectivity_matrix)],
            ]
            if self.cfg.ARCH.ENTROPY_REGULARIZATION:
                entropy_los = entropy_loss(arch_parameters)
                entropy_weight = poly(start=0., end=self.cfg.ARCH.ENTROPY_REGULARIZATION_WEIGHT,
                                    steps=self._step, total_steps=self.cfg.TRAIN.EPOCHS,
                                    period=self.cfg.ARCH.ENTROPY_PERIOD,
                                    power=1.)
                loss['loss'] += entropy_weight * entropy_los
            if self.cfg.ARCH.L1_REGULARIZATION:
                if self.cfg.ARCH.WEIGHTED_L1:
                    l1_los = l1_loss(arch_parameters, self.path_costs)
                else:
                    l1_los = l1_loss(arch_parameters)
                l1_weight = poly(start=0., end=self.cfg.ARCH.L1_REGULARIZATION_WEIGHT,
                                steps=self._step, total_steps=self.cfg.TRAIN.EPOCHS,
                                period=self.cfg.ARCH.L1_PERIOD,
                                power=1.)
                if float(self._step) / self.cfg.TRAIN.EPOCHS > self.cfg.ARCH.L1_PERIOD[1] and self.cfg.ARCH.L1_OFF:
                    l1_weight = 0.
                loss['loss'] += l1_weight * l1_los
        return loss

    def new(self):
        return copy.deepcopy(self)

    def step(self):  # update temperature
        self._step += 1

    def get_temperature(self):
        return poly(start=self.cfg.ARCH.INIT_TEMP, end=0.,
                    steps=self._step, total_steps=self.cfg.TRAIN.EPOCHS,
                    period=self.cfg.ARCH.TEMPERATURE_PERIOD,
                    power=self.cfg.ARCH.TEMPERATURE_POWER)

    def arch_parameters(self):
        return self._arch_parameters.values()

    def named_arch_parameters(self):
        return self._arch_parameters.items()

    def net_parameters(self):
        return self._net_parameters.values()

    def named_net_parameters(self):
        return self._net_parameters.items()

    def arch_train(self):
        self.arch_training = True

    def arch_eval(self):
        self.arch_training = False

    def retrain(self):
        self.retraining = True

    def gumbel_connectivity(self, path_weights):
        temp = self.get_temperature()
        net_dist = dist.relaxed_bernoulli.RelaxedBernoulli(temp, logits=path_weights)
        path_connectivity = net_dist.rsample()
        return path_connectivity
    
    def bernoulli_connectivity(self, path_weights):
        net_dist = dist.bernoulli.Bernoulli(logits=path_weights)
        path_connectivity = net_dist.sample()
        return path_connectivity

    def sigmoid_connectivity(self, path_weights):
        return torch.sigmoid(path_weights)

    def onehot_connectivity(self, path_weights):
        path_connectivity = (path_weights > 0.).float()
        return path_connectivity
    
    def all_connectivity(self, path_weights):
        return torch.ones_like(path_weights)

    def forward(self, x):
        size = x.size()[2:]
        if self.aux:
            x, y, auxout1, auxout2 = self.net_branch(x)
        else:
            x, y = self.net_branch(x)
        x = self.net1.base(x)
        y = self.net2.base(y)
        xs, ys = [], []
        for stage_id in range(self.num_stages):
            x = self.net1.stages[stage_id](x)
            y = self.net2.stages[stage_id](y)
            if isinstance(x, list):
                xs.append(x[0])
                ys.append(y[0])
            else:
                xs.append(x)
                ys.append(y)

            net1_path_ids = np.nonzero(self.net1_connectivity_matrix[stage_id])[0]
            net2_path_ids = np.nonzero(self.net2_connectivity_matrix[stage_id])[0]
            net1_path_weights = self.net1_alphas[stage_id][net1_path_ids]
            net2_path_weights = self.net2_alphas[stage_id][net2_path_ids]

            # Calculating path strength based on weights
            if self.training:
                if self.supernet:
                    connectivity = 'all'
                elif self.retraining:
                    connectivity = 'onehot'
                else:
                    if self.arch_training:  # Training architecture
                        if self.hard_arch_training:
                            connectivity = 'gumbel'
                        else:
                            connectivity = 'sigmoid'
                    else:  # Training weights
                        if self.hard_weight_training:
                            connectivity = 'gumbel'
                        else:
                            connectivity = 'sigmoid'
            else:
                if self.supernet:
                    connectivity = 'all'
                elif self.retraining:
                    connectivity = 'onehot'
                elif self.stochastic_evaluation:
                    assert not self.hard_evaluation
                    connectivity = 'bernoulli'
                elif self.hard_evaluation:
                    connectivity = 'onehot'
                else:
                    connectivity = 'sigmoid'

            if connectivity == 'gumbel':
                net1_path_connectivity = self.gumbel_connectivity(net1_path_weights)
                net2_path_connectivity = self.gumbel_connectivity(net2_path_weights)
            elif connectivity == 'sigmoid':
                net1_path_connectivity = self.sigmoid_connectivity(net1_path_weights)
                net2_path_connectivity = self.sigmoid_connectivity(net2_path_weights)
            elif connectivity == 'all':
                net1_path_connectivity = self.all_connectivity(net1_path_weights)
                net2_path_connectivity = self.all_connectivity(net2_path_weights)
            elif connectivity == 'bernoulli':
                net1_path_connectivity = self.bernoulli_connectivity(net1_path_weights)
                net2_path_connectivity = self.bernoulli_connectivity(net2_path_weights)
            else:
                assert connectivity == 'onehot'
                net1_path_connectivity = self.onehot_connectivity(net1_path_weights)
                net2_path_connectivity = self.onehot_connectivity(net2_path_weights)
                
            if isinstance(x, list):
                net1_fusion_input = [x[0]]
                net2_fusion_input = [y[0]]
            else:
                net1_fusion_input = [x]
                net2_fusion_input = [y]
                
            for idx, input_id in enumerate(net1_path_ids):
                net1_fusion_input.append(net1_path_connectivity[idx]*ys[input_id])
            for idx, input_id in enumerate(net2_path_ids):
                net2_fusion_input.append(net2_path_connectivity[idx]*xs[input_id])
            
            if isinstance(x, list):
                x[0] = self.paths['net1_paths'][stage_id](net1_fusion_input)
                y[0] = self.paths['net2_paths'][stage_id](net2_fusion_input)
            else:
                x = self.paths['net1_paths'][stage_id](net1_fusion_input)
                y = self.paths['net2_paths'][stage_id](net2_fusion_input)

        x = self.feature_fusion(x, y)
        x = self.classifier(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        
        if self.aux:
            auxout1 = F.interpolate(auxout1, size, mode='bilinear', align_corners=True)
            auxout2 = F.interpolate(auxout2, size, mode='bilinear', align_corners=True)
            
        return tuple([x,auxout1,auxout2])

