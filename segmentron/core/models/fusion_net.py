import torch
import torch.nn as nn
from ..models.common_layers import Stage
from ..config import cfg
from ..models.common_layers import SeparableConv2d

def get_fusion_net(net1_dim=64, net2_dim=128, net1_net2_factor=4):
    net1 = BranchNet(64, 64)
    net2 = BranchNet(128, 128)
    return [net1,net2,net1_net2_factor]

class BranchNet(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, factor=1, *args, **kwargs):
        super(BranchNet, self).__init__(*args, **kwargs)
        self.base = lambda x: x  # required by NAS
        
        self.stages = []
        layers = []
        stages = [
            (in_dim, SeparableConv2d(in_dim, in_dim)),
            (in_dim, SeparableConv2d(in_dim, in_dim)),
            (out_dim, SeparableConv2d(in_dim, in_dim)),
        ]

        for channels, stage in stages:
            if isinstance(stage, list):
                layers.extend(stage)
            else:
                layers.append(stage)
            self.stages.append(Stage(channels, stage))
        self.stages = nn.ModuleList(self.stages)

        # Used for backward compatibility with weight loading
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        
        return x

