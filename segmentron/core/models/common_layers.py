import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage(nn.Module):
    def __init__(self, out_channels, layers):
        super(Stage, self).__init__()
        if isinstance(layers, list):
            self.feature = nn.Sequential(*layers)
        else:
            self.feature = layers
        self.out_channels = out_channels
        
    def forward(self, x):
        return self.feature(x)
    
    
def batch_norm(num_features, eps=1e-3, momentum=0.05):
    bn = nn.BatchNorm2d(num_features, eps, momentum)
    nn.init.constant_(bn.weight, 1)
    nn.init.constant_(bn.bias, 0)
    return bn


def get_nddr_bn(cfg):
    if cfg.MODEL.NDDR_BN_TYPE == 'default':
        return lambda width: batch_norm(width, eps=1e-03, momentum=cfg.MODEL.BATCH_NORM_MOMENTUM)
    else:
        raise NotImplementedError
        

def get_nddr(cfg, in_channels, out_channels, factor = 1):
    
    if cfg.ARCH.SEARCHSPACE == '':
        assert in_channels == out_channels
        if cfg.MODEL.NDDR_TYPE == '':
            return NDDR(cfg, out_channels)
        elif cfg.MODEL.NDDR_TYPE == 'single_side':
            return SingleSideNDDR(cfg, out_channels, False)
        elif cfg.MODEL.NDDR_TYPE == 'single_side_reverse':
            return SingleSideNDDR(cfg, out_channels, True)
        elif cfg.MODEL.NDDR_TYPE == 'cross_stitch':
            return CrossStitch(cfg, out_channels)
        else:
            raise NotImplementedError
    elif cfg.ARCH.SEARCHSPACE == 'GeneralizedMTLNAS':
        if cfg.MODEL.NDDR_TYPE == '':
            return SingleSidedAsymmetricNDDR(cfg, in_channels, out_channels)
        elif cfg.MODEL.NDDR_TYPE == 'cross_stitch':
            return SingleSidedAsymmetricCrossStitch(cfg, in_channels, out_channels)
        else:
            raise NotImplementedError
    elif cfg.ARCH.SEARCHSPACE == 'GeneralizedFastSCNN':
        return SingleSidedAsymmetricFeatureFusion(cfg, in_channels, out_channels, factor)
    else:
        raise NotImplementedError


class CrossStitch(nn.Module):
    def __init__(self, cfg, out_channels):
        super(CrossStitch, self).__init__()
        init_weights = cfg.MODEL.INIT
        
        self.a11 = nn.Parameter(torch.tensor(init_weights[0]))
        self.a22 = nn.Parameter(torch.tensor(init_weights[0]))
        self.a12 = nn.Parameter(torch.tensor(init_weights[1]))
        self.a21 = nn.Parameter(torch.tensor(init_weights[1]))

    def forward(self, feature1, feature2):
        out1 = self.a11 * feature1 + self.a21 * feature2
        out2 = self.a12 * feature1 + self.a22 * feature2
        return out1, out2
    
    
class NDDR(nn.Module):
    def __init__(self, cfg, out_channels):
        super(NDDR, self).__init__()
        init_weights = cfg.MODEL.INIT
        norm = get_nddr_bn(cfg)
        
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        
        # Initialize weight
        if len(init_weights):
            self.conv1.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[0],
                torch.eye(out_channels) * init_weights[1]
            ], dim=1).view(out_channels, -1, 1, 1))
            self.conv2.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[1],
                torch.eye(out_channels) * init_weights[0]
            ], dim=1).view(out_channels, -1, 1, 1))
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

        self.activation = nn.ReLU()

        self.bn1 = norm(out_channels)
        self.bn2 = norm(out_channels)

    def forward(self, feature1, feature2):
        x = torch.cat([feature1, feature2], 1)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out1 = self.bn1(out1)
        out2 = self.bn2(out2)
        out1 = self.activation(out1)
        out2 = self.activation(out2)
        return out1, out2


class SingleSideNDDR(nn.Module):
    def __init__(self, cfg, out_channels, reverse):
        """
        Net1 is main task, net2 is aux task
        """
        super(SingleSideNDDR, self).__init__()
        init_weights = cfg.MODEL.INIT
        norm = get_nddr_bn(cfg)

        self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)

        # Initialize weight
        if len(init_weights):
            self.conv.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[0],
                torch.eye(out_channels) * init_weights[1]
            ], dim=1).view(out_channels, -1, 1, 1))
        else:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.activation = nn.ReLU()

        self.bn = norm(out_channels)

        self.reverse = reverse

    def forward(self, feature1, feature2):
        if self.reverse:
            out2 = feature2
            out1 = torch.cat([feature1, feature2], 1)
            out1 = self.conv(out1)
            out1 = self.bn(out1)
        else:
            out1 = feature1
            out2 = torch.cat([feature2, feature1], 1)
            out2 = self.conv(out2)
            out2 = self.bn(out2)
        return out1, out2


class SingleSidedAsymmetricCrossStitch(nn.Module):
    def __init__(self, cfg, in_channels, out_channels):
        super(SingleSidedAsymmetricCrossStitch, self).__init__()
        init_weights = cfg.MODEL.INIT
        
        assert in_channels >= out_channels
        # check if out_channel divides in_channels
        assert in_channels % out_channels == 0
        multipiler = in_channels / out_channels - 1
        self.a = nn.Parameter(torch.tensor([init_weights[0]] +\
                                            [init_weights[1] / float(multipiler) for _ in range(int(multipiler))]))

    def forward(self, features):
        """

        :param features: upstream feature maps
        :return:
        """
        out = 0.
        for i, feature in enumerate(features):
            out += self.a[i] * feature
        return out
    
    
class SingleSidedAsymmetricNDDR(nn.Module):
    def __init__(self, cfg, in_channels, out_channels):
        super(SingleSidedAsymmetricNDDR, self).__init__()
        init_weights = cfg.MODEL.INIT
        norm = get_nddr_bn(cfg)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        assert in_channels >= out_channels
        # check if out_channel divides in_channels
        assert in_channels % out_channels == 0
        multipiler = in_channels / out_channels - 1
        
        # Initialize weight
        if len(init_weights):
            weight = [torch.eye(out_channels) * init_weights[0]] +\
                 [torch.eye(out_channels) * init_weights[1] / float(multipiler) for _ in range(int(multipiler))]
            self.conv.weight = nn.Parameter(torch.cat(weight, dim=1).view(out_channels, -1, 1, 1))
        else:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        
        self.activation = nn.ReLU()
        self.bn = norm(out_channels)
        nn.init.constant_(self.bn.weight, 1.)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, features):
        """

        :param features: upstream feature maps
        :return:
        """
        x = torch.cat(features, 1)
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out

class SingleSidedAsymmetricFeatureFusion(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, factor):
        super(SingleSidedAsymmetricFeatureFusion, self).__init__()
        init_weights = cfg.MODEL.INIT
        norm = get_nddr_bn(cfg)
        self.factor = factor

        self.localConv = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1,
                padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=1, stride=1,
                padding=0, bias=False),
            Swish()
        )
        self.conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) for i in range(3)])
        # assert in_channels >= out_channels
        # check if out_channel divides in_channels
        # assert in_channels % out_channels == 0
        # multipiler = in_channels / out_channels - 1
        
        # Initialize weight
        # if len(init_weights):
        #     weight = [torch.eye(out_channels) * init_weights[0]] +\
        #          [torch.eye(out_channels) * init_weights[1] / float(multipiler) for _ in range(int(multipiler))]
        #     self.conv.weight = nn.Parameter(torch.cat(weight, dim=1).view(out_channels, -1, 1, 1))
        # else:
        #     nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        
        self.activation = Swish()
        self.bn = norm(out_channels)
        nn.init.constant_(self.bn.weight, 1.)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, features):
        """

        :param features: upstream feature maps
        :return:
        """
        # x = torch.cat(features, 1)
        local_feature = self.localConv(features[0])
        shared_features = features[1:]
        size = local_feature.size()[2:]
        channel = local_feature.size()[1]
        sum = torch.zeros(size=local_feature.size()).cuda()
        
        for index, shared_feature in enumerate(shared_features):
            shared_feature = self.bn(self.conv[index](shared_feature))
            if self.factor != 1:
                shared_feature = F.interpolate(shared_feature, size=size, mode= 'bilinear')
            shered_feature = self.activation(shared_feature)
            sum += sum + shered_feature * local_feature
        
        out = sum
        # out = self.conv(x)
        # out = self.bn(out)
        # out = self.activation(out)
        return out


class Swish(nn.Module):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.mul_(x.sigmoid())
        else:
            return x * x.sigmoid()
