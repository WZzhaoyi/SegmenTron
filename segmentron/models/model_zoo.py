import logging
import torch
import numpy as np

from collections import OrderedDict
from segmentron.utils.registry import Registry
from ..config import cfg
from ..core.models import GeneralizedFastSCNNNet
from fast_scnn import FastSCNNBranch
from ..core.models.fusion_net import get_fusion_net

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for segment model, i.e. the whole model.

The registered object will be called with `obj()`
and expected to return a `nn.Module` object.
"""


def get_segmentation_model():
    """
    Built the whole model, defined by `cfg.MODEL.META_ARCHITECTURE`.
    """
    model_name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(model_name)()
    load_model_pretrain(model)
    return model

def get_supernet():
    model_name = cfg.MODEL.MODEL_NAME
    if cfg.ARCH.SEARCHSPACE == 'GeneralizedFastSCNN' and model_name == 'FastSCNN':
        connectivity = feature_fusion_connectivity()
        fusion_net = get_fusion_net()
        net1_net2_factor = cfg.MODEL.MODEL_FACTOR
        model = GeneralizedFastSCNNNet(cfg, FastSCNNBranch, fusion_net, net1_connectivity_matrix=connectivity(), net2_connectivity_matrix=connectivity(), net1_net2_factor=net1_net2_factor)
        return model
    else:
        return NotImplementedError

def load_model_pretrain(model):
    if cfg.PHASE == 'train':
        if cfg.TRAIN.PRETRAINED_MODEL_PATH:
            logging.info('load pretrained model from {}'.format(cfg.TRAIN.PRETRAINED_MODEL_PATH))
            state_dict_to_load = torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH)
            keys_wrong_shape = []
            state_dict_suitable = OrderedDict()
            state_dict = model.state_dict()
            for k, v in state_dict_to_load.items():
                if v.shape == state_dict[k].shape:
                    state_dict_suitable[k] = v
                else:
                    keys_wrong_shape.append(k)
            logging.info('Shape unmatched weights: {}'.format(keys_wrong_shape))
            msg = model.load_state_dict(state_dict_suitable, strict=False)
            logging.info(msg)
    else:
        if cfg.TEST.TEST_MODEL_PATH:
            logging.info('load test model from {}'.format(cfg.TEST.TEST_MODEL_PATH))
            msg = model.load_state_dict(torch.load(cfg.TEST.TEST_MODEL_PATH), strict=False)
            logging.info(msg)

def depth_limited_connectivity_matrix(stage_config, limit=3):
    """

    :param stage_config: list of number of layers in each stage
    :param limit: limit of depth difference between connected layers, pass in -1 to disable
    :return: connectivity matrix
    """
    network_depth = np.sum(stage_config)
    stage_depths = np.cumsum([0] + stage_config)
    matrix = np.zeros((network_depth, network_depth)).astype('int')
    for i in range(network_depth):
        j_limit = stage_depths[np.argmax(stage_depths > i) - 1]
        for j in range(network_depth):
            if j <= i and i - j < limit and j >= j_limit:
                matrix[i, j] = 1.
    return matrix


def feature_fusion_connectivity():
    return depth_limited_connectivity_matrix([3])
