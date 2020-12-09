import numpy as np
import torch
from collections import OrderedDict

def show_model_alpha(model_path):
    state_dict_to_load = torch.load(model_path)
    paraDic = OrderedDict()
    for k, v in state_dict_to_load.items():
        if 'alpha' in k:
            paraDic[k] = v
    print(paraDic)

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


if __name__ == '__main__':
    # out = depth_limited_connectivity_matrix([3])
    # skip = out - np.eye(3,dtype=int)
    # print(out)
    # print(out.shape)
    # print(out.sum(axis=1))
    # print(np.nonzero(out[1]))
    # print(skip)
    # print(np.nonzero(skip[0]))
    path = '/home/szy/code/SegmenTron/runs/checkpoints/FastSCNN__cityscape_2020-12-03-22-15/best_model.pth'
    show_model_alpha(path)
