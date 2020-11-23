from typing import Any

import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiStream(nn.Module):
    '''
    Multi-Stream model.
    '''

    def __init__(self, model_dict):
        super(MultiStream, self).__init__()

        self.endpoints = {}

        for model_name, model in model_dict.items():
            self.endpoints[model_name] = model
            self.add_module(model_name, model)

        self.num_models = len(model_dict.items())

        self.fc_name = 'Linear_layer'
        self.fc_linear = nn.Linear(self.num_models, 1)
        self.endpoints[self.fc_name] = self.fc_linear
        self.add_module(self.fc_name, self.fc_linear)

    def forward(self, x):
        assert len(x) == self.num_models

        y = []

        for inp, model_name in zip(x, self.endpoints.keys()):
            y.append(self._modules[model_name](inp))
        y = torch.stack(y, dim=1).squeeze()
        y = self._modules[self.fc_name](F.relu(y))
        return y


class MultiStreamShared(nn.Module):
    '''
    Multi-Stream model with shared weights.
    '''

    def __init__(self, model_img, model_flow, num_m):
        super(MultiStreamShared, self).__init__()

        self.model_img_name = 'Model_img'
        self.model_img = model_img
        self.add_module(self.model_img_name, self.model_img)

        self.model_flow_name = 'Model_flow'
        self.model_flow = model_flow
        self.add_module(self.model_flow_name, self.model_flow)

        self.num_models = num_m

        self.fc_name = 'Linear_layer'
        self.fc_linear = nn.Linear(self.num_models, 1)
        self.add_module(self.fc_name, self.fc_linear)

    def forward(self, x):
        assert len(x) == self.num_models

        y = []

        for i, inp in enumerate(x):
            if i % 2 == 0:
                y.append(self._modules[self.model_img_name](inp))
            else:
                y.append(self._modules[self.model_flow_name](inp))
        y = torch.stack(y, dim=1).squeeze()
        y = self._modules[self.fc_name](F.relu(y))
        return y
