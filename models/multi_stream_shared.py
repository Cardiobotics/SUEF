from typing import Any

import torch.nn as nn
import torch
import torch.nn.functional as F


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
                y.append(self.model_img(inp))
            else:
                y.append(self.model_flow(inp))
        y = torch.stack(y, dim=1).squeeze()
        y = self.fc_linear(F.relu(y))
        return y
