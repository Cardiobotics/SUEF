from typing import Any

import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiViewLinear(nn.Module):
    def __init__(self, input_size):
        super(MultiViewLinear, self).__init__()
        self.fc_linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc_linear(x)
