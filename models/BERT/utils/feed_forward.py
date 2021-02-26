import torch.nn as nn
import torch.nn.functional as F
from .gelu import GELU
from torch.cuda.amp import autocast


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        #self.activation = GELU()

    
    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))
