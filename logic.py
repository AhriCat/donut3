"""
Logic Bias (Donut3)
===================
Soft AND/OR inductive bias. Unchanged from Donut2 except imports fixed.
"""

import torch
import torch.nn as nn


class LogicBias(nn.Module):
    def __init__(self, dim, strength=0.08):
        super().__init__()
        self.strength = strength
        self.weight_and = nn.Parameter(torch.ones(dim))
        self.weight_or = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [B, N, D]
        sigmoid_x = torch.sigmoid(x)
        x_and = x * sigmoid_x
        x_or = 1 - (1 - sigmoid_x) ** 2
        return x + self.strength * (x_and + x_or - x)
