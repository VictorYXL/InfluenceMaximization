import torch
import torch.nn as nn
import torch.nn.functional as F

class LTNet(nn.Module):
    """Backbone for LT influence model"""
    def __init__(self, node_num: int):
        super(LTNet, self).__init__()
        # Define a fully connected layer with bias
        self.fc = nn.Linear(node_num, node_num, bias=True)
       
    def forward(self, x):
        # Pass the input through the fully connected layer, then apply ReLU
        x = self.fc(x)
        return x
