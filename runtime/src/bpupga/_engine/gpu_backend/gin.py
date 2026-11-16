"""Author GradAlign GIN architecture without unrelated CPU sklearn imports."""
import torch
from torch import nn
from torch_geometric.nn import GINConv


class myGIN(nn.Module):
    def __init__(self,in_channels,hidden_channels,num_layers):
        super().__init__()
        self.num_layers=num_layers
        self.convs=nn.ModuleList()
        for layer in range(num_layers):
            width=in_channels if layer==0 else hidden_channels
            self.convs.append(GINConv(nn.Sequential(nn.Linear(width,hidden_channels),
                nn.ReLU(),nn.Linear(hidden_channels,hidden_channels),nn.ReLU(),
                nn.BatchNorm1d(hidden_channels)),train_eps=False,aggr='add'))
        for module in self.modules():
            if isinstance(module,nn.Linear):nn.init.xavier_uniform_(module.weight)

    def full_forward(self,x,edge_index):
        values=[x]
        for convolution in self.convs:
            x=convolution(x,edge_index).tanh();values.append(x)
        return values
