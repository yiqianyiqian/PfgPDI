import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn.pool import SAGPooling

class GCN(nn.Module):
    def __init__(self, in_fea, num_nodes, out_fea):
        super(GCN, self).__init__()
        self.conv1 = DenseGCNConv(in_fea, 32)
        # self.conv2 = GraphConv(16, 13, activation=nn.ReLU())
        self.conv2 = DenseGCNConv(32, 64)
        self.conv3 = DenseGCNConv(64, 128)
        self.conv4 = DenseGCNConv(128, 256)
        self.conv5 = DenseGCNConv(256, out_fea)

        self.bn1 = nn.BatchNorm1d(num_nodes)
        self.bn2 = nn.BatchNorm1d(num_nodes)
        self.bn3 = nn.BatchNorm1d(num_nodes)
        self.bn4 = nn.BatchNorm1d(num_nodes)
        self.bn5 = nn.BatchNorm1d(num_nodes)
        self.sag1 = SAGPooling(32 ,0.1)
        self.sag2 = SAGPooling(64 ,1e-4)
        # self.linear = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(0.5)
        for param in self.parameters():
            print(type(param), param.size())

        self.fc1 = nn.Linear(16, 16)
        self.active = nn.Tanh()
        # self.fc2 = nn.Linear(13, 13)

    def forward(self, x, edge_index, batch=None):
        # x = x.reshape((-1, x.shape[-1]))
        # edge_index = edge_index.reshape((-1, edge_index.shape[-1])).type(torch.int64)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.active(x)  # 激活函数
        # x = F.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.active(x)
        # x = F.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.active(x)

        # y = self.sag1(x, edge_index, batch = batch)
        # x = y[0]
        # batch = y[3]
        # edge_index = y[1]
        x = self.conv4(x, edge_index)
        # x = F.dropout(x)
        x = self.bn4(x)
        x = self.active(x)
        x = self.conv5(x, edge_index)
        # x = F.dropout(x)
        # x = self.bn5(x)
        # x = F.tanh(x)
        # x = self.sag2(x, edge_index, batch = batch)
        return x