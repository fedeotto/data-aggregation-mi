import numpy as np
import torch

import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_, calculate_gain

from torch_geometric.utils import scatter
 
def softmax_weights(alpha, row, weights, col):
        weights=weights[col].unsqueeze(-1)
        alpha=torch.exp(alpha)
        alpha=weights*alpha
        alpha_rowsum = scatter(alpha, row, dim=0, dim_size=None,reduce='add')
        alpha_rowsum_inv = alpha_rowsum.pow(-1)+1e-16
        alpha = alpha * alpha_rowsum_inv[row]
        return alpha

class GATLayer(nn.Module):
    def __init__(self,in_channels, out_channels, heads, negative_slope=0.2, 
                dropout=0.0, bias=False, concat=True):
        super(GATLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if concat:
            assert out_channels % heads == 0
            self.hidden_channels = out_channels // heads
        else:
            self.hidden_channels = out_channels
        
        self.lin = nn.Linear(in_channels, heads * self.hidden_channels, bias=bias)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * self.hidden_channels))

        self.reset_parameters()
    
    def reset_parameters(self):
        gain=calculate_gain('leaky_relu', self.negative_slope)
        xavier_uniform_(self.lin.weight, gain=gain)
        xavier_uniform_(self.att, gain=gain)
        if self.lin.bias is not None:
            zeros_(self.lin.bias)

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, x, edge_index, pos, edge_weight=None):
        """
        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, in_channels].
            edge_index (Tensor): Graph connectivity in
        
        """
        x = self.lin(x).view(-1, self.heads, self.hidden_channels) # [batch_num_nodes, heads, hidden_channels]
        x = x.unsqueeze(-1) if x.dim() == 2 else x 
        
        # Compute attention coefficients.
        row, col = edge_index
        weights=pos

        alpha = torch.cat([x[row], x[col]], dim=-1) # [batch_num_edges, heads, 2 * hidden_channels]
        alpha = (alpha * self.att).sum(dim=-1) # [batch_num_edges, heads]
        alpha = self.leaky_relu(alpha)

        # Sample attention coefficients stochastically.
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Normalize attention coefficients.
        alpha = softmax_weights(alpha, row, weights, col) # [batch_num_edges, heads]

        # Calculate node updates
        out = alpha.unsqueeze(-1) * x[col] 
        out = scatter(out, row, dim=0, dim_size=x.size(0), reduce='add')
        if self.concat:
            out = out.view(-1, self.heads * self.hidden_channels)
        else:
            out = out.mean(dim=1)
        
        return out # [batch_num_nodes, heads * hidden_channels] or [batch_num_nodes, hidden_channels]
    

class GATv2Layer(nn.Module):
    def __init__(self,in_channels, out_channels, heads, negative_slope=0.2, 
                dropout=0.0, bias=False, concat=True):
        super(GATv2Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if concat:
            assert out_channels % heads == 0
            self.hidden_channels = out_channels // heads
        else:
            self.hidden_channels = out_channels
        
        self.lin = nn.Linear(in_channels, heads * self.hidden_channels, bias=bias)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * self.hidden_channels))

        self.reset_parameters()
    
    def reset_parameters(self):
        gain=calculate_gain('leaky_relu', self.negative_slope)
        xavier_uniform_(self.lin.weight,gain=gain)
        xavier_uniform_(self.att,gain=gain)
        if self.lin.bias is not None:
            zeros_(self.lin.bias)

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, x, edge_index, pos, edge_weight=None):
        """
        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, in_channels].
            edge_index (Tensor): Graph connectivity in
        
        """
        x = self.lin(x).view(-1, self.heads, self.hidden_channels)
        x = x.unsqueeze(-1) if x.dim() == 2 else x
        
        # Compute attention coefficients.
        row, col = edge_index
        weights=pos

        alpha = torch.cat([x[row], x[col]], dim=-1)
        alpha = (alpha * self.att).sum(dim=-1)
        alpha = self.leaky_relu(alpha)

        # Sample attention coefficients stochastically.
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Normalize attention coefficients.
        alpha = softmax_weights(alpha, row, weights, col)

        # Calculate node updates
        out = alpha.unsqueeze(-1) * x[col]
        out = scatter(out, row, dim=0, dim_size=x.size(0), reduce='add')
        if self.concat:
            out = out.view(-1, self.heads * self.hidden_channels)
        else:
            out = out.mean(dim=1)
        
        return out
    

class Simple_linear(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, negative_slope=0.2,bias=False):
        super(Simple_linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.negative_slope = negative_slope
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.lin1 = nn.Linear(self.input_size, self.hidden_dim, bias)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_size, bias)
        # self.reset_parameters()
    
    def reset_parameters(self):
        gain=calculate_gain('leaky_relu', self.negative_slope)
        xavier_uniform_(self.lin1.weight, gain=gain)
        xavier_uniform_(self.lin2.weight, gain=gain)
        if self.lin1.bias is not None:
            zeros_(self.lin1.bias)
        if self.lin2.bias is not None:
            zeros_(self.lin2.bias)
    
    def __repr__(self):
        return self.__class__.__name__

    def forward(self, x):
        """
        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, in_channels].
        
        """
        x = self.leaky_relu(self.lin1(x))
        x = self.leaky_relu(self.lin2(x))

        return x
    
class WeightedAttentionPooling(nn.Module):
    def __init__(self, gate_nn, message_nn):
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        # self.reset_parameters()

    def reset_parameters(self):
        self.gate_nn.reset_parameters()
        self.message_nn.reset_parameters()
        
    def forward(self, x, edge_index, pos):
        row, col = edge_index
        weights = pos

        alpha = torch.cat([x[row], x[col]], dim=-1)
        alpha = self.gate_nn(alpha)
        alpha = softmax_weights(alpha, row, weights, col)

        beta = torch.cat([x[row], x[col]], dim=-1)
        beta = self.message_nn(beta)
        out = alpha * beta
        out = scatter(out, row, dim=0, reduce='add')
        return out
    

class GATRoostLayer(nn.Module):
    def __init__(self,in_channels, out_channels, g_elem_dim, f_elem_dim, 
                 heads, negative_slope=0.2, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.bias = bias
    
        # here I have a mistake as I need individual function for each head
        self.pooling = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=Simple_linear(2 * self.in_channels, 1, f_elem_dim, negative_slope,bias),
                    message_nn=Simple_linear(2 * self.in_channels, self.out_channels, g_elem_dim, negative_slope,bias),
                )
                for _ in range(heads)
            ]
        )
        
        # self.reset_parameters()
        
    def reset_parameters(self):
        for head in self.pooling:
            head.reset_parameters()
            
    def __repr__(self):
        return self.__class__.__name__
    
    def forward(self, x, edge_index, pos, edge_weight=None):
        """
        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, in_channels].
            edge_index (Tensor): Graph connectivity in
        
        """
        head_fea = []
        for attnhead in self.pooling:
            head_fea.append(attnhead(x, edge_index, pos))

        # average the attention heads
        out = torch.mean(torch.stack(head_fea), dim=0)

        return out
    

#this AttentionPooling for pooling of the whole graph

class WeightedAttentionPooling_comp(nn.Module):
    def __init__(self, gate_nn, message_nn):
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        # self.reset_parameters()

    def reset_parameters(self):
        self.gate_nn.reset_parameters()
        self.message_nn.reset_parameters()
        
    def forward(self, x, edge_index, pos, batch_index=None):
        weights = pos

        alpha = x
        alpha = self.gate_nn(alpha)
        alpha = weights*alpha.exp().squeeze(-1)
        # probably here we need to do it only for each compound
        if batch_index is not None:
            alpha_sum = scatter(alpha, batch_index, dim=0, reduce='sum')
            alpha=alpha/alpha_sum[batch_index]
        else:
            alpha_sum = alpha.sum(dim=0)+1e-10
            alpha=alpha/alpha_sum
        
        beta = x
        beta = self.message_nn(beta)
        if batch_index is not None:
            out = scatter(alpha.unsqueeze(-1) * beta, batch_index, dim=0, reduce='add')
        else:
            out = torch.mean(alpha.unsqueeze(-1) * beta, dim=0)
        return  out