from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

class MP_GNN(torch.nn.Module):
    def __init__(self, 
                 in_channels_node: int, 
                 in_channels_edge: int, 
                 hidden_channels: int, 
                 out_channels: int, 
                 n_mlp_encode: int, 
                 n_mlp_mp: int, 
                 n_mp: int, 
                 act: Optional[Callable] = F.elu):
        super().__init__()
        self.edge_aggregator = EdgeAggregation() # EdgeAggregation() object (defined below)
        self.in_channels_node = in_channels_node # number of node attributes in input
        self.in_channels_edge = in_channels_edge # number of edge attributes in input 
        self.hidden_channels = hidden_channels # dimensionality of latent nodes and edges
        self.out_channels = out_channels #  number of node attributes in output
        self.act = act # activation function used in hidden layers of MLPs 
        self.n_mlp_encode = n_mlp_encode # number of MLP layers in node/edge encoding stage 
        self.n_mlp_mp = n_mlp_mp # number of MLP layers in node/edge update functions in message passing layer
        self.n_mp = n_mp # number of message passing layers

        # ~~~~ Node encoder
        self.node_encode = torch.nn.ModuleList() 
        for i in range(self.n_mlp_encode): 
            if i == 0:
                input_features = in_channels_node 
                output_features = hidden_channels 
            else:
                input_features = hidden_channels 
                output_features = hidden_channels 
            self.node_encode.append( nn.Linear(input_features, output_features) )
        self.node_encode_norm = nn.LayerNorm(output_features)
       
        # ~~~~ Edge encoder 
        self.edge_encode = torch.nn.ModuleList() 
        for i in range(self.n_mlp_encode): 
            if i == 0:
                input_features = in_channels_edge
                output_features = hidden_channels 
            else:
                input_features = hidden_channels 
                output_features = hidden_channels 
            self.edge_encode.append( nn.Linear(input_features, output_features) )
        self.edge_encode_norm = nn.LayerNorm(output_features)

            
        # ~~~~ Message Passing
        # 1) concatenate edge + neighbor + owner 
        # 2) edge update mlp + layernorm  
        # 3) edge aggregation 
        # 4) cocnatenate node + avg edge feature 
        # 5) node update mlp + layernorm

        # Edge updates 
        self.edge_mp = torch.nn.ModuleList()
        self.edge_mp_norm = torch.nn.ModuleList()
        for i in range(self.n_mp):
            temp = torch.nn.ModuleList()
            for j in range(self.n_mlp_mp):
                if j == 0:
                    input_features = hidden_channels*3
                    output_features = hidden_channels 
                else:
                    input_features = hidden_channels
                    output_features = hidden_channels
                temp.append( nn.Linear(input_features, output_features) )
            self.edge_mp.append(temp)
            self.edge_mp_norm.append( nn.LayerNorm(output_features) )

        # Node updates 
        self.node_mp = torch.nn.ModuleList()
        self.node_mp_norm = torch.nn.ModuleList()
        for i in range(self.n_mp):
            temp = torch.nn.ModuleList()
            for j in range(self.n_mlp_mp):
                if j == 0:
                    input_features = hidden_channels*2
                    output_features = hidden_channels 
                else:
                    input_features = hidden_channels
                    output_features = hidden_channels 
                temp.append( nn.Linear(input_features, output_features) )
            self.node_mp.append(temp)
            self.node_mp_norm.append( nn.LayerNorm(output_features) )
        

        # ~~~~ Node decoder
        self.node_decode = torch.nn.ModuleList() 
        self.node_decode_norm = torch.nn.ModuleList()
        for i in range(self.n_mlp_encode): 
            if i == self.n_mlp_encode - 1:
                input_features = hidden_channels 
                output_features = in_channels_node
            else:
                input_features = hidden_channels 
                output_features = hidden_channels 
                self.node_decode_norm.append( nn.LayerNorm(output_features) )
            self.node_decode.append( nn.Linear(input_features, output_features) )

        self.reset_parameters()

    def reset_parameters(self):
        # Node encoding
        for module in self.node_encode:
            module.reset_parameters()

        # Edge encoding 
        for module in self.edge_encode:
            module.reset_parameters()

        # Message passing - edge update 
        for module in self.edge_mp:
            for submod in module:
                submod.reset_parameters()

        # Message passing - node update 
        for module in self.node_mp:
            for submod in module:
                submod.reset_parameters()

        # Node decoder: 
        for module in self.node_decode:
            module.reset_parameters()


    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # ~~~~ Node Encoder: 
        for i in range(self.n_mlp_encode):
            x = self.node_encode[i](x) 
            if i < self.n_mlp_encode - 1:
                x = self.act(x)
        x = self.node_encode_norm(x)

        # ~~~~ Edge Encoder: 
        for i in range(self.n_mlp_encode):
            edge_attr = self.edge_encode[i](edge_attr)
            if i < self.n_mlp_encode - 1:
                edge_attr = self.act(edge_attr)
            else:
                edge_attr = edge_attr
        edge_attr = self.edge_encode_norm(edge_attr)

        # ~~~~ Message passing: 
        for i in range(self.n_mp):
            # 1) concatenate edge + sender + receiver into new edge_attr 
            x_own = x[edge_index[0,:], :]
            x_nei = x[edge_index[1,:], :]
            edge_attr = torch.cat((x_own, x_nei, edge_attr), axis=1)

            # 2) edge update mlp
            for j in range(self.n_mlp_mp):
                edge_attr = self.edge_mp[i][j](edge_attr) 
                if j < self.n_mlp_mp - 1:
                    edge_attr = self.act(edge_attr)

            # 3) edge layer norm
            edge_attr = self.edge_mp_norm[i](edge_attr)

            # 4) get the aggregate edge features as node features: mean aggregation 
            edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

            # 5) concatenate owner node + edge 
            x = torch.cat((x, edge_agg), axis=1)

            # 6) node update mlp
            for j in range(self.n_mlp_mp):
                x = self.node_mp[i][j](x)
                if j < self.n_mlp_mp - 1:
                    x = self.act(x) 
            
            # 7) node layer norm 
            x = self.node_mp_norm[i](x)

        # ~~~~ Node Decoder: 
        for i in range(self.n_mlp_encode):
            x = self.node_decode[i](x)
            if i < self.n_mlp_encode - 1:
                x = self.act(x)
        
        return x


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels_node},{self.hidden_channels},{self.out_channels})')

class EdgeAggregation(MessagePassing):
    r"""This is a custom class that returns node quantities that represent the neighborhood-averaged edge features.
    Args:
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or 
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`, 
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        x_j = edge_attr
        return x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
