from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_scatter import scatter_mean
from torch_geometric.nn.conv import MessagePassing
from pooling import TopKPooling_Mod, avg_pool_mod, avg_pool_mod_no_x

class Simple_MP_Layer(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.edge_aggregator = EdgeAggregation() # EdgeAggregation() object (defined below)
        self.input_channels = input_channels
        self.output_channels = input_channels
        self.hidden_channels = hidden_channels
        self.n_mlp_layers = 2
        self.act = F.elu 

        # ~~~~ GLL-based point-wise encoder 
        self.gll_encoder = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers):
            if j == 0:
                input_features = self.input_channels
                output_features = self.hidden_channels 
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.gll_encoder.append( nn.Linear(input_features, output_features, bias=False) )

        # ~~~~ GLL-based point-wise decoder 
        self.gll_decoder = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers):
            if j == 0:
                input_features = self.hidden_channels
                output_features = self.output_channels 
            else:
                input_features = self.output_channels
                output_features = self.output_channels
            self.gll_decoder.append( nn.Linear(input_features, output_features, bias=False) )


        # ~~~~ Element-based message passing 
        # For edge feature update: 
        self.edge_updater = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers):
            if j == 0:
                input_features = self.hidden_channels*3
                output_features = self.hidden_channels 
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.edge_updater.append( nn.Linear(input_features, output_features, bias=False) )

        # For node feature update: 
        self.node_updater = torch.nn.ModuleList()
        for j in range(self.n_mlp_layers):
            if j == 0:
                input_features = self.hidden_channels*2
                output_features = self.hidden_channels 
            else:
                input_features = self.hidden_channels
                output_features = self.hidden_channels
            self.node_updater.append( nn.Linear(input_features, output_features, bias=False) )

    def run_messagepassing(
            self, 
            x: Tensor, 
            edge_index: LongTensor) -> Tensor:

        # concatenate edge + neighbor + owner into new edge_attr 
        x_own = x[edge_index[0,:], :]
        x_nei = x[edge_index[1,:], :]
        edge_attr = (x_own - x_nei)
        edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)

        # edge update mlp
        for j in range(self.n_mlp_layers):
            edge_attr_t = self.edge_updater[j](edge_attr_t) 
            if j < self.n_mlp_layers - 1:
                edge_attr_t = self.act(edge_attr_t)
            else:
                edge_attr_t = edge_attr_t

        # edge residual connection
        edge_attr = edge_attr + edge_attr_t
       
        # get the aggregate edge features as node features: mean aggregation 
        edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

        # concatenate owner node + edge 
        x_t = torch.cat((x, edge_agg), axis=1)

        # node update mlp
        for j in range(self.n_mlp_layers):
            x_t = self.node_updater[j](x_t)
            if j < self.n_mlp_layers - 1:
                x_t = self.act(x_t) 
            else:
                x_t = x_t
        
        # node residual connection
        x = x + x_t

        return x 

    def run_encoder(self, x: Tensor, edge_index: LongTensor, cluster: LongTensor) -> Tensor:

        # gll point-wise update mlp: 
        for j in range(self.n_mlp_layers):
            x = self.gll_encoder[j](x)
            if j < self.n_mlp_layers - 1:
                x = self.act(x)
            else:
                x = x

        # restriction: gll-to-element representation 
        x = scatter_mean(x, cluster, dim=0)
        
        return x

    def run_decoder(self, x: Tensor, pos_x: Tensor, pos_y: Tensor) -> Tensor:

        # interpolation: element-to-gll representation
        x = tgnn.knn_interpolate(x = x, 
                                 pos_x = pos_x, 
                                 pos_y = pos_y, 
                                 k = 2)

        # gll point-wise update mlp :
        for j in range(self.n_mlp_layers):
            x = self.gll_decoder[j](x)
            if j < self.n_mlp_layers - 1:
                x = self.act(x)
            else:
                x = x

        return x


    def forward(
            self, 
            x: Tensor, 
            edge_index: LongTensor) -> Tensor:

        # concatenate edge + neighbor + owner into new edge_attr 
        x_own = x[edge_index[0,:], :]
        x_nei = x[edge_index[1,:], :]
        edge_attr = (x_own - x_nei)
        edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)

        # edge update mlp
        for j in range(self.n_mlp_layers):
            edge_attr_t = self.edge_updater[j](edge_attr_t) 
            if j < self.n_mlp_layers - 1:
                edge_attr_t = self.act(edge_attr_t)
            else:
                edge_attr_t = edge_attr_t

        # edge residual connection
        edge_attr = edge_attr + edge_attr_t
       
        # get the aggregate edge features as node features: mean aggregation 
        edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

        # concatenate owner node + edge 
        x_t = torch.cat((x, edge_agg), axis=1)

        # node update mlp
        for j in range(self.n_mlp_layers):
            x_t = self.node_updater[j](x_t)
            if j < self.n_mlp_layers - 1:
                x_t = self.act(x_t) 
            else:
                x_t = x_t
        
        # node residual connection
        x = x + x_t

        return x 


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
