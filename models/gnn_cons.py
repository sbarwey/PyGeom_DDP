from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable, List
import torch
from torch import Tensor
import torch_geometric.nn as tgnn
from torch_scatter import scatter_mean
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from pooling import TopKPooling_Mod, avg_pool_mod, avg_pool_mod_no_x

SPACEDIM = 2

class ConsGNN(torch.nn.Module):
    def __init__(self, 
                 input_node_channels: int, 
                 hidden_channels: int, 
                 output_node_channels: int, 
                 n_mlp_hidden_layers: int, 
                 n_messagePassing_layers: int,
                 identical_messagePassing: Optional[bool] = False,
                 name: Optional[str] = 'gnn'):
        super().__init__()
        
        self.input_node_channels = input_node_channels
        self.hidden_channels = hidden_channels
        self.output_node_channels = output_node_channels 
        self.n_mlp_hidden_layers = n_mlp_hidden_layers
        self.n_messagePassing_layers = n_messagePassing_layers
        self.identical_messagePassing = identical_messagePassing
        self.name = name 

        # ~~~~ node encoder MLP  
        self.node_encoder = MLP(
                input_channels = self.input_node_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.hidden_channels,
                activation_layer = torch.nn.ReLU(),
                norm_layer = torch.nn.LayerNorm(self.hidden_channels)
                )

        # ~~~~ node decoder MLP  
        self.node_decoder = MLP(
                input_channels = self.hidden_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.output_node_channels,
                activation_layer = torch.nn.ReLU(),
                )
        
        # ~~~~ Processor 
        self.processor = torch.nn.ModuleList()
        if not self.identical_messagePassing: 
            """
            Use independently parameterized message passing layers. 
            """
            for i in range(self.n_messagePassing_layers):
                self.processor.append( 
                              ConsMessagePassingLayer(
                                         channels = hidden_channels,
                                         n_mlp_hidden_layers = self.n_mlp_hidden_layers, 
                                         ) 
                                      )
        else:
            """
            Use the same message passing parameterization for all layers.
            """
            self.processor.append( 
                          ConsMessagePassingLayer(
                                     channels = hidden_channels,
                                     n_mlp_hidden_layers = self.n_mlp_hidden_layers, 
                                     ) 
                                  )
       
        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: LongTensor,
            pos: Tensor,
            edge_attr: Tensor,
            nvec: Tensor,
            batch: Optional[LongTensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # ~~~~ Node encoder 
        x = self.node_encoder(x) 

        # ~~~~ Processor 
        for i in range(self.n_messagePassing_layers):
            if not self.identical_messagePassing: 
                x = self.processor[i](x,edge_attr,nvec,edge_index,batch)
            else:
                x = self.processor[0](x,edge_attr,nvec,edge_index,batch)

        # ~~~~ Node decoder 
        x = self.node_decoder(x)
        
        return x 

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.node_decoder.reset_parameters()
        for module in self.processor:
            module.reset_parameters()
        return

    def input_dict(self) -> dict:
        a = {'input_node_channels': self.input_node_channels,
             'hidden_channels': self.hidden_channels, 
             'output_node_channels': self.output_node_channels,
             'n_mlp_hidden_layers': self.n_mlp_hidden_layers,
             'n_messagePassing_layers': self.n_messagePassing_layers,
             'name': self.name} 
        return a

    def get_save_header(self) -> str:
        a = self.input_dict()
        header = a['name']
        
        for key in a.keys():
            if key != 'name': 
                header += '_' + str(a[key])

        #for item in self.input_dict():
        return header

class MLP(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: List[int],
                 output_channels: int,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU(),
                 bias: bool = True):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels 
        self.output_channels = output_channels 
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer

        self.ic = [input_channels] + hidden_channels # input channel dimensions for each layer
        self.oc = hidden_channels + [output_channels] # output channel dimensions for each layer 

        self.mlp = torch.nn.ModuleList()
        for i in range(len(self.ic)):
            self.mlp.append( torch.nn.Linear(self.ic[i], self.oc[i], bias=bias) )

        self.reset_parameters()

        return

    def forward(self, x: Tensor) -> Tensor:
        for i in range(len(self.ic)):
            x = self.mlp[i](x) 
            if i < (len(self.ic) - 1):
                x = self.activation_layer(x)
        x = self.norm_layer(x) if self.norm_layer else x
        return x  

    def reset_parameters(self):
        for module in self.mlp:
            module.reset_parameters()
        if self.norm_layer:
            self.norm_layer.reset_parameters()
        return

    def copy(self, mlp_bl, freeze_params=False):
        """
        Copy parameters from another identically structured MLP, given as "mlp_bl". 
        """
        if self.norm_layer:
            if freeze_params:
                self.norm_layer.weight.requires_grad=False
                self.norm_layer.bias.requires_grad=False
            self.norm_layer.weight[:] = mlp_bl.norm_layer.weight.detach().clone()
            self.norm_layer.bias[:] = mlp_bl.norm_layer.bias.detach().clone()
        for k in range(len(self.mlp)):
            if freeze_params:
                self.mlp[k].weight.requires_grad = False
                self.mlp[k].bias.requires_grad = False
            self.mlp[k].weight[:,:] = mlp_bl.mlp[k].weight.detach().clone()
            self.mlp[k].bias[:] = mlp_bl.mlp[k].bias.detach().clone()
        return

class ConsMessagePassingLayer(torch.nn.Module):
    def __init__(self, 
                 channels: int, 
                 n_mlp_hidden_layers: int):
        super().__init__()

        self.edge_aggregator = EdgeAggregation(aggr='add')
        self.channels = channels
        self.n_mlp_hidden_layers = n_mlp_hidden_layers 

        # Flux MLP 
        self.flux_mlp = MLP(
                input_channels = self.channels,
                hidden_channels = [self.channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.channels * SPACEDIM,
                activation_layer = torch.nn.ReLU(),
                norm_layer = torch.nn.LayerNorm(self.channels * SPACEDIM)
                )

        # Wavespeed MLP 
        self.a_mlp = MLP(
                input_channels = self.channels,
                hidden_channels = [self.channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.channels,
                activation_layer = torch.nn.ReLU(),
                norm_layer = torch.nn.LayerNorm(self.channels)
                )

        self.reset_parameters()

        return 

    def forward(
            self,
            x: Tensor,
            e: Tensor,
            nvec: Tensor, 
            edge_index: LongTensor,
            batch: Optional[LongTensor] = None) -> Tensor:
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # KT Flux = 0.5*{ [ (F(xi) + F(xj)) ] - a_ij*[ xi - xj ] }
        x_send = x[edge_index[0,:]] # send
        x_recv = x[edge_index[1,:]] # recv 
        flux_send = torch.bmm(self.flux_mlp(x_send).view(-1,x.shape[1],SPACEDIM), 
                              nvec.unsqueeze(-1)).squeeze()
        flux_recv = torch.bmm(self.flux_mlp(x_recv).view(-1,x.shape[1],SPACEDIM), 
                              nvec.unsqueeze(-1)).squeeze()

        # wavespeed: 
        a_ws = self.a_mlp((x_send + x_recv)/2.0)

        # final flux:
        flux = 0.5*(flux_send + flux_recv) - 0.5*a_ws*(x_send - x_recv)

        # node update: x = x + sum(flux)
        x += self.edge_aggregator(x, edge_index, flux)

        return x

    def reset_parameters(self):
        return

    def copy(self, mp_layer_bl, freeze_params=False):
        """
        Copy parameters from another identically structured mesage passing layer, given as "mp_layer_bl". 
        """
        self.edge_updater.copy(mp_layer_bl.edge_updater, freeze_params)
        self.node_updater.copy(mp_layer_bl.node_updater, freeze_params)
        return

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

    propagate_type = {'x': Tensor, 'edge_attr': Tensor}

    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        x_j = edge_attr
        return x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
