from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable, List
import torch
from torch import Tensor
import torch_geometric.nn as tgnn
from torch_scatter import scatter_mean
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from pooling import TopKPooling_Mod, avg_pool_mod, avg_pool_mod_no_x


class TopkMultiscaleGNN(torch.nn.Module):
    """
    max_level_mmp: the number of levels in a single mmp layer 
    l_char: characteristic lengthscale of original graph (used in mmp coarsening) 
    max_level_topk: the number of topk levels 
    rf_topk: node reduction factor used in topk levels 
    E.g., if max_level=1, we have this:
        (L x Down MMP) 0 ---------------> (L x Up MMP) 0 --->
                |                               | 
                |                               |
                |----> (L x Down MMP) 1  ------>|
        where "L" is the number of MMP layers (n_mmp_layers) in each block  
    """
    def __init__(self, 
                 input_node_channels: int, 
                 input_edge_channels: int, 
                 hidden_channels: int, 
                 output_node_channels: int, 
                 n_mlp_hidden_layers: int, 
                 n_mmp_layers: int, 
                 n_messagePassing_layers: int,
                 max_level_mmp: int, 
                 l_char: float,
                 max_level_topk: int,
                 rf_topk: int,
                 name: Optional[str] = 'gnn'):
        super().__init__()
        
        self.input_node_channels = input_node_channels
        self.input_edge_channels = input_edge_channels
        self.hidden_channels = hidden_channels
        self.output_node_channels = output_node_channels 
        self.n_mlp_hidden_layers = n_mlp_hidden_layers
        self.n_mmp_layers = n_mmp_layers
        self.n_messagePassing_layers = n_messagePassing_layers
        self.max_level_mmp = max_level_mmp
        self.l_char = l_char
        self.max_level_topk = max_level_topk
        self.rf_topk = rf_topk
        self.n_levels = max_level_topk + 1
        self.name = name 

        # ~~~~ TopK factor for levels 
        self.pool_ratios = torch.zeros(self.max_level_topk)
        for i in range(self.max_level_topk):
            self.pool_ratios[i] = 1./self.rf_topk

        # ~~~~ node encoder MLP  
        self.node_encoder = MLP(
                input_channels = self.input_node_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.hidden_channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.hidden_channels)
                )

        # ~~~~ edge encoder MLP 
        self.edge_encoder = MLP(
                input_channels = self.input_edge_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.hidden_channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.hidden_channels)
                )

        # ~~~~ node decoder MLP  
        self.node_decoder = MLP(
                input_channels = self.hidden_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.output_node_channels,
                activation_layer = torch.nn.ELU(),
                )
        
        # ~~~~ down mmp layers
        self.mmp_down = torch.nn.ModuleList()
        for i in range(self.n_levels):
            mmp = torch.nn.ModuleList()
            for j in range(self.n_mmp_layers):
                mmp.append(
                            MultiscaleMessagePassingLayer(
                                channels = self.hidden_channels,
                                n_mlp_hidden_layers = self.n_mlp_hidden_layers,
                                n_messagePassing_layers = self.n_messagePassing_layers, 
                                max_level = self.max_level_mmp,
                                l_char = self.l_char
                              )   
                           )
            self.mmp_down.append(mmp)

        # ~~~~ up mmp layers
        self.mmp_up = torch.nn.ModuleList()
        for i in range(self.n_levels - 1):
            mmp = torch.nn.ModuleList()
            for j in range(self.n_mmp_layers):
                mmp.append(
                            MultiscaleMessagePassingLayer(
                                channels = self.hidden_channels,
                                n_mlp_hidden_layers = self.n_mlp_hidden_layers,
                                n_messagePassing_layers = self.n_messagePassing_layers, 
                                max_level = self.max_level_mmp,
                                l_char = self.l_char
                              )   
                           )
            self.mmp_up.append(mmp)

        # ~~~~ topk layers 
        self.topk = torch.nn.ModuleList()
        for i in range(self.max_level_topk):
            self.topk.append(
                    TopKPooling_Mod(self.hidden_channels, 
                                    self.pool_ratios[i])
                            )
        
        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: LongTensor,
            pos: Tensor,
            edge_attr: Tensor,
            batch: Optional[LongTensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # ~~~~ Mask initialization
        mask = x.new_zeros(x.size(0))

        # ~~~~ Node encoder 
        x = self.node_encoder(x) 

        # ~~~~ Edge encoder 
        edge_attr = self.edge_encoder(edge_attr) 

        # ~~~~ MMP down 
        i = 0 # on level 0 
        for j in range(self.n_mmp_layers):
            x,edge_attr = self.mmp_down[i][j](x, edge_attr, edge_index, pos, batch)
        xs = [x]
        eis = [edge_index]
        eas = [edge_attr]
        poss = [pos]
        batches = [batch]
        perms = [] 
        edge_masks = []
        for i in range(1,self.n_levels): # on levels > 0
            # pool 
            x, edge_index, edge_attr, batch, perm, edge_mask, _ = self.topk[i-1](
                    x,
                    edge_index,
                    edge_attr,
                    batch)
            pos = pos[perm]
            # append pos, batch, perm, edge_mask list 
            poss += [pos]
            batches += [batch]
            perms += [perm]
            edge_masks += [edge_mask]
            # mmp 
            for j in range(self.n_mmp_layers):
                x,edge_attr = self.mmp_down[i][j](x, edge_attr, edge_index, pos, batch)
            # append other lists if there are coarser levels  
            if i < self.max_level_topk:
                xs += [x] 
                eis += [edge_index]
                eas += [edge_attr]

        # ~~~~ populate mask 
        if self.max_level_topk > 0:
            perm_global = perms[0]
            mask[perm_global] = 1
            for i in range(1,self.max_level_topk):
                perm_global = perm_global[perms[i]]
                mask[perm_global] = i+1

        # ~~~~ MMP up
        for i in range(self.n_levels-1):
            fine = self.max_level_topk - 1 - i # gets the fine level index 
            # fill in node features at fine level  
            xf = torch.zeros_like(xs[fine]) # init as zeros 
            xf[perms[fine]] = x # fill in topk locations with nodes at current level 
            x = xf + xs[fine] # add a skip connection  
            # fill in edge features at fine level 
            eaf = torch.zeros_like(eas[fine]) # init as zeros 
            eaf[edge_masks[fine]] = edge_attr # fill in with edge features at current level 
            edge_attr = eaf + eas[fine] # add a skip connection
            # mmp
            edge_index = eis[fine]
            pos = poss[fine]
            batch = batches[fine]
            for j in range(self.n_mmp_layers):
                x,edge_attr = self.mmp_up[i][j](x, edge_attr, edge_index, pos, batch)

        # ~~~~ Node decoder 
        x = self.node_decoder(x)

        return x, mask 

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.node_decoder.reset_parameters()
        for mmp in self.mmp_down:
            for module in mmp: 
                module.reset_parameters()
        for mmp in self.mmp_up:
            for module in mmp:
                module.reset_parameters()
        for module in self.topk:
            module.reset_parameters()
        return

    def input_dict(self) -> dict:
        a = {'input_node_channels': self.input_node_channels,
             'input_edge_channels': self.input_edge_channels,
             'hidden_channels': self.hidden_channels, 
             'output_node_channels': self.output_node_channels,
             'n_mlp_hidden_layers': self.n_mlp_hidden_layers,
             'n_messagePassing_layers': self.n_messagePassing_layers,
             'max_level_mmp': self.max_level_mmp,
             'l_char': self.l_char, 
             'max_level_topk': self.max_level_topk,
             'rf_topk': self.rf_topk,
             'name': self.name} 
        return a

    def get_save_header(self) -> str:
        a = self.input_dict()
        header = a['name']
        
        for key in a.keys():
            if key not in ['name','l_char']:
                header += '_' + str(a[key])

        #for item in self.input_dict():
        return header


class MultiscaleGNN(torch.nn.Module):
    def __init__(self, 
                 input_node_channels: int, 
                 input_edge_channels: int, 
                 hidden_channels: int, 
                 output_node_channels: int, 
                 n_mlp_hidden_layers: int, 
                 n_mmp_layers: int, 
                 n_messagePassing_layers: int,
                 max_level: int, 
                 l_char: float,
                 name: Optional[str] = 'gnn'):
        super().__init__()
        
        self.input_node_channels = input_node_channels
        self.input_edge_channels = input_edge_channels
        self.hidden_channels = hidden_channels
        self.output_node_channels = output_node_channels 
        self.n_mlp_hidden_layers = n_mlp_hidden_layers
        self.n_mmp_layers = n_mmp_layers
        self.n_messagePassing_layers = n_messagePassing_layers
        self.max_level = max_level
        self.l_char = l_char
        self.name = name 

        # ~~~~ node encoder MLP  
        self.node_encoder = MLP(
                input_channels = self.input_node_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.hidden_channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.hidden_channels)
                )

        # ~~~~ edge encoder MLP 
        self.edge_encoder = MLP(
                input_channels = self.input_edge_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.hidden_channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.hidden_channels)
                )

        # ~~~~ node decoder MLP  
        self.node_decoder = MLP(
                input_channels = self.hidden_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.output_node_channels,
                activation_layer = torch.nn.ELU(),
                )
        
        # ~~~~ Processor 
        self.mmp_processor = torch.nn.ModuleList()
        for i in range(self.n_mmp_layers):
            self.mmp_processor.append( 
                          MultiscaleMessagePassingLayer(
                                     channels = self.hidden_channels,
                                     n_mlp_hidden_layers = self.n_mlp_hidden_layers, 
                                     n_messagePassing_layers = self.n_messagePassing_layers,
                                     max_level = self.max_level,
                                     l_char = self.l_char
                                     ) 
                                  )
        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: LongTensor,
            pos: Tensor,
            edge_attr: Tensor,
            batch: Optional[LongTensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # ~~~~ Node encoder 
        x = self.node_encoder(x) 

        # ~~~~ Edge encoder 
        e = self.edge_encoder(edge_attr) 

        # ~~~~ Processor 
        for i in range(self.n_mmp_layers):
            x,e = self.mmp_processor[i](x,e,edge_index,pos,batch)

        # ~~~~ Node decoder 
        x = self.node_decoder(x)
        
        return x 

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.node_decoder.reset_parameters()
        for module in self.mmp_processor:
            module.reset_parameters()
        return

    def input_dict(self) -> dict:
        a = {'input_node_channels': self.input_node_channels,
             'input_edge_channels': self.input_edge_channels,
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

class SinglescaleGNN(torch.nn.Module):
    def __init__(self, 
                 input_node_channels: int, 
                 input_edge_channels: int, 
                 hidden_channels: int, 
                 output_node_channels: int, 
                 n_mlp_hidden_layers: int, 
                 n_messagePassing_layers: int,
                 name: Optional[str] = 'gnn'):
        super().__init__()
        
        self.input_node_channels = input_node_channels
        self.input_edge_channels = input_edge_channels
        self.hidden_channels = hidden_channels
        self.output_node_channels = output_node_channels 
        self.n_mlp_hidden_layers = n_mlp_hidden_layers
        self.n_messagePassing_layers = n_messagePassing_layers
        self.name = name 

        # ~~~~ node encoder MLP  
        self.node_encoder = MLP(
                input_channels = self.input_node_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.hidden_channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.hidden_channels)
                )

        # ~~~~ edge encoder MLP 
        self.edge_encoder = MLP(
                input_channels = self.input_edge_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.hidden_channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.hidden_channels)
                )

        # ~~~~ node decoder MLP  
        self.node_decoder = MLP(
                input_channels = self.hidden_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.output_node_channels,
                activation_layer = torch.nn.ELU(),
                )
        
        # ~~~~ Processor 
        self.processor = torch.nn.ModuleList()
        for i in range(self.n_messagePassing_layers):
            self.processor.append( 
                          MessagePassingLayer(
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
            batch: Optional[LongTensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # ~~~~ Node encoder 
        x = self.node_encoder(x) 

        # ~~~~ Edge encoder 
        e = self.edge_encoder(edge_attr) 

        # ~~~~ Processor 
        for i in range(self.n_messagePassing_layers):
            x,e = self.processor[i](x,e,edge_index,batch)

        # ~~~~ Node decoder 
        x = self.node_decoder(x)
        
        return x 

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.node_decoder.reset_parameters()
        for module in self.processor:
            module.reset_parameters()
        return

    def input_dict(self) -> dict:
        a = {'input_node_channels': self.input_node_channels,
             'input_edge_channels': self.input_edge_channels,
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

class MessagePassingLayer(torch.nn.Module):
    def __init__(self, 
                 channels: int, 
                 n_mlp_hidden_layers: int):
        super().__init__()

        self.edge_aggregator = EdgeAggregation(aggr='add')
        self.channels = channels
        self.n_mlp_hidden_layers = n_mlp_hidden_layers 

        # Edge update MLP 
        self.edge_updater = MLP(
                input_channels = self.channels*3,
                hidden_channels = [self.channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.channels)
                )

        # Node update MLP
        self.node_updater = MLP(
                input_channels = self.channels*2,
                hidden_channels = [self.channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.channels)
                )

        self.reset_parameters()

        return 

    def forward(
            self,
            x: Tensor,
            e: Tensor,
            edge_index: LongTensor,
            batch: Optional[LongTensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # ~~~~ Edge update 
        e += self.edge_updater(
                torch.cat((x[edge_index[0,:],:], x[edge_index[1,:],:], e), dim=1)
                )
        
        # ~~~~ Edge aggregation
        edge_agg = self.edge_aggregator(x, edge_index, e)

        # ~~~~ Node update 
        x += self.node_updater(
                torch.cat((x, edge_agg), dim=1)
                )

        return x,e  

    def reset_parameters(self):
        self.edge_updater.reset_parameters()
        self.node_updater.reset_parameters()
        return

class MultiscaleMessagePassingLayer(torch.nn.Module):
    """
    max_level: the highest level index for multiscale operations.
    l_char: characteristic lengthscale. This can be, e.g. average edge length at level 0. 
    
    E.g., if max_level=1, we have this:
        
        Down Processor 0 ---------------> Up Processor 0 --->
                |                               | 
                |                               |
                |----> Down Processor 1  ------>|

        where a "processor" is (L x MessagePassingLayer), and L = n_messagePassing_layers
    """
    def __init__(self, 
                 channels: int, 
                 n_mlp_hidden_layers: int,
                 n_messagePassing_layers: int,
                 max_level: int,
                 l_char: float):
        super().__init__()

        self.channels = channels
        self.n_mlp_hidden_layers = n_mlp_hidden_layers 
        self.n_messagePassing_layers = n_messagePassing_layers
        self.max_level = max_level
        self.n_levels = max_level+1
        self.l_char = l_char
        coarsen_factor = 2 

        # lengthscales for levels 
        self.lengthscales = torch.zeros(self.n_levels)
        self.lengthscales[0] = self.l_char
        for i in range(1,self.n_levels):
            self.lengthscales[i] = self.lengthscales[i-1] * coarsen_factor

        # down processors 
        self.processors_down = torch.nn.ModuleList()
        for i in range(self.n_levels):
            processor = torch.nn.ModuleList()
            for j in range(self.n_messagePassing_layers):
                processor.append( 
                          MessagePassingLayer(
                                     channels = channels,
                                     n_mlp_hidden_layers = self.n_mlp_hidden_layers, 
                                     )   
                                )   
            self.processors_down.append(processor)

        # up processors 
        self.processors_up = torch.nn.ModuleList()
        for i in range(self.n_levels - 1):
            processor = torch.nn.ModuleList()
            for j in range(self.n_messagePassing_layers):
                processor.append(
                          MessagePassingLayer(
                                     channels = channels,
                                     n_mlp_hidden_layers = self.n_mlp_hidden_layers,
                                     )
                                )
            self.processors_up.append(processor)
        self.reset_parameters()
        return 

    def forward(
            self,
            x: Tensor,
            edge_attr: Tensor,
            edge_index: LongTensor,
            pos: Tensor,
            batch: Optional[LongTensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # Downward processor on level 0 
        for j in range(self.n_messagePassing_layers):
            x,edge_attr = self.processors_down[0][j](x,edge_attr,edge_index,batch)
        xs = [x]
        eis = [edge_index]
        eas = [edge_attr]
        poss = [pos]
        batches = [batch]
        # Downward processors on levels > 0
        for i in range(1,self.n_levels):
            # voxel
            cluster = tgnn.voxel_grid(
                    pos = pos,
                    size = self.lengthscales[i],
                    batch = batch)
            # coarse graph
            x, edge_index, edge_attr, batch, pos, cluster, perm = avg_pool_mod(
                    cluster,
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    pos)
            # append pos and batch list 
            poss += [pos]
            batches += [batch]
            # processor 
            for j in range(self.n_messagePassing_layers):
                x,edge_attr = self.processors_down[i][j](x,edge_attr,edge_index,batch)
            # append other lists 
            if i < self.max_level:
                xs += [x] 
                eis += [edge_index]
                eas += [edge_attr]

        # upward 
        for i in range(self.n_levels-1):
            fine = self.max_level - 1 - i # gets the fine level index 
            # Interpolate node features  
            x = xs[fine] + tgnn.knn_interpolate(
                                 x = x,
                                 pos_x = poss[fine+1],
                                 pos_y = poss[fine],
                                 batch_x = batches[fine+1],
                                 batch_y = batches[fine],
                                 k = 4)
            # processor 
            edge_attr = eas[fine]
            edge_index = eis[fine]
            batch = batches[fine]
            for j in range(self.n_messagePassing_layers):
                x,edge_attr = self.processors_up[i][j](x,edge_attr,edge_index,batch)
            
        return x,edge_attr 

    def reset_parameters(self):
        for processor in self.processors_down:
            for module in processor:
                module.reset_parameters()
        for processor in self.processors_up:
            for module in processor:
                module.reset_parameters()
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
