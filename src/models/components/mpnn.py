import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import einsum
from torch_geometric.nn import GATv2Conv


class GATv2Conv_module(nn.Module):
    """Simplified GNN module with GATv2Conv."""
    def __init__(self, 
                 node_feat_channels, 
                 edge_feat_channels, 
                 atten_heads=4, 
                 dropout_rate=0.1, 
                 atten_dropout_rate=0.1, 
                 add_self_loops=True, 
                 activ_fn=nn.GELU(), 
                 residual=True, 
                 return_atten_weights=False):
        
        super().__init__()
        self.node_feat_channels = node_feat_channels
        self.edge_feat_channels = edge_feat_channels
        self.atten_heads = atten_heads
        self.dropout_rate = dropout_rate
        self.atten_dropout_rate = atten_dropout_rate
        self.add_self_loops = add_self_loops
        self.residual = residual
        self.return_atten_weights = return_atten_weights
        self.activ_fn = activ_fn if isinstance(activ_fn, nn.Module) else nn.GELU()

        # GATv2Conv layer
        self.gat_layer = GATv2Conv(in_channels=node_feat_channels,
                                   out_channels=node_feat_channels // atten_heads,
                                   heads=atten_heads,
                                   concat=True,
                                   dropout=atten_dropout_rate,
                                   add_self_loops=add_self_loops,
                                   edge_dim=edge_feat_channels)

        # Layer normalization and MLP
        self.norm1 = nn.LayerNorm(node_feat_channels)
        self.norm2 = nn.LayerNorm(node_feat_channels)
        self.mlp = nn.Sequential(
            nn.Linear(node_feat_channels, node_feat_channels * 2, bias=False),
            self.activ_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(node_feat_channels * 2, node_feat_channels, bias=False)
        )

    def forward(self, node_feats, edge_index, edge_attr):
        node_feats_res = node_feats
        # GATv2Conv layer
        if self.return_atten_weights:
            node_feats, attn_weights = self.gat_layer(node_feats, edge_index, edge_attr, self.return_atten_weights)
        else:
            node_feats = self.gat_layer(node_feats, edge_index, edge_attr)
        node_feats = nn.Dropout(self.dropout_rate)(node_feats)

        # Residual connection and normalization
        if self.residual:
            node_feats += node_feats_res
        node_feats = self.norm1(node_feats)

        # MLP with residual
        node_feats_res = node_feats
        node_feats = self.mlp(node_feats)
        if self.residual:
            node_feats += node_feats_res
        node_feats = self.norm2(node_feats)

        return (node_feats, attn_weights) if self.return_atten_weights else node_feats
        
class graph_MPNN(nn.Module):
    """Simplified GNN module with GATv2Conv."""
    def __init__(self, 
                 input_node_size, 
                 input_edge_size, 
                 lm_hidden_size=512, 
                 node_feat_channels=128, 
                 edge_feat_channels=128,
                 atten_heads=4, 
                 dropout_rate=0.1,
                 num_layers=4):
        
        super().__init__()
        self.init_node_trans = nn.Linear(input_node_size, node_feat_channels, bias=False)
        self.init_edge_trans = nn.Linear(input_edge_size, edge_feat_channels, bias=False)
        self.graph_modules = nn.ModuleList([
            GATv2Conv_module(node_feat_channels, edge_feat_channels, atten_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        self.final_node_trans = nn.Linear(node_feat_channels, lm_hidden_size, bias=False)

    def preprocess_edge_attr(self, all_local_attn_weights, edge_index):
        num_edges = edge_index.shape[1]
        
        layers, heads, seq_len, _ = all_local_attn_weights.shape
        num_features = layers * heads  
        
        edge_attr = all_local_attn_weights.permute(2, 3, 0, 1).reshape(seq_len, seq_len, num_features)  
        
        selected_edge_attr = edge_attr[edge_index[0], edge_index[1], :] 
        return selected_edge_attr 

    def forward(self, node_feats, edge_index, all_local_attn_weights, mask):
        node_feats = self.init_node_trans(node_feats) 
        
        edge_attr = self.preprocess_edge_attr(all_local_attn_weights, edge_index) 
    
        edge_feats = self.init_edge_trans(edge_attr)  # Shape: [num_edges, edge_feat_channels]
            
        for mp_layer in self.graph_modules:
            node_feats = mp_layer(node_feats, edge_index, edge_feats)

    
        node_feats = self.final_node_trans(node_feats)
        
        # Apply mask if provided
        if mask is not None:
            node_feats = node_feats * mask.unsqueeze(-1)  # Keep it 2D
    
        return node_feats


