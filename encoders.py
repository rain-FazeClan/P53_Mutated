# encoders.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool, NNConv, PointNetConv # Choose appropriate point conv
from torch_geometric.nn.pool import fps # Farthest Point Sampling
from utils import build_mlp
import math

# --- Image Encoder ---
class ImageEncoder(nn.Module):
    def __init__(self, input_channels, channels, fc_dims):
        super().__init__()
        conv_layers = []
        current_channels = input_channels
        for out_channels in channels[1:]:
            conv_layers.append(nn.Conv3d(current_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.BatchNorm3d(out_channels))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            current_channels = out_channels
        self.conv_part = nn.Sequential(*conv_layers)

        # Need to calculate the flattened size after conv layers
        # Example: Assuming input size (B, C, D, H, W) = (B, 4, 128, 128, 128)
        # This needs to be adjusted based on actual input size
        # dummy_input = torch.randn(1, input_channels, 128, 128, 128) # Adjust dimensions
        # with torch.no_grad():
        #     flattened_size = self.conv_part(dummy_input).view(1, -1).size(1)
        flattened_size = 256 * 4 * 4 * 4 # Placeholder - MUST BE CALCULATED BASED ON INPUT SIZE
        print(f"Image Encoder flattened size (placeholder): {flattened_size}")

        fc_dims[0] = flattened_size # Set the first FC layer input dim
        self.fc_part = build_mlp(fc_dims)

    def forward(self, x): # x: [B, C, D, H, W]
        x = self.conv_part(x)
        x = x.view(x.size(0), -1) # Flatten
        uI = self.fc_part(x) # Image features
        return uI

# --- Geometric Encoder ---
# Using NNConv as mentioned in paper discussion
class GeometricEncoder(nn.Module):
    def __init__(self, input_point_dim, conv_dims, fc_dims, num_points_out=None):
        super().__init__()
        # NN for edge features (e.g., distance)
        nn1 = build_mlp([input_point_dim * 2, 25, conv_dims[1]]) # Example edge nn
        self.conv1 = NNConv(conv_dims[0], conv_dims[1], nn1, aggr='mean')

        nn2 = build_mlp([input_point_dim * 2, 50, conv_dims[1] * conv_dims[2]]) # Example edge nn
        self.conv2 = NNConv(conv_dims[1], conv_dims[2], nn2, aggr='mean')

        # Add more layers as needed matching GEOM_ENCODER_DIMS
        # ...

        self.global_pool = nn.AdaptiveAvgPool1d(1) # Simpler global pooling
        fc_dims[0] = conv_dims[-1] # Input to FC is the last conv dim
        self.fc_part = build_mlp(fc_dims)
        # Paper mentions global attention pooling - could use attention mechanism here

    def forward(self, data): # data is PyG Data(x=[N, 3], pos=[N, 3], batch=[N]) or just points tensor
        # Assuming input is points tensor [B, N, 3]
        # Need to convert to PyG Batch format for NNConv or adapt NNConv
        # Or use PointNet-style architecture directly
        # --- Placeholder using PointNet features ---
        # Requires PointNetConv or similar
        # x = F.relu(self.conv1(data.x, data.pos, data.edge_index)) # Need edges for NNConv
        # ...
        # Placeholder: simple MLP on points
        x = data.view(data.size(0), -1) # Flatten points [B, N*3]
        # This is NOT what the paper does, just a placeholder
        uP = self.fc_part(x) # Replace with actual PointNet/NNConv forward pass
        aP = torch.ones(data.size(0), data.size(1), device=data.device) / data.size(1) # Dummy attention
        print("GeometricEncoder forward pass is a placeholder!")
        return uP, aP # Geometric features, point attention


# --- Brain Network Encoder ---
class BrainNetworkEncoder(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, gat_dims, gat_heads, fc_dims, dropout=0.5):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        in_channels = node_feat_dim
        for i, out_channels in enumerate(gat_dims[1:]):
            heads = gat_heads[i]
            # Note: GATConv expects edge_index, not explicit edge features in standard implementation.
            # To use edge features, need GATv2Conv or modify GATConv/use edge attributes in attention calc.
            # Let's assume standard GAT for now, ignoring bE during convolution.
            self.gat_layers.append(GATConv(in_channels * heads if i > 0 else in_channels,
                                           out_channels, heads=heads, dropout=dropout, concat=True))
            # Update in_channels for next layer if concat=True
            # in_channels = out_channels # If concat=False or last layer

        # Calculate input dimension for FC layers based on final GAT output
        final_gat_dim = gat_dims[-1] * gat_heads[-1]
        fc_dims[0] = final_gat_dim
        self.fc_part = build_mlp(fc_dims)
        self.dropout = dropout

    def forward(self, data, node_attention=None): # data is PyG Batch(x_n, edge_index, edge_attr, batch)
        x = data.x_n # Node features (bN)
        edge_index = data.edge_index

        # Apply edge attention if GAT version supports it or modify attention calculation
        # edge_attr_attended = data.edge_attr * edge_attention # Element-wise - depends on how attention is used

        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index) # Add edge_attr if layer supports it
            if i < len(self.gat_layers) - 1:
                 x = F.elu(x)
                 x = F.dropout(x, p=self.dropout, training=self.training)

        # x now contains node embeddings (zN_m from paper, before projection)
        # Store node embeddings for attention calculation if needed externally
        node_embeddings = x # Shape: [TotalNodes, FinalGatDim]

        # Global Pooling using Node Attention (Eq 3 related)
        if node_attention is not None:
            # node_attention shape: [TotalNodes, 1]
            # Weighted sum pooling
            x_pooled = global_add_pool(x * node_attention, data.batch) # [BatchSize, FinalGatDim]
        else:
            # Default: simple global mean pooling if no attention
            x_pooled = global_add_pool(x, data.batch) # [BatchSize, FinalGatDim]

        uB = self.fc_part(x_pooled) # Brain network features

        return uB, node_embeddings