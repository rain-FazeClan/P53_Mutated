# attention.py
import torch
import torch.nn as nn
from utils import cosine_similarity
from torch_geometric.nn import knn_graph # For finding edges crossed by points

def project_points_to_edges(points_pos, edge_index, nodes_pos, k=1):
     """Find the k nearest edges for each point (simplistic projection)."""
     # Represent edges by their midpoints or endpoints
     edge_midpoints = (nodes_pos[edge_index[0]] + nodes_pos[edge_index[1]]) / 2
     # Find k nearest edge midpoints to each point
     # This is a simplification. True projection might involve line segment distance.
     assignment = knn_graph(edge_midpoints, k=k, query=points_pos) # query=points, x=edges
     # assignment[0] = edge indices, assignment[1] = point indices
     return assignment # Returns edge index [num_edges * k], point index [num_edges * k]

def calculate_edge_attention(point_attention, points_pos, edge_index, nodes_pos, num_edges, device):
     """ Approximates Eq 2 by assigning point attention to nearest edges. """
     # points_pos: [B, N_points, 3]
     # point_attention: [B, N_points]
     # edge_index: [2, N_edges] (Global, assumed same for all batch items for simplicity)
     # nodes_pos: [N_nodes, 3] (Global node positions)

     batch_size = points_pos.shape[0]
     all_edge_attentions = torch.zeros(batch_size, num_edges, device=device)

     # This projection needs careful batch handling if edge_index varies or nodes_pos varies.
     # Assuming fixed atlas for now.
     for i in range(batch_size):
         # Find which edges are "close" to which points
         # Use KNN between points and edge representations (e.g., midpoints)
         edge_indices_mapped, point_indices_mapped = project_points_to_edges(
             points_pos[i], edge_index, nodes_pos
         )

         # Aggregate attention: Sum attention of points mapped to the same edge
         # Use scatter_add
         point_att_flat = point_attention[i][point_indices_mapped]
         edge_att_aggregated = torch.zeros(num_edges, device=device).scatter_add_(
             0, edge_indices_mapped, point_att_flat
         )

         # Normalize attention per edge (average over points mapped to it)
         point_counts = torch.zeros(num_edges, device=device).scatter_add_(
             0, edge_indices_mapped, torch.ones_like(point_att_flat)
         )
         edge_attention_normalized = edge_att_aggregated / point_counts.clamp(min=1.0) # Eq 2 (K=point_counts)
         all_edge_attentions[i] = edge_attention_normalized

     # Reshape edge attention to match edge_attr in PyG Batch if needed
     # Example: If PyG Batch concatenates edges, attention needs to be [TotalEdges, 1]
     # This depends on how BrainNetworkEncoder expects it. Let's return [B, N_edges] for now.
     return all_edge_attentions # aE: [B, N_edges]

class NodeAttentionModule(nn.Module):
     """ Calculates node attention based on similarity (Eq 3). """
     def __init__(self, node_embedding_dim, tumor_feature_dim, projection_dim):
         super().__init__()
         # Projection heads g_node (ga in paper), g_focal (gF in paper, part of contrastive learning)
         self.g_node = build_mlp([node_embedding_dim, projection_dim // 2, projection_dim])
         # g_focal is assumed to be part of the main contrastive learning module

     def forward(self, node_embeddings, focal_features_projected, batch_index):
         # node_embeddings (zN_m): [TotalNodes, NodeEmbDim] from BrainNetworkEncoder
         # focal_features_projected (gF(uF)): [BatchSize, ProjectionDim]
         # batch_index: [TotalNodes] mapping node to batch item

         # Project node embeddings
         node_embeddings_projected = self.g_node(node_embeddings) # [TotalNodes, ProjectionDim]

         # Expand focal features to match node batch index
         focal_features_expanded = focal_features_projected[batch_index] # [TotalNodes, ProjectionDim]

         # Calculate similarity (Eq 3)
         # Cosine similarity expects [N, D] inputs
         similarity = cosine_similarity(node_embeddings_projected, focal_features_expanded, dim=1) # [TotalNodes]

         # Normalize similarity (e.g., Softmax per graph) - Paper doesn't specify normalization here
         # Could use softmax or just the raw similarity scores
         # Applying softmax per graph in the batch:
         # Use scatter_softmax from torch_scatter if available, otherwise loop or use workarounds
         # For simplicity, let's return raw similarity for now, apply softmax in pooling if needed
         node_attention = similarity # aN: [TotalNodes]

         return node_attention