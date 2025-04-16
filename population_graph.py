# population_graph.py
import torch
from torch_geometric.data import Data
from utils import pearson_correlation # Or use torch.corrcoef
import numpy as np
from config import POP_GRAPH_CORR_THRESHOLD

def build_population_graph(all_features, patient_ids, labels_df, config):
    """ Constructs the population graph based on extracted features. """
    print("Building population graph...")
    num_patients = len(patient_ids)
    id_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    node_features = []
    edge_indices = [[], []]
    edge_weights = []
    y_labels = []

    # Select features based on config
    node_weight_type = config.POP_GRAPH_NODE_WEIGHT_TYPE
    edge_weight_type = config.POP_GRAPH_EDGE_WEIGHT_TYPE

    feature_map = {} # Temp store tensors for correlation calc

    for i, patient_id in enumerate(patient_ids):
        features = all_features[patient_id]

        # Determine node feature w_node (Table II)
        if node_weight_type == 'UF':
            node_feat = features['uF']
        elif node_weight_type == 'UB':
            node_feat = features['uB']
        elif node_weight_type == 'concat_UF_UB':
            node_feat = torch.cat([features['uF'], features['uB']], dim=0)
        else:
            raise ValueError(f"Unknown node_weight_type: {node_weight_type}")

        node_features.append(node_feat)
        feature_map[patient_id] = features # Store all for edge calculation
        y_labels.append(labels_df.loc[patient_id, 'IDH_status'])

    x = torch.stack(node_features) # Node features tensor [NumPatients, NodeFeatDim]
    y = torch.tensor(y_labels, dtype=torch.long) # Labels tensor [NumPatients]

    # Calculate edge weights w_edge (Table II, Eq 8)
    for i in range(num_patients):
        for j in range(i + 1, num_patients): # Avoid self-loops and duplicates
            id_i = patient_ids[i]
            id_j = patient_ids[j]

            feat_i = feature_map[id_i]
            feat_j = feature_map[id_j]

            # Determine features for correlation based on edge_weight_type
            if edge_weight_type == 'corr_UF':
                corr_feat_i = feat_i['uF']
                corr_feat_j = feat_j['uF']
            elif edge_weight_type == 'corr_UB':
                corr_feat_i = feat_i['uB']
                corr_feat_j = feat_j['uB']
            elif edge_weight_type == 'corr_concat_UF_UB':
                corr_feat_i = torch.cat([feat_i['uF'], feat_i['uB']], dim=0)
                corr_feat_j = torch.cat([feat_j['uF'], feat_j['uB']], dim=0)
            else:
                 raise ValueError(f"Unknown edge_weight_type: {edge_weight_type}")

            # Calculate correlation r(ui, uj) - Use Pearson
            # Ensure tensors are 1D or correctly shaped for correlation function
            # Using PyTorch's corrcoef requires [feature_dim, num_observations] -> stack and transpose
            # Or use custom pearson_correlation expecting [N, FeatureDim] -> need N=1
            # Let's assume features are 1D vectors [FeatureDim]
            # correlation = pearson_correlation(corr_feat_i.unsqueeze(0), corr_feat_j.unsqueeze(0)).item() # Returns single value
            # Using torch.corrcoef for potentially multi-dimensional features (treat patient as observation)
            combined = torch.stack([corr_feat_i, corr_feat_j], dim=1) # [FeatDim, 2]
            if combined.shape[0] > 1 and combined.shape[1] > 1:
                # Clamp to avoid numerical issues if features are constant
                std_dev = torch.std(combined, dim=0)
                if torch.any(std_dev < 1e-6):
                     correlation_matrix = torch.eye(2) # Treat as uncorrelated if no variance
                else:
                    correlation_matrix = torch.corrcoef(combined)
                correlation = correlation_matrix[0, 1].item()
            elif combined.shape[0] == 1: # Single feature
                 correlation = 1.0 if torch.isclose(corr_feat_i, corr_feat_j) else 0.0 # Simplistic
            else:
                 correlation = 0.0 # Should not happen with valid features


            # Apply threshold (Eq 8)
            if correlation >= config.POP_GRAPH_CORR_THRESHOLD:
                edge_weight = correlation
                # Add edges in both directions for undirected graph
                edge_indices[0].extend([i, j])
                edge_indices[1].extend([j, i])
                edge_weights.extend([edge_weight, edge_weight])

    edge_index = torch.tensor(edge_indices, dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1) # Edge weights as attributes

    # Create masks for semi-supervised learning
    # Use train_ids, val_ids, test_ids passed from main script
    # train_mask = torch.zeros(num_patients, dtype=torch.bool)
    # val_mask = torch.zeros(num_patients, dtype=torch.bool)
    # test_mask = torch.zeros(num_patients, dtype=torch.bool)
    # train_idx = [id_to_idx[pid] for pid in train_ids]
    # val_idx = [id_to_idx[pid] for pid in val_ids]
    # test_idx = [id_to_idx[pid] for pid in test_ids]
    # train_mask[train_idx] = True
    # val_mask[val_idx] = True
    # test_mask[test_idx] = True

    # Create PyG Data object
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                    # train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    print(f"Population graph built with {graph_data.num_nodes} nodes and {graph_data.num_edges // 2} unique edges.")
    return graph_data, id_to_idx