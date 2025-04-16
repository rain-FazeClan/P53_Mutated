# classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv, global_mean_pool
from config import (DEVICE, CLASSIFIER_GAT_DIMS, CLASSIFIER_GAT_HEADS, CLASSIFIER_N_CLASSES,
                    CLASSIFIER_LR, CLASSIFIER_WD, CLASSIFIER_EPOCHS, CLASSIFIER_MODEL_PATH)
from utils import calculate_metrics

class PopulationGNNClassifier(nn.Module):
    def __init__(self, input_dim, gat_dims, gat_heads, n_classes, dropout=0.5):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        in_channels = input_dim
        gat_dims[0] = input_dim # Set input dim for first layer
        for i, out_channels in enumerate(gat_dims[1:]):
            heads = gat_heads[i]
            # GATConv using edge_attr (weights) for attention scores
            self.gat_layers.append(GATConv(in_channels * heads if i > 0 else in_channels,
                                           out_channels, heads=heads, dropout=dropout,
                                           concat=True, add_self_loops=True, edge_dim=1)) # edge_dim=1 for edge weights
            # Update in_channels for next layer if concat=True
            # in_channels = out_channels # If concat=False or last layer

        final_gat_dim = gat_dims[-1] * gat_heads[-1]
        # Final classification layer (acting on node embeddings)
        self.class_head = nn.Linear(final_gat_dim, n_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index, edge_attr=edge_attr) # Pass edge weights
            if i < len(self.gat_layers) - 1:
                 x = F.elu(x)
                 x = F.dropout(x, p=self.dropout, training=self.training)
            else: # Last GAT layer often doesn't need activation/dropout before classifier head
                 pass

        # x now contains node embeddings for classification
        logits = self.class_head(x)
        # No global pooling needed, classify each node
        return logits

def train_classifier(model, graph_data, train_mask, val_mask, config):
    """Trains the GNN classifier in a semi-supervised manner."""
    print("Training Population GNN Classifier...")
    optimizer = optim.Adam(model.parameters(), lr=config.CLASSIFIER_LR, weight_decay=config.CLASSIFIER_WD)
    criterion = nn.CrossEntropyLoss()

    graph_data = graph_data.to(config.DEVICE)
    best_val_auc = 0.0

    for epoch in range(config.CLASSIFIER_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(graph_data)
        loss = criterion(logits[train_mask], graph_data.y[train_mask])
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(graph_data)
            val_loss = criterion(logits[val_mask], graph_data.y[val_mask])
            probs = F.softmax(logits[val_mask], dim=1)[:, 1] # Prob of class 1
            metrics = calculate_metrics(graph_data.y[val_mask].cpu().numpy(), probs.cpu().numpy())
            val_auc = metrics['AUC']

        print(f"Epoch [{epoch+1}/{config.CLASSIFIER_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), config.CLASSIFIER_MODEL_PATH)
            print(f"Best classifier model saved to {config.CLASSIFIER_MODEL_PATH}")

    print("Classifier training finished.")
    model.load_state_dict(torch.load(config.CLASSIFIER_MODEL_PATH)) # Load best model
    return model

def evaluate_classifier(model, graph_data, test_mask, config):
    """Evaluates the trained classifier on the test set."""
    print("Evaluating classifier on test set...")
    model.eval()
    graph_data = graph_data.to(config.DEVICE)
    with torch.no_grad():
        logits = model(graph_data)
        probs = F.softmax(logits[test_mask], dim=1)[:, 1] # Prob of class 1
        metrics = calculate_metrics(graph_data.y[test_mask].cpu().numpy(), probs.cpu().numpy())

    print("-" * 30)
    print("Test Set Performance:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("-" * 30)
    return metrics