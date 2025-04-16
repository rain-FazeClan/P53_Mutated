# contrastive_learning.py
import torch
import torch.nn as nn
import torch.optim as optim
from config import (DEVICE, IMG_ENCODER_CHANNELS, IMG_ENCODER_FC_DIMS, IMG_FEATURE_DIM,
                    GEOM_ENCODER_DIMS, GEOM_ENCODER_FC_DIMS, GEOM_FEATURE_DIM,
                    BRAIN_NET_ENCODER_GAT_DIMS, BRAIN_NET_ENCODER_GAT_HEADS,
                    BRAIN_NET_ENCODER_FC_DIMS, BRAIN_NODE_FEATURE_DIM, BRAIN_EDGE_FEATURE_DIM,
                    PROJECTION_HEAD_DIMS, CONTRASTIVE_LATENT_DIM, BRAIN_NET_FEATURE_DIM_OUT,
                    CONTRASTIVE_TEMP, CONTRASTIVE_LAMBDA, CONTRASTIVE_LR, CONTRASTIVE_WD,
                    CONTRASTIVE_EPOCHS, CONTRASTIVE_MODEL_PATH)
from encoders import ImageEncoder, GeometricEncoder, BrainNetworkEncoder
from attention import NodeAttentionModule, calculate_edge_attention
from losses import BiLevelContrastiveLoss
from utils import build_mlp
from torch_geometric.data import Batch # To handle graph batches

class MultiModalContrastiveModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- Encoders ---
        self.image_encoder = ImageEncoder(config.IMG_INPUT_CHANNELS, config.IMG_ENCODER_CHANNELS,
                                          config.IMG_ENCODER_FC_DIMS.copy())
        self.geometric_encoder = GeometricEncoder(3, config.GEOM_ENCODER_DIMS, # Input dim=3 (coords)
                                                  config.GEOM_ENCODER_FC_DIMS.copy())
        self.brain_network_encoder = BrainNetworkEncoder(config.BRAIN_NODE_FEATURE_DIM, config.BRAIN_EDGE_FEATURE_DIM,
                                                         config.BRAIN_NET_ENCODER_GAT_DIMS, config.BRAIN_NET_ENCODER_GAT_HEADS,
                                                         config.BRAIN_NET_ENCODER_FC_DIMS.copy())

        # --- Attention Modules ---
        # Edge attention calculation needs fixed node positions if using atlas
        # self.node_positions = load_node_positions(config.NODE_ATLAS_FILE) # Implement this
        self.node_positions = torch.randn(config.BRAIN_NODE_N_REGIONS, 3).to(DEVICE) # Placeholder

        self.node_attention_module = NodeAttentionModule(config.BRAIN_NET_ENCODER_GAT_DIMS[-1] * config.BRAIN_NET_ENCODER_GAT_HEADS[-1],
                                                         config.CONTRASTIVE_LATENT_DIM, # Uses projected focal features
                                                         config.CONTRASTIVE_LATENT_DIM) # Projects node embeddings


        # --- Projection Heads (gI, gP, gB, gF) ---
        proj_head_dims = config.PROJECTION_HEAD_DIMS.copy()

        proj_head_dims[0] = config.IMG_FEATURE_DIM
        self.gI = build_mlp(proj_head_dims)

        proj_head_dims[0] = config.GEOM_FEATURE_DIM
        self.gP = build_mlp(proj_head_dims)

        proj_head_dims[0] = config.BRAIN_NET_FEATURE_DIM_OUT
        self.gB = build_mlp(proj_head_dims)

        # gF projects concatenated focal features (uI, uP)
        proj_head_dims[0] = config.IMG_FEATURE_DIM + config.GEOM_FEATURE_DIM
        self.gF = build_mlp(proj_head_dims)

        # --- Loss ---
        self.contrastive_loss = BiLevelContrastiveLoss(config.CONTRASTIVE_TEMP, config.CONTRASTIVE_LAMBDA)

    def forward(self, batch):
        # Extract data from batch
        img = batch['image'].to(DEVICE)         # [B, C, D, H, W]
        pts = batch['points'].to(DEVICE)       # [B, N_points, 3]
        # Brain network data needs batching using torch_geometric.data.Batch
        network_batch = Batch.from_data_list([data for data in batch['network']]).to(DEVICE)

        # --- Encode Features ---
        uI = self.image_encoder(img)                # [B, ImgFeatDim]
        uP, aP = self.geometric_encoder(pts)        # [B, GeomFeatDim], [B, N_points]

        # --- Hierarchical Attention for Brain Network ---
        # 1. Calculate Edge Attention (aE)
        # Requires global edge_index, assuming fixed atlas
        global_edge_index = network_batch.edge_index[:, :self.config.BRAIN_EDGE_N_TRACTS] # Get one copy
        aE = calculate_edge_attention(aP, pts, global_edge_index, self.node_positions,
                                      self.config.BRAIN_EDGE_N_TRACTS, DEVICE)
        # aE needs to be expanded/repeated to match number of edges in the PyG Batch object
        # Example: aE_expanded = aE.repeat_interleave(torch.bincount(network_batch.batch), dim=0) # Check dimensions
        aE_expanded = None # Placeholder - needs correct expansion based on PyG batching

        # 2. Encode Brain Network with potential Edge Attention (depends on GAT implementation)
        # Pass aE_expanded to encoder if it uses it
        uB, node_embeddings = self.brain_network_encoder(network_batch) # [B, BrainFeatDim], [TotalNodes, NodeEmbDim]

        # 3. Calculate Node Attention (aN) using projected focal features
        uF_concat = torch.cat([uI, uP], dim=1)       # [B, ImgFeatDim + GeomFeatDim]
        zF = self.gF(uF_concat)                     # [B, ProjDim] - Projected Focal Tumor
        aN = self.node_attention_module(node_embeddings, zF, network_batch.batch) # [TotalNodes]

        # 4. Re-encode Brain Network using Node Attention for Pooling
        uB_attended, _ = self.brain_network_encoder(network_batch, node_attention=aN.unsqueeze(1)) # [B, BrainFeatDim]

        # --- Project Features ---
        zI = self.gI(uI)                            # [B, ProjDim]
        zP = self.gP(uP)                            # [B, ProjDim]
        zB = self.gB(uB_attended)                   # [B, ProjDim]
        # zF was calculated earlier for node attention

        # --- Calculate Loss ---
        loss = self.contrastive_loss(zI, zP, zB, zF)

        # Return loss and features for potential inspection or downstream tasks
        return loss, {'uI': uI, 'uP': uP, 'uB': uB_attended, 'uF': uF_concat,
                       'zI': zI, 'zP': zP, 'zB': zB, 'zF': zF}

    def extract_features(self, batch):
        """Extracts features uI, uP, uB after training."""
        self.eval()
        with torch.no_grad():
            # Simplified forward pass just for feature extraction
            img = batch['image'].to(DEVICE)
            pts = batch['points'].to(DEVICE)
            network_batch = Batch.from_data_list([data for data in batch['network']]).to(DEVICE)

            uI = self.image_encoder(img)
            uP, aP = self.geometric_encoder(pts)

            # Need to compute attention again for consistent uB
            uF_concat = torch.cat([uI, uP], dim=1)
            zF = self.gF(uF_concat) # Requires gF to be trained

            _, node_embeddings = self.brain_network_encoder(network_batch)
            aN = self.node_attention_module(node_embeddings, zF, network_batch.batch)

            uB_attended, _ = self.brain_network_encoder(network_batch, node_attention=aN.unsqueeze(1))

            uF = uF_concat # Combined focal features

            # Return features per batch item (needs de-batching)
            # This part requires mapping features back to individual patients
            # For simplicity, returning batched features for now
            return {'uI': uI.cpu(), 'uP': uP.cpu(), 'uB': uB_attended.cpu(), 'uF': uF.cpu(), 'ids': batch['id']}


def train_contrastive_model(model, train_loader, val_loader, config):
    """Trains the MultiModalContrastiveModel."""
    print("Training Multi-Modal Contrastive Model...")
    optimizer = optim.SGD(model.parameters(), lr=config.CONTRASTIVE_LR, weight_decay=config.CONTRASTIVE_WD, momentum=0.9)
    # Add scheduler if desired (e.g., reduce LR on plateau or cosine annealing)

    best_val_loss = float('inf')

    for epoch in range(config.CONTRASTIVE_EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _ = model(batch)
            if torch.isnan(loss):
                print("Warning: NaN loss detected. Skipping batch.")
                continue
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss, _ = model(batch)
                if not torch.isnan(loss):
                    val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{config.CONTRASTIVE_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.CONTRASTIVE_MODEL_PATH)
            print(f"Best model saved to {config.CONTRASTIVE_MODEL_PATH}")

    print("Contrastive training finished.")
    # Load best model for feature extraction
    model.load_state_dict(torch.load(config.CONTRASTIVE_MODEL_PATH))
    return model

def extract_all_features(model, feature_loader, config):
    """Extracts features (uI, uP, uB, uF) for all patients using the trained model."""
    print("Extracting features for all patients...")
    model.eval()
    all_features = {} # Dict to store features per patient_id
    with torch.no_grad():
        for batch in feature_loader:
            features = model.extract_features(batch)
            # De-batch and store features by patient ID
            for i, patient_id in enumerate(features['ids']):
                all_features[patient_id] = {
                    'uI': features['uI'][i],
                    'uP': features['uP'][i],
                    'uB': features['uB'][i],
                    'uF': features['uF'][i] # Concatenated uI, uP
                }
    print(f"Features extracted for {len(all_features)} patients.")
    return all_features