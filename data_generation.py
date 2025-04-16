# data_generation.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import (DEVICE, BRAIN_NET_NODE_AE_PARAMS, BRAIN_NET_EDGE_ENCODER_PARAMS_ANAT,
                    BRAIN_NET_EDGE_ENCODER_PARAMS_FA, BRAIN_NET_EDGE_PROJECTION_DIM,
                    BRAIN_NET_CONTRASTIVE_TEMP_EDGE, BRAIN_NET_LR_NODE, BRAIN_NET_WD_NODE,
                    BRAIN_NET_EPOCHS_NODE, BRAIN_NET_LR_EDGE, BRAIN_NET_WD_EDGE,
                    BRAIN_NET_EPOCHS_EDGE, BRAIN_NET_BATCH_SIZE, BRAIN_NET_NODE_MODEL_PATH,
                    BRAIN_NET_EDGE_MODEL_PATH)
from utils import build_mlp, cosine_similarity
from datasets import BrainNetworkDataset
from torch.utils.data import DataLoader

# --- Node Attribute Autoencoder ---
class NodeAutoencoder(nn.Module):
    def __init__(self, encoder_dims, decoder_dims):
        super().__init__()
        self.encoder = build_mlp(encoder_dims)
        self.decoder = build_mlp(decoder_dims)

    def forward(self, x):
        bN = self.encoder(x)
        x_recon = self.decoder(bN)
        return x_recon, bN # Return reconstruction and bottleneck features (node attributes)

# --- Edge Attribute Contrastive Model ---
class EdgeContrastiveNet(nn.Module):
    def __init__(self, anat_encoder_dims, fa_encoder_dims, projection_dim):
        super().__init__()
        self.anat_encoder = build_mlp(anat_encoder_dims) # f_E
        self.fa_encoder = build_mlp(fa_encoder_dims)   # f_E'
        # Projection heads gE, gE'
        self.anat_projector = build_mlp([anat_encoder_dims[-1], projection_dim // 2, projection_dim])
        self.fa_projector = build_mlp([fa_encoder_dims[-1], projection_dim // 2, projection_dim])

    def forward(self, x_anat, x_fa):
        bE = self.anat_encoder(x_anat) # Edge attributes from anatomical
        bE_prime = self.fa_encoder(x_fa)

        zE = self.anat_projector(bE) # Projected anatomical features
        zE_prime = self.fa_projector(bE_prime) # Projected FA features

        return bE, zE, zE_prime # Return anat edge attr, projected anat, projected fa

def contrastive_loss_edge(zE, zE_prime, temperature):
    """ Eq 1: Contrastive loss for edge attributes """
    batch_size = zE.shape[0]
    # Calculate similarity for positive pairs (same patient, same tract)
    sim_positive = cosine_similarity(zE, zE_prime, dim=1) / temperature

    # Calculate similarity for negative pairs (different patients or different tracts within batch)
    # This requires careful indexing if batch contains multiple tracts per patient
    # Simplest: treat all other pairs in batch as negative
    z_all = torch.cat([zE, zE_prime], dim=0) # 2N x D
    sim_all = torch.mm(z_all, z_all.t().contiguous()) / temperature # 2N x 2N

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=zE.device)
    sim_all.masked_fill_(mask, -float('inf')) # Or very small number

    # Numerator: Positive pair similarities
    pos = torch.cat([sim_positive, sim_positive], dim=0) # 2N

    # Denominator: Sum over all similarities for each anchor (row sums)
    log_sum_exp = torch.logsumexp(sim_all, dim=1) # 2N

    loss = -torch.mean(pos - log_sum_exp)
    return loss

def train_node_ae(config, train_patient_ids_for_brainnet):
    """Train the Node Autoencoder."""
    print("Training Node Autoencoder...")
    model = NodeAutoencoder(config.BRAIN_NET_NODE_AE_PARAMS['encoder_dims'],
                            config.BRAIN_NET_NODE_AE_PARAMS['decoder_dims']).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.BRAIN_NET_LR_NODE, weight_decay=config.BRAIN_NET_WD_NODE)
    criterion = nn.MSELoss()

    dataset = BrainNetworkDataset(train_patient_ids_for_brainnet, config.MRI_DATA_DIR, None, # No FA needed
                                  config.NODE_ATLAS_FILE, None, mode='node')
    loader = DataLoader(dataset, batch_size=config.BRAIN_NET_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    model.train()
    for epoch in range(config.BRAIN_NET_EPOCHS_NODE):
        epoch_loss = 0.0
        for batch_voxels in loader:
            batch_voxels = batch_voxels.to(config.DEVICE)
            optimizer.zero_grad()
            recon_voxels, _ = model(batch_voxels)
            loss = criterion(recon_voxels, batch_voxels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{config.BRAIN_NET_EPOCHS_NODE}], Node AE Loss: {epoch_loss/len(loader):.4f}")
        # Add scheduler if needed

    torch.save(model.state_dict(), config.BRAIN_NET_NODE_MODEL_PATH)
    print(f"Node AE model saved to {config.BRAIN_NET_NODE_MODEL_PATH}")
    return model

def train_edge_contrastive(config, train_patient_ids_for_brainnet):
    """Train the Edge Contrastive Network."""
    print("Training Edge Contrastive Network...")
    model = EdgeContrastiveNet(config.BRAIN_NET_EDGE_ENCODER_PARAMS_ANAT['dims'],
                               config.BRAIN_NET_EDGE_ENCODER_PARAMS_FA['dims'],
                               config.BRAIN_NET_EDGE_PROJECTION_DIM).to(config.DEVICE)
    # Paper uses SGD, Adam might also work
    optimizer = optim.SGD(model.parameters(), lr=config.BRAIN_NET_LR_EDGE, weight_decay=config.BRAIN_NET_WD_EDGE, momentum=0.9)

    dataset = BrainNetworkDataset(train_patient_ids_for_brainnet, config.MRI_DATA_DIR, config.FA_DATA_DIR,
                                  None, config.TRACT_ATLAS_FILE, mode='edge')
    loader = DataLoader(dataset, batch_size=config.BRAIN_NET_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    model.train()
    for epoch in range(config.BRAIN_NET_EPOCHS_EDGE):
        epoch_loss = 0.0
        for anat_voxels, fa_voxels in loader:
            anat_voxels = anat_voxels.to(config.DEVICE)
            fa_voxels = fa_voxels.to(config.DEVICE)
            optimizer.zero_grad()
            _, zE, zE_prime = model(anat_voxels, fa_voxels)
            loss = contrastive_loss_edge(zE, zE_prime, config.BRAIN_NET_CONTRASTIVE_TEMP_EDGE)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{config.BRAIN_NET_EPOCHS_EDGE}], Edge Contrastive Loss: {epoch_loss/len(loader):.4f}")
        # Add scheduler if needed

    torch.save(model.state_dict(), config.BRAIN_NET_EDGE_MODEL_PATH)
    print(f"Edge Contrastive model saved to {config.BRAIN_NET_EDGE_MODEL_PATH}")
    return model

def generate_brain_network_features(patient_ids, node_ae_model, edge_contrastive_model, config):
    """Generate bN and bE features for a list of patients using trained models."""
    print("Generating brain network features (bN, bE)...")
    node_ae_model.eval()
    edge_contrastive_model.eval() # We only need the anat_encoder part for inference

    brain_networks = {}
    node_atlas_data = nib.load(config.NODE_ATLAS_FILE).get_fdata()
    tract_atlas_data = nib.load(config.TRACT_ATLAS_FILE).get_fdata()
    node_regions = range(1, config.BRAIN_NODE_N_REGIONS + 1)
    edge_tracts = range(1, config.BRAIN_EDGE_N_TRACTS + 1)

    with torch.no_grad():
        for patient_id in patient_ids:
            print(f" Processing patient: {patient_id}")
            # --- Generate Node Attributes (bN) ---
            node_attrs = []
            anat_mri_path = os.path.join(config.MRI_DATA_DIR, f"{patient_id}_T1_preprocessed.nii.gz")
            anat_mri_data = nib.load(anat_mri_path).get_fdata()

            for region_label in node_regions:
                voxels = get_atlas_voxels(anat_mri_data, node_atlas_data, region_label)
                sampled_voxels = sample_voxels(voxels, config.BRAIN_NODE_VOXEL_SAMPLE_DIM)
                voxel_tensor = torch.FloatTensor(sampled_voxels).unsqueeze(0).to(config.DEVICE)
                _, bN = node_ae_model(voxel_tensor)
                node_attrs.append(bN.squeeze(0).cpu())
            patient_bN = torch.stack(node_attrs) # [N_nodes, NodeFeatDim]

            # --- Generate Edge Attributes (bE) ---
            edge_attrs = []
            for tract_label in edge_tracts:
                anat_voxels = get_atlas_voxels(anat_mri_data, tract_atlas_data, tract_label)
                sampled_anat = sample_voxels(anat_voxels, config.BRAIN_EDGE_VOXEL_SAMPLE_DIM_ANAT)
                anat_tensor = torch.FloatTensor(sampled_anat).unsqueeze(0).to(config.DEVICE)
                # Use only the anatomical encoder for inference
                bE = edge_contrastive_model.anat_encoder(anat_tensor)
                edge_attrs.append(bE.squeeze(0).cpu())
            patient_bE = torch.stack(edge_attrs) # [N_edges, EdgeFeatDim]

            # --- Get Edge Index (Connectivity) ---
            # This needs to be derived from the tract atlas definition
            # Placeholder: Assume edge_index is fixed for all patients based on atlas
            edge_index = get_atlas_connectivity(config.TRACT_ATLAS_FILE, config.BRAIN_NODE_N_REGIONS) # Implement this function

            brain_networks[patient_id] = {
                'x_n': patient_bN,      # Node features
                'edge_attr': patient_bE,  # Edge features
                'edge_index': edge_index # Connectivity
            }
    print("Brain network feature generation complete.")
    return brain_networks

def get_atlas_connectivity(tract_atlas_path, num_nodes):
    """ Reads tract atlas or definition file to determine node pairs for each edge/tract. """
    print("Determining atlas connectivity...")
    # Placeholder: Implement logic to map tract labels to connected node pairs (0-indexed)
    # Example: Tract 1 connects node 0 and node 5 -> [0, 5]
    #          Tract 2 connects node 0 and node 8 -> [0, 8] ...
    # Result should be a tensor of shape [2, num_edges]
    num_edges = BRAIN_EDGE_N_TRACTS
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    print("Atlas connectivity determined (stub).")
    return edge_index