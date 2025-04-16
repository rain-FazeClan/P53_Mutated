# datasets.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import nibabel as nib
from config import (IMG_INPUT_CHANNELS, GEOM_N_POINTS, BRAIN_NODE_N_REGIONS,
                    BRAIN_NODE_VOXEL_SAMPLE_DIM, BRAIN_EDGE_N_TRACTS,
                    BRAIN_EDGE_VOXEL_SAMPLE_DIM_ANAT, BRAIN_EDGE_VOXEL_SAMPLE_DIM_FA,
                    NODE_ATLAS_FILE, TRACT_ATLAS_FILE)
from preprocessing import generate_point_cloud, get_atlas_voxels, sample_voxels

class BrainNetworkDataset(Dataset):
    """Dataset for self-supervised brain network generation."""
    def __init__(self, patient_ids, anat_mri_dir, fa_map_dir, node_atlas_path, tract_atlas_path, mode='node'):
        self.patient_ids = patient_ids
        self.anat_mri_dir = anat_mri_dir # Assuming T1 preprocessed files are here
        self.fa_map_dir = fa_map_dir
        self.mode = mode # 'node' or 'edge'

        # Load atlases once
        self.node_atlas_data = nib.load(node_atlas_path).get_fdata() if node_atlas_path else None
        self.tract_atlas_data = nib.load(tract_atlas_path).get_fdata() if tract_atlas_path else None
        if self.node_atlas_data is None and mode == 'node':
            raise ValueError("Node atlas required for node mode")
        if self.tract_atlas_data is None and mode == 'edge':
             raise ValueError("Tract atlas required for edge mode")

        self.node_regions = range(1, BRAIN_NODE_N_REGIONS + 1) if mode == 'node' else []
        self.edge_tracts = range(1, BRAIN_EDGE_N_TRACTS + 1) if mode == 'edge' else [] # Assuming tract labels are 1 to N

    def __len__(self):
        if self.mode == 'node':
            return len(self.patient_ids) * BRAIN_NODE_N_REGIONS
        elif self.mode == 'edge':
            return len(self.patient_ids) * BRAIN_EDGE_N_TRACTS
        return 0

    def __getitem__(self, idx):
        if self.mode == 'node':
            patient_idx = idx // BRAIN_NODE_N_REGIONS
            region_label = self.node_regions[idx % BRAIN_NODE_N_REGIONS]
            patient_id = self.patient_ids[patient_idx]

            # Load anatomical MRI (assuming T1 for node attributes)
            anat_mri_path = os.path.join(self.anat_mri_dir, f"{patient_id}_T1_preprocessed.nii.gz")
            anat_mri_data = nib.load(anat_mri_path).get_fdata()

            voxels = get_atlas_voxels(anat_mri_data, self.node_atlas_data, region_label)
            sampled_voxels = sample_voxels(voxels, BRAIN_NODE_VOXEL_SAMPLE_DIM)
            return torch.FloatTensor(sampled_voxels)

        elif self.mode == 'edge':
            patient_idx = idx // BRAIN_EDGE_N_TRACTS
            tract_label = self.edge_tracts[idx % BRAIN_EDGE_N_TRACTS]
            patient_id = self.patient_ids[patient_idx]

            # Load anatomical MRI
            anat_mri_path = os.path.join(self.anat_mri_dir, f"{patient_id}_T1_preprocessed.nii.gz")
            anat_mri_data = nib.load(anat_mri_path).get_fdata()
            # Load FA map
            fa_path = os.path.join(self.fa_map_dir, f"{patient_id}_FA.nii.gz")
            fa_data = nib.load(fa_path).get_fdata()

            anat_voxels = get_atlas_voxels(anat_mri_data, self.tract_atlas_data, tract_label)
            fa_voxels = get_atlas_voxels(fa_data, self.tract_atlas_data, tract_label)

            sampled_anat = sample_voxels(anat_voxels, BRAIN_EDGE_VOXEL_SAMPLE_DIM_ANAT)
            sampled_fa = sample_voxels(fa_voxels, BRAIN_EDGE_VOXEL_SAMPLE_DIM_FA)

            return torch.FloatTensor(sampled_anat), torch.FloatTensor(sampled_fa)

class GliomaDataset(Dataset):
    """Dataset for multi-modal contrastive learning and classification."""
    def __init__(self, patient_ids, mri_dir, mask_dir, labels_df, brain_networks=None):
        self.patient_ids = patient_ids
        self.mri_dir = mri_dir
        self.mask_dir = mask_dir
        self.labels_df = labels_df.set_index('patient_id')
        self.brain_networks = brain_networks # Dict: {patient_id: {'nodes': tensor, 'edges': tensor, 'edge_index': tensor}}

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # --- Load MRI Modalities ---
        # Assuming filenames like patient_id_t1.nii.gz, patient_id_t1ce.nii.gz, etc.
        # Load preprocessed data
        mri_modalities = []
        for mod in ['t1', 't1ce', 't2', 'flair']: # Or match actual preprocessed filenames
             path = os.path.join(self.mri_dir, f"{patient_id}_{mod}_preprocessed.nii.gz")
             mri_data = nib.load(path).get_fdata()
             # Apply tumor mask for focal image data? Paper implies yes.
             # mask_path = os.path.join(self.mask_dir, f"{patient_id}_mask.nii.gz")
             # mask_data = nib.load(mask_path).get_fdata()
             # mri_data = mri_data * mask_data # Or crop to bounding box
             mri_modalities.append(mri_data)

        # Stack modalities -> (Channels, D, H, W)
        img_data = np.stack(mri_modalities, axis=0)
        img_tensor = torch.FloatTensor(img_data)

        # --- Load Tumor Mask & Generate Point Cloud ---
        mask_path = os.path.join(self.mask_dir, f"{patient_id}_mask.nii.gz")
        mask_data = nib.load(mask_path).get_fdata()
        points_data = generate_point_cloud(mask_data, num_points=GEOM_N_POINTS)
        points_tensor = torch.FloatTensor(points_data)

        # --- Load Pre-generated Brain Network ---
        # This assumes brain networks (node/edge attributes) were generated separately
        # and stored, possibly as PyG Data objects.
        if self.brain_networks and patient_id in self.brain_networks:
             network_data = self.brain_networks[patient_id] # Should be a PyG Data object or similar dict
        else:
             # Create placeholder if generation failed or not done yet
             print(f"Warning: Brain network not found for {patient_id}. Using placeholder.")
             network_data = {
                 'x_n': torch.zeros(BRAIN_NODE_N_REGIONS, BRAIN_NODE_FEATURE_DIM), # Node features bN
                 'edge_attr': torch.zeros(BRAIN_EDGE_N_TRACTS, BRAIN_EDGE_FEATURE_DIM), # Edge features bE
                 'edge_index': torch.zeros(2, BRAIN_EDGE_N_TRACTS, dtype=torch.long) # Connectivity
             }
             # Note: Need actual edge_index based on the atlas connectivity

        # --- Load Label ---
        label = self.labels_df.loc[patient_id, 'IDH_status'] # Assuming column name
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'id': patient_id,
            'image': img_tensor,          # xI
            'points': points_tensor,      # xP
            'network': network_data,      # xB (contains bN, bE, edge_index)
            'label': label_tensor
        }

def get_data_loaders(all_patient_ids, labels_df, brain_networks, config):
    # Split patient IDs
    np.random.shuffle(all_patient_ids)
    n_test = int(len(all_patient_ids) * config.TEST_SPLIT_RATIO)
    n_train_val = len(all_patient_ids) - n_test
    n_val = int(n_train_val * config.VALIDATION_SPLIT_RATIO) # Validation from train
    n_train = n_train_val - n_val

    train_ids = all_patient_ids[:n_train]
    val_ids = all_patient_ids[n_train:n_train+n_val]
    test_ids = all_patient_ids[n_train+n_val:]

    print(f"Train size: {len(train_ids)}, Val size: {len(val_ids)}, Test size: {len(test_ids)}")

    train_dataset = GliomaDataset(train_ids, config.MRI_DATA_DIR, config.TUMOR_MASK_DIR, labels_df, brain_networks)
    val_dataset = GliomaDataset(val_ids, config.MRI_DATA_DIR, config.TUMOR_MASK_DIR, labels_df, brain_networks)
    test_dataset = GliomaDataset(test_ids, config.MRI_DATA_DIR, config.TUMOR_MASK_DIR, labels_df, brain_networks)

    # DataLoader for contrastive learning (usually requires shuffling)
    train_loader_contrastive = DataLoader(train_dataset, batch_size=config.CONTRASTIVE_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader_contrastive = DataLoader(val_dataset, batch_size=config.CONTRASTIVE_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # For population graph GNN, we often use the full graph structure at once,
    # but can still iterate through nodes for loss calculation if needed.
    # The datasets can be used to extract features first.
    # Loader for feature extraction (no shuffle needed)
    full_dataset = GliomaDataset(all_patient_ids, config.MRI_DATA_DIR, config.TUMOR_MASK_DIR, labels_df, brain_networks)
    feature_loader = DataLoader(full_dataset, batch_size=config.CONTRASTIVE_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)


    # For classifier training, we might need loaders depending on the semi-supervised strategy
    # Usually train on train_ids, evaluate on val_ids/test_ids within the population graph context
    # So, the primary loaders might be train_loader_contrastive, val_loader_contrastive, feature_loader

    return train_loader_contrastive, val_loader_contrastive, feature_loader, train_ids, val_ids, test_ids