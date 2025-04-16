# main.py
import os
import pandas as pd
import numpy as np
import torch
import argparse

import config as cfg
from utils import set_seed, calculate_metrics
from datasets import get_data_loaders, BrainNetworkDataset # Import necessary dataset class
from data_generation import (train_node_ae, train_edge_contrastive,
                             generate_brain_network_features, NodeAutoencoder, EdgeContrastiveNet)
from contrastive_learning import (MultiModalContrastiveModel, train_contrastive_model,
                                  extract_all_features)
from population_graph import build_population_graph
from classifier import PopulationGNNClassifier, train_classifier, evaluate_classifier

def main(args):
    set_seed(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # --- Load Patient IDs and Labels ---
    labels_df = pd.read_csv(cfg.LABEL_FILE)
    all_patient_ids = labels_df['patient_id'].tolist() # Assuming 'patient_id' column

    # --- Stage 1: Brain Network Generation (Optional - can be skipped if pre-computed) ---
    brain_networks = None
    if args.run_brain_net_gen:
        # Need a subset of patients with FA data for edge training
        # Placeholder: Use first 20 patients assuming they have FA data
        brain_net_train_ids = all_patient_ids[:20] # Adjust as needed

        # Train Node AE
        node_ae_model = train_node_ae(cfg, brain_net_train_ids)
        # Train Edge Contrastive
        edge_contrastive_model = train_edge_contrastive(cfg, brain_net_train_ids)

        # Generate features for ALL patients
        brain_networks = generate_brain_network_features(all_patient_ids, node_ae_model, edge_contrastive_model, cfg)
        # Optionally save brain_networks dict to disk
    elif args.load_brain_nets:
        # Load pre-computed brain_networks dict from disk
        print("Loading pre-computed brain networks...")
        # brain_networks = load_from_disk(...) # Implement loading logic
        pass # Skip if not implemented

    # --- Prepare Data Loaders for Contrastive Learning & Feature Extraction ---
    # Split data *after* potential brain net generation
    train_loader, val_loader, feature_loader, train_ids, val_ids, test_ids = get_data_loaders(
        all_patient_ids, labels_df, brain_networks, cfg
    )

    # --- Stage 2: Multi-Modal Contrastive Learning ---
    contrastive_model = MultiModalContrastiveModel(cfg).to(cfg.DEVICE)
    if args.run_contrastive_train:
        contrastive_model = train_contrastive_model(contrastive_model, train_loader, val_loader, cfg)
    else:
        print(f"Loading pre-trained contrastive model from {cfg.CONTRASTIVE_MODEL_PATH}")
        contrastive_model.load_state_dict(torch.load(cfg.CONTRASTIVE_MODEL_PATH, map_location=cfg.DEVICE))

    # Extract features for all patients using the trained contrastive model
    all_extracted_features = extract_all_features(contrastive_model, feature_loader, cfg)

    # --- Stage 3: Population Graph Construction & Classification ---
    graph_data, id_to_idx = build_population_graph(all_extracted_features, all_patient_ids, labels_df, cfg)

    # Create masks for semi-supervised training/evaluation
    num_patients = graph_data.num_nodes
    train_mask = torch.zeros(num_patients, dtype=torch.bool)
    val_mask = torch.zeros(num_patients, dtype=torch.bool)
    test_mask = torch.zeros(num_patients, dtype=torch.bool)
    train_idx = [id_to_idx[pid] for pid in train_ids if pid in id_to_idx]
    val_idx = [id_to_idx[pid] for pid in val_ids if pid in id_to_idx]
    test_idx = [id_to_idx[pid] for pid in test_ids if pid in id_to_idx]
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    graph_data.train_mask = train_mask
    graph_data.val_mask = val_mask
    graph_data.test_mask = test_mask


    # Initialize Classifier
    input_node_dim = graph_data.num_node_features
    classifier_model = PopulationGNNClassifier(input_node_dim, cfg.CLASSIFIER_GAT_DIMS.copy(),
                                              cfg.CLASSIFIER_GAT_HEADS, cfg.CLASSIFIER_N_CLASSES).to(cfg.DEVICE)

    if args.run_classifier_train:
        classifier_model = train_classifier(classifier_model, graph_data, train_mask, val_mask, cfg)
    else:
        print(f"Loading pre-trained classifier model from {cfg.CLASSIFIER_MODEL_PATH}")
        classifier_model.load_state_dict(torch.load(cfg.CLASSIFIER_MODEL_PATH, map_location=cfg.DEVICE))

    # Evaluate final model
    evaluate_classifier(classifier_model, graph_data, test_mask, cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Modal Glioma Genotype Prediction")
    parser.add_argument('--run_brain_net_gen', action='store_true', help="Run self-supervised training for brain networks.")
    parser.add_argument('--load_brain_nets', action='store_true', help="Load pre-generated brain network features.")
    parser.add_argument('--run_contrastive_train', action='store_true', help="Run multi-modal contrastive learning training.")
    parser.add_argument('--run_classifier_train', action='store_true', help="Run population graph classifier training.")
    # Add more arguments as needed (e.g., specify config file)

    args = parser.parse_args()

    # Basic workflow logic based on flags
    if not args.run_brain_net_gen and not args.load_brain_nets:
        print("Warning: No brain networks will be generated or loaded. Brain network features will be placeholders.")
    if not args.run_contrastive_train:
        if not os.path.exists(cfg.CONTRASTIVE_MODEL_PATH):
             raise FileNotFoundError(f"Contrastive model path not found: {cfg.CONTRASTIVE_MODEL_PATH}. Use --run_contrastive_train.")
    if not args.run_classifier_train:
        if not os.path.exists(cfg.CLASSIFIER_MODEL_PATH):
             raise FileNotFoundError(f"Classifier model path not found: {cfg.CLASSIFIER_MODEL_PATH}. Use --run_classifier_train.")

    main(args)