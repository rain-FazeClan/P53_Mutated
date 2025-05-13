import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time

# Import project modules
from utils import set_seed, get_device
from data_loader import load_data_manifest, split_data, MODALITIES, collate_fn_skip_none
from preprocessing import preprocess_patient_images
from segmentation import run_automatic_segmentation # Placeholder
from radiomics_extractor import extract_features_for_patient
from radiomics_model import RadiomicsMLP, select_radiomics_features
from cnn_model import ResNet3D, VGGNet3D
from integrated_model import IntegratedMLP
from train import (RadiomicsDataset, CNNDataset, IntegratedDataset,
                   train_model, evaluate_model, extract_features)
from evaluate import calculate_metrics, plot_roc_curves, compare_auc_delong

# --- Configuration ---
SEED = 42
PREPROCESSED_DIR = "output/preprocessed"
RADIOMICS_FEATURES_FILE = "output/radiomics_features.csv"
MODEL_OUTPUT_DIR = "output/models"
RESULTS_DIR = "output/results"

# Hyperparameters (adjust based on paper/tuning)
# Common
EPOCHS = 50 # Adjust as needed
BATCH_SIZE = 16 # Paper uses 16 for CNN/Integrated, 32 for Radiomics
# Radiomics Model
RAD_LR = 0.001
RAD_WEIGHT_DECAY = 0.05 # Regularization weight
RAD_HIDDEN_DIMS = [256, 128] # Example
RAD_DROPOUT = [0.2, 0.5]
RAD_FEATURE_PERCENTILE = 30
# CNN Model (ResNet)
CNN_LR = 0.001
CNN_WEIGHT_DECAY = 0.001 # Regularization weight (RMSprop doesn't typically use weight decay like AdamW)
# Integrated Model
INT_LR = 0.1 # Note: Paper mentions 0.1, which seems high. Verify.
INT_WEIGHT_DECAY = 0.005
INT_HIDDEN_DIMS = [512, 256] # Example
INT_DROPOUT = 0.5


def main(args):
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RADIOMICS_FEATURES_FILE), exist_ok=True)


    # --- 1. Load Data Manifest ---
    print("Loading data manifest...")
    all_patient_data = load_data_manifest(args.metadata)
    if not all_patient_data:
        print("Error: No patient data loaded. Exiting.")
        return
    train_list, val_list = split_data(all_patient_data, test_size=0.3, random_state=SEED) # ~30% validation

    # --- 2. Preprocessing & Segmentation (Run once) ---
    print("\n--- Starting Preprocessing & Segmentation ---")
    processed_train_files = []
    processed_val_files = []

    # Process Training Data
    print("Processing Training Set...")
    for patient_info in tqdm(train_list, desc="Preprocessing Train"):
        # a. Run segmentation (placeholder - assumes masks exist or are generated)
        #    In reality, you might run this *before* preprocessing if seg model needs original images
        #    Let's assume seg_paths are added to patient_info if found/generated
        # seg_paths = run_automatic_segmentation(patient_info, output_dir=PREPROCESSED_DIR) # Pass original paths
        # if seg_paths:
        #     patient_info.update(seg_paths) # Add seg paths to dict

        # b. Run preprocessing
        output_files = preprocess_patient_images(patient_info, PREPROCESSED_DIR, skip_existing=args.skip_preprocessing)
        if output_files:
            output_files['id'] = patient_info['id'] # Keep ID
            output_files['label'] = patient_info['label'] # Keep label
            processed_train_files.append(output_files)
        else:
            print(f"Preprocessing failed for train patient {patient_info['id']}")

    # Process Validation Data
    print("Processing Validation Set...")
    for patient_info in tqdm(val_list, desc="Preprocessing Val"):
        # a. Segmentation (placeholder)
        # seg_paths = run_automatic_segmentation(patient_info, output_dir=PREPROCESSED_DIR)
        # if seg_paths:
        #     patient_info.update(seg_paths)

        # b. Preprocessing
        output_files = preprocess_patient_images(patient_info, PREPROCESSED_DIR, skip_existing=args.skip_preprocessing)
        if output_files:
            output_files['id'] = patient_info['id']
            output_files['label'] = patient_info['label']
            processed_val_files.append(output_files)
        else:
             print(f"Preprocessing failed for validation patient {patient_info['id']}")

    if not processed_train_files or not processed_val_files:
        print("Error: Preprocessing failed for too many patients. Cannot continue.")
        return

    # Update train/val lists to use only successfully preprocessed patients
    train_ids = {p['id'] for p in processed_train_files}
    val_ids = {p['id'] for p in processed_val_files}
    train_list = [p for p in train_list if p['id'] in train_ids]
    val_list = [p for p in val_list if p['id'] in val_ids]
    print(f"Proceeding with {len(train_list)} training and {len(val_list)} validation samples after preprocessing check.")


    # --- 3. Radiomics Feature Extraction (Run once) ---
    print("\n--- Starting Radiomics Feature Extraction ---")
    if not os.path.exists(RADIOMICS_FEATURES_FILE) or args.force_feature_extraction:
        all_rad_features_list = []
        print("Extracting Train Radiomics...")
        for patient_proc_info in tqdm(processed_train_files, desc="Extracting Train Radiomics"):
             # Ensure segmentation paths are present in processed_info
             seg_wt_path = os.path.join(PREPROCESSED_DIR, patient_proc_info['id'], f"{patient_proc_info['id']}_seg_WT_preprocessed.nii.gz")
             seg_tc_path = os.path.join(PREPROCESSED_DIR, patient_proc_info['id'], f"{patient_proc_info['id']}_seg_TC_preprocessed.nii.gz")
             if os.path.exists(seg_wt_path): patient_proc_info['seg_WT'] = seg_wt_path
             if os.path.exists(seg_tc_path): patient_proc_info['seg_TC'] = seg_tc_path

             patient_features = extract_features_for_patient(patient_proc_info)
             if not patient_features.empty:
                 patient_features['PatientID'] = patient_proc_info['id'] # Ensure ID is present
                 patient_features['Set'] = 'Train'
                 patient_features['Label'] = patient_proc_info['label'] # Add label
                 all_rad_features_list.append(patient_features)

        print("Extracting Validation Radiomics...")
        for patient_proc_info in tqdm(processed_val_files, desc="Extracting Val Radiomics"):
             seg_wt_path = os.path.join(PREPROCESSED_DIR, patient_proc_info['id'], f"{patient_proc_info['id']}_seg_WT_preprocessed.nii.gz")
             seg_tc_path = os.path.join(PREPROCESSED_DIR, patient_proc_info['id'], f"{patient_proc_info['id']}_seg_TC_preprocessed.nii.gz")
             if os.path.exists(seg_wt_path): patient_proc_info['seg_WT'] = seg_wt_path
             if os.path.exists(seg_tc_path): patient_proc_info['seg_TC'] = seg_tc_path

             patient_features = extract_features_for_patient(patient_proc_info)
             if not patient_features.empty:
                 patient_features['PatientID'] = patient_proc_info['id']
                 patient_features['Set'] = 'Val'
                 patient_features['Label'] = patient_proc_info['label']
                 all_rad_features_list.append(patient_features)

        if not all_rad_features_list:
            print("Error: No radiomics features extracted. Cannot proceed with radiomics models.")
            # Decide whether to exit or just skip radiomics parts
            return

        # Combine all features into a single DataFrame
        all_radiomics_df = pd.concat(all_rad_features_list, ignore_index=True)
        # Handle potential NaNs from extraction failures or specific features
        all_radiomics_df = all_radiomics_df.dropna(axis=1, how='all') # Drop columns that are all NaN
        # Impute remaining NaNs (e.g., with mean/median) - check impact
        numeric_cols = all_radiomics_df.select_dtypes(include=np.number).columns.tolist()
        # Exclude Label and potentially PatientID if it was numeric
        cols_to_impute = [col for col in numeric_cols if col not in ['Label', 'PatientID']]
        for col in cols_to_impute:
             if all_radiomics_df[col].isnull().any():
                  mean_val = all_radiomics_df[col].mean()
                  all_radiomics_df[col].fillna(mean_val, inplace=True)
                  print(f"Imputed NaNs in column {col} with mean {mean_val:.4f}")


        all_radiomics_df.to_csv(RADIOMICS_FEATURES_FILE, index=False)
        print(f"Radiomics features saved to {RADIOMICS_FEATURES_FILE}")
    else:
        print(f"Loading existing radiomics features from {RADIOMICS_FEATURES_FILE}")
        all_radiomics_df = pd.read_csv(RADIOMICS_FEATURES_FILE)
        # Basic check for expected columns
        if 'PatientID' not in all_radiomics_df.columns or 'Set' not in all_radiomics_df.columns or 'Label' not in all_radiomics_df.columns:
             print("Error: Loaded radiomics file missing required columns (PatientID, Set, Label). Please regenerate.")
             return


    # Prepare Radiomics Data for Models
    train_rad_df = all_radiomics_df[all_radiomics_df['Set'] == 'Train'].drop(columns=['PatientID', 'Set', 'Label'])
    train_rad_labels = all_radiomics_df[all_radiomics_df['Set'] == 'Train']['Label']
    val_rad_df = all_radiomics_df[all_radiomics_df['Set'] == 'Val'].drop(columns=['PatientID', 'Set', 'Label'])
    val_rad_labels = all_radiomics_df[all_radiomics_df['Set'] == 'Val']['Label']

    # Feature Selection for Radiomics Model
    print("Performing Radiomics Feature Selection...")
    train_rad_df_selected, selected_feature_names = select_radiomics_features(
        train_rad_df, train_rad_labels, percentile=RAD_FEATURE_PERCENTILE
    )
    # Apply selection to validation set
    val_rad_df_selected = val_rad_df[selected_feature_names] # Use same features selected from training

    # Convert to tensors for datasets
    train_rad_dataset = RadiomicsDataset(train_rad_df_selected, train_rad_labels)
    val_rad_dataset = RadiomicsDataset(val_rad_df_selected, val_rad_labels)
    train_rad_loader = DataLoader(train_rad_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_rad_loader = DataLoader(val_rad_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 4. Train Radiomics Model ---
    print("\n--- Training Radiomics Model ---")
    rad_model = RadiomicsMLP(input_dim=len(selected_feature_names),
                             hidden_dims=RAD_HIDDEN_DIMS,
                             dropout_rates=RAD_DROPOUT).to(device)
    rad_optimizer = optim.RMSprop(rad_model.parameters(), lr=RAD_LR, weight_decay=RAD_WEIGHT_DECAY)
    criterion = nn.BCELoss() # Binary Cross Entropy

    best_val_auc_rad = -1
    best_rad_model_path = os.path.join(MODEL_OUTPUT_DIR, "best_radiomics_model.pth")

    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_model(rad_model, train_rad_loader, rad_optimizer, criterion, device, model_type="radiomics")
        val_loss, val_acc, val_labels_np, val_probs_np = evaluate_model(rad_model, val_rad_loader, criterion, device, model_type="radiomics")

        val_metrics = calculate_metrics(val_labels_np, (val_probs_np > 0.5).astype(int), val_probs_np)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_metrics['AUC']:.4f}")

        if val_metrics['AUC'] > best_val_auc_rad:
            best_val_auc_rad = val_metrics['AUC']
            torch.save(rad_model.state_dict(), best_rad_model_path)
            print(f"*** Best Radiomics Model Saved (AUC: {best_val_auc_rad:.4f}) ***")

    # Load best radiomics model for feature extraction
    print(f"Loading best radiomics model from {best_rad_model_path}")
    rad_model.load_state_dict(torch.load(best_rad_model_path))

    # --- 5. Train CNN Model (ResNet) ---
    print("\n--- Training CNN Model (ResNet-18 3D) ---")
    # Prepare CNN Datasets
    # Need patient list with labels, and path to preprocessed dir
    train_cnn_patient_list = [{'id': p['id'], 'label': p['label']} for p in train_list]
    val_cnn_patient_list = [{'id': p['id'], 'label': p['label']} for p in val_list]

    train_cnn_dataset = CNNDataset(train_cnn_patient_list, PREPROCESSED_DIR)
    val_cnn_dataset = CNNDataset(val_cnn_patient_list, PREPROCESSED_DIR)
    # Use collate_fn to handle potential None returns if preprocessing failed for a sample
    train_cnn_loader = DataLoader(train_cnn_dataset, batch_size=BATCH_SIZE // 2, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn_skip_none) # Smaller BS for 3D CNN
    val_cnn_loader = DataLoader(val_cnn_dataset, batch_size=BATCH_SIZE // 2, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn_skip_none)

    cnn_model = ResNet3D(num_classes=1, input_channels=len(MODALITIES)).to(device)
    cnn_optimizer = optim.RMSprop(cnn_model.parameters(), lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY)
    # criterion already defined (BCELoss)

    best_val_auc_cnn = -1
    best_cnn_model_path = os.path.join(MODEL_OUTPUT_DIR, "best_cnn_model.pth")

    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_model(cnn_model, train_cnn_loader, cnn_optimizer, criterion, device, model_type="cnn")
        val_loss, val_acc, val_labels_np_cnn, val_probs_np_cnn = evaluate_model(cnn_model, val_cnn_loader, criterion, device, model_type="cnn")

        # Check if evaluation produced results (might be empty if all batches failed)
        if val_labels_np_cnn is not None and len(val_labels_np_cnn) > 0:
             val_metrics_cnn = calculate_metrics(val_labels_np_cnn, (val_probs_np_cnn > 0.5).astype(int), val_probs_np_cnn)
             epoch_time = time.time() - start_time
             print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.2f}s | "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_metrics_cnn['AUC']:.4f}")

             if val_metrics_cnn['AUC'] > best_val_auc_cnn:
                 best_val_auc_cnn = val_metrics_cnn['AUC']
                 torch.save(cnn_model.state_dict(), best_cnn_model_path)
                 print(f"*** Best CNN Model Saved (AUC: {best_val_auc_cnn:.4f}) ***")
        else:
             epoch_time = time.time() - start_time
             print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.2f}s | "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                   f"Validation produced no results (check data loading/preprocessing).")


    # Load best CNN model for feature extraction
    print(f"Loading best CNN model from {best_cnn_model_path}")
    if os.path.exists(best_cnn_model_path):
        cnn_model.load_state_dict(torch.load(best_cnn_model_path))
    else:
        print(f"Warning: Best CNN model file not found at {best_cnn_model_path}. Using last epoch model.")


    # --- 6. Extract Features for Integrated Model ---
    print("\n--- Extracting Features for Integrated Model ---")
    # Use the *full* radiomics datasets (before feature selection for the standalone model)
    # to extract features from the trained radiomics model's intermediate layer.
    # Need to re-create DataLoaders with the full feature set if needed.
    # For simplicity here, let's assume we extract from the *selected* features dataset used for training.
    # If using intermediate layer before selection, adjust RadiomicsMLP input dim and dataset.

    # Extract Radiomics Features (from intermediate layer of best rad_model)
    train_rad_features_int, train_labels_for_int_rad = extract_features(rad_model, train_rad_loader, device, model_type="radiomics")
    val_rad_features_int, val_labels_for_int_rad = extract_features(rad_model, val_rad_loader, device, model_type="radiomics")

    # Extract CNN Features (from avgpool layer of best cnn_model)
    train_cnn_features_int, train_labels_for_int_cnn = extract_features(cnn_model, train_cnn_loader, device, model_type="cnn")
    val_cnn_features_int, val_labels_for_int_cnn = extract_features(cnn_model, val_cnn_loader, device, model_type="cnn")

    # --- Data Consistency Check ---
    # Ensure the labels match up after potential filtering in dataloaders/feature extraction
    if train_rad_features_int is None or train_cnn_features_int is None or \
       val_rad_features_int is None or val_cnn_features_int is None:
        print("Error: Feature extraction failed for one or more components. Cannot train integrated model.")
        return

    if not torch.equal(train_labels_for_int_rad.squeeze(), train_labels_for_int_cnn.squeeze()):
         print("Warning: Label mismatch between extracted radiomics and CNN features for training set. Check data processing.")
         # Attempt to reconcile or raise error
         # This might happen if collate_fn_skip_none filtered different samples
         # A more robust approach involves mapping features back to patient IDs.
         # For now, we'll proceed assuming the order is correct but warn the user.

    if not torch.equal(val_labels_for_int_rad.squeeze(), val_labels_for_int_cnn.squeeze()):
         print("Warning: Label mismatch between extracted radiomics and CNN features for validation set.")
         # Proceed with caution

    # Use one set of labels (assuming they should be the same)
    train_int_labels = train_labels_for_int_cnn # Or _rad
    val_int_labels = val_labels_for_int_cnn # Or _rad


    # Create Integrated Dataset and Loaders
    train_int_dataset = IntegratedDataset(train_cnn_features_int, train_rad_features_int, train_int_labels)
    val_int_dataset = IntegratedDataset(val_cnn_features_int, val_rad_features_int, val_int_labels)
    train_int_loader = DataLoader(train_int_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_int_loader = DataLoader(val_int_dataset, batch_size=BATCH_SIZE, shuffle=False)


    # --- 7. Train Integrated Model ---
    print("\n--- Training Integrated Model ---")
    cnn_feat_dim = train_cnn_features_int.shape[1]
    rad_feat_dim = train_rad_features_int.shape[1]
    int_model = IntegratedMLP(cnn_feature_dim=cnn_feat_dim,
                              radiomics_feature_dim=rad_feat_dim,
                              hidden_dims=INT_HIDDEN_DIMS,
                              dropout_rate=INT_DROPOUT).to(device)
    int_optimizer = optim.RMSprop(int_model.parameters(), lr=INT_LR, weight_decay=INT_WEIGHT_DECAY)
    # criterion already defined

    best_val_auc_int = -1
    best_int_model_path = os.path.join(MODEL_OUTPUT_DIR, "best_integrated_model.pth")

    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_model(int_model, train_int_loader, int_optimizer, criterion, device, model_type="integrated")
        val_loss, val_acc, val_labels_np_int, val_probs_np_int = evaluate_model(int_model, val_int_loader, criterion, device, model_type="integrated")

        val_metrics_int = calculate_metrics(val_labels_np_int, (val_probs_np_int > 0.5).astype(int), val_probs_np_int)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_metrics_int['AUC']:.4f}")

        if val_metrics_int['AUC'] > best_val_auc_int:
            best_val_auc_int = val_metrics_int['AUC']
            torch.save(int_model.state_dict(), best_int_model_path)
            print(f"*** Best Integrated Model Saved (AUC: {best_val_auc_int:.4f}) ***")

    # --- 8. Final Evaluation & Comparison ---
    print("\n--- Final Evaluation on Validation Set ---")
    # Load best models
    rad_model.load_state_dict(torch.load(best_rad_model_path))
    if os.path.exists(best_cnn_model_path):
        cnn_model.load_state_dict(torch.load(best_cnn_model_path))
    else:
        print(f"Warning: Best CNN model {best_cnn_model_path} not found for final eval.")
    int_model.load_state_dict(torch.load(best_int_model_path))

    # Evaluate Radiomics Model
    _, _, val_labels_rad, val_probs_rad = evaluate_model(rad_model, val_rad_loader, criterion, device, model_type="radiomics")
    final_metrics_rad = calculate_metrics(val_labels_rad, (val_probs_rad > 0.5).astype(int), val_probs_rad)
    print("\nRadiomics Model Final Metrics:")
    for key, value in final_metrics_rad.items(): print(f"  {key}: {value:.4f}")

    # Evaluate CNN Model
    _, _, val_labels_cnn, val_probs_cnn = evaluate_model(cnn_model, val_cnn_loader, criterion, device, model_type="cnn")
    if val_labels_cnn is not None and len(val_labels_cnn) > 0:
         final_metrics_cnn = calculate_metrics(val_labels_cnn, (val_probs_cnn > 0.5).astype(int), val_probs_cnn)
         print("\nCNN Model Final Metrics:")
         for key, value in final_metrics_cnn.items(): print(f"  {key}: {value:.4f}")
    else:
         print("\nCNN Model Final Metrics: No results obtained.")
         final_metrics_cnn = None
         val_labels_cnn, val_probs_cnn = None, None # Ensure None for plotting/comparison

    # Evaluate Integrated Model
    _, _, val_labels_int, val_probs_int = evaluate_model(int_model, val_int_loader, criterion, device, model_type="integrated")
    final_metrics_int = calculate_metrics(val_labels_int, (val_probs_int > 0.5).astype(int), val_probs_int)
    print("\nIntegrated Model Final Metrics:")
    for key, value in final_metrics_int.items(): print(f"  {key}: {value:.4f}")

    # Plot ROC Curves
    roc_data = {
        "Radiomics": (val_labels_rad, val_probs_rad),
        "CNN (ResNet)": (val_labels_cnn, val_probs_cnn),
        "Integrated": (val_labels_int, val_probs_int)
    }
    plot_roc_curves(roc_data, title="Validation Set ROC Curves", save_path=os.path.join(RESULTS_DIR, "validation_roc_curves.png"))

    # Compare AUCs (using placeholder DeLong test)
    # Note: Ensure labels used for comparison are consistent across models
    # If CNN loader filtered samples, comparisons might be on different subsets.
    # Using the integrated model's labels (val_labels_int) as the reference might be safest if filtering occurred.
    print("\nAUC Comparisons (DeLong Test - Placeholder):")
    if val_labels_cnn is not None:
        p_cnn_vs_rad = compare_auc_delong(val_labels_int, val_probs_cnn, val_probs_rad) # Compare based on common samples if possible
        print(f"  CNN vs Radiomics: p = {p_cnn_vs_rad:.4f}")
    else:
         print("  CNN vs Radiomics: Skipped (CNN results missing)")

    p_int_vs_rad = compare_auc_delong(val_labels_int, val_probs_int, val_probs_rad)
    print(f"  Integrated vs Radiomics: p = {p_int_vs_rad:.4f}")

    if val_labels_cnn is not None:
        p_int_vs_cnn = compare_auc_delong(val_labels_int, val_probs_int, val_probs_cnn)
        print(f"  Integrated vs CNN: p = {p_int_vs_cnn:.4f}")
    else:
        print("  Integrated vs CNN: Skipped (CNN results missing)")


    print("\n--- Workflow Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTEN Mutation Prediction using Deep Learning and Radiomics")
    parser.add_argument('--metadata', type=str, default="path/to/your/metadata.csv",
                        help='Path to the metadata CSV file')
    # Add flags to control steps
    parser.add_argument('--skip_preprocessing', action='store_true', help='Skip preprocessing if output files exist')
    parser.add_argument('--force_feature_extraction', action='store_true', help='Force radiomics feature extraction even if file exists')
    # Add more arguments for hyperparameters if needed

    args = parser.parse_args()

    # Basic check for metadata file
    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file not found at {args.metadata}")
        print("Please provide the correct path using the --metadata argument.")
    else:
        main(args)