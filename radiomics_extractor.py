from radiomics import featureextractor
import SimpleITK as sitk
import pandas as pd
import os
import numpy as np

# --- Configuration ---
# Path to pyradiomics parameter file (optional, can configure in code)
# PARAMS_FILE = 'path/to/pyradiomics_params.yaml'
# Or define settings here:
RADIOMICS_SETTINGS = {
  'setting': {
    'binWidth': 25,
    'resampledPixelSpacing': None, # Already resampled in preprocessing
    'interpolator': 'sitkLinear',
    'label': 1, # Assume tumor label in mask is 1
    'force2D': False,
    'normalize': True, # Normalize image before feature extraction
    'normalizeScale': 100, # Scale after normalization
    'removeOutliers': None, # Consider setting a value e.g., 3 sigma
  },
  'imageType': {
    'Original': {},
    'Wavelet': {} # Enable wavelet features as per paper
    # Add other filters if needed (e.g., LoG)
  },
  'featureClass': {
    # As per paper sections/IBSI: Geometry (Shape), Intensity (FirstOrder), Texture
    'shape': None, # Geometry features
    'firstorder': None, # Intensity features
    'glcm': None, # Gray Level Co-occurrence Matrix
    'glrlm': None, # Gray Level Run Length Matrix
    'glszm': None, # Gray Level Size Zone Matrix
    'ngtdm': None, # Neighbouring Gray Tone Difference Matrix
    # 'gldm': None # Gray Level Dependence Matrix (optional)
  }
}
# --- ---

def extract_radiomics_features(image_path: str, mask_path: str, settings=RADIOMICS_SETTINGS):
    """
    Extracts radiomics features for a given image and mask pair.

    Args:
        image_path (str): Path to the preprocessed NIfTI image file.
        mask_path (str): Path to the corresponding preprocessed NIfTI mask file.
        settings (dict): PyRadiomics settings dictionary.

    Returns:
        pd.Series: A pandas Series containing the extracted features, or None if extraction fails.
                   Feature names are prefixed with 'radiomics_'.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found for radiomics: {image_path}")
        return None
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found for radiomics: {mask_path}")
        return None

    try:
        # Initialize extractor
        # extractor = featureextractor.RadiomicsFeatureExtractor(PARAMS_FILE)
        extractor = featureextractor.RadiomicsFeatureExtractor(settings) # Use dict settings

        # print("Enabled input images types: ", extractor.enabledImagetypes)
        # print("Enabled features: ", extractor.enabledFeatures)

        # Execute extraction
        # Ensure mask has integer type
        mask_img = sitk.ReadImage(mask_path)
        if mask_img.GetPixelIDValue() not in [1, 2, 4]: # Check for common integer types
             mask_img = sitk.Cast(mask_img, sitk.sitkUInt32) # Cast to unsigned int

        result = extractor.execute(image_path, mask_img) # Pass mask as SimpleITK object or path

        # Filter out non-feature results and diagnostics
        feature_values = {f'radiomics_{key}': val for key, val in result.items() if not key.startswith('diagnostics_')}

        # Convert numpy arrays to float (some shape features might be arrays)
        for key, val in feature_values.items():
            if isinstance(val, np.ndarray):
                 # Handle array features (e.g., take mean, flatten, etc.) - Check pyradiomics docs
                 # For simplicity, let's try converting to float if it's scalar-like
                 try:
                     feature_values[key] = float(val)
                 except TypeError:
                     print(f"Warning: Could not convert feature {key} (type {type(val)}) to float. Setting to NaN.")
                     feature_values[key] = np.nan


        return pd.Series(feature_values)

    except Exception as e:
        print(f"Error during radiomics extraction for {os.path.basename(image_path)} with mask {os.path.basename(mask_path)}: {e}")
        # print traceback for debugging
        import traceback
        traceback.print_exc()
        return None


def extract_features_for_patient(patient_processed_files: dict, modalities=MODALITIES):
    """
    Extracts radiomics features for all modalities and both segmentation regions (WT, TC) for a patient.

    Args:
        patient_processed_files (dict): Paths to preprocessed files ('T1w', 'T1c', ..., 'seg_WT', 'seg_TC').
        modalities (list): List of modalities to extract features from.

    Returns:
        pd.DataFrame: DataFrame where each row corresponds to a modality/region combination,
                      or None if essential files are missing. Returns an empty DataFrame if no features extracted.
    """
    all_features = []
    patient_id = patient_processed_files.get('id', 'unknown_patient')

    for seg_key, seg_name in [('seg_WT', 'WT'), ('seg_TC', 'TC')]:
        mask_path = patient_processed_files.get(seg_key)
        if not mask_path or not os.path.exists(mask_path):
            print(f"Skipping radiomics for region {seg_name} due to missing mask for patient {patient_id}.")
            continue

        for mod in modalities:
            image_path = patient_processed_files.get(mod)
            if not image_path or not os.path.exists(image_path):
                print(f"Skipping radiomics for modality {mod} (region {seg_name}) due to missing image for patient {patient_id}.")
                continue

            print(f"Extracting radiomics: Patient {patient_id}, Modality {mod}, Region {seg_name}")
            features = extract_radiomics_features(image_path, mask_path)

            if features is not None:
                # Add identifiers
                features['PatientID'] = patient_id
                features['Modality'] = mod
                features['Region'] = seg_name
                # Add prefix to feature names to avoid clashes between modality/region
                features = features.add_prefix(f"{mod}_{seg_name}_")
                # Rename ID columns back
                features.rename(columns={f"{mod}_{seg_name}_PatientID": "PatientID",
                                         f"{mod}_{seg_name}_Modality": "Modality",
                                         f"{mod}_{seg_name}_Region": "Region"}, inplace=True)
                all_features.append(features)
            else:
                print(f"Feature extraction failed for {patient_id}, {mod}, {seg_name}")


    if not all_features:
        print(f"No radiomics features extracted for patient {patient_id}.")
        return pd.DataFrame() # Return empty dataframe

    # Combine features from different modalities/regions for the patient
    # This creates one very wide row per patient
    patient_df = pd.DataFrame([pd.concat(all_features)]) # Concatenate Series into one row

    # Alternative: Keep modality/region info if needed for specific analysis
    # patient_df = pd.DataFrame(all_features)

    # Clean column names if needed (e.g., remove special characters if any)
    patient_df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in patient_df.columns]

    # Set PatientID as index maybe?
    # patient_df = patient_df.set_index('PatientID') # Be careful if PatientID column name got mangled

    return patient_df