import SimpleITK as sitk
import numpy as np
import os
from utils import load_nifti_image, save_nifti_image

# --- Configuration ---
# Target properties after preprocessing
TARGET_SPACING = (1.0, 1.0, 1.0)  # Isotropic voxel resampling (example, adjust if needed)
TARGET_SIZE = (240, 240, 155)     # Target size after resizing (as per paper)
CNN_INPUT_SIZE = (224, 224, 32)   # Target size for CNN input (H, W, D) - from paper ResNet input shape (4, 32, 224, 224) -> (modal, D, H, W)
# --- ---

def n4_bias_field_correction(image: sitk.Image) -> sitk.Image:
    """Applies N4 Bias Field Correction."""
    print("Applying N4 Bias Field Correction...")
    # Create a mask covering the brain (simple thresholding, replace with better mask if available)
    mask_image = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # Parameters can be tuned, these are common defaults
    corrector.SetMaximumNumberOfIterations([50] * 4)
    corrected_image = corrector.Execute(image, mask_image)
    print("N4 Correction Done.")
    return corrected_image

def resample_image(image: sitk.Image, target_spacing=TARGET_SPACING, interpolator=sitk.sitkLinear) -> sitk.Image:
    """Resamples image to target spacing."""
    print(f"Resampling to spacing: {target_spacing}...")
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [int(round(original_size[i] * (original_spacing[i] / target_spacing[i]))) for i in range(3)]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputSpacing(target_spacing)
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetOutputOrigin(image.GetOrigin())
    resample_filter.SetOutputDirection(image.GetDirection())
    resample_filter.SetDefaultPixelValue(image.GetPixelIDValue()) # Use appropriate default value

    resampled_image = resample_filter.Execute(image)
    print("Resampling Done.")
    return resampled_image

def register_images(fixed_image: sitk.Image, moving_image: sitk.Image) -> sitk.Image:
    """Performs rigid registration of moving_image to fixed_image."""
    # NOTE: This is a simplified rigid registration. Real-world registration
    # often requires more complex setup, potentially non-rigid methods,
    # and careful parameter tuning. Using a dedicated library/tool is often better.
    print("Performing Rigid Registration...")
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))

    print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
    print(f"Final metric value: {registration_method.GetMetricValue()}")

    # Apply transform to the moving image
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetReferenceImage(fixed_image) # Use fixed image grid
    resample_filter.SetInterpolator(sitk.sitkLinear)
    resample_filter.SetDefaultPixelValue(moving_image.GetPixelIDValue())
    resample_filter.SetTransform(final_transform)

    registered_image = resample_filter.Execute(moving_image)
    print("Registration Done.")
    return registered_image

def normalize_intensity(image: sitk.Image, method="zscore") -> sitk.Image:
    """Normalizes image intensity."""
    print(f"Normalizing intensity ({method})...")
    img_array = sitk.GetArrayFromImage(image)

    if method == "zscore":
        # Normalize only non-zero voxels (assuming background is 0)
        mask = img_array > (img_array.min() + 1e-5) # Avoid pure background
        if np.any(mask):
            mean = img_array[mask].mean()
            std = img_array[mask].std()
            if std > 1e-5: # Avoid division by zero
                 normalized_array = np.where(mask, (img_array - mean) / std, 0)
            else:
                 normalized_array = np.zeros_like(img_array) # Or handle as constant
        else:
             normalized_array = np.zeros_like(img_array) # Handle empty mask case

    elif method == "minmax":
        min_val = img_array.min()
        max_val = img_array.max()
        if max_val > min_val:
            normalized_array = (img_array - min_val) / (max_val - min_val)
        else:
            normalized_array = np.zeros_like(img_array)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image) # Keep metadata
    print("Normalization Done.")
    return normalized_image


def resize_image(image: sitk.Image, target_size=TARGET_SIZE, interpolator=sitk.sitkLinear) -> sitk.Image:
    """Resizes image to target size using padding/cropping or resampling."""
    print(f"Resizing to size: {target_size}...")
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    # Calculate new spacing based on target size
    new_spacing = [original_spacing[i] * (original_size[i] / target_size[i]) for i in range(3)]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(target_size)
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetOutputOrigin(image.GetOrigin())
    resample_filter.SetOutputDirection(image.GetDirection())
    resample_filter.SetDefaultPixelValue(image.GetPixelIDValue())

    resized_image = resample_filter.Execute(image)
    print("Resizing Done.")
    return resized_image

def preprocess_patient_images(patient_files: dict, output_dir: str, skip_existing=True):
    """Runs the full preprocessing pipeline for a patient's modalities."""
    patient_id = patient_files['id']
    patient_output_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)

    processed_files = {}
    reference_image = None # Use T1c or T1w as reference for registration

    # --- Step 0: Load Reference Image ---
    ref_modality = 'T1c' if 'T1c' in patient_files else 'T1w'
    if ref_modality in patient_files:
        print(f"Loading reference image: {ref_modality}")
        reference_image = load_nifti_image(patient_files[ref_modality])
    else:
        print(f"Error: Cannot find reference modality ({ref_modality}) for patient {patient_id}")
        return None # Cannot proceed without reference

    # --- Preprocess Reference Image ---
    ref_output_path = os.path.join(patient_output_dir, f"{patient_id}_{ref_modality}_preprocessed.nii.gz")
    if not skip_existing or not os.path.exists(ref_output_path):
        print(f"--- Preprocessing {ref_modality} ---")
        img = reference_image
        # 1. N4 Bias Correction (Optional but recommended)
        # img = n4_bias_field_correction(img) # Can be slow
        # 2. Resample to Isotropic Spacing
        img = resample_image(img, target_spacing=TARGET_SPACING)
        # 3. Resize to Target Dimensions (before normalization for consistency)
        img = resize_image(img, target_size=TARGET_SIZE)
        # 4. Normalize Intensity
        img = normalize_intensity(img, method="zscore")
        save_nifti_image(img, ref_output_path)
        reference_image = img # Update reference to the preprocessed version
        processed_files[ref_modality] = ref_output_path
        print(f"--- Finished {ref_modality} ---")
    else:
        print(f"Skipping existing preprocessed file: {ref_output_path}")
        processed_files[ref_modality] = ref_output_path
        reference_image = load_nifti_image(ref_output_path) # Load preprocessed as reference

    # --- Preprocess Other Modalities ---
    for mod in MODALITIES:
        if mod == ref_modality: continue # Already processed

        output_path = os.path.join(patient_output_dir, f"{patient_id}_{mod}_preprocessed.nii.gz")
        if not skip_existing or not os.path.exists(output_path):
             if mod in patient_files:
                print(f"--- Preprocessing {mod} ---")
                img = load_nifti_image(patient_files[mod])
                # 1. N4 Bias Correction (Optional)
                # img = n4_bias_field_correction(img)
                # 2. Register to Reference Image
                img = register_images(reference_image, img) # Register to preprocessed reference
                # 3. Resample (already done via registration to reference grid)
                # 4. Resize (already done via registration to reference grid)
                # 5. Normalize Intensity
                img = normalize_intensity(img, method="zscore")
                save_nifti_image(img, output_path)
                processed_files[mod] = output_path
                print(f"--- Finished {mod} ---")
             else:
                 print(f"Warning: Modality {mod} not found for patient {patient_id}")
        else:
            print(f"Skipping existing preprocessed file: {output_path}")
            processed_files[mod] = output_path


    # --- Preprocess Segmentation Masks (Resample/Resize only) ---
    for seg_key in ['seg_WT', 'seg_TC']:
        if patient_files.get(seg_key): # Check if mask exists
            mask_path = patient_files[seg_key]
            mask_output_path = os.path.join(patient_output_dir, f"{patient_id}_{seg_key}_preprocessed.nii.gz")
            if not skip_existing or not os.path.exists(mask_output_path):
                print(f"--- Preprocessing {seg_key} Mask ---")
                mask = load_nifti_image(mask_path)
                # Apply transforms using the reference image grid
                # Resample/Resize using Nearest Neighbor interpolation for masks
                resample_filter = sitk.ResampleImageFilter()
                resample_filter.SetReferenceImage(reference_image) # Use preprocessed reference grid
                resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
                resample_filter.SetDefaultPixelValue(0) # Background label
                # No specific transform needed if mask was generated on original space
                # If mask was generated on a different space, registration transform might be needed
                processed_mask = resample_filter.Execute(mask)

                # Ensure mask size matches the preprocessed image size
                if processed_mask.GetSize() != reference_image.GetSize():
                     print(f"Warning: Processed mask size {processed_mask.GetSize()} differs from reference {reference_image.GetSize()}. Resizing mask again.")
                     processed_mask = resize_image(processed_mask, target_size=reference_image.GetSize(), interpolator=sitk.sitkNearestNeighbor)


                save_nifti_image(processed_mask, mask_output_path)
                processed_files[seg_key] = mask_output_path
                print(f"--- Finished {seg_key} Mask ---")
            else:
                print(f"Skipping existing preprocessed mask: {mask_output_path}")
                processed_files[seg_key] = mask_output_path

    return processed_files

# --- CNN Specific Preprocessing ---
def crop_or_pad_to_size(img_array: np.ndarray, target_size: tuple) -> np.ndarray:
    """Crops or pads a numpy array to the target size (H, W, D)."""
    current_size = img_array.shape # (D, H, W) from SimpleITK
    target_size_torch = (target_size[2], target_size[0], target_size[1]) # D, H, W

    # Calculate padding/cropping amounts
    delta = [target_size_torch[i] - current_size[i] for i in range(3)]
    paddings = []
    croppings = []

    for i in range(3):
        if delta[i] > 0: # Pad
            pad_before = delta[i] // 2
            pad_after = delta[i] - pad_before
            paddings.append((pad_before, pad_after))
            croppings.append((0, current_size[i])) # No cropping on this dim
        else: # Crop
            crop_start = abs(delta[i]) // 2
            crop_end = current_size[i] - (abs(delta[i]) - crop_start)
            paddings.append((0, 0)) # No padding on this dim
            croppings.append((crop_start, crop_end))

    # Apply cropping first
    cropped_array = img_array[croppings[0][0]:croppings[0][1],
                              croppings[1][0]:croppings[1][1],
                              croppings[2][0]:croppings[2][1]]

    # Apply padding
    padded_array = np.pad(cropped_array, paddings, mode='constant', constant_values=0)

    return padded_array


def prepare_cnn_input(patient_processed_files: dict, modalities=MODALITIES, target_size=CNN_INPUT_SIZE):
    """Loads preprocessed images, stacks modalities, crops/pads for CNN."""
    stacked_data = []
    valid = True
    for mod in modalities:
        if mod in patient_processed_files:
            img_path = patient_processed_files[mod]
            try:
                img = load_nifti_image(img_path)
                img_array = sitk.GetArrayFromImage(img) # Shape (D, H, W)
                # Crop/Pad to CNN input size (H, W, D specified) -> (D, H, W target)
                processed_array = crop_or_pad_to_size(img_array, target_size)
                stacked_data.append(processed_array)
            except Exception as e:
                print(f"Error loading/processing {mod} for CNN input: {e}")
                valid = False
                break
        else:
            print(f"Error: Preprocessed modality {mod} not found for CNN input.")
            valid = False
            break

    if not valid or len(stacked_data) != len(modalities):
        return None

    # Stack along channel dimension (new first dimension)
    cnn_input_array = np.stack(stacked_data, axis=0) # Shape (Num_Modalities, D, H, W)
    # Convert to PyTorch tensor
    cnn_input_tensor = torch.from_numpy(cnn_input_array).float()

    return cnn_input_tensor