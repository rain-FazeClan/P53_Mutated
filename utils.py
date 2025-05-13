import torch
import numpy as np
import random
import os
import SimpleITK as sitk

def set_seed(seed=42):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def load_nifti_image(path):
    """Loads a NIfTI image using SimpleITK."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    return sitk.ReadImage(path)

def save_nifti_image(image, path):
    """Saves a SimpleITK image to NIfTI format."""
    sitk.WriteImage(image, path)

def get_device():
    """Gets the appropriate device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")