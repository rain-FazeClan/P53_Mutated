import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Paths ---
MRI_DATA_DIR = "/path/to/preprocessed/mri_data/"
TUMOR_MASK_DIR = "/path/to/tumor_masks/"
LABEL_FILE = "/path/to/labels.csv" # Patient ID, IDH_status (0=wildtype, 1=mutant)
FA_DATA_DIR = "/path/to/fa_maps/" # For self-supervised edge training
NODE_ATLAS_FILE = "/path/to/node_atlas.nii.gz" # e.g., 90 regions
TRACT_ATLAS_FILE = "/path/to/tract_atlas.nii.gz" # e.g., 2309 tracts mapping between nodes
OUTPUT_DIR = "./output/"
BRAIN_NET_NODE_MODEL_PATH = OUTPUT_DIR + "brain_node_ae.pth"
BRAIN_NET_EDGE_MODEL_PATH = OUTPUT_DIR + "brain_edge_contrastive.pth"
CONTRASTIVE_MODEL_PATH = OUTPUT_DIR + "contrastive_model.pth"
CLASSIFIER_MODEL_PATH = OUTPUT_DIR + "classifier_model.pth"

# --- Brain Network Generation ---
BRAIN_NODE_N_REGIONS = 90
BRAIN_EDGE_N_TRACTS = 2309 # Number of connections/edges
BRAIN_NODE_VOXEL_SAMPLE_DIM = 4000
BRAIN_EDGE_VOXEL_SAMPLE_DIM_ANAT = 4000
BRAIN_EDGE_VOXEL_SAMPLE_DIM_FA = 1000
BRAIN_NODE_FEATURE_DIM = 16 # bN dimension
BRAIN_EDGE_FEATURE_DIM = 16 # bE dimension
BRAIN_NET_NODE_AE_PARAMS = {
    'encoder_dims': [BRAIN_NODE_VOXEL_SAMPLE_DIM, 2048, 1024, 512, 128, 32, BRAIN_NODE_FEATURE_DIM],
    'decoder_dims': [BRAIN_NODE_FEATURE_DIM, 32, 128, 512, 1024, 2048, BRAIN_NODE_VOXEL_SAMPLE_DIM]
}
BRAIN_NET_EDGE_ENCODER_PARAMS_ANAT = {
    'dims': [BRAIN_EDGE_VOXEL_SAMPLE_DIM_ANAT, 2048, 1024, 512, 128, 32, BRAIN_EDGE_FEATURE_DIM]
}
BRAIN_NET_EDGE_ENCODER_PARAMS_FA = {
    'dims': [BRAIN_EDGE_VOXEL_SAMPLE_DIM_FA, 1024, 512, 128, 32, BRAIN_EDGE_FEATURE_DIM]
}
BRAIN_NET_EDGE_PROJECTION_DIM = 128 # Latent dim for contrastive edge loss
BRAIN_NET_CONTRASTIVE_TEMP_EDGE = 0.1 # Tau for edge contrastive loss
BRAIN_NET_LR_NODE = 0.001
BRAIN_NET_LR_EDGE = 0.001
BRAIN_NET_WD_NODE = 5e-4
BRAIN_NET_WD_EDGE = 5e-4
BRAIN_NET_EPOCHS_NODE = 1000
BRAIN_NET_EPOCHS_EDGE = 1000
BRAIN_NET_BATCH_SIZE = 50

# --- Multi-Modal Contrastive Learning ---
IMG_INPUT_CHANNELS = 4 # T1, T1CE, T2, FLAIR
IMG_FEATURE_DIM = 32   # uI dimension
GEOM_N_POINTS = 1024 # Number of points sampled for point cloud
GEOM_FEATURE_DIM = 32  # uP dimension
BRAIN_NET_FEATURE_DIM_OUT = 32 # uB dimension (after GNN encoder)
CONTRASTIVE_LATENT_DIM = 128 # z dimension for contrastive losses
CONTRASTIVE_TEMP = 0.1 # Tau for bi-level contrastive loss
CONTRASTIVE_LAMBDA = 0.8 # Weight between tumor-level and brain-level loss (Eq 7)

# Image Encoder Params (Example 3D CNN)
IMG_ENCODER_CHANNELS = [IMG_INPUT_CHANNELS, 64, 128, 128, 256, 256]
IMG_ENCODER_FC_DIMS = [None, 512, 256, IMG_FEATURE_DIM] # Auto-calculate first dim based on conv output

# Geometric Encoder Params (Example PointNet-like/NNConv)
# Dimensions for NNConv layers or PointNet set abstraction modules
GEOM_ENCODER_DIMS = [3, 32, 64, 64, 128, 128] # Input coord dim = 3
GEOM_ENCODER_FC_DIMS = [128, 256, 128, GEOM_FEATURE_DIM]

# Brain Network Encoder Params (GAT)
BRAIN_NET_ENCODER_GAT_DIMS = [BRAIN_NODE_FEATURE_DIM, 64, 128, 128, 256, 256, 256] # Input node dim
BRAIN_NET_ENCODER_GAT_HEADS = [4] * 6 # Example: 4 heads per layer
BRAIN_NET_ENCODER_FC_DIMS = [256, 512, 256, BRAIN_NET_FEATURE_DIM_OUT] # Input from GAT output

# Projection Heads
PROJECTION_HEAD_DIMS = [None, 64, 128, CONTRASTIVE_LATENT_DIM] # Input dim depends on feature type

CONTRASTIVE_LR = 0.001
CONTRASTIVE_WD = 5e-4
CONTRASTIVE_EPOCHS = 1000
CONTRASTIVE_BATCH_SIZE = 20

# --- Population Graph & Classification ---
POP_GRAPH_CORR_THRESHOLD = 0.5 # Theta
POP_GRAPH_NODE_WEIGHT_TYPE = 'concat_UF_UB' # Options based on Table II
POP_GRAPH_EDGE_WEIGHT_TYPE = 'corr_UB'     # Options based on Table II

# GNN Classifier Params (GAT)
CLASSIFIER_GAT_DIMS = [None, 64, 128, 128, 128] # Input node dim depends on POP_GRAPH_NODE_WEIGHT_TYPE
CLASSIFIER_GAT_HEADS = [4] * 4
CLASSIFIER_N_CLASSES = 2 # IDH mutant vs wildtype
CLASSIFIER_LR = 0.001
CLASSIFIER_WD = 5e-4
CLASSIFIER_EPOCHS = 200
CLASSIFIER_BATCH_SIZE = 20 # Note: For semi-supervised, the graph contains all nodes

# --- Training ---
SEED = 42
TEST_SPLIT_RATIO = 0.3
VALIDATION_SPLIT_RATIO = 0.1 # From the training set
NUM_WORKERS = 4