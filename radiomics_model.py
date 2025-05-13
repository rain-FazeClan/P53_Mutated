import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif # Using f_classif as proxy for MIC

class RadiomicsMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], output_dim=1, dropout_rates=[0.2, 0.5]):
        """
        Four-layer MLP for radiomics features (Input -> Hidden1 -> Hidden2 -> Output).
        Paper implies 4 layers total, maybe Input -> H1 -> H2 -> H3 -> Output?
        Let's stick to Input -> H1 -> H2 -> Output for now (3 weight layers).
        Adjust hidden_dims list if a different number of hidden layers is intended.
        """
        super(RadiomicsMLP, self).__init__()
        self.input_dim = input_dim
        layers = []
        current_dim = input_dim

        # Input layer to first hidden layer
        layers.append(nn.Linear(current_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        if dropout_rates[0] > 0:
            layers.append(nn.Dropout(dropout_rates[0]))
        current_dim = hidden_dims[0]

        # Second hidden layer
        layers.append(nn.Linear(current_dim, hidden_dims[1]))
        layers.append(nn.ReLU())
        if dropout_rates[1] > 0:
            layers.append(nn.Dropout(dropout_rates[1]))
        current_dim = hidden_dims[1]

        # Output layer
        self.output_layer = nn.Linear(current_dim, output_dim)

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()


    def _initialize_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.xavier_uniform_(m.weight) # Glorot uniform
                 if m.bias is not None:
                     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Get features from the third layer (output of second hidden layer)
        # This requires running layer by layer or accessing intermediate outputs
        hidden_features = self.network(x) # Output of last dropout/relu before final linear layer
        out = self.output_layer(hidden_features)

        # Apply sigmoid for binary classification probability
        if self.output_layer.out_features == 1:
            out = torch.sigmoid(out)

        return out, hidden_features # Return final output and intermediate features

def select_radiomics_features(features_df: pd.DataFrame, labels: pd.Series, percentile=30):
    """
    Selects top features based on Maximal Information Coefficient (MIC) proxy (f_classif).

    Args:
        features_df (pd.DataFrame): DataFrame of radiomics features (rows=samples, cols=features).
        labels (pd.Series): Series of corresponding labels (0 or 1).
        percentile (int): Percentage of top features to select.

    Returns:
        pd.DataFrame: DataFrame with selected features.
        list: List of selected feature names.
    """
    if features_df.isnull().any().any():
        print("Warning: Input features contain NaNs. Imputing with mean.")
        features_df = features_df.fillna(features_df.mean()) # Simple imputation

    # Ensure labels are numpy array
    labels_np = labels.to_numpy()

    # Use f_classif (ANOVA F-value) as a proxy for MIC - requires sklearn
    # For actual MIC, you might need `minepy` or other libraries.
    selector = SelectPercentile(score_func=f_classif, percentile=percentile)

    try:
        selector.fit(features_df, labels_np)
        selected_features_mask = selector.get_support()
        selected_feature_names = features_df.columns[selected_features_mask].tolist()
        selected_features_df = features_df.loc[:, selected_feature_names]

        print(f"Selected {len(selected_feature_names)} features (top {percentile}%)")
        return selected_features_df, selected_feature_names

    except Exception as e:
        print(f"Error during feature selection: {e}")
        # Fallback: return all features? Or raise error?
        return features_df, features_df.columns.tolist()