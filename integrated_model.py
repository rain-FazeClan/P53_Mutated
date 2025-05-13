import torch
import torch.nn as nn
import torch.nn.functional as F

class IntegratedMLP(nn.Module):
    def __init__(self, cnn_feature_dim, radiomics_feature_dim, hidden_dims=[512, 256], output_dim=1, dropout_rate=0.5):
        """
        MLP to combine CNN features and Radiomics features.
        Paper mentions a four-layer network. Assuming Input -> H1 -> H2 -> H3 -> Output.
        Adjust hidden_dims if needed.
        """
        super(IntegratedMLP, self).__init__()
        self.input_dim = cnn_feature_dim + radiomics_feature_dim
        layers = []
        current_dim = self.input_dim

        # Hidden Layer 1
        layers.append(nn.Linear(current_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate)) # Dropout can be applied differently
        current_dim = hidden_dims[0]

        # Hidden Layer 2
        layers.append(nn.Linear(current_dim, hidden_dims[1]))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        current_dim = hidden_dims[1]

        # Hidden Layer 3 (if needed to make it 4 layers total)
        # Example: Assuming 3 hidden layers based on "four-layer" description
        if len(hidden_dims) > 2:
             layers.append(nn.Linear(current_dim, hidden_dims[2]))
             layers.append(nn.ReLU())
             if dropout_rate > 0:
                 layers.append(nn.Dropout(dropout_rate))
             current_dim = hidden_dims[2]


        # Output Layer
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

    def forward(self, cnn_features, radiomics_features):
        # Concatenate features
        combined_features = torch.cat((cnn_features, radiomics_features), dim=1)

        # Pass through the network
        hidden_output = self.network(combined_features)
        out = self.output_layer(hidden_output)

        # Apply sigmoid for binary classification probability
        if self.output_layer.out_features == 1:
            out = torch.sigmoid(out)

        return out