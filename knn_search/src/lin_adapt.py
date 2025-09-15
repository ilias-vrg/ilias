import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAdapter(nn.Module):
    def __init__(self, in_features, out_features):
        """Initialize a linear adapter layer.
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        super(LinearAdapter, self).__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize the layer parameters."""
        nn.init.xavier_uniform_(self.layer.weight)
        if self.layer.bias is not None:
            nn.init.constant_(self.layer.bias, 0.0)

    def forward(self, x):
        """Forward pass through the linear adapter layer.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Normalized output tensor.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.layer(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


def load_lin_adapt_layer(weights_path, use_gpu=False):
    """Load a linear adaptation layer from a given weights path.
    Args:
        weights_path (str): Path to the weights file.
        use_gpu (bool): If True, load the layer on GPU; otherwise, load on CPU.
    Returns:
        LinearAdapter: An instance of the LinearAdapter with loaded weights.
    """
    lin_adapt_layer = nn.Identity()
    if weights_path is not None:
        weights = torch.load(weights_path, map_location="cpu")
        out_features, in_features = weights["layer.weight"].shape
        lin_adapt_layer = LinearAdapter(
            in_features=in_features, out_features=out_features
        )
        lin_adapt_layer.load_state_dict(weights)
        print("\n> linear adaptation layer loaded from:", weights_path)

        lin_adapt_layer.eval()
        if use_gpu:
            lin_adapt_layer = lin_adapt_layer.cuda()
        else:
            lin_adapt_layer = lin_adapt_layer.cpu()
    return lin_adapt_layer
