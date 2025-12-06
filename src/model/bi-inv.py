import torch
import torch.nn as nn

class RelativePositionND(nn.Module):
    """ Calculate the relative position between two sets of coordinates in N dimensions. """

    def __init__(self, num_dims: int):
        super().__init__()
        self.dim = num_dims
        self.num_x_pos_dims = num_dims
        self.num_z_pos_dims = num_dims

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, num_x_pos_dims] (Query coordinates)
        p: [B, Z, num_z_pos_dims] (Latent coordinates)
        returns: [B, C, Z, num_x_pos_dims]
        """
        return x[:, :, None, :self.num_x_pos_dims] - p[:, None, :, :self.num_z_pos_dims]