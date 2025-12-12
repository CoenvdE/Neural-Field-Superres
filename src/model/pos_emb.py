import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CoordinateEncoder(nn.Module):
    """Encodes coordinates into the attention embedding space via RFF."""

    def __init__(
        self,
        *,
        coord_dim: int,
        embed_dim: int,
        coordinate_system: str = "cartesian",
        learnable_coefficients: bool = True,
        init_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.coord_dim = coord_dim
        self.coordinate_system = coordinate_system

        self.encoder = RFFNet(
            in_dim=self.coord_dim,
            output_dim=embed_dim,
            hidden_dim=embed_dim, #TODO: needed and expansion?
            learnable_coefficients=learnable_coefficients,
            std=init_std,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        processed = self._preprocess(coords)
        return self.encoder(processed)

    def _preprocess(self, coords: torch.Tensor) -> torch.Tensor:
        if self.coordinate_system == "cartesian":
            return coords

        if self.coordinate_system == "latlon":
            # if self.coord_dim != 2:
            #     raise ValueError("latlon coordinates require at least 2 dims") #TODO: at least 2? exactly 2 right?
            # lat = coords[..., 0].clamp(-1.0, 1.0) * (math.pi / 2)
            # lon = coords[..., 1].clamp(-1.0, 1.0) * math.pi
            # lat_features = torch.stack(
            #     (torch.sin(lat), torch.cos(lat)), dim=-1)
            # lon_features = torch.stack(
            #     (torch.sin(lon), torch.cos(lon)), dim=-1)
            # return torch.cat((lat_features, lon_features), dim=-1)
            raise NotImplementedError

        if self.coordinate_system == "polar": #TODO: polar for 3d
            # if self.coord_dim != 3:
            #     raise ValueError("polar coordinates require at least 2 dims") #TODO: at least 2? exactly 3 right?
            # radius = coords[..., 0]
            # angle = coords[..., 1] * math.pi
            # x = radius * torch.cos(angle)
            # y = radius * torch.sin(angle)
            # return torch.stack((x, y, angle), dim=-1)
            raise NotImplementedError

        return coords

class RFFNet(nn.Module):
    """RFF-based embedding network: encoding + MLP + final linear."""
    def __init__(
        self,
        in_dim: int,
        output_dim: int,
        hidden_dim: int, #TODO: needed?
        learnable_coefficients: bool,
        std: float
    ):
        super().__init__()
        self.encoding = RFFEmbedding(
            in_dim=in_dim,
            hidden_dim=hidden_dim, #TODO: needed?
            learnable_coefficients=learnable_coefficients,
            std=std
        )
        self.mlp = PointwiseFFN(hidden_dim, hidden_dim, hidden_dim) #TODO: needed and expansion?
        self.linear_final = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoding(x)
        x = self.mlp(x)
        return self.linear_final(x)

class RFFEmbedding(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, learnable_coefficients: bool, std: float):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be even"
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.learnable = learnable_coefficients
        # weight shape [in_dim, hidden_dim//2]
        coeff = torch.randn(in_dim, hidden_dim // 2) * std
        self.coefficients = nn.Parameter(coeff, requires_grad=learnable_coefficients)
        self.pi = 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]
        coeff = self.coefficients if self.learnable else self.coefficients.detach() #TODO: why detach
        x_proj = self.pi * x @ coeff #TODO: why times pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PointwiseFFN(nn.Module):
    """Two-layer FFN with GELU and LayerNorm."""
    def __init__(self, num_in: int, num_hidden: int, num_out: int):
        super().__init__()
        self.fc1 = nn.Linear(num_in, num_hidden)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.ln(x)
        return self.fc2(x)
