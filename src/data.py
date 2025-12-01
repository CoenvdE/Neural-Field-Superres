import torch
from torch.utils.data import Dataset
import numpy as np

from typing import Optional

class SyntheticWeatherDataset(Dataset):
    def __init__(self, num_samples=1000, 
                    coord_dim=2, 
                    num_queries=200, 
                    num_latents=50, 
                    num_input_features: Optional[int] = None):
        

        self.num_samples = num_samples
        self.num_latents = num_latents
        self.num_input_features = num_input_features
        self.num_queries = num_queries
        self.coord_dim = coord_dim
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        samples = {}
        # Generate a random field function (e.g. sum of sines)
        # f(x) = sum(A_i * sin(w_i * x + b_i))
        
        # Random parameters for the field
        num_freqs = 5
        freqs = torch.randn(num_freqs, self.coord_dim) * 2
        phases = torch.rand(num_freqs) * 2 * np.pi
        amplitudes = torch.randn(num_freqs)
        
        def field_fn(pos):
            # pos: [N, D]
            # result: [N, 1]
            val = 0
            for i in range(num_freqs):
                # dot product
                phase_shift = (pos @ freqs[i]) + phases[i]
                val += amplitudes[i] * torch.sin(phase_shift)
            return val.unsqueeze(-1)
        
        # Sample latent positions (randomly distributed in [-1, 1])
        latent_pos = torch.rand(self.num_latents, self.coord_dim) * 2 - 1
        latents = field_fn(latent_pos) # The "features" are just the field values for now
        
        # Sample query positions
        query_pos = torch.rand(self.num_queries, self.coord_dim) * 2 - 1
        targets = field_fn(query_pos)

        if self.num_input_features is not None:
            input_pos = torch.rand(self.num_input_features, self.coord_dim) * 2 - 1
            input_features = field_fn(input_pos)

            samples['input_features'] = input_features.float()
            samples['input_pos'] = input_pos.float()
        else:
            # Default to empty or same as latents if not specified? 
            # Or just use latents as input features if not specified?
            # For now let's use latents to avoid UnboundLocalError
            input_pos = latent_pos
            input_features = latents
        
        samples['latents'] = latents.float()       # [Z, 1]
        samples['latent_pos'] = latent_pos.float() # [Z, D]
        samples['query_pos'] = query_pos.float()   # [C, D]
        samples['query_features'] = targets.float() # [C, 1]
        return samples


if __name__ == "__main__":
    dataset = SyntheticWeatherDataset(
        num_samples=10,
        num_latents=50,
        num_queries=200,
        coord_dim=2,
        num_input_features=1000
    )
    
    print("Dataset sample shapes:")
    sample = dataset[0]
    for key, value in sample.items():
        print(f"{key}: {value.shape}")

# usage:
# python data.py