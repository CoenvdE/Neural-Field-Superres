import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import NeuralFieldSuperRes
from src.data import SyntheticWeatherDataset
import time

def train():
    # Hyperparameters
    BATCH_SIZE = 16
    LR = 1e-3
    EPOCHS = 5
    LATENT_DIM = 1
    COORD_DIM = 2
    NUM_HIDDEN = 64
    NUM_HEADS = 4
    NUM_QUERY_FEATURES = 64 # Assuming query features match hidden dim for now
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    dataset = SyntheticWeatherDataset(num_samples=500, num_latents=50, num_queries=100, coord_dim=COORD_DIM)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Test Configuration 1: Equivariant Decoder + Encoder
    print("\n--- Testing Config 1: Equivariant Decoder + Encoder ---")
    model = NeuralFieldSuperRes(
        latent_dim=LATENT_DIM,
        out_dim=1,
        num_query_features=NUM_QUERY_FEATURES,
        num_hidden=NUM_HIDDEN,
        num_heads=NUM_HEADS,
        coord_dim=COORD_DIM,
        use_self_attention=True,
        use_encoder=True,
        num_encoder_layers=1,
        decoder_type='equivariant',
        num_decoder_layers=1
    ).to(device)
    
    run_training(model, dataloader, device, EPOCHS, LR, NUM_QUERY_FEATURES)

    # Test Configuration 2: Attention Decoder (RoPE) + Encoder
    print("\n--- Testing Config 2: Attention Decoder (RoPE) + Encoder ---")
    model = NeuralFieldSuperRes(
        latent_dim=LATENT_DIM,
        out_dim=1,
        num_query_features=NUM_QUERY_FEATURES,
        num_hidden=NUM_HIDDEN,
        num_heads=NUM_HEADS,
        coord_dim=COORD_DIM,
        use_self_attention=True,
        use_encoder=True,
        num_encoder_layers=1,
        decoder_type='attention',
        num_decoder_layers=2,
        use_rope=True
    ).to(device)
    
    run_training(model, dataloader, device, EPOCHS, LR, NUM_QUERY_FEATURES)

    # Test Configuration 3: KNN Attention Decoder + Encoder
    print("\n--- Testing Config 3: KNN Attention Decoder + Encoder ---")
    model = NeuralFieldSuperRes(
        latent_dim=LATENT_DIM,
        out_dim=1,
        num_query_features=NUM_QUERY_FEATURES,
        num_hidden=NUM_HIDDEN,
        num_heads=NUM_HEADS,
        coord_dim=COORD_DIM,
        use_self_attention=True,
        use_encoder=True,
        num_encoder_layers=1,
        decoder_type='knn attn',
        num_decoder_layers=1,
        use_rope=True
    ).to(device)
    
    run_training(model, dataloader, device, EPOCHS, LR, NUM_QUERY_FEATURES)

def run_training(model, dataloader, device, epochs, lr, num_query_features):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            latents = batch['latents'].to(device)       # [B, Z, 1]
            latent_pos = batch['latent_pos'].to(device) # [B, Z, D]
            query_pos = batch['query_pos'].to(device)   # [B, C, D]
            targets = batch['targets'].to(device)       # [B, C, 1]
            
            # Mock grid features: [B, Z, num_hidden]
            # In real usage, this would be a grid, but for testing shapes, latents works
            grid_features = torch.randn(latents.shape[0], latents.shape[1], 64).to(device)
            grid_pos = latent_pos # Mock grid pos
            
            # Mock query features
            query_features = torch.randn(latents.shape[0], query_pos.shape[1], num_query_features).to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(
                latents=latents, 
                latent_pos=latent_pos, 
                query_pos=query_pos,
                grid_features=grid_features,
                grid_pos=grid_pos,
                query_features=query_features
            )
            
            # Loss
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
    print(f"Training finished in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    train()
