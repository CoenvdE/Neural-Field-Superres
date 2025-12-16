# Neural-Field-Superres

Neural Field Super-Resolution for high-resolution weather data reconstruction from ERA5 latents.

## Quick Start on Snellius

### 1. Clone Repository

```bash
cd /home/$USER
git clone <your-repo-url> Neural-Field-Superres
cd Neural-Field-Superres
```

### 2. Setup Environment (One-Time)

Submit the environment setup job:

```bash
sbatch cluster_workflow/setup_env.job
```

This creates both a **conda** and **uv** environment. Check the output:

```bash
cat cluster_workflow/slurm_outputs/setup_env_*.out
```

### 3. Run Training

```bash
sbatch cluster_workflow/train_europe.job
```

---

## Installation Options

### Option A: Conda (Recommended for Snellius)

```bash
# On Snellius
module load 2023
module load Anaconda3/2023.07-2

# Create environment
conda env create -f environment.yaml

# Activate
source activate neural-field-superres

# Run training
python -m src.train fit --config config/default_europe.yaml
```

### Option B: UV/Pip

```bash
# Create virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install PyTorch with CUDA (for GPU)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project
uv pip install -e .

# Run training
python -m src.train fit --config config/default_europe.yaml
```

### Option C: Plain Pip

```bash
python -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project
pip install -e .
```

---

## Project Structure

```
Neural-Field-Superres/
├── config/                     # Training configurations
│   ├── default_europe.yaml     # Full Europe region
│   └── default.yaml            # Scandinavia subset
├── cluster_workflow/           # SLURM job files
│   ├── setup_env.job           # Environment setup
│   ├── train_europe.job        # Training job (Europe)
│   └── train_scandinavia.job   # Training job (Scandinavia)
├── src/
│   ├── train.py                # Training entry point
│   ├── model/                  # Neural network modules
│   ├── data/                   # Data loading
│   └── callbacks/              # Training callbacks
├── environment.yaml            # Conda environment
└── pyproject.toml              # UV/pip dependencies
```

---

## Training

### Run with LightningCLI

```bash
# Default config
python -m src.train fit --config config/default_europe.yaml

# Override parameters
python -m src.train fit --config config/default_europe.yaml \
    --trainer.max_epochs 50 \
    --data.batch_size 32

# Quick test run
python -m src.train fit --config config/default_europe.yaml --trainer.fast_dev_run true
```

### Monitor with WandB

Training automatically logs to [Weights & Biases](https://wandb.ai). Set your API key:

```bash
wandb login
```

---

## Data Paths (Snellius)

| Dataset | Path |
|---------|------|
| Latents | `/projects/prjs1858/latents_europe_2018_2020.zarr` |
| HRES | `/projects/prjs1858/hres_europe_2018_2020.zarr` |
| Static | `/projects/prjs1858/static_hres_europe.zarr` |
| Statistics | `/projects/prjs1858/hres_europe_2018_2020_statistics.json` |