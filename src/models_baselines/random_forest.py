"""
Random Forest Baseline for Super-Resolution.

Uses sklearn RandomForestRegressor trained on low-resolution features + coordinates
to predict high-resolution values. Can use either Aurora predictions or latent features.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from typing import Dict, Optional, List
import joblib
from pathlib import Path
from tqdm import tqdm


class RandomForestBaseline:
    """
    Random Forest baseline for super-resolution.
    
    Trains a Random Forest to predict high-resolution values from:
    - Interpolated low-res features at query positions
    - Query position coordinates (lat, lon)
    - Optional: auxiliary features (z, lsm, slt)
    
    This is a non-parametric ML baseline that can capture non-linear relationships.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        n_jobs: int = -1,
        random_state: int = 42,
        use_coordinates: bool = True,
        use_interpolated_features: bool = True,
    ):
        """
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in leaf
            n_jobs: Number of parallel jobs (-1 = all cores)
            random_state: Random seed for reproducibility
            use_coordinates: Include (lat, lon) as input features
            use_interpolated_features: Include interpolated low-res values as input
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_coordinates = use_coordinates
        self.use_interpolated_features = use_interpolated_features
        
        self.model = None
        self.is_fitted = False
        
        # For interpolating low-res features to query positions
        from .linear_interpolation import LinearInterpolationBaseline
        self.interpolator = LinearInterpolationBaseline(method="linear")
    
    def _build_features(
        self,
        low_res_data: np.ndarray,
        low_res_lat: np.ndarray,
        low_res_lon: np.ndarray,
        query_lat: np.ndarray,
        query_lon: np.ndarray,
        query_aux: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build feature matrix for Random Forest.
        
        Args:
            low_res_data: [lat, lon, channels] low-res data
            low_res_lat, low_res_lon: Low-res coordinate arrays
            query_lat, query_lon: Query position coordinates
            query_aux: Optional [Q, num_aux] auxiliary features
            
        Returns:
            Feature matrix [Q, num_features]
        """
        features = []
        
        # Interpolate low-res features to query positions
        if self.use_interpolated_features:
            interp_features = self.interpolator.interpolate(
                low_res_data, low_res_lat, low_res_lon,
                query_lat, query_lon
            )
            # Reshape to [Q, channels]
            if interp_features.ndim == 3:
                interp_features = interp_features.reshape(-1, interp_features.shape[-1])
            elif interp_features.ndim == 2 and query_lat.ndim == 1:
                interp_features = interp_features.ravel()[:, None]
            features.append(interp_features)
        
        # Add coordinates
        if self.use_coordinates:
            if query_lat.ndim == 1 and query_lon.ndim == 1:
                lon_grid, lat_grid = np.meshgrid(query_lon, query_lat)
                coords = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
            else:
                coords = np.stack([query_lat.ravel(), query_lon.ravel()], axis=-1)
            features.append(coords)
        
        # Add auxiliary features
        if query_aux is not None:
            features.append(query_aux)
        
        return np.concatenate(features, axis=-1)
    
    def fit(
        self,
        dataset,
        num_samples: Optional[int] = 100,
        samples_per_timestep: int = 1000,
    ) -> "RandomForestBaseline":
        """
        Fit Random Forest on dataset samples.
        
        Args:
            dataset: EraPredictionHresDataset or EraLatentHresDataset
            num_samples: Number of timesteps to use for training
            samples_per_timestep: Number of query points to sample per timestep
            
        Returns:
            self
        """
        n_samples = len(dataset) if num_samples is None else min(num_samples, len(dataset))
        
        # Get coordinate arrays from dataset
        low_res_lat = dataset.pred_lat if hasattr(dataset, 'pred_lat') else dataset.latent_lat
        low_res_lon = dataset.pred_lon if hasattr(dataset, 'pred_lon') else dataset.latent_lon
        hres_lat = dataset.hres_lat
        hres_lon = dataset.hres_lon
        
        all_X = []
        all_y = []
        
        print(f"Building training data from {n_samples} samples...")
        for idx in tqdm(range(n_samples), desc="Collecting training data"):
            sample = dataset[idx]
            
            # Get low-res features
            if 'low_res_features' in sample:
                low_res_data = sample['low_res_features'].numpy()
                low_res_data = low_res_data.reshape(len(low_res_lat), len(low_res_lon), -1)
            else:
                low_res_data = sample['latents'].numpy()
                low_res_data = low_res_data.reshape(len(low_res_lat), len(low_res_lon), -1)
            
            # Get targets (already sampled or full grid)
            query_fields = sample['query_fields'].numpy()  # [Q, num_vars]
            query_pos = sample['query_pos'].numpy()  # [Q, 2]
            
            # Subsample if needed
            n_points = query_fields.shape[0]
            if samples_per_timestep < n_points:
                indices = np.random.choice(n_points, samples_per_timestep, replace=False)
                query_fields = query_fields[indices]
                query_pos = query_pos[indices]
            
            query_aux = sample.get('query_auxiliary_features')
            if query_aux is not None:
                query_aux = query_aux.numpy()
                if samples_per_timestep < n_points:
                    query_aux = query_aux[indices]
            
            # Denormalize query positions to get lat/lon
            # Assuming normalized to [-1, 1]
            q_lat = query_pos[:, 0] * (hres_lat.max() - hres_lat.min()) / 2 + (hres_lat.max() + hres_lat.min()) / 2
            q_lon = query_pos[:, 1] * (hres_lon.max() - hres_lon.min()) / 2 + (hres_lon.max() + hres_lon.min()) / 2
            
            # Build features
            X = self._build_features(
                low_res_data, low_res_lat, low_res_lon,
                q_lat, q_lon, query_aux
            )
            
            all_X.append(X)
            all_y.append(query_fields)
        
        X_train = np.concatenate(all_X, axis=0)
        y_train = np.concatenate(all_y, axis=0)
        
        print(f"Training Random Forest on {X_train.shape[0]} samples, {X_train.shape[1]} features...")
        
        # Create and fit model
        base_rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
        )
        
        # Use MultiOutputRegressor for multiple target variables
        if y_train.shape[1] > 1:
            self.model = MultiOutputRegressor(base_rf, n_jobs=1)  # Already parallel inside RF
        else:
            self.model = base_rf
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        print("Training complete!")
        return self
    
    def predict(
        self,
        low_res_data: np.ndarray,
        low_res_lat: np.ndarray,
        low_res_lon: np.ndarray,
        query_lat: np.ndarray,
        query_lon: np.ndarray,
        query_aux: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict high-resolution values.
        
        Args:
            low_res_data: [lat, lon, channels] low-res data
            low_res_lat, low_res_lon: Low-res coordinates
            query_lat, query_lon: Query positions
            query_aux: Optional auxiliary features
            
        Returns:
            Predicted values [Q, num_vars]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = self._build_features(
            low_res_data, low_res_lat, low_res_lon,
            query_lat, query_lon, query_aux
        )
        
        return self.model.predict(X)
    
    def evaluate_on_dataset(
        self,
        dataset,
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate Random Forest on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            num_samples: Number of samples to evaluate
            
        Returns:
            Dict with MSE and RMSE metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        n_samples = len(dataset) if num_samples is None else min(num_samples, len(dataset))
        
        low_res_lat = dataset.pred_lat if hasattr(dataset, 'pred_lat') else dataset.latent_lat
        low_res_lon = dataset.pred_lon if hasattr(dataset, 'pred_lon') else dataset.latent_lon
        hres_lat = dataset.hres_lat
        hres_lon = dataset.hres_lon
        
        all_mse = []
        
        for idx in tqdm(range(n_samples), desc="Random Forest evaluation"):
            sample = dataset[idx]
            
            if 'low_res_features' in sample:
                low_res_data = sample['low_res_features'].numpy()
                low_res_data = low_res_data.reshape(len(low_res_lat), len(low_res_lon), -1)
            else:
                low_res_data = sample['latents'].numpy()
                low_res_data = low_res_data.reshape(len(low_res_lat), len(low_res_lon), -1)
            
            query_fields = sample['query_fields'].numpy()
            query_pos = sample['query_pos'].numpy()
            
            query_aux = sample.get('query_auxiliary_features')
            if query_aux is not None:
                query_aux = query_aux.numpy()
            
            # Denormalize positions
            q_lat = query_pos[:, 0] * (hres_lat.max() - hres_lat.min()) / 2 + (hres_lat.max() + hres_lat.min()) / 2
            q_lon = query_pos[:, 1] * (hres_lon.max() - hres_lon.min()) / 2 + (hres_lon.max() + hres_lon.min()) / 2
            
            # Predict
            pred = self.predict(
                low_res_data, low_res_lat, low_res_lon,
                q_lat, q_lon, query_aux
            )
            
            # Compute MSE
            mse = np.mean((pred - query_fields) ** 2)
            all_mse.append(mse)
        
        mean_mse = np.mean(all_mse)
        
        return {
            "mse": float(mean_mse),
            "rmse": float(np.sqrt(mean_mse)),
        }
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> "RandomForestBaseline":
        """Load model from disk."""
        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"Model loaded from {path}")
        return self
