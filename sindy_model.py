"""
sindy_model.py  —  SINDy Model for ETF Dynamics
"""

import numpy as np
from scipy import linalg
from scipy.optimize import nnls
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, List, Tuple, Optional


class SINDyModel:
    """
    Sparse Identification of Nonlinear Dynamics (SINDy).
    Learns governing equations: dX/dt = Θ(X) · Ξ
    """
    
    def __init__(self, poly_order: int = 2, threshold: float = 0.01, 
                 alpha: float = 0.1, use_trig: bool = False):
        self.poly_order = poly_order
        self.threshold = threshold
        self.alpha = alpha
        self.use_trig = use_trig
        self.coefficients = None
        self.feature_names = None
        self.feature_library = None
        
    def _build_library(self, X: np.ndarray) -> np.ndarray:
        """
        Build candidate function library Θ(X).
        Includes: constants, polynomials, and optionally trig functions.
        """
        n_samples, n_features = X.shape
        library = []
        
        # Constant term (ones)
        library.append(np.ones((n_samples, 1)))
        
        # Polynomial features
        poly = PolynomialFeatures(degree=self.poly_order, include_bias=False)
        poly_features = poly.fit_transform(X)
        library.append(poly_features)
        
        # Trig functions (optional)
        if self.use_trig:
            for i in range(n_features):
                library.append(np.sin(X[:, i:i+1]))
                library.append(np.cos(X[:, i:i+1]))
        
        # Combine
        theta = np.hstack(library)
        
        # Store feature names for interpretability
        self.feature_library = library
        self.feature_names = self._generate_feature_names(n_features)
        
        return theta
    
    def _generate_feature_names(self, n_features: int) -> List[str]:
        """Generate human-readable feature names."""
        names = ['1']  # constant
        
        # Polynomial terms
        for i in range(n_features):
            names.append(f'x{i+1}')
        
        for i in range(n_features):
            for j in range(i, n_features):
                names.append(f'x{i+1}*x{j+1}')
        
        if self.poly_order >= 3:
            for i in range(n_features):
                for j in range(i, n_features):
                    for k in range(j, n_features):
                        names.append(f'x{i+1}*x{j+1}*x{k+1}')
        
        # Trig terms
        if self.use_trig:
            for i in range(n_features):
                names.append(f'sin(x{i+1})')
                names.append(f'cos(x{i+1})')
        
        return names
    
    def _compute_derivative(self, X: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Compute time derivative using finite differences."""
        return np.gradient(X, dt, axis=0)
    
    def _sparse_regression(self, theta: np.ndarray, dX: np.ndarray) -> np.ndarray:
        """
        Sparse regression to find Ξ.
        Uses Lasso with thresholding (Sequential Thresholded Least Squares).
        """
        n_features = dX.shape[1]
        coefficients = np.zeros((theta.shape[1], n_features))
        
        for i in range(n_features):
            # Initial Lasso fit
            lasso = Lasso(alpha=self.alpha, max_iter=1000, fit_intercept=False)
            lasso.fit(theta, dX[:, i])
            coef = lasso.coef_.copy()
            
            # Sequential thresholding
            for _ in range(10):
                mask = np.abs(coef) > self.threshold
                if not mask.any():
                    break
                # Refit using only selected features
                theta_selected = theta[:, mask]
                coef_selected, _, _, _ = linalg.lstsq(theta_selected, dX[:, i])
                coef_new = np.zeros_like(coef)
                coef_new[mask] = coef_selected
                coef = coef_new
            
            coefficients[:, i] = coef
        
        return coefficients
    
    def fit(self, X: np.ndarray, dt: float = 1.0) -> Dict:
        """
        Fit SINDy model to data.
        X: (n_samples, n_features) time series data
        dt: time step
        """
        # Compute derivative
        dX = self._compute_derivative(X, dt)
        
        # Build library
        theta = self._build_library(X)
        
        # Sparse regression
        self.coefficients = self._sparse_regression(theta, dX)
        
        # Get active features
        active = np.abs(self.coefficients).sum(axis=1) > 1e-6
        self.active_features = [self.feature_names[i] for i in range(len(self.feature_names)) if active[i]]
        
        return {
            "coefficients": self.coefficients,
            "active_features": self.active_features,
            "n_features": len(self.active_features),
            "sparsity": 1 - len(self.active_features) / len(self.feature_names)
        }
    
    def predict(self, X: np.ndarray, dt: float = 1.0, steps: int = 1) -> np.ndarray:
        """
        Predict future states by integrating the discovered dynamics.

        The forward integration is seeded with the MEAN of X (the state
        representative of the window/regime that was fit), not just the
        single most-recent raw observation. Seeding on the last raw value
        made predictions dominated by one noisy day's return and made them
        almost independent of the training window size, since the most
        recent observation is identical no matter how far back training
        data goes. Seeding on the window mean makes the training window
        size actually influence the forecast, while for callers that pass
        in a single-row X (e.g. the 1-step walk-forward backtest), the mean
        of one row is just that row, so behavior there is unchanged.
        """
        X_current = X.mean(axis=0, keepdims=True)
        predictions = [X_current]
        
        for _ in range(steps):
            # Build library for current state
            theta = self._build_library(X_current)
            
            # Compute derivative
            dX = theta @ self.coefficients
            
            # Euler integration
            X_next = X_current + dt * dX
            predictions.append(X_next)
            X_current = X_next
        
        return np.vstack(predictions[1:])
    
    def get_governing_equations(self) -> str:
        """Return human-readable governing equations."""
        if self.coefficients is None:
            return "Model not fitted yet."
        
        equations = []
        for i in range(self.coefficients.shape[1]):
            terms = []
            for j, coef in enumerate(self.coefficients[:, i]):
                if abs(coef) > 1e-6:
                    terms.append(f"{coef:.4f} * {self.feature_names[j]}")
            if terms:
                equations.append(f"d/dt x{i+1} = " + " + ".join(terms))
            else:
                equations.append(f"d/dt x{i+1} = 0")
        
        return "\n".join(equations)


def get_sindy_predictions(prices: np.ndarray, config: Dict) -> Dict:
    """
    Run SINDy analysis on ETF price data and generate predictions.
    """
    # Calculate returns
    returns = np.diff(np.log(prices), axis=0)
    
    # Fit SINDy model
    model = SINDyModel(
        poly_order=config.get("poly_order", 2),
        threshold=config.get("threshold", 0.01),
        alpha=config.get("alpha", 0.1),
        use_trig=config.get("use_trig", False)
    )
    
    # Use last window of data
    window = config.get("window", 252)
    if len(returns) > window:
        X = returns[-window:]
    else:
        X = returns
    
    # Fit model
    fit_results = model.fit(X)
    
    # Predict next step
    next_return = model.predict(X, steps=1)[-1]
    
    # Calculate confidence (based on model fit quality)
    residual = np.std(np.diff(X, axis=0) - (model._build_library(X[:-1]) @ model.coefficients))
    confidence = "High" if residual < 0.01 else "Medium" if residual < 0.02 else "Low"
    
    return {
        "next_return": next_return,
        "confidence": confidence,
        "coefficients": fit_results["coefficients"],
        "active_features": fit_results["active_features"],
        "sparsity": fit_results["sparsity"],
        "governing_equations": model.get_governing_equations()
    }
