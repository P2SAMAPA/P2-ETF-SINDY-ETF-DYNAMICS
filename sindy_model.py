"""
SINDy-style sparse feature selection for ETF return dynamics.

Design notes (v3)
------------------
v1 fit d(return)/dt (via np.gradient) and integrated forward to forecast.
That has two structural problems for daily financial returns:

1. Differencing amplifies noise: for i.i.d.-ish noise, the standard
   deviation of a first difference is ~sqrt(2) times the standard
   deviation of the raw series. Meanwhile genuine macro-driven signal is
   in the LEVEL of returns, not in their day-to-day change. So
   differencing buries real macro signal under amplified noise -- in
   testing, this caused macro terms to be thresholded to zero even in a
   deliberately strong, obvious synthetic relationship.
2. The forward-integration seed (the most recent single return) is the
   same value regardless of training window size, so window size barely
   affected forecasts unless patched around (v2 seeded on the window
   mean as a workaround).

v3 instead fits a direct one-step predictive regression:

    return[t+1]  ~  f( return[t], macro[t] )

selected sparsely via Lasso + sequential thresholding over a candidate
feature library (own return, own return^2, macro levels, and
return*macro interaction terms), still standardized so heterogeneous
scales (VIX ~10-30 vs. returns ~0.01) are regularized fairly. This
avoids the noise-doubling problem, and window size now legitimately
affects only what it should: which training data is used to fit f (the
"today" state used at prediction time is, correctly, the same real
observation no matter the window -- what should and does vary by window
is the fitted relationship applied to it).

For a joint multi-ticker fit (used in backtesting), features are built
per-ticker only (no cross-ticker products): a universe of 36+ tickers
previously produced full pairwise polynomial cross-terms (~700 candidate
features fit on as few as ~125 training rows), a serious overfitting /
rank-deficiency risk. Per-ticker-only terms scale linearly instead of
quadratically with the number of tickers.
"""

import numpy as np
from sklearn.linear_model import Lasso
from typing import Dict, List, Optional


class SINDyModel:
    def __init__(self, poly_order: int = 2, threshold: float = 0.0005,
                 alpha: float = 0.0001, max_iter: int = 100,
                 use_trig: bool = False, use_macro: bool = True):
        self.poly_order = poly_order
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.use_trig = use_trig
        self.use_macro = use_macro

        self.coefficients = None
        self.feature_names: List[str] = []
        self.active_features: List[str] = []
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None
        self.residual_std_: float = 0.0
        self.n_features = 0
        self.n_macro = 0

    # ------------------------------------------------------------------ #
    # Library construction
    # ------------------------------------------------------------------ #
    def _build_raw_library(self, X: np.ndarray, Z: Optional[np.ndarray] = None) -> np.ndarray:
        """
        X: (n_samples, n_features) -- each column is one ticker's return
           (the PREDICTOR state, i.e. "today's" return).
        Z: (n_samples, n_macro) or None -- macro features contemporaneous
           with X (i.e. also known "today").
        """
        n_samples, n_features = X.shape
        blocks = [np.ones((n_samples, 1))]

        for order in range(1, self.poly_order + 1):
            blocks.append(X ** order)

        if self.use_trig:
            blocks.append(np.sin(X))
            blocks.append(np.cos(X))

        if self.use_macro and Z is not None and Z.shape[1] > 0:
            blocks.append(Z)
            for i in range(n_features):
                blocks.append(X[:, i:i + 1] * Z)

        return np.hstack(blocks)

    def _generate_feature_names(self, n_features: int, macro_names: Optional[List[str]] = None) -> List[str]:
        names = ["1"]
        for order in range(1, self.poly_order + 1):
            for i in range(n_features):
                names.append(f"x{i+1}" if order == 1 else f"x{i+1}^{order}")

        if self.use_trig:
            for i in range(n_features):
                names.append(f"sin(x{i+1})")
            for i in range(n_features):
                names.append(f"cos(x{i+1})")

        if self.use_macro and self.n_macro > 0:
            macro_names = macro_names or [f"m{j+1}" for j in range(self.n_macro)]
            names.extend(macro_names)
            for i in range(n_features):
                for m in macro_names:
                    names.append(f"x{i+1}*{m}")

        return names

    def _sparse_regression(self, theta: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Lasso fit + sequential thresholding, per output column."""
        n_features_lib = theta.shape[1]
        n_targets = Y.shape[1]
        coefficients = np.zeros((n_features_lib, n_targets))

        for k in range(n_targets):
            y = Y[:, k]
            active = np.ones(n_features_lib, dtype=bool)

            for _ in range(10):
                if active.sum() == 0:
                    break
                # alpha is small relative to typical Lasso defaults, which
                # makes the coordinate-descent objective very flat near the
                # optimum; the default tol (1e-4, relative to y's variance)
                # combined with max_iter=100 silently failed to converge on
                # essentially every call. A tighter/looser tol tradeoff:
                # loosen tol slightly so convergence is reached in practice,
                # while max_iter provides a generous ceiling.
                lasso = Lasso(alpha=self.alpha, max_iter=self.max_iter,
                               tol=1e-2, fit_intercept=False)
                lasso.fit(theta[:, active], y)
                coefs = lasso.coef_

                small = np.abs(coefs) < self.threshold
                if not small.any():
                    coefficients[active, k] = coefs
                    break

                active_idx = np.where(active)[0]
                active[active_idx[small]] = False
                if active.sum() == 0:
                    break
                coefficients[:, k] = 0
            else:
                if active.sum() > 0:
                    lasso = Lasso(alpha=self.alpha, max_iter=self.max_iter,
                                   tol=1e-2, fit_intercept=False)
                    lasso.fit(theta[:, active], y)
                    coefficients[active, k] = lasso.coef_

        return coefficients

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, Y: np.ndarray, Z: Optional[np.ndarray] = None,
            macro_names: Optional[List[str]] = None) -> Dict:
        """
        X: (n_samples, n_features) predictor state ("today's" return per ticker).
        Y: (n_samples, n_features) target -- the NEXT period's return per
           ticker, i.e. Y[i] should equal the ticker's return that follows
           the one in X[i].
        Z: (n_samples, n_macro) macro features contemporaneous with X.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n_samples, n_features = X.shape
        self.n_features = n_features
        self.n_macro = Z.shape[1] if (self.use_macro and Z is not None) else 0

        raw_theta = self._build_raw_library(X, Z if self.use_macro else None)

        mean = raw_theta.mean(axis=0)
        std = raw_theta.std(axis=0)
        std[std < 1e-8] = 1.0
        mean[0] = 0.0  # keep the bias column as a literal constant 1
        std[0] = 1.0
        theta = (raw_theta - mean) / std
        self.feature_mean_ = mean
        self.feature_std_ = std

        self.coefficients = self._sparse_regression(theta, Y)
        self.feature_names = self._generate_feature_names(n_features, macro_names)

        active_mask = np.abs(self.coefficients).sum(axis=1) > 1e-10
        self.active_features = [self.feature_names[i] for i in range(len(self.feature_names)) if active_mask[i]]

        fitted_Y = theta @ self.coefficients
        residuals = Y - fitted_Y
        self.residual_std_ = float(np.std(residuals)) if residuals.size else 0.0

        n_active = int(active_mask.sum())
        sparsity = 1.0 - (n_active / len(self.feature_names)) if self.feature_names else 1.0

        return {
            "n_features": n_active,
            "sparsity": sparsity,
            "active_features": self.active_features,
            "residual_std": self.residual_std_,
        }

    def predict(self, X_current: np.ndarray, Z_current: Optional[np.ndarray] = None,
                steps: int = 1) -> np.ndarray:
        """
        X_current: (1, n_features) the most recently known state (e.g. the
                   most recent actual daily return).
        Z_current: (1, n_macro) the most recently known macro reading.
        steps: number of periods to forecast forward. For steps > 1, each
               predicted return becomes the "current" state fed into the
               next step; macro is held at Z_current throughout (no macro
               dynamics are modeled -- macro is treated as persistent).
        """
        if X_current.ndim == 1:
            X_current = X_current.reshape(1, -1)
        X_cur = X_current[-1:].copy()
        Z_cur = Z_current[-1:].copy() if (self.use_macro and Z_current is not None and Z_current.shape[0] > 0) else None

        predictions = []
        for _ in range(steps):
            raw_theta = self._build_raw_library(X_cur, Z_cur)
            theta = (raw_theta - self.feature_mean_) / self.feature_std_
            Y_pred = theta @ self.coefficients
            predictions.append(Y_pred)
            X_cur = Y_pred  # the forecast becomes "today's known return" for the next step

        return np.vstack(predictions)

    def get_governing_equations(self, target_names: Optional[List[str]] = None) -> List[str]:
        if self.coefficients is None:
            return []
        n_targets = self.coefficients.shape[1]
        target_names = target_names or [f"x{i+1}(t+1)" for i in range(n_targets)]
        equations = []
        for k in range(n_targets):
            terms = []
            for i, name in enumerate(self.feature_names):
                c = self.coefficients[i, k]
                if abs(c) > 1e-10:
                    terms.append(f"{c:+.4f}*{name}")
            eq = f"{target_names[k]} = " + (" ".join(terms) if terms else "0")
            equations.append(eq)
        return equations


def get_sindy_predictions(prices: np.ndarray, config: Dict,
                           macro: Optional[np.ndarray] = None) -> Dict:
    """
    Fit a SINDy-style model on a single ticker's price history (optionally
    with aligned macro features) and produce a 1-step-ahead return forecast.

    prices: (n_samples, 1) price series for one ticker.
    macro:  (n_samples, n_macro) macro readings row-aligned to the SAME
            dates as `prices` (row i of macro corresponds to row i of
            prices). May be None to disable macro features for this call.
    """
    returns_full = np.diff(np.log(prices), axis=0)  # (n-1, 1)

    use_macro = config.get("use_macro", True) and macro is not None
    macro_full = macro[1:] if use_macro else None  # aligned with returns_full

    window = config.get("window", 252)
    if len(returns_full) > window:
        r = returns_full[-window:]
        z = macro_full[-window:] if use_macro else None
    else:
        r = returns_full
        z = macro_full

    if len(r) < 12:
        raise ValueError("Not enough data to fit a predictive model.")

    X_pred = r[:-1]
    Y_target = r[1:]
    Z_pred = z[:-1] if use_macro else None

    model = SINDyModel(
        poly_order=config.get("poly_order", 2),
        threshold=config.get("threshold", 0.0005),
        alpha=config.get("alpha", 0.0001),
        max_iter=config.get("max_iter", 100),
        use_trig=config.get("use_trig", False),
        use_macro=use_macro,
    )

    fit_result = model.fit(X_pred, Y_target, Z=Z_pred, macro_names=config.get("macro_names"))

    X_current = r[-1:]
    Z_current = z[-1:] if use_macro else None
    next_return = model.predict(X_current, Z_current=Z_current, steps=1)[-1]

    residual = model.residual_std_
    if residual < 0.0005:
        confidence = "High"
    elif residual < 0.0015:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "next_return": next_return,
        "confidence": confidence,
        "sparsity": fit_result["sparsity"],
        "active_features": fit_result["active_features"],
        "residual_std": residual,
    }
