"""
trainer.py  —  SINDy ETF Dynamics Trainer with Multi-Window Backtesting

Macro alignment convention
---------------------------
`prices` and `macro` arrays passed around this module are assumed to be
row-aligned to the SAME dates (same index, same length). Internally, for
a price slice prices[a:b], the return series has length (b - a - 1), and
we pair return[k] (the return ending on date a+k+1) with macro[a+k+1]
(the macro reading known as of that same date). That means, for a slice
prices[a:b], the correctly-aligned macro slice is macro[a+1:b]. This
keeps every training/test pairing strictly causal: a return is only ever
paired with macro data that was already known by the time that return
finished.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, validate_data
from sindy_model import SINDyModel, get_sindy_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def backtest_window(prices: np.ndarray, window: int, config: Dict,
                     macro: Optional[np.ndarray] = None) -> Dict:
    """
    Backtest SINDy model on a specific window size.
    Uses rolling walk-forward validation.

    prices: (n_samples, n_etfs) price matrix for a universe.
    macro:  (n_samples, n_macro) macro readings row-aligned with `prices`,
            or None to disable macro features.
    """
    n_samples = len(prices)

    if n_samples < window + 50:
        return {"error": "Insufficient data", "window": window}

    use_macro = config.get("use_macro", True) and macro is not None

    # Precompute the full return series once. returns_full[k] is the return
    # ending on the date at prices index k+1; macro_full[k] is the macro
    # reading known as of that same date (macro[1:] shifted to align).
    returns_full = np.diff(np.log(prices), axis=0)
    macro_full = macro[1:] if use_macro else None
    n_returns = len(returns_full)

    # Use 80% for training, 20% for testing (in return-index space)
    train_size = int(n_returns * 0.8)

    predictions = []
    actuals = []

    # Walk-forward: at each i, fit on a `window`-length slice of past
    # returns ending at i, then forecast the return that follows returns_full[i]
    # (i.e. returns_full[i+1]) using returns_full[i] itself as "today's" state.
    for i in range(train_size, n_returns - 1):
        train_start = i - window
        if train_start < 0:
            continue

        r_train = returns_full[train_start:i]
        if len(r_train) < 12:
            continue

        try:
            X_pred_train = r_train[:-1]
            Y_target_train = r_train[1:]
            Z_train = macro_full[train_start:i - 1] if use_macro else None

            model = SINDyModel(
                poly_order=config.get("poly_order", 2),
                threshold=config.get("threshold", 0.0005),
                alpha=config.get("alpha", 0.0001),
                use_trig=config.get("use_trig", False),
                use_macro=use_macro,
            )
            model.fit(X_pred_train, Y_target_train, Z=Z_train)

            X_current = returns_full[i:i + 1]
            Z_current = macro_full[i:i + 1] if use_macro else None
            pred_return = model.predict(X_current, Z_current=Z_current, steps=1)[-1]
            actual_return = returns_full[i + 1]

            predictions.append(pred_return)
            actuals.append(actual_return)
        except Exception:
            continue

    if len(predictions) < 10:
        return {"error": "Not enough predictions", "window": window}

    # Calculate performance metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Correlation between predicted and actual
    correlation = np.corrcoef(predictions.flatten(), actuals.flatten())[0, 1]

    # Mean squared error
    mse = np.mean((predictions - actuals) ** 2)

    # Directional accuracy (sign prediction)
    pred_sign = np.sign(predictions.flatten())
    actual_sign = np.sign(actuals.flatten())
    directional_accuracy = np.mean(pred_sign == actual_sign)

    # Sharpe ratio of strategy (if we trade based on predictions)
    # Trade: long if pred > 0, short if pred < 0
    returns_strategy = actuals.flatten() * pred_sign
    sharpe = np.mean(returns_strategy) / (np.std(returns_strategy) + 1e-8) * np.sqrt(252)

    return {
        "window": window,
        "n_predictions": len(predictions),
        "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
        "mse": float(mse),
        "directional_accuracy": float(directional_accuracy),
        "sharpe": float(sharpe),
        "mean_return": float(np.mean(returns_strategy)),
        "std_return": float(np.std(returns_strategy)),
    }


def compute_ticker_picks(prices: np.ndarray, tickers: List[str], window: int,
                          sindy_config: Dict, top_n: int,
                          macro: Optional[np.ndarray] = None) -> Tuple[List[Dict], Dict]:
    """
    Run SINDy on every ticker in a universe using a specific training window,
    and return the top-N picks by predicted next-day return along with the
    full per-ticker result dict.

    macro: (n_samples, n_macro) macro readings row-aligned with `prices`,
           or None to disable macro features.
    """
    config_copy = sindy_config.copy()
    config_copy["window"] = window

    ticker_predictions = {}
    ticker_results = {}

    for i, ticker in enumerate(tickers):
        try:
            result = get_sindy_predictions(prices[:, i:i + 1], config_copy, macro=macro)
            ticker_predictions[ticker] = result["next_return"][0]
            ticker_results[ticker] = {
                "next_return": float(result["next_return"][0]),
                "confidence": result["confidence"],
                "sparsity": result["sparsity"],
                "active_features": result["active_features"][:8]
            }
        except Exception as e:
            logger.error(f"  Error on {ticker} (window={window}): {e}")
            ticker_predictions[ticker] = 0.0
            ticker_results[ticker] = {
                "next_return": 0.0,
                "confidence": "Low",
                "sparsity": 0.0,
                "active_features": []
            }

    sorted_picks = sorted(ticker_predictions.items(), key=lambda x: x[1], reverse=True)
    top_picks = sorted_picks[:top_n]

    picks = []
    for ticker, pred in top_picks:
        picks.append({
            "ticker": ticker,
            "expected_return": round(float(pred) * 100, 2),
            "confidence": ticker_results[ticker].get("confidence", "Low"),
            "sparsity": ticker_results[ticker].get("sparsity", 0),
            "active_features": ticker_results[ticker].get("active_features", [])
        })

    return picks, ticker_results


def run_trainer() -> Dict:
    """Main SINDy trainer with multi-window backtesting."""

    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}

    macro_names = list(macro_df.columns) if macro_df is not None else []
    use_macro_globally = config.SINDY_CONFIG.get("use_macro", True) and len(macro_names) > 0
    if use_macro_globally:
        logger.info(f"📈 Using macro features: {macro_names}")
    else:
        logger.info("📈 Macro features disabled or unavailable.")

    run_date = datetime.now().strftime("%Y-%m-%d")
    results = {
        "run_date": run_date,
        "top_picks": {},
        "backtest_results": {},
        "best_window": {},
        "universes": {},
        "window_picks": {}
    }

    # Test each window
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {universe_name}...")

        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            continue

        # Select this universe's tickers, and keep only rows where ALL of
        # them are non-null. Macro data must be filtered with the EXACT
        # SAME row mask so `prices` and `macro` stay row-aligned -- using
        # prices_df[available].dropna() alone (as before) could drop a
        # different set of rows per universe and silently desync macro data.
        universe_prices_df = prices_df[available]
        valid_mask = ~universe_prices_df.isna().any(axis=1)
        universe_prices_df = universe_prices_df[valid_mask]

        prices = universe_prices_df.values

        macro = None
        if use_macro_globally:
            universe_macro_df = macro_df.loc[universe_prices_df.index]
            macro = universe_macro_df.values

        if len(prices) < 200:
            logger.warning(f"Not enough data for {universe_name}")
            continue

        sindy_config = config.SINDY_CONFIG.copy()
        sindy_config["macro_names"] = macro_names
        if not use_macro_globally:
            sindy_config["use_macro"] = False

        # Backtest each window
        window_results = {}
        for window in config.WINDOWS:
            logger.info(f"  Testing window {window}...")
            result = backtest_window(prices, window, sindy_config, macro=macro)

            if "error" not in result:
                window_results[window] = result
                logger.info(f"    Correlation: {result['correlation']:.3f}, "
                           f"Directional: {result['directional_accuracy']:.2%}, "
                           f"Sharpe: {result['sharpe']:.2f}")
            else:
                logger.warning(f"    {result['error']}")

        # Find best window -- selected by RETURN-PREDICTION quality
        # (correlation between predicted and actual returns), not by
        # backtested Sharpe. See config.BEST_WINDOW_METRIC for why.
        if window_results:
            select_metric = config.BEST_WINDOW_METRIC
            best_window = max(window_results.items(),
                             key=lambda x: x[1].get(select_metric, -999))
            results["best_window"][universe_name] = {
                "window": best_window[0],
                "metrics": best_window[1],
                "selected_by": select_metric,
            }
            logger.info(f"  ✅ Best window for {universe_name}: {best_window[0]} "
                       f"(selected by {select_metric}={best_window[1][select_metric]:.4f}; "
                       f"Sharpe: {best_window[1]['sharpe']:.2f})")

        results["backtest_results"][universe_name] = window_results

        best_win = results["best_window"].get(universe_name, {}).get("window", 252)

        # Run SINDy on each ticker for EVERY window, so the UI can show ETF
        # picks per window, not just for the single "best" one.
        results["window_picks"][universe_name] = {}
        best_win_ticker_results = {}

        for window in config.WINDOWS:
            picks, ticker_results = compute_ticker_picks(
                prices, available, window, sindy_config, config.TOP_N, macro=macro
            )
            results["window_picks"][universe_name][window] = picks
            if window == best_win:
                best_win_ticker_results = ticker_results

        # Top picks / universe summary mirror the best window's results
        picks = results["window_picks"][universe_name].get(best_win, [])
        if not best_win_ticker_results:
            # Fallback in case best_win isn't in config.WINDOWS for some reason
            picks, best_win_ticker_results = compute_ticker_picks(
                prices, available, best_win, sindy_config, config.TOP_N, macro=macro
            )

        results["top_picks"][universe_name] = picks
        results["universes"][universe_name] = {
            "tickers": available,
            "best_window": best_win,
            "ticker_results": best_win_ticker_results
        }

        logger.info(f"  ✅ Top picks for {universe_name}:")
        for pick in picks:
            logger.info(f"     {pick['ticker']}: {pick['expected_return']}% ({pick['confidence']})")

    # Save results
    output_path = f"sindy_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n💾 Saved: {output_path}")

    # Upload to HuggingFace
    try:
        from push_results import upload_results
        upload_results(output_path, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")

    return results


if __name__ == "__main__":
    run_trainer()
