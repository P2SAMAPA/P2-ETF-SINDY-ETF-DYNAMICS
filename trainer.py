"""
trainer.py  —  SINDy ETF Dynamics Trainer with Multi-Window Backtesting
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
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


def backtest_window(prices: np.ndarray, window: int, config: Dict) -> Dict:
    """
    Backtest SINDy model on a specific window size.
    Uses rolling walk-forward validation.
    """
    n_samples = len(prices)
    n_etfs = prices.shape[1]
    
    if n_samples < window + 50:
        return {"error": "Insufficient data", "window": window}
    
    # Use 80% for training, 20% for testing
    train_size = int(n_samples * 0.8)
    test_size = n_samples - train_size
    
    # Rolling predictions
    predictions = []
    actuals = []
    
    # Walk-forward: train on window, predict next day, roll forward
    for i in range(train_size, n_samples - 1):
        # Training data: from (i - window) to i
        train_start = i - window
        if train_start < 0:
            continue
        
        X_train = prices[train_start:i]
        X_test = prices[i-1:i+1]  # need 2 rows to compute the return that seeds the prediction
        
        # Fit SINDy on training data
        try:
            # Calculate returns
            returns_train = np.diff(np.log(X_train), axis=0)
            if len(returns_train) < 10:
                continue
            
            # Fit model
            model = SINDyModel(
                poly_order=config.get("poly_order", 2),
                threshold=config.get("threshold", 0.01),
                alpha=config.get("alpha", 0.1),
                use_trig=config.get("use_trig", False)
            )
            model.fit(returns_train)
            
            # Predict next return
            returns_test = np.diff(np.log(X_test), axis=0)
            if len(returns_test) > 0:
                pred_return = model.predict(returns_test, steps=1)[-1]
                actual_return = np.diff(np.log(prices[i:i+2]), axis=0)[0]
                
                predictions.append(pred_return)
                actuals.append(actual_return)
        except:
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
        "std_return": float(np.std(returns_strategy))
    }


def compute_ticker_picks(prices: np.ndarray, tickers: List[str], window: int,
                          sindy_config: Dict, top_n: int) -> Tuple[List[Dict], Dict]:
    """
    Run SINDy on every ticker in a universe using a specific training window,
    and return the top-N picks by predicted next-day return along with the
    full per-ticker result dict.
    """
    config_copy = sindy_config.copy()
    config_copy["window"] = window

    ticker_predictions = {}
    ticker_results = {}

    for i, ticker in enumerate(tickers):
        try:
            result = get_sindy_predictions(prices[:, i:i + 1], config_copy)
            ticker_predictions[ticker] = result["next_return"][0]
            ticker_results[ticker] = {
                "next_return": float(result["next_return"][0]),
                "confidence": result["confidence"],
                "sparsity": result["sparsity"],
                "active_features": result["active_features"][:5]
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
        
        prices = prices_df[available].dropna().values
        
        if len(prices) < 200:
            logger.warning(f"Not enough data for {universe_name}")
            continue
        
        # Backtest each window
        window_results = {}
        for window in config.WINDOWS:
            logger.info(f"  Testing window {window}...")
            result = backtest_window(prices, window, config.SINDY_CONFIG)
            
            if "error" not in result:
                window_results[window] = result
                logger.info(f"    Correlation: {result['correlation']:.3f}, "
                           f"Directional: {result['directional_accuracy']:.2%}, "
                           f"Sharpe: {result['sharpe']:.2f}")
            else:
                logger.warning(f"    {result['error']}")
        
        # Find best window
        if window_results:
            best_window = max(window_results.items(), 
                             key=lambda x: x[1].get('sharpe', -999))
            results["best_window"][universe_name] = {
                "window": best_window[0],
                "metrics": best_window[1]
            }
            logger.info(f"  ✅ Best window for {universe_name}: {best_window[0]} "
                       f"(Sharpe: {best_window[1]['sharpe']:.2f})")
        
        results["backtest_results"][universe_name] = window_results
        
        best_win = results["best_window"].get(universe_name, {}).get("window", 252)
        
        # Run SINDy on each ticker for EVERY window, so the UI can show ETF
        # picks per window, not just for the single "best" one.
        results["window_picks"][universe_name] = {}
        best_win_ticker_results = {}
        
        for window in config.WINDOWS:
            picks, ticker_results = compute_ticker_picks(
                prices, available, window, config.SINDY_CONFIG, config.TOP_N
            )
            results["window_picks"][universe_name][window] = picks
            if window == best_win:
                best_win_ticker_results = ticker_results
        
        # Top picks / universe summary mirror the best window's results
        picks = results["window_picks"][universe_name].get(best_win, [])
        if not best_win_ticker_results:
            # Fallback in case best_win isn't in config.WINDOWS for some reason
            picks, best_win_ticker_results = compute_ticker_picks(
                prices, available, best_win, config.SINDY_CONFIG, config.TOP_N
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
