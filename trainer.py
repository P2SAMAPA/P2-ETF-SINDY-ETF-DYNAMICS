"""
trainer.py  —  SINDy ETF Dynamics Trainer
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List

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


def run_trainer() -> Dict:
    """Main SINDy trainer."""
    
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
        "universes": {}
    }
    
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {universe_name}...")
        
        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            continue
        
        prices = prices_df[available].dropna().values
        
        if len(prices) < 100:
            logger.warning(f"Not enough data for {universe_name}")
            continue
        
        # Run SINDy on each ticker
        ticker_predictions = {}
        ticker_results = {}
        
        for i, ticker in enumerate(available):
            try:
                result = get_sindy_predictions(prices[:, i:i+1], config.SINDY_CONFIG)
                ticker_predictions[ticker] = result["next_return"][0]
                ticker_results[ticker] = {
                    "next_return": float(result["next_return"][0]),
                    "confidence": result["confidence"],
                    "sparsity": result["sparsity"],
                    "active_features": result["active_features"]
                }
            except Exception as e:
                logger.error(f"  Error on {ticker}: {e}")
                ticker_predictions[ticker] = 0.0
                ticker_results[ticker] = {"next_return": 0.0, "confidence": "Low", "error": str(e)}
        
        # Sort by predicted return
        sorted_picks = sorted(ticker_predictions.items(), key=lambda x: x[1], reverse=True)
        top_picks = sorted_picks[:config.TOP_N]
        
        picks = []
        for ticker, pred in top_picks:
            picks.append({
                "ticker": ticker,
                "expected_return": round(float(pred) * 100, 2),
                "confidence": ticker_results[ticker].get("confidence", "Low"),
                "sparsity": ticker_results[ticker].get("sparsity", 0)
            })
        
        results["top_picks"][universe_name] = picks
        results["universes"][universe_name] = {
            "tickers": available,
            "ticker_results": ticker_results
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
