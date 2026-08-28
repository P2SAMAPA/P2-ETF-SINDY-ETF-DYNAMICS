"""
config.py  —  Configuration for SINDy ETF Dynamics
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-sindy-etf-dynamics-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]
}

# Multiple windows for analysis
WINDOWS = [126, 252, 504, 756, 1008]

# SINDy configuration
# NOTE: alpha/threshold are on the scale of daily log-returns (~0.005-0.02).
# The previous values (alpha=0.1, threshold=0.01) were an order of magnitude
# too large for that scale, so the Lasso step zeroed out every coefficient on
# real data -> the model always degenerated to "predict no change", which is
# why every window size and every ticker produced identical results.
SINDY_CONFIG = {
    "poly_order": 2,
    "threshold": 0.0005,
    "alpha": 0.0001,
    "max_iter": 100,
    "use_trig": False,
}

TOP_N = 3
