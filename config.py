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
# NOTE on alpha/threshold: the full feature library (own-return terms +
# macro terms) is standardized (zero mean, unit variance) before fitting,
# so a single alpha/threshold can regularize fairly across heterogeneous
# input scales (e.g. VIX ~10-30 vs. daily returns ~0.01). After
# standardization, fitted coefficients are on roughly the same order as
# the return-derivative target (~1e-3 to 1e-4), which is what these
# defaults are tuned against.
SINDY_CONFIG = {
    "poly_order": 2,
    "threshold": 0.0005,
    "alpha": 0.0001,
    "max_iter": 5000,     # small alpha needs many more coordinate-descent
                          # iterations to actually converge; 100 was too low
                          # and silently under-fit every single Lasso call
    "use_trig": False,
    "use_macro": True,   # include macro variables (VIX, yield curve, DXY,
                          # credit spreads, ...) as exogenous SINDy features
}

TOP_N = 3
