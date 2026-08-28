# P2-SINDY-ETF-DYNAMICS

## Sparse Identification of Nonlinear Dynamics for ETF Selection

### Concept

This repository applies **SINDy (Sparse Identification of Nonlinear Dynamics)** to discover governing equations from ETF price data.
dX/dt = Θ(X) · Ξ

text

Where:
- `Θ(X)` is a library of candidate functions (polynomials, trig, etc.)
- `Ξ` is a sparse coefficient matrix found via sparse regression

### How It Works

1. **Load Data**: ETF prices from HuggingFace dataset
2. **Compute Returns**: Log returns for each ETF
3. **Build Library**: Polynomial and trigonometric candidate functions
4. **Sparse Regression**: Identify the most important terms
5. **Predict**: Use discovered dynamics to forecast returns
6. **Rank**: Select top ETFs by predicted return# P2-ETF-SINDY-ETF-DYNAMICS
