# Data Directory Policy & Attribution

This directory contains reference datasets and samples used by VolSurfaceLab.

## Data Governance & Redistribution Policy
- **Synthetic Quotes (`data/samples/synthetic_quotes.csv`):** Generated deterministically from a mathematically certified SSVI volatility surface via `volsurfacelab.data.synthetic`. This file contains simulated data with known ground truth and carries no copyright or redistribution restrictions.
- **External Market Data:** VolSurfaceLab does **NOT** redistribute proprietary, licensed, or commercial real-time market data (such as Cboe DataShop proprietary feeds or exchange NBBO feeds).
- **Importing Authorised Data:** Users wishing to calibrate against historical European index quotes (e.g. SPX or STOXX) may import their authorized CSV files conforming to the schema documented in `docs/DATA_CONTRACT.md`.
