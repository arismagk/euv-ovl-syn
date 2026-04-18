# CLAUDE.md

## Project Overview

**EUV-OVL-SYN** is a standalone dataset-generation repository producing a synthetic EUV lithography overlay benchmark dataset. It is self-contained — it does not depend on any ML training code.

## Repository Contents

| Path | Purpose |
|---|---|
| `src/generate_synthetic_data.py` | Dataset generator (200 wafers, 4 lots) |
| `src/model.py` | Brink/van den Brink 22-parameter overlay model |
| `src/generate_figures.py` | Publication figures for the dataset paper |
| `dat_synthetic/` | Generated CSV files (excluded from git) |
| `paper/dataset_paper.tex` | Self-contained IEEE conference paper (LaTeX) |
| `paper/figs/` | PDF figures included in the paper |

## Running

All commands from the repo root:

```bash
# Generate the dataset (creates dat_synthetic/Lot<N>Wafer<M>.csv)
python src/generate_synthetic_data.py

# Regenerate publication figures (requires dat_synthetic/ to exist)
python src/generate_figures.py
```

## Dataset Schema

Each `Lot<N>Wafer<M>.csv` has columns:
- `WaferCenterX_mm_`, `WaferCenterY_mm_` — inter-field stage coordinates (mm)
- `DieCenterX_mm_`, `DieCenterY_mm_` — intra-field die offset (mm)
- `OverlayX`, `OverlayY` — measured overlay (nm)

## Dependencies

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
```

No ML frameworks required.
