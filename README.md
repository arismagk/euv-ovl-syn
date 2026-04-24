# EUV-OVL-SYN

A physically grounded synthetic overlay benchmark dataset for EUV lithography research.

## Overview

EUV-OVL-SYN contains 200 wafers (4 lots × 50 wafers) of simulated overlay measurements on a 300 mm HVM wafer geometry. Each wafer has 1,420 measurement sites (71 fields × 20 dies/field), giving **284,000 overlay vectors** in total.

The dataset is generated from a two-component ground-truth model:

1. **Van den Brink model** — a 22-parameter polynomial capturing systematic inter- and intra-field overlay (translations, rotations, magnification, radial distortions up to 5th order). Parameters drift wafer-to-wafer according to four lot-specific physical scenarios.
2. **Nonlinear residual** — an analytically verified component outside the Brink polynomial column space, representing lens-heating-induced intra-field distortion. This is the learning target for residual correction models.

### Lot Drift Scenarios

| Lot | Scenario |
|-----|----------|
| 1 | Linear scanner heating (T_x creep, slight M_x drift) |
| 2 | Reticle alignment drift (T_y + M_y monotonic) |
| 3 | Multi-parameter monotonic drift (largest amplitude) |
| 4 | Sinusoidal chiller-cycling oscillation (~2.5 cycles) |

## Quick Start

```bash
# Install dependencies
pip install numpy pandas matplotlib

# Generate the dataset
python src/generate_synthetic_data.py

# Regenerate publication figures
python src/generate_figures.py
```

The generator creates `dat_synthetic/Lot<N>Wafer<M>.csv` for N ∈ {1,2,3,4} and M ∈ {1..50}.

## Dataset Schema

Each CSV file has the following columns:

| Column | Units | Description |
|--------|-------|-------------|
| `WaferCenterX_mm_` | mm | Field centre X (inter-field stage coordinate) |
| `WaferCenterY_mm_` | mm | Field centre Y (inter-field stage coordinate) |
| `DieCenterX_mm_` | mm | Die offset X within field (intra-field coordinate) |
| `DieCenterY_mm_` | mm | Die offset Y within field (intra-field coordinate) |
| `OverlayX` | nm | Measured overlay in X |
| `OverlayY` | nm | Measured overlay in Y |

This schema is compatible with the HVM 300 mm wafer geometry.

## Geometry

- Wafer diameter: 300 mm, fields within 138 mm radius
- Field pitch: 26 × 33 mm 
- 71 fields per wafer
- Die grid: 4 × 5 per field (offsets ±3, ±9 mm in X; ±6, ±12, 0 mm in Y)
- Metrology noise: σ = 0.15 nm (Gaussian)

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{magklaras2025euv,
  author    = {Magklaras, A. Tsirogiannis G., Alefragis, P., Gogos, C. and Birbas, A.},
  title     = {{EUV-OVL-SYN}: A Physically Grounded Synthetic Overlay Benchmark Dataset
               for {EUV} Lithography},
  booktitle = {Proc. IEEE},
  year      = {2025},
}
```

## License

MIT — see [LICENSE](LICENSE).
