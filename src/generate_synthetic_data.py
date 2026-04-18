"""
Synthetic EUV overlay dataset generator.

Produces physically realistic 300 mm wafer overlay data with:
  - Proper field / die geometry derived from ASML scanner step-and-repeat
  - Strong lot-level Brink fingerprint (consistent beta, CV < 0.15)
  - Wafer-to-wafer temporal drift (varies by lot)
  - Nonlinear spatial residual outside the Brink polynomial basis (for PINN)
  - Gaussian metrology noise (σ = 0.10 nm)

Output layout: dat_synthetic/Lot<N>Wafer<M>.csv (same schema as real data)
Expected aggregate performance after synthesis:
  Raw 3σ     ~ 5-8 nm
  Brink 3σ   ~ 1.5-3 nm  (cross-wafer, n_train=10)
  PINN 3σ    ~ 0.5-1.5 nm

Usage:
    python src/generate_synthetic_data.py
    python src/generate_synthetic_data.py --outdir dat_synthetic --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model import build_design_matrix, PARAM_NAMES

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RNG_SEED = 42

# ---------------------------------------------------------------------------
# Scanner / wafer geometry constants
# ---------------------------------------------------------------------------

FIELD_STEP_X   = 26.0    # mm  (standard ASML NXE field pitch X)
FIELD_STEP_Y   = 33.0    # mm  (standard ASML NXE field pitch Y)
MAX_FIELD_R    = 138.0   # mm  field centres must sit within this radius
WAFER_RADIUS_R = 300.0   # mm  normalisation radius (matches constants.py)

# Intra-field die grid (mm) — 4×5 sites per field → 20 sites/field
DIE_X = np.array([-9.0, -3.0,  3.0,  9.0])           # 4 columns
DIE_Y = np.array([-12.0, -6.0,  0.0,  6.0, 12.0])    # 5 rows

N_LOTS   = 4
N_WAFERS = 50
MEAS_NOISE_SIGMA = 0.15  # nm  (1σ metrology noise)


# ---------------------------------------------------------------------------
# 1. Scanner field grid
# ---------------------------------------------------------------------------

def make_field_grid(
    step_x: float = FIELD_STEP_X,
    step_y: float = FIELD_STEP_Y,
    max_r:  float = MAX_FIELD_R,
) -> np.ndarray:
    """Return (N_fields, 2) array of field centres within max_r."""
    xs = np.arange(-5 * step_x, 5 * step_x + 1, step_x)
    ys = np.arange(-5 * step_y, 5 * step_y + 1, step_y)
    grid = np.array([[x, y] for x in xs for y in ys
                     if np.sqrt(x**2 + y**2) <= max_r])
    return grid


def make_site_table(field_grid: np.ndarray, die_x: np.ndarray, die_y: np.ndarray) -> np.ndarray:
    """
    Return (N_sites, 4) array: [WaferCenterX, WaferCenterY, DieCenterX, DieCenterY].
    WaferCenter = field centre; DieCenter = die offset within field.
    """
    rows = []
    for fx, fy in field_grid:
        for dx in die_x:
            for dy in die_y:
                rows.append([fx, fy, dx, dy])
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# 2. Brink parameter sets per lot (nm and nm/mm units)
# ---------------------------------------------------------------------------

# Baseline beta vector (22 values in PARAM_NAMES order):
#   x-params: T_x, M_x, R_x, B_x, m_x, r_x, t1_x, t2_x, w_x, D3_x, D5_x
#   y-params: T_y, R_y, M_y, B_y, m_y, r_y, t1_y, t2_y, w_y, D3_y, D5_y
#
# Design matrix rows at a typical site (xstar=0.3, ystar=0.2, x=6mm, y=4mm):
#   hx = [1, 0.3, -0.2, 0.04,  6,  4, -36, -24, 16,  6*52,  6*52²]
#        → T_x adds  ~1.5nm, M_x*xstar ~8*0.3=2.4nm, m_x*x ~0.12*6=0.72nm
#
# Physical magnitudes chosen to give raw 3σ ≈ 6-8nm after combining all terms.

_BASE_BETA = {
    # x parameters
    "T_x":  1.50,     # nm  scanner X translation
    "M_x":  7.50,     # nm  inter-field X magnification (xstar ≤ 0.46 → up to 3.5nm)
    "R_x":  1.20,     # nm  rotation (coupling to -ystar)
    "B_x": -2.50,     # nm  intra-field shear (ystar²)
    "m_x":  0.070,    # nm/mm  intra-field X magnification
    "r_x":  0.025,    # nm/mm  intra-field rotation
    "t1_x":-0.003,    # nm/mm² intra-field quadratic
    "t2_x": 0.002,
    "w_x":  0.001,
    "D3_x": 8e-5,     # nm/mm³  3rd-order radial (x·r²; max contribution ~0.5nm)
    "D5_x": 2e-9,     # nm/mm⁵  5th-order radial
    # y parameters
    "T_y": -1.20,
    "R_y": -1.20,     # coupling to -xstar
    "M_y":  8.00,
    "B_y":  2.20,     # xstar²
    "m_y":  0.065,
    "r_y":  0.030,
    "t1_y":-0.002,
    "t2_y": 0.002,
    "w_y":  0.001,
    "D3_y": 7e-5,
    "D5_y": 2e-9,
}


def base_beta_vector() -> np.ndarray:
    return np.array([_BASE_BETA[p] for p in PARAM_NAMES])


# Per-lot perturbations: each lot gets a slightly different fingerprint so that
# cross-lot generalisation is non-trivial but within-lot cross-wafer correction works.
_LOT_OFFSETS = {
    1: {"T_x":  0.00, "M_x":  0.00, "T_y":  0.00, "M_y":  0.00},
    2: {"T_x": +0.40, "M_x": -0.80, "T_y": -0.30, "M_y": +0.70, "B_x": +0.60},
    3: {"T_x": -0.60, "M_x": +1.20, "T_y": +0.50, "M_y": -1.10, "R_x": +0.40},
    4: {"T_x": +0.20, "M_x": +0.40, "T_y": +0.20, "M_y": -0.40, "B_y": -0.50},
}


def lot_base_beta(lot: int) -> np.ndarray:
    beta = _BASE_BETA.copy()
    for param, delta in _LOT_OFFSETS.get(lot, {}).items():
        beta[param] = beta[param] + delta
    return np.array([beta[p] for p in PARAM_NAMES])


# ---------------------------------------------------------------------------
# 3. Temporal drift: wafer-to-wafer evolution of Brink parameters
# ---------------------------------------------------------------------------

# Per-parameter noise scale (1σ) for the iid wafer-to-wafer process jitter.
# Calibrated so each parameter contributes ~0.05 nm of wafer-to-wafer overlay noise,
# computed as sigma / effective_gain where effective_gain = mean|H_col| across sites.
# Gains: T~1, M/R~0.23, B~0.07, m/r~6-7mm, t/w~45-72mm², D3~800mm³, D5~1e5mm⁵
_PARAM_JITTER = np.array([
    # x-equation: T_x, M_x, R_x, B_x, m_x, r_x, t1_x, t2_x, w_x, D3_x, D5_x
    0.050,   # T_x  (nm)
    0.200,   # M_x  (nm)
    0.200,   # R_x  (nm)
    0.600,   # B_x  (nm)
    0.008,   # m_x  (nm/mm)
    0.007,   # r_x  (nm/mm)
    0.0010,  # t1_x (nm/mm²)
    0.0010,  # t2_x (nm/mm²)
    0.0007,  # w_x  (nm/mm²)
    6.0e-5,  # D3_x (nm/mm³)
    5.0e-7,  # D5_x (nm/mm⁵)
    # y-equation: T_y, R_y, M_y, B_y, m_y, r_y, t1_y, t2_y, w_y, D3_y, D5_y
    0.050,   # T_y
    0.200,   # R_y
    0.200,   # M_y
    0.600,   # B_y
    0.008,   # m_y
    0.007,   # r_y
    0.0010,  # t1_y
    0.0010,  # t2_y
    0.0007,  # w_y
    6.0e-5,  # D3_y
    5.0e-7,  # D5_y
], dtype=float)


def wafer_drift(lot: int, wafer_idx: int, n_wafers: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return a 22-element delta-beta vector capturing thermal / process drift.

    Lot 1 — linear scanner heating (T_x creep, slight M_x drift)
    Lot 2 — linear T_y + M_y drift (reticle alignment drift)
    Lot 3 — multi-parameter linear drift (largest amplitude)
    Lot 4 — sinusoidal oscillation (chiller cycling artefact)
    """
    t = wafer_idx / (n_wafers - 1)   # normalised time ∈ [0, 1]
    delta = np.zeros(22)
    idx = {p: i for i, p in enumerate(PARAM_NAMES)}

    if lot == 1:
        # Linear scanner-heating drift
        delta[idx["T_x"]] = +1.50 * t
        delta[idx["M_x"]] = -0.60 * t
        delta[idx["T_y"]] = +0.70 * t
        delta[idx["R_x"]] = +0.25 * t

    elif lot == 2:
        # Reticle alignment drift
        delta[idx["T_y"]] = +1.80 * t
        delta[idx["M_y"]] = -0.80 * t
        delta[idx["R_x"]] = +0.35 * t
        delta[idx["T_x"]] = +0.40 * t

    elif lot == 3:
        # Multi-parameter monotonic drift
        delta[idx["T_x"]] = +2.00 * t
        delta[idx["T_y"]] = -1.60 * t
        delta[idx["M_x"]] = +0.90 * t
        delta[idx["M_y"]] = -0.90 * t
        delta[idx["R_x"]] = +0.50 * t
        delta[idx["B_x"]] = -0.70 * t

    elif lot == 4:
        # Sinusoidal oscillation (chiller cycling artefact, ~2.5 cycles)
        amp_t = 0.90
        amp_m = 0.45
        phase = 2.0 * np.pi * t * 2.5
        delta[idx["T_x"]] = amp_t * np.sin(phase)
        delta[idx["T_y"]] = amp_t * np.cos(phase + 0.8)
        delta[idx["M_x"]] = amp_m * np.sin(phase + np.pi / 3)

    # Per-parameter iid process jitter — scaled per-parameter to give ~0.05nm each
    delta += rng.normal(0, 1.0, size=22) * _PARAM_JITTER
    return delta


# ---------------------------------------------------------------------------
# 4. Nonlinear residual (outside the Brink polynomial basis)
# ---------------------------------------------------------------------------

def nonlinear_residual(
    Xw: np.ndarray, Yw: np.ndarray,
    xd: np.ndarray, yd: np.ndarray,
    lot: int,
    wafer_idx: int,
    n_wafers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (res_x, res_y) in nm — contributions that the Brink model cannot represent.

    IMPORTANT: residual is a function of intra-field die coordinates (xd, yd)
    only — matching the PINN input domain (pipeline.py uses DieCenterX/Y).

    Terms used (verified NOT in the Brink polynomial basis):
      a) sin(π·xd/x_max)·cos(π·yd/(2·y_max))
             — sinusoidal intra-field fingerprint (outside any polynomial basis)
      b) xd²·yd   — mixed cubic, NOT in Brink x-equation
             (Brink x-eq contains x·r²=x³+xy² but not x²·y separately)
      c) xd·yd²   — mixed cubic, NOT in Brink y-equation
             (Brink y-eq contains y·r²=x²y+y³ but not xy² separately)

    Physical motivation: localised lens-heating hot-spot causes a non-radial
    intra-field distortion whose spatial pattern is fixed to the reticle but
    grows slowly over the lot as the lens warms up.

    Amplitudes give residual 1σ ≈ 1.1-1.4 nm per axis, so the PINN has a
    clear learning target above the 0.15 nm noise floor.
    """
    X_MAX = 9.0    # mm  (die half-width)
    Y_MAX = 12.0   # mm  (die half-height)

    xn = xd / X_MAX   # normalised die X: ±1
    yn = yd / Y_MAX   # normalised die Y: ±1

    # Lot-specific amplitude
    amp = {1: 1.0, 2: 0.80, 3: 1.30, 4: 0.95}[lot]

    # Temporal modulation: grows +30% over the lot (for with_time PINN benefit)
    t = wafer_idx / max(n_wafers - 1, 1)
    t_amp = 1.0 + 0.30 * t

    res_x = amp * t_amp * (
        1.80 * np.sin(np.pi * xn) * np.cos(0.5 * np.pi * yn)   # sinusoidal
      + 1.20 * xn**2 * yn                                        # cubic x²y (not in Brink x-eq)
    )

    res_y = amp * t_amp * (
        1.50 * np.sin(np.pi * yn) * np.cos(0.5 * np.pi * xn)
      + 1.00 * xn * yn**2                                        # cubic xy² (not in Brink y-eq)
    )

    return res_x.astype(float), res_y.astype(float)


# ---------------------------------------------------------------------------
# 5. Wafer generator
# ---------------------------------------------------------------------------

def generate_wafer(
    sites: np.ndarray,          # (N_sites, 4): [Xw, Yw, xd, yd]
    lot:   int,
    wafer_idx: int,
    n_wafers:  int,
    beta_base: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate one wafer's overlay data."""
    Xw, Yw, xd, yd = sites[:, 0], sites[:, 1], sites[:, 2], sites[:, 3]
    N = len(Xw)

    # Instantaneous beta for this wafer (lot fingerprint + temporal drift)
    drift  = wafer_drift(lot, wafer_idx, n_wafers, rng)
    beta_w = beta_base + drift

    # Brink model prediction
    H = build_design_matrix(Xw, Yw, xd, yd, 0.0, 0.0, WAFER_RADIUS_R)
    y_stacked = H @ beta_w   # [ox1, oy1, ox2, oy2, ...]
    ox_brink  = y_stacked[0::2]
    oy_brink  = y_stacked[1::2]

    # Nonlinear residual (PINN target)
    res_x, res_y = nonlinear_residual(Xw, Yw, xd, yd, lot, wafer_idx, n_wafers)

    # Metrology noise
    noise_x = rng.normal(0.0, MEAS_NOISE_SIGMA, N)
    noise_y = rng.normal(0.0, MEAS_NOISE_SIGMA, N)

    # Total observed overlay
    ox = ox_brink + res_x + noise_x
    oy = oy_brink + res_y + noise_y

    df = pd.DataFrame({
        "WaferCenterX_mm_": Xw,
        "WaferCenterY_mm_": Yw,
        "DieCenterX_mm_":   xd,
        "DieCenterY_mm_":   yd,
        "OverlayX":         ox,
        "OverlayY":         oy,
    })
    return df


# ---------------------------------------------------------------------------
# 6. Main generation loop
# ---------------------------------------------------------------------------

def generate_all(
    outdir: str = "dat_synthetic",
    seed: int = RNG_SEED,
    verbose: bool = True,
) -> None:
    rng = np.random.default_rng(seed)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    field_grid = make_field_grid()
    die_x_grid, die_y_grid = DIE_X, DIE_Y
    sites = make_site_table(field_grid, die_x_grid, die_y_grid)

    n_fields = len(field_grid)
    n_sites  = len(sites)

    if verbose:
        print(f"Field grid: {n_fields} fields, {n_sites} sites/wafer")
        print(f"WaferCenter X range: {sites[:,0].min():.1f} to {sites[:,0].max():.1f} mm")
        print(f"DieCenter   X range: {sites[:,2].min():.1f} to {sites[:,2].max():.1f} mm")
        print(f"Generating {N_LOTS} lots × {N_WAFERS} wafers …\n")

    for lot in range(1, N_LOTS + 1):
        beta_base = lot_base_beta(lot)
        lot_rmse_raw  = []
        lot_rmse_pred = []

        for w in range(1, N_WAFERS + 1):
            wafer_rng = np.random.default_rng(seed + lot * 1000 + w)
            df = generate_wafer(sites, lot, w - 1, N_WAFERS, beta_base, wafer_rng)

            fname = out / f"Lot{lot}Wafer{w}.csv"
            df.to_csv(fname, index=False)

            ox_std = df["OverlayX"].std()
            lot_rmse_raw.append(ox_std)

        if verbose:
            print(
                f"Lot {lot}: {N_WAFERS} wafers written  "
                f"OverlayX std (raw): "
                f"min={min(lot_rmse_raw):.3f}  "
                f"max={max(lot_rmse_raw):.3f}  "
                f"mean={np.mean(lot_rmse_raw):.3f} nm"
            )

    print(f"\nAll data written to {out.resolve()}/")

    # Quick sanity print
    _sanity_check(out, seed)


def _sanity_check(outdir: Path, seed: int) -> None:
    """Fit Brink on wafers 1-10 of Lot 1 and report cross-wafer R²."""
    import sys
    sys.path.insert(0, str(outdir.parent / "src"))
    from model import build_design_matrix, fit_beta, predict_overlay, overlay_metrics

    lot = 1
    train_wafers = list(range(1, 11))
    test_wafers  = list(range(11, 21))

    dfs_tr = [pd.read_csv(outdir / f"Lot{lot}Wafer{w}.csv") for w in train_wafers]
    df_tr  = pd.concat(dfs_tr, ignore_index=True)

    Xw = df_tr["WaferCenterX_mm_"].values
    Yw = df_tr["WaferCenterY_mm_"].values
    xd = df_tr["DieCenterX_mm_"].values
    yd = df_tr["DieCenterY_mm_"].values
    ox = df_tr["OverlayX"].values
    oy = df_tr["OverlayY"].values

    H_tr  = build_design_matrix(Xw, Yw, xd, yd, 0.0, 0.0, 300.0)
    y_tr  = np.empty(2 * len(ox)); y_tr[0::2] = ox; y_tr[1::2] = oy
    beta, _ = fit_beta(H_tr, y_tr)

    dfs_te = [pd.read_csv(outdir / f"Lot{lot}Wafer{w}.csv") for w in test_wafers]

    print("\n=== SANITY CHECK: cross-wafer Brink (Lot 1, train=10, test=10-19) ===")
    for df_te in dfs_te:
        Xwt = df_te["WaferCenterX_mm_"].values
        Ywt = df_te["WaferCenterY_mm_"].values
        xdt = df_te["DieCenterX_mm_"].values
        ydt = df_te["DieCenterY_mm_"].values
        oxt = df_te["OverlayX"].values
        oyt = df_te["OverlayY"].values
        Ht  = build_design_matrix(Xwt, Ywt, xdt, ydt, 0.0, 0.0, 300.0)
        ypx, ypy = predict_overlay(Ht, beta)
        obs  = np.concatenate([oxt, oyt])
        pred = np.concatenate([ypx, ypy])
        m_raw   = overlay_metrics(obs, np.zeros_like(obs))
        m_brink = overlay_metrics(obs, pred)
        print(
            f"  raw 3σ={m_raw['ThreeSigma']:.3f}  "
            f"Brink 3σ={m_brink['ThreeSigma']:.3f}  "
            f"R²={m_brink['R2']:.3f}  "
            f"improvement={100*(m_raw['RMSE']-m_brink['RMSE'])/m_raw['RMSE']:.1f}%"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic EUV overlay dataset.")
    ap.add_argument("--outdir", default="dat_synthetic",
                    help="Output directory (default: dat_synthetic)")
    ap.add_argument("--seed",   type=int, default=RNG_SEED)
    ap.add_argument("--quiet",  action="store_true")
    args = ap.parse_args()
    generate_all(outdir=args.outdir, seed=args.seed, verbose=not args.quiet)
