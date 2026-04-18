"""
Generate publication-quality figures for the EUV-OVL-SYN dataset paper.

Figures written to paper/figs/:
  fig_overlay_panels.pdf    -- 4-panel quiver: raw / Brink prediction /
                               Brink residual / ground-truth NL residual
  fig_temporal_profiles.pdf -- 4-subplot per-wafer drift, one panel per lot
  fig_residual_pattern.pdf  -- heatmap of the nonlinear residual in die space

Run from the repository root:
    python src/generate_dataset_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.dirname(__file__))

from model import (
    build_design_matrix,
    fit_beta,
    predict_overlay,
    PARAM_NAMES,
)
from generate_synthetic_data import (
    nonlinear_residual,
    make_field_grid,
    make_site_table,
    lot_base_beta,
    wafer_drift,
    N_WAFERS,
    DIE_X,
    DIE_Y,
    WAFER_RADIUS_R,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO   = Path(__file__).resolve().parent.parent
SYNDIR = REPO / "dat_synthetic"
FIGDIR = REPO / "paper" / "figs"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "figure.dpi":       150,
    "axes.linewidth":   0.7,
    "lines.linewidth":  1.0,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# ---------------------------------------------------------------------------
# Helper: fit Brink on training wafers
# ---------------------------------------------------------------------------

def fit_brink(lot: int, train_wafers: list[int]) -> np.ndarray:
    frames = []
    for w in train_wafers:
        df = pd.read_csv(SYNDIR / f"Lot{lot}Wafer{w}.csv")
        frames.append(df)
    df_tr = pd.concat(frames, ignore_index=True)

    Xw = df_tr["WaferCenterX_mm_"].values
    Yw = df_tr["WaferCenterY_mm_"].values
    xd = df_tr["DieCenterX_mm_"].values
    yd = df_tr["DieCenterY_mm_"].values
    ox = df_tr["OverlayX"].values
    oy = df_tr["OverlayY"].values

    H = build_design_matrix(Xw, Yw, xd, yd, 0.0, 0.0, WAFER_RADIUS_R)
    y = np.empty(2 * len(ox))
    y[0::2] = ox
    y[1::2] = oy
    beta, _ = fit_beta(H, y)
    return beta


# ===========================================================================
# Figure 1: overlay quiver panels (raw / Brink / Brink residual / NL truth)
# ===========================================================================

def fig_overlay_panels() -> None:
    print("  Generating fig_overlay_panels.pdf …")

    LOT         = 1
    WAFER_ID    = 25                     # mid-lot wafer
    TRAIN_WS    = list(range(1, 11))

    beta = fit_brink(LOT, TRAIN_WS)

    df = pd.read_csv(SYNDIR / f"Lot{LOT}Wafer{WAFER_ID}.csv")
    Xw = df["WaferCenterX_mm_"].values
    Yw = df["WaferCenterY_mm_"].values
    xd = df["DieCenterX_mm_"].values
    yd = df["DieCenterY_mm_"].values
    ox = df["OverlayX"].values
    oy = df["OverlayY"].values

    H = build_design_matrix(Xw, Yw, xd, yd, 0.0, 0.0, WAFER_RADIUS_R)
    px, py = predict_overlay(H, beta)

    # Ground-truth nonlinear residual (wafer_idx = WAFER_ID - 1)
    nl_x, nl_y = nonlinear_residual(Xw, Yw, xd, yd, LOT, WAFER_ID - 1, N_WAFERS)

    panels = [
        ("(a) Raw overlay",           ox,      oy),
        ("(b) Brink prediction",       px,      py),
        ("(c) Brink residual",         ox - px, oy - py),
        ("(d) Nonlinear component",    nl_x,    nl_y),
    ]

    # Common scale for quiver arrows (units: nm → scaled to mm display)
    SCALE = 0.006   # mm per nm

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2))
    axes = axes.flat

    for ax, (title, vx, vy) in zip(axes, panels):
        mag = np.sqrt(vx**2 + vy**2)
        sc = ax.scatter(Xw, Yw, c=mag, cmap="plasma", s=4,
                        vmin=0, vmax=np.percentile(mag, 97))
        ax.quiver(Xw, Yw, vx * SCALE, vy * SCALE,
                  scale=1, scale_units="xy",
                  width=0.003, headwidth=3, headlength=4,
                  color="white", alpha=0.7)
        # Wafer boundary
        ax.add_patch(Circle((0, 0), 150, fill=False, lw=0.8, ls="--",
                             edgecolor="0.5"))
        ax.set_aspect("equal")
        ax.set_xlim(-165, 165)
        ax.set_ylim(-165, 165)
        ax.set_title(title, pad=3)
        ax.set_xlabel("$X_w$ (mm)")
        ax.set_ylabel("$Y_w$ (mm)")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("$|$OVL$|$ (nm)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        # RMS annotation
        rms = np.sqrt(np.mean(vx**2 + vy**2))
        ax.text(0.03, 0.97, f"RMS = {rms:.2f} nm",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    fig.suptitle(
        f"Lot {LOT}, Wafer {WAFER_ID} — overlay decomposition",
        fontsize=9, y=1.01
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_overlay_panels.pdf", bbox_inches="tight")
    plt.close(fig)
    print("    → paper/figs/fig_overlay_panels.pdf")


# ===========================================================================
# Figure 2: per-wafer temporal drift profiles, one panel per lot
# ===========================================================================

def fig_temporal_profiles() -> None:
    print("  Generating fig_temporal_profiles.pdf …")

    lot_labels = {
        1: "Lot 1 — linear scanner heating",
        2: "Lot 2 — reticle alignment drift",
        3: "Lot 3 — multi-parameter drift",
        4: "Lot 4 — chiller-cycling oscillation",
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharey=False)
    axes = axes.flat

    for lot, ax in zip(range(1, 5), axes):
        wafer_ids = []
        mean_ox   = []
        sigma3_ox = []

        for w in range(1, N_WAFERS + 1):
            df = pd.read_csv(SYNDIR / f"Lot{lot}Wafer{w}.csv")
            ox = df["OverlayX"].values
            wafer_ids.append(w)
            mean_ox.append(ox.mean())
            sigma3_ox.append(3.0 * ox.std())

        wafer_ids = np.array(wafer_ids)
        mean_ox   = np.array(mean_ox)
        sigma3_ox = np.array(sigma3_ox)

        ax.fill_between(wafer_ids,
                        mean_ox - sigma3_ox / 3,
                        mean_ox + sigma3_ox / 3,
                        alpha=0.25, color="royalblue", label=r"$\pm\sigma$")
        ax.plot(wafer_ids, mean_ox, lw=1.2, color="royalblue",
                label=r"$\langle\mathrm{OVL}_X\rangle$")
        ax.plot(wafer_ids, sigma3_ox, lw=1.0, ls="--", color="tomato",
                label=r"$3\sigma(\mathrm{OVL}_X)$")

        ax.axhline(0, color="0.7", lw=0.6, ls=":")
        ax.set_title(lot_labels[lot], pad=3)
        ax.set_xlabel("Wafer index")
        ax.set_ylabel("Overlay (nm)")
        ax.legend(loc="upper left", ncol=1, framealpha=0.7, handlelength=1.5)
        ax.set_xlim(1, N_WAFERS)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_temporal_profiles.pdf", bbox_inches="tight")
    plt.close(fig)
    print("    → paper/figs/fig_temporal_profiles.pdf")


# ===========================================================================
# Figure 3: nonlinear residual spatial pattern in die-coordinate space
# ===========================================================================

def fig_residual_pattern() -> None:
    print("  Generating fig_residual_pattern.pdf …")

    # Evaluate on a dense grid of die coordinates
    xd_vec = np.linspace(-9.0, 9.0, 120)
    yd_vec = np.linspace(-12.0, 12.0, 120)
    XD, YD = np.meshgrid(xd_vec, yd_vec)
    shape  = XD.shape
    xd_flat = XD.ravel()
    yd_flat = YD.ravel()
    # Dummy inter-field coords (residual is independent of these)
    Xw_flat = np.zeros_like(xd_flat)
    Yw_flat = np.zeros_like(xd_flat)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.8))

    lot_amps = {1: 1.0, 2: 0.80, 3: 1.30, 4: 0.95}
    lot_names = {1: "Lot 1", 2: "Lot 2", 3: "Lot 3", 4: "Lot 4"}
    WAFER_MID = 25   # mid-lot time amplitude

    for idx, lot in enumerate([1, 2, 3, 4]):
        ax = axes.flat[idx]
        rx, ry = nonlinear_residual(
            Xw_flat, Yw_flat, xd_flat, yd_flat,
            lot, WAFER_MID - 1, N_WAFERS
        )
        mag = np.sqrt(rx**2 + ry**2).reshape(shape)

        im = ax.pcolormesh(xd_vec, yd_vec, mag,
                           cmap="inferno", shading="gouraud",
                           vmin=0)
        # Overlay quiver arrows on a coarser grid
        skip = 12
        ax.quiver(XD[::skip, ::skip], YD[::skip, ::skip],
                  rx.reshape(shape)[::skip, ::skip],
                  ry.reshape(shape)[::skip, ::skip],
                  scale=25, scale_units="xy",
                  color="white", alpha=0.8, width=0.005,
                  headwidth=3, headlength=4)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$|f_{\rm NL}|$ (nm)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        ax.set_title(lot_names[lot], pad=3)
        ax.set_xlabel("Die $x_d$ (mm)")
        ax.set_ylabel("Die $y_d$ (mm)")
        ax.set_aspect("equal")

    fig.suptitle("Nonlinear intra-field residual spatial pattern (mid-lot wafer)",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_residual_pattern.pdf", bbox_inches="tight")
    plt.close(fig)
    print("    → paper/figs/fig_residual_pattern.pdf")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print(f"Reading synthetic data from: {SYNDIR}")
    print(f"Writing figures to:          {FIGDIR}\n")
    fig_overlay_panels()
    fig_temporal_profiles()
    fig_residual_pattern()
    print("\nDone.")
