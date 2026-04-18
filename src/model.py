"""
Classical Brink / van den Brink overlay model.

The model captures systematic inter-field overlay patterns using a 22-parameter
polynomial that includes translations, rotations, magnification, intra-field
bilinear/quadratic terms, and two radial distortion terms (3rd- and 5th-order).

Typical usage
-------------
::

    H = build_design_matrix(Xw, Yw, x, y, Xc=0.0, Yc=0.0, R=300.0)
    beta, beta_dict = fit_beta(H, y_stacked)
    ovl_x_pred, ovl_y_pred = predict_overlay(H, beta)
    metrics = overlay_metrics(ovl_x_true, ovl_x_pred)
"""

import numpy as np
from typing import Tuple, Dict, Optional

# ---------- Parameter (column) names in H (Brink/van den Brink extension) ----------
# 11 parameters per axis (x, y), 22 total.
# D3 = 3rd-order radial term (x·r²), D5 = 5th-order radial term (x·r⁴).
PARAM_NAMES = [
    # x-equation (11)
    "T_x", "M_x", "R_x", "B_x", "m_x", "r_x", "t1_x", "t2_x", "w_x", "D3_x", "D5_x",
    # y-equation (11)
    "T_y", "R_y", "M_y", "B_y", "m_y", "r_y", "t1_y", "t2_y", "w_y", "D3_y", "D5_y",
]

def normalize_to_unit_disk(X: np.ndarray, Y: np.ndarray, Xc: float, Yc: float, R: float) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize inter-field wafer coordinates to the unit disk."""
    xstar = (X - Xc) / R
    ystar = (Y - Yc) / R
    return xstar, ystar

def design_rows_for_site_brvdb(xstar: float, ystar: float, x: float, y: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the two design row vectors (Brink/van den Brink extension).
    Base you already had:
      h_x_base = [1, x*, -y*, (y*)^2, x, y, -x^2, -(x y), y^2]
      h_y_base = [1, -x*, y*, (x*)^2, y, x, -y^2, -(y x), x^2]
    Added radial distortion terms (r^2 = x^2 + y^2, r^4 = (r^2)^2):
      x-equation adds: [ x*r^2, x*r^4 ]
      y-equation adds: [ y*r^2, y*r^4 ]
    """
    # base (same as your current model)
    hx = np.array([1.0, xstar, -ystar, ystar**2, x, y, -x**2, -(x*y), y**2], dtype=float)
    hy = np.array([1.0, -xstar, ystar, xstar**2, y, x, -y**2, -(y*x), x**2], dtype=float)

    # radial terms
    r2 = x*x + y*y
    r4 = r2*r2
    hx_ext = np.concatenate([hx, np.array([x*r2, x*r4], dtype=float)])  # D3_x, D5_x
    hy_ext = np.concatenate([hy, np.array([y*r2, y*r4], dtype=float)])  # D3_y, D5_y
    return hx_ext, hy_ext

def build_design_matrix(
    X: np.ndarray, Y: np.ndarray,  # inter-field wafer coords
    x: np.ndarray, y: np.ndarray,  # intra-field coords
    Xc: float, Yc: float, R: float
) -> np.ndarray:
    """
    Build the stacked design matrix H (shape: 2N x 22) for all sites.
    Row order: [ovl_x_1, ovl_y_1, ovl_x_2, ovl_y_2, ..., ovl_x_N, ovl_y_N].
    Column order: PARAM_NAMES defined above (x-equation 11, then y-equation 11).
    """
    assert X.shape == Y.shape == x.shape == y.shape
    N = X.size
    xstar, ystar = normalize_to_unit_disk(X, Y, Xc, Yc, R)

    H = np.zeros((2 * N, 22), dtype=float)
    for i in range(N):
        hx, hy = design_rows_for_site_brvdb(xstar[i], ystar[i], x[i], y[i])

        # Place hx into x-block columns [0..10], hy into y-block columns [11..21]
        H[2*i,   0:11]  = hx
        H[2*i,   11:22] = 0.0
        H[2*i+1, 0:11]  = 0.0
        H[2*i+1, 11:22] = hy
    return H

def fit_beta(H: np.ndarray, y_stack: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Solve for β via (weighted) least squares for the stacked problem:
      H (2N x 22) @ beta (22,) ≈ y_stack (2N,)
    - y_stack is [ovl_x_1, ovl_y_1, ..., ovl_x_N, ovl_y_N]
    - weights (optional) is length 2N for per-row weights
    Returns (beta_vector, beta_dict).
    """
    assert H.shape[0] == y_stack.shape[0]
    if weights is not None:
        assert weights.shape == y_stack.shape
        w = np.sqrt(weights).reshape(-1, 1)
        Hw = H * w
        yw = y_stack * w.ravel()
        beta, *_ = np.linalg.lstsq(Hw, yw, rcond=None)
    else:
        beta, *_ = np.linalg.lstsq(H, y_stack, rcond=None)

    beta_dict = {name: float(val) for name, val in zip(PARAM_NAMES, beta)}
    return beta, beta_dict

def predict_overlay(H: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given H and beta, return predicted (ovl_x, ovl_y) arrays of length N each.
    """
    yhat = H @ beta  # stacked [x1, y1, x2, y2, ...]
    ovl_x_pred = yhat[0::2]
    ovl_y_pred = yhat[1::2]
    return ovl_x_pred, ovl_y_pred

def overlay_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute a comprehensive set of overlay error metrics.

    All metrics are computed on finite (non-NaN, non-Inf) residuals only.

    Parameters
    ----------
    y_true:
        Ground-truth overlay values (nm), shape ``(M,)``.
    y_pred:
        Model-predicted overlay values (nm), shape ``(M,)``.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``"RMSE"``       — root mean squared error (nm)
        * ``"MAE"``        — mean absolute error (nm)
        * ``"MeanError"``  — signed mean residual / bias (nm)
        * ``"Sigma"``      — residual standard deviation (ddof=1, nm)
        * ``"ThreeSigma"`` — 3× residual standard deviation (nm)
        * ``"P95_abs"``    — 95th percentile of |residual| (nm)
        * ``"MaxAbs"``     — maximum absolute residual (nm)
        * ``"R2"``         — coefficient of determination (dimensionless)
    """
    r = y_true - y_pred  # residuals
    r = r[np.isfinite(r)]
    y_true = y_true[np.isfinite(y_true)]
    y_pred = y_pred[np.isfinite(y_pred)]

    rmse = float(np.sqrt(np.mean(r**2)))
    mae  = float(np.mean(np.abs(r)))
    me   = float(np.mean(r))                     # mean error (bias)
    std  = float(np.std(r, ddof=1))              # residual std
    p95  = float(np.percentile(np.abs(r), 95))   # 95th percentile (abs)
    maxabs = float(np.max(np.abs(r)))
    # R^2 optional (less standard for overlay, but useful)
    denom = np.sum((y_true - np.mean(y_true))**2)
    r2 = float(1.0 - np.sum(r**2) / denom) if denom > 0 else np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MeanError": me,
        "Sigma": std,
        "ThreeSigma": 3*std,
        "P95_abs": p95,
        "MaxAbs": maxabs,
        "R2": r2,
    }