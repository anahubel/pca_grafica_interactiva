# utils/stats.py
import numpy as np
import pandas as pd
from itertools import combinations

def eps2_kw(H: float, n: int, k: int) -> float:
    if pd.isna(H) or n <= k or k < 2:
        return np.nan
    return max(0.0, (H - k + 1.0) / (n - k))

def magnitude_eps2(e: float) -> str:
    if pd.isna(e):
        return ""
    if e < 0.01:
        return "Muy pequeña"
    if e < 0.06:
        return "Pequeña"
    if e < 0.14:
        return "Media"
    return "Grande"

def magnitude_cramers_v(v: float) -> str:
    if pd.isna(v):
        return ""
    if v < 0.10:
        return "Pequeña"
    if v < 0.30:
        return "Media"
    if v < 0.50:
        return "Grande"
    return "Muy grande"

# --- Post-hoc numéricas ---
def p_adjust_holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        val = min(1.0, max(val, prev))
        adj[idx] = val
        prev = val
    return adj.tolist()

def p_adjust_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    return [min(1.0, float(p) * m) for p in pvals]

def magnitude_delta(d: float) -> str:
    if pd.isna(d):
        return ""
    ad = abs(float(d))
    if ad < 0.147:
        return "Despreciable"
    if ad < 0.33:
        return "Pequeño"
    if ad < 0.474:
        return "Medio"
    return "Grande"

def compute_posthoc_mwu(data_by: dict, mannwhitneyu_fn, cliffs_delta_fn, sig_stars_fn, adjust: str = "Holm"):
    clusters = list(data_by.keys())
    pairs = list(combinations(clusters, 2))

    rows = []
    pvals = []
    for (a, b) in pairs:
        xa, xb = data_by.get(a, np.array([])), data_by.get(b, np.array([]))
        if len(xa) < 2 or len(xb) < 2:
            p, d = np.nan, np.nan
        else:
            try:
                _, p = mannwhitneyu_fn(xa, xb, alternative="two-sided")
            except Exception:
                p = np.nan
            try:
                d = cliffs_delta_fn(xa, xb)
            except Exception:
                d = np.nan

        pvals.append(1.0 if pd.isna(p) else float(p))
        rows.append({"Comparación": f"{a}-{b}", "P-valor": p, "δ (Cliff)": d, "Magnitud": magnitude_delta(d)})

    padj = p_adjust_holm(pvals) if adjust == "Holm" else p_adjust_bonferroni(pvals)

    for i in range(len(rows)):
        rows[i]["P-ajustada"] = padj[i]
        rows[i]["Sig."] = sig_stars_fn(padj[i])

    return pd.DataFrame(rows)