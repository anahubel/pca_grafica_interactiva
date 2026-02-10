# utils/fmt.py
import numpy as np
import pandas as pd
from utils.config import LOGGED_IN_MODEL

def to_display_scale(var: str, s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if var in LOGGED_IN_MODEL:
        return np.expm1(s)
    return s

def fmt_num(x) -> str:
    if pd.isna(x):
        return ""
    try:
        ax = abs(float(x))
        if ax >= 1_000_000:
            return f"{x:,.0f}"
        if ax >= 10_000:
            return f"{x:,.0f}"
        if ax >= 100:
            return f"{x:,.1f}"
        if ax >= 1:
            return f"{x:,.3f}"
        return f"{x:.4f}"
    except Exception:
        return str(x)

def fmt_p(p):
    if pd.isna(p):
        return ""
    p = float(p)
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def sig_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    p = float(p)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""