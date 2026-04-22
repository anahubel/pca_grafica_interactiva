# utils/data_io.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import streamlit as st

from utils.config import VARS_CLUSTER


# =========================
# Helpers
# =========================
def _mtime(path: str | Path) -> float:
    """Devuelve mtime del fichero (0 si no existe). Útil para invalidar cache."""
    p = Path(path)
    return p.stat().st_mtime if p.exists() else 0.0


def _read_any(path: str) -> pd.DataFrame:
    """Lee CSV/Parquet/Excel de forma robusta."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el fichero: {path}")

    ext = p.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(p)

    if ext == ".csv":
        try:
            return pd.read_csv(p, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(p, encoding="latin1")

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(p)

    raise ValueError(f"Formato no soportado: {ext} ({path})")


def _type_app_df(df: pd.DataFrame) -> pd.DataFrame:
    """Tipa columnas clave y numéricas para la app."""
    df = df.copy()

    for c in ["nombre", "cluster_label", "codigo_nif"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    for c in VARS_CLUSTER + ["PC1", "PC2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# =========================
# Dataset APP (cached)
# =========================
@st.cache_data(show_spinner=True)
def load_data_cached(path: str, mtime: float) -> pd.DataFrame:
    """
    Lee el dataset de la app y tipa columnas.
    El parámetro mtime fuerza recarga cuando cambia el fichero.
    """
    df = _read_any(path)
    return _type_app_df(df)


def load_app_dataset(path: str) -> pd.DataFrame:
    """Wrapper cómodo: llama a cache pasando mtime automáticamente."""
    return load_data_cached(path, _mtime(path))


# =========================
# Base completa + clusters (cached)
# =========================
@st.cache_data(show_spinner=True)
def load_base_with_clusters_cached(base_path: str, base_mtime: float, df_app: pd.DataFrame) -> pd.DataFrame:
    """
    Lee la base completa (parquet/csv/xlsx) y añade cluster_label desde df_app usando codigo_nif.
    base_mtime fuerza recarga cuando cambia el fichero base.
    """
    base_df = None
    last_err = None

    # 1) Intento leer tal cual (parquet/csv/xlsx...)
    try:
        base_df = _read_any(base_path)
    except Exception as e:
        last_err = e
        base_df = None

    # 2) Fallback: si pedían parquet y falla, intentar CSV con mismo nombre
    if base_df is None and base_path.lower().endswith(".parquet"):
        csv_path = base_path[:-8] + ".csv"  # reemplaza .parquet por .csv
        if os.path.exists(csv_path):
            try:
                base_df = _read_any(csv_path)
                st.warning(f"No pude leer el parquet; usando CSV: {os.path.basename(csv_path)}")
            except Exception as e:
                last_err = e
                base_df = None

    if base_df is None:
        raise RuntimeError(
            "No he podido leer la base completa. "
            f"Error: {last_err}. "
            "Si tienes CSV, guárdalo con el mismo nombre que el parquet pero extensión .csv."
        )

    if "codigo_nif" not in base_df.columns:
        raise ValueError(f"En la base completa no existe 'codigo_nif': {base_path}")

    base_df = base_df.copy()
    base_df["codigo_nif"] = base_df["codigo_nif"].astype(str)

    # clusters desde df_app
    if "codigo_nif" not in df_app.columns or "cluster_label" not in df_app.columns:
        raise ValueError("df_app debe contener 'codigo_nif' y 'cluster_label'.")

    cl = df_app[["codigo_nif", "cluster_label"]].dropna().copy()
    cl["codigo_nif"] = cl["codigo_nif"].astype(str)
    cl = cl.drop_duplicates(subset=["codigo_nif"])

    out = base_df.merge(cl, on="codigo_nif", how="left")
    return out


def load_base_with_clusters(base_path: str, df_app: pd.DataFrame) -> pd.DataFrame:
    """Wrapper cómodo: invalida cache cuando cambia la base."""
    return load_base_with_clusters_cached(base_path, _mtime(base_path), df_app)