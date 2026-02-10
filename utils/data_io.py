# utils/data_io.py
import os
import pandas as pd
import numpy as np
import streamlit as st

from utils.config import VARS_CLUSTER

@st.cache_data(show_spinner=True)
def load_data(path: str, mtime: float) -> pd.DataFrame:
    """
    Lee el dataset de la app (CSV) y tipa columnas clave.
    El parámetro mtime fuerza recarga cuando cambia el fichero.
    """
    df = pd.read_csv(path)

    for c in ["nombre", "cluster_label", "codigo_nif"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    for c in VARS_CLUSTER + ["PC1", "PC2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


@st.cache_data(show_spinner=True)
def load_base_with_clusters(base_path: str, df_app: pd.DataFrame) -> pd.DataFrame:
    """
    Lee la base completa (parquet o csv) y le añade cluster_label desde df_app usando codigo_nif.
    Incluye fallback parquet -> csv si el parquet falla.
    """
    base_df = None
    last_err = None

    # 1) Intento parquet si toca
    if base_path.endswith(".parquet"):
        try:
            base_df = pd.read_parquet(base_path)
        except Exception as e:
            last_err = e
            base_df = None

    # 2) Fallback a CSV (mismo nombre)
    if base_df is None:
        csv_path = base_path.replace(".parquet", ".csv")
        if os.path.exists(csv_path):
            try:
                try:
                    base_df = pd.read_csv(csv_path, encoding="utf-8")
                except UnicodeDecodeError:
                    base_df = pd.read_csv(csv_path, encoding="latin1")
                st.warning(f"No pude leer el parquet; usando CSV: {os.path.basename(csv_path)}")
            except Exception as e:
                last_err = e
                base_df = None

    if base_df is None:
        raise RuntimeError(
            f"No he podido leer la base completa. "
            f"Parquet falló con: {last_err}. "
            f"Si tienes CSV, guárdalo como {base_path.replace('.parquet', '.csv')}."
        )

    if "codigo_nif" not in base_df.columns:
        raise ValueError(f"En la base inicial no existe 'codigo_nif': {base_path}")

    base_df["codigo_nif"] = base_df["codigo_nif"].astype(str)

    # clusters desde df_app
    if "codigo_nif" not in df_app.columns or "cluster_label" not in df_app.columns:
        raise ValueError("df_app debe contener 'codigo_nif' y 'cluster_label'.")

    cl = df_app[["codigo_nif", "cluster_label"]].dropna().copy()
    cl["codigo_nif"] = cl["codigo_nif"].astype(str)
    cl = cl.drop_duplicates(subset=["codigo_nif"])

    out = base_df.merge(cl, on="codigo_nif", how="left")
    return out