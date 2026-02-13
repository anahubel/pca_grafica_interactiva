# utils/subgroups.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, cut_tree

from utils.config import VARS_CLUSTER

@st.cache_data(show_spinner=False)
def add_subclusters_within_general(
    df: pd.DataFrame,
    general_col: str = "cluster_label",
    vars_cluster: list[str] | None = None,
    k: int = 3,
    prefix: str = "subcluster_",
) -> pd.DataFrame:
    """
    Añade subgrupos (k) dentro de cada cluster general.
    Crea columnas: prefix + <valor_cluster>  (ej: subcluster_C1, subcluster_C2, subcluster_C3)
    Solo se rellena para filas del cluster correspondiente.
    """
    vars_cluster = vars_cluster or VARS_CLUSTER

    out = df.copy()

    # Garantiza numérico en vars_cluster
    for v in vars_cluster:
        if v in out.columns:
            out[v] = pd.to_numeric(out[v], errors="coerce")

    clusters = [c for c in out[general_col].dropna().unique()]
    # ordenar: si son C1/C2/C3, orden natural; si son numéricos, orden numérico
    try:
        clusters = sorted(clusters, key=lambda x: int(str(x).replace("C","")))
    except Exception:
        clusters = sorted(clusters)

    for c in clusters:
        mask = out[general_col] == c
        df_c = out.loc[mask, vars_cluster].copy().dropna()
        if len(df_c) < k:
            out.loc[mask, f"{prefix}{c}"] = np.nan
            continue

        X = df_c.values.astype(float)
        Xs = StandardScaler().fit_transform(X)
        Z = linkage(Xs, method="ward")
        labels = cut_tree(Z, n_clusters=k).reshape(-1) + 1  # 1..k

        # asignar respetando el índice tras dropna
        out.loc[df_c.index, f"{prefix}{c}"] = labels.astype(int)

    return out

def build_subgroup_column(df: pd.DataFrame, general_cluster_value: str, general_col: str = "cluster_label", prefix: str = "subcluster_") -> pd.DataFrame:
    """
    Devuelve df filtrado al cluster general (p.ej. C1) y con una columna común 'subcluster'
    que toma los valores de subcluster_<C1>.
    """
    out = df[df[general_col] == general_cluster_value].copy()
    src_col = f"{prefix}{general_cluster_value}"
    if src_col not in out.columns:
        out["subcluster"] = np.nan
    else:
        out["subcluster"] = out[src_col]
    return out
