# utils/views/cluster_textil.py
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans

from utils.config import VARS_CLUSTER, LABELS
from utils.fmt import fmt_num
from utils.views.resumen import render_resumen
from utils.views.estadistica import render_estadistica
from utils.views.arbol_decision import render_arbol_decision


def _find_sector_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "sector", "Sector",
        "sector_label",
        "sector_agrupado", "Sector agrupado", "Sector_agrupado",
        "sector_grp",
        "sector_agrupado_2", "sector_agrupado_2_label",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _filter_textil(df: pd.DataFrame, sector_col: str) -> pd.DataFrame:
    s = df[sector_col].astype(str).str.strip().str.lower()
    return df.loc[s.str.contains("textil", na=False)].copy()


def _build_textil_story(df_in: pd.DataFrame, cluster_col: str = "cluster_label") -> dict:
    """
    Crea un story_map tipo render_resumen: { "1": {...}, "2": {...}, "3": {...} }
    Basado en medianas relativas (↑/↓) en variables clave.
    """
    key_vars = [
        "rotacion_stocks",
        "productividad_va_pax",
        "inmovilizado_empleado",
        "nofs_ventas",
        "competitividad",
    ]
    key_vars = [v for v in key_vars if v in df_in.columns]

    if cluster_col not in df_in.columns:
        return {}

    clusters = sorted(df_in[cluster_col].dropna().astype(str).unique())
    if len(clusters) < 2:
        return {}

    # medianas
    med = {}
    for v in key_vars:
        med[v] = (
            df_in[[cluster_col, v]]
            .assign(_v=pd.to_numeric(df_in[v], errors="coerce"))
            .groupby(df_in[cluster_col].astype(str))["_v"]
            .median()
        )

    # ranking alto/bajo por variable
    def _rank_label(v: str, cl: str) -> str:
        s = med[v].dropna()
        if len(s) < 2 or cl not in s.index:
            return "≈"
        order = s.sort_values()
        low = order.index[0]
        high = order.index[-1]
        if cl == high:
            return "↑"
        if cl == low:
            return "↓"
        return "≈"

    # nombre bonito
    pretty = {c: f"T{idx+1}" for idx, c in enumerate(clusters)}

    story_map = {}
    for c in clusters:
        bullets_struct = []
        for v in key_vars:
            lab = _rank_label(v, c)
            bullets_struct.append(f"{LABELS.get(v, v)}: {lab}")

        # 2–3 frases automáticas según patrón
        flags = {v: _rank_label(v, c) for v in key_vars}
        lectura = []
        implic = []

        # heurísticas sencillas y entendibles
        if flags.get("rotacion_stocks") == "↑" and flags.get("nofs_ventas") == "↓":
            lectura.append("Perfil ágil en circulante: rota más rápido y necesita menos financiación operativa.")
            implic.append("Prioridad: sostener rotación y trabajar margen/eficiencia para capturar valor.")
        if flags.get("inmovilizado_empleado") == "↑" and flags.get("productividad_va_pax") == "↑":
            lectura.append("Perfil intensivo en capital con alta productividad: estructura fuerte y mayor valor por persona.")
            implic.append("Foco: asegurar utilización de capacidad y disciplina de inversión (capex/costes fijos).")
        if flags.get("productividad_va_pax") == "↓":
            lectura.append("Eficiencia más baja en generación de valor por persona dentro del textil.")
            implic.append("Palanca: procesos/organización para elevar productividad (y revisar estructura de costes).")

        if not lectura:
            lectura.append("Perfil intermedio dentro del textil (valores no extremos en los indicadores clave).")
            implic.append("Foco: identificar 1–2 palancas (circulante, productividad o capital) para mover el desempeño.")

        story_map[str(c)] = {
            "titulo": f"{pretty[c]} (Textil) — caracterización del clúster {c}",
            "rasgos_estructurales": bullets_struct,
            "rasgos_economicos": [],
            "lectura_economica": lectura[:3],
            "implicaciones": implic[:3],
        }

    return story_map


def render_cluster_textil(
    df: pd.DataFrame,
    base_path: str | None,
    comparar_con: str,
    zoom: bool,
    k: int = 3,
):
    """
    Sector textil: repetir clustering k=3 SOLO en textil,
    y mostrar MISMA estructura que Subgrupos (tabs).
    """
    st.header("Sector textil — análisis de clustering (misma estructura que Subgrupos)")

    # 1) Base completa (para sector)
    df_full = st.session_state.get("df_base_full", None)
    if df_full is None or not isinstance(df_full, pd.DataFrame):
        st.error("Base completa no disponible (df_base_full). Revisa load_base_with_clusters.")
        st.stop()

    sector_col = _find_sector_col(df_full)
    if sector_col is None:
        st.error("No encuentro ninguna columna de sector (sector / sector_agrupado / ...).")
        st.stop()

    df_textil_full = _filter_textil(df_full, sector_col)
    if df_textil_full.empty:
        st.warning(f"No hay registros Textil (buscando 'textil' en `{sector_col}`).")
        st.stop()

    # 2) Merge con df_app para PC1/PC2/cluster_label/nombre
    merge_key = None
    if "codigo_nif" in df.columns and "codigo_nif" in df_textil_full.columns:
        merge_key = "codigo_nif"
    elif "nombre" in df.columns and "nombre" in df_textil_full.columns:
        merge_key = "nombre"

    if merge_key is None:
        st.error("No puedo cruzar df_app con base completa: no hay clave común (codigo_nif o nombre).")
        st.stop()

    cols_app_needed = [c for c in ["PC1", "PC2", "cluster_label", "nombre", "codigo_nif"] if c in df.columns]
    df_app_min = df[cols_app_needed].copy()

    df_textil = df_textil_full.merge(df_app_min, on=merge_key, how="left", suffixes=("", "_app"))

    # normaliza columnas necesarias
    if "cluster_label" not in df_textil.columns and "cluster_label_app" in df_textil.columns:
        df_textil["cluster_label"] = df_textil["cluster_label_app"]
    if "PC1" not in df_textil.columns and "PC1_app" in df_textil.columns:
        df_textil["PC1"] = df_textil["PC1_app"]
    if "PC2" not in df_textil.columns and "PC2_app" in df_textil.columns:
        df_textil["PC2"] = df_textil["PC2_app"]

    # 3) Clustering k=3 dentro del textil con VARS_CLUSTER
    vars_ok = [v for v in VARS_CLUSTER if v in df_textil.columns]
    if len(vars_ok) < 2:
        st.error("No hay suficientes variables numéricas para clusterizar en Textil (revisa VARS_CLUSTER).")
        st.stop()

    X = df_textil[vars_ok].apply(pd.to_numeric, errors="coerce")
    mask = X.notna().all(axis=1)
    df_textil2 = df_textil.loc[mask].copy()
    X2 = X.loc[mask].values

    if len(df_textil2) < (k * 10):
        st.warning(f"Textil tiene pocos casos válidos para k={k} (n={len(df_textil2)}).")
        # seguimos igualmente, pero aviso

    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    df_textil2["textil_cluster"] = km.fit_predict(X2) + 1   # 1..k

    # df "app" para reutilizar vistas: cluster_label = textil_cluster
    df_app_textil = df_textil2.copy()
    df_app_textil["cluster_label"] = df_app_textil["textil_cluster"].astype(str)

    st.caption(f"Filtrado por `{sector_col}` contiene 'textil' · n={len(df_app_textil)} (con datos completos en VARS_CLUSTER)")

    # 4) Story map para interpretación por clúster en Textil
    story_map = _build_textil_story(df_app_textil, cluster_col="cluster_label")

    # 5) Tabs: igual que Subgrupos
    tab_resumen, tab_estad, tab_arbol = st.tabs(["Resumen", "Estadística del modelo", "Árbol de decisión"])

    with tab_resumen:
        render_resumen(
            df=df_app_textil,
            comparar_con=comparar_con,
            zoom=zoom,
            base_path=base_path,
            show_subgroup_interpretation=False,   # 👈 nunca subgrupos aquí
            story_map_override=story_map,          # 👈 textos textil
            story_title="Interpretación del clúster (Textil)",
        )

    with tab_estad:
        try:
            render_estadistica(df=df_app_textil, base_path=base_path, group_col="cluster_label", title=None)
        except TypeError:
            render_estadistica(df=df_app_textil, base_path=base_path)

    with tab_arbol:
        try:
            render_arbol_decision(df_app=df_app_textil, base_path=base_path)
        except TypeError:
            render_arbol_decision(df=df_app_textil, base_path=base_path)

    st.divider()
    st.subheader("Descargas")
    st.download_button(
        "Descargar datos (Textil + clusters k=3)",
        data=df_app_textil.to_csv(index=False).encode("utf-8"),
        file_name="textil_clusters_k3.csv",
        mime="text/csv",
    )