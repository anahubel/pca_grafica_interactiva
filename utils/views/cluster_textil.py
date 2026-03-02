# utils/views/cluster_textil.py
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from utils.config import VARS_CLUSTER, LABELS
from utils.views.resumen import render_resumen
from utils.views.estadistica import render_estadistica
from utils.views.arbol_decision import render_arbol_decision

import os
import glob
from utils.config import VARS_CLUSTER, LABELS, DATA_PATH


# ============================================================
# Helpers: sector
# ============================================================
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


# ============================================================
# Pipeline (mismo espíritu que 02_clustering)
# ============================================================
def _log1p_like_r(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    # en R log1p acepta 0 y positivos; si hay negativos, quedan NaN
    x = x.where(x >= 0)
    return np.log1p(x)


def _winsorize_p1_p99(df: pd.DataFrame, cols: list[str], p1: float = 0.01, p99: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce")
        lo = x.quantile(p1)
        hi = x.quantile(p99)
        out[c] = x.clip(lower=lo, upper=hi)
    return out


def _apply_cluster_pipeline(
    df_in: pd.DataFrame,
    vars_model: list[str],
    do_trim_99: bool = True,   # recorte multivariante suave
    trim_percentile: float = 99,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Devuelve:
      - df_out: df con complete-cases y transforms aplicadas (mismas filas que X_scaled)
      - X_scaled: matriz escalada usada para clusterizar
    """
    # 1) nos quedamos con columnas necesarias + copiamos
    df = df_in.copy()

    # 2) transformaciones
    if "rotacion_stocks" in df.columns and "rotacion_stocks" in vars_model:
        df["rotacion_stocks"] = _log1p_like_r(df["rotacion_stocks"])

    # 3) complete cases
    df = df.dropna(subset=vars_model).copy()

    # 4) winsor P1-P99
    df = _winsorize_p1_p99(df, vars_model, p1=0.01, p99=0.99)

    # 5) escalado
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = df[vars_model].apply(pd.to_numeric, errors="coerce").to_numpy()
    X_scaled = scaler.fit_transform(X)

    # 6) recorte multivariante suave (quita 1% más extremo) si se activa
    if do_trim_99 and len(df) >= 20:
        centro = X_scaled.mean(axis=0, keepdims=True)
        dist = cdist(X_scaled, centro).ravel()
        umbral = np.percentile(dist, trim_percentile)
        mask = dist <= umbral
        df = df.loc[mask].copy()
        X_scaled = X_scaled[mask]

    return df, X_scaled


# ============================================================
# Story map Textil (texto por clúster)
# ============================================================
def _build_textil_story(df_in: pd.DataFrame, cluster_col: str = "cluster_label") -> dict:
    key_vars = [
        "rotacion_stocks",
        "productividad_va_pax",
        "inmovilizado_empleado",
        "nofs_ventas",
        "competitividad",
    ]
    key_vars = [v for v in key_vars if v in df_in.columns]

    if cluster_col not in df_in.columns or len(key_vars) == 0:
        return {}

    clusters = sorted(df_in[cluster_col].dropna().astype(str).unique())
    if len(clusters) < 2:
        return {}

    # medianas por cluster y variable
    med = {}
    for v in key_vars:
        tmp = df_in[[cluster_col, v]].copy()
        tmp[v] = pd.to_numeric(tmp[v], errors="coerce")
        med[v] = tmp.groupby(tmp[cluster_col].astype(str))[v].median()

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

    pretty = {c: f"T{idx+1}" for idx, c in enumerate(clusters)}

    story_map = {}
    for c in clusters:
        bullets = []
        flags = {}
        for v in key_vars:
            lab = _rank_label(v, c)
            flags[v] = lab
            bullets.append(f"{LABELS.get(v, v)}: {lab}")

        lectura = []
        implic = []

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
            "rasgos_estructurales": bullets,
            "rasgos_economicos": [],
            "lectura_economica": lectura[:3],
            "implicaciones": implic[:3],
        }

    return story_map

def _read_any(path: str) -> pd.DataFrame | None:
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".parquet":
            return pd.read_parquet(path)
        if ext == ".csv":
            return pd.read_csv(path)
        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(path)
    except Exception:
        return None
    return None


def _list_candidate_files(root: str) -> list[str]:
    if not root:
        return []
    if os.path.isfile(root):
        return [root]
    if os.path.isdir(root):
        patterns = ["**/*.parquet", "**/*.csv", "**/*.xlsx", "**/*.xls"]
        out = []
        for p in patterns:
            out.extend(glob.glob(os.path.join(root, p), recursive=True))
        return out
    return []


@st.cache_data(show_spinner=False)
def _find_base_with_sector(base_path: str | None) -> tuple[pd.DataFrame | None, str | None]:
    """
    Busca una base que contenga alguna columna de sector.
    Prioriza base_path y rutas típicas del proyecto.
    """
    search_roots: list[str] = []

    if base_path:
        search_roots.append(base_path)
        if os.path.isfile(base_path):
            search_roots.append(os.path.dirname(base_path))

    if DATA_PATH:
        search_roots.append(os.path.dirname(DATA_PATH))

    if os.path.isdir("data"):
        search_roots.append("data")

    candidates = []
    for r in search_roots:
        candidates.extend(_list_candidate_files(r))

    # quitar duplicados manteniendo orden
    candidates = list(dict.fromkeys(candidates))

    # scoring: prioriza "clean / completa / raw / base"
    def score(p: str) -> int:
        name = os.path.basename(p).lower()
        s = 0
        if any(k in name for k in ["clean", "completa", "full", "raw", "base", "original"]):
            s += 10
        if "clusters" in name:
            s -= 5
        return -s

    candidates = sorted(candidates, key=score)

    for f in candidates:
        df_try = _read_any(f)
        if df_try is None:
            continue
        if _find_sector_col(df_try) is not None:
            return df_try, f

    return None, None

# ============================================================
# Vista principal
# ============================================================
def render_cluster_textil(
    df: pd.DataFrame,                  # df "app" (suele tener PC1/PC2/cluster_label/nombre)
    base_path: str | None,
    comparar_con: str,
    zoom: bool,
    k: int = 3,
):
    st.header("Sector textil — clustering interno (k=3)")

    # 1) Cargar base completa SIEMPRE desde disco (NO session_state)
    #    -> evita el comportamiento “si pasé por otra pestaña cambia”
    #    Usamos base_path si viene; si no, usamos el df actual como fallback (pero avisamos).
    df_full, src = _find_base_with_sector(base_path)

    if df_full is None:
        st.error(
            "No pude encontrar ninguna base con columna de sector.\n"
            "Asegúrate de que base_path apunta a un fichero/carpeta donde esté la base completa "
            "(p.ej. data/interim/data_clean.parquet o tu base original con 'sector')."
        )
        st.stop()

    st.caption(f"Base usada para sector: {src}")

    # 2) Detectar sector y filtrar textil
    sector_col = _find_sector_col(df_full)
    if sector_col is None:
        st.error("No encuentro ninguna columna de sector (sector / sector_agrupado / ...).")
        st.stop()

    df_textil_full = _filter_textil(df_full, sector_col)
    if df_textil_full.empty:
        st.warning(f"No hay registros Textil (buscando 'textil' en `{sector_col}`).")
        st.stop()

    # 3) Merge para traer PC1/PC2/nombre del df de la app (si existe)
    merge_key = None
    if "codigo_nif" in df.columns and "codigo_nif" in df_textil_full.columns:
        merge_key = "codigo_nif"
    elif "nombre" in df.columns and "nombre" in df_textil_full.columns:
        merge_key = "nombre"

    if merge_key is not None:
        cols_app = [c for c in ["PC1", "PC2", "nombre", "codigo_nif"] if c in df.columns]
        df_app_min = df[cols_app].drop_duplicates(subset=[merge_key]).copy()
        df_textil = df_textil_full.merge(df_app_min, on=merge_key, how="left", suffixes=("", "_app"))

        # normaliza PC1/PC2 si quedaron con sufijo
        if "PC1" not in df_textil.columns and "PC1_app" in df_textil.columns:
            df_textil["PC1"] = df_textil["PC1_app"]
        if "PC2" not in df_textil.columns and "PC2_app" in df_textil.columns:
            df_textil["PC2"] = df_textil["PC2_app"]
    else:
        df_textil = df_textil_full.copy()
        st.warning("No pude cruzar con df_app (no hay codigo_nif/nombre común). PCA podría no mostrarse.")

    # 4) Aplicar pipeline igual que 02_clustering sobre VARS_CLUSTER
    vars_model = [v for v in VARS_CLUSTER if v in df_textil.columns]
    if len(vars_model) < 2:
        st.error("No hay suficientes variables numéricas para clusterizar (revisa VARS_CLUSTER).")
        st.stop()

    do_trim = st.checkbox("Recorte multivariante suave (quita 1% más extremo)", value=True)

    df_pipe, X_scaled = _apply_cluster_pipeline(
        df_textil,
        vars_model=vars_model,
        do_trim_99=do_trim,
        trim_percentile=99,
    )

    if len(df_pipe) < max(20, k * 10):
        st.warning(f"Pocos casos válidos en Textil tras pipeline: n={len(df_pipe)} (k={k}).")

    # 5) KMeans
    km = KMeans(n_clusters=k, n_init=50, random_state=123, algorithm="lloyd")
    labels = km.fit_predict(X_scaled) + 1  # 1..k

    df_app_textil = df_pipe.copy()
    df_app_textil["textil_cluster"] = labels
    df_app_textil["cluster_label"] = df_app_textil["textil_cluster"].astype(str)  # para reutilizar vistas

    st.caption(
        f"Filtrado por `{sector_col}` contiene 'textil' · "
        f"n={len(df_textil_full)} (antes) → n={len(df_app_textil)} (tras pipeline + complete cases)"
    )
    st.write("Tamaños de cluster (textil):")
    st.dataframe(df_app_textil["textil_cluster"].value_counts().sort_index().rename("N").to_frame(), use_container_width=True)

    # 6) Story map (texto por cluster en Textil)
    story_map = _build_textil_story(df_app_textil, cluster_col="cluster_label")

    # 7) Tabs
    tab_resumen, tab_estad, tab_arbol = st.tabs(["Resumen", "Estadística del modelo", "Árbol de decisión"])

    with tab_resumen:
        render_resumen(
            df=df_app_textil,
            comparar_con=comparar_con,
            zoom=zoom,
            base_path=base_path,
            show_subgroup_interpretation=False,
            story_map_override=story_map,
            story_title="Interpretación del clúster (Textil)",
            normalize_cluster_labels=False,  # aquí cluster_label es "1","2","3"
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

    # 8) Descarga
    st.divider()
    st.subheader("Descargas")
    st.download_button(
        "Descargar datos (Textil + pipeline + clusters)",
        data=df_app_textil.to_csv(index=False).encode("utf-8"),
        file_name="textil_clusters_k3_pipeline.csv",
        mime="text/csv",
    )
