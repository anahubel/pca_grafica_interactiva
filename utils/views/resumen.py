# utils/views/resumen.py
import io
import os
import glob

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.preprocessing import StandardScaler

from utils.config import VARS_CLUSTER, LABELS, DATA_PATH
from utils.fmt import to_display_scale, fmt_num
from utils.ui import plotly_layout_base, anchor

from sklearn.linear_model import LinearRegression

# ============================================================
# Helpers: encontrar y mergear ingresos_de_explotacion
# ============================================================
def _read_any(path: str) -> pd.DataFrame | None:
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(path)
        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        if ext == ".parquet":
            return pd.read_parquet(path)
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
        files: list[str] = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(root, p), recursive=True))
        return files
    return []

@st.cache_data(show_spinner=False)
def _find_base_with_ingresos(base_path: str | None) -> tuple[pd.DataFrame | None, str | None]:
    search_roots: list[str] = []

    if base_path:
        search_roots.append(base_path)
        if os.path.isfile(base_path):
            search_roots.append(os.path.dirname(base_path))

    if DATA_PATH:
        search_roots.append(os.path.dirname(DATA_PATH))

    if os.path.isdir("data"):
        search_roots.append("data")

    candidates: list[str] = []
    for r in search_roots:
        candidates.extend(_list_candidate_files(r))

    def score(p: str) -> int:
        name = os.path.basename(p).lower()
        s = 0
        if any(k in name for k in ["original", "completa", "full", "raw", "base"]):
            s += 10
        if "clean" in name:
            s -= 2
        if "interim" in p.lower():
            s -= 1
        return -s  # sort asc

    candidates = sorted(list(dict.fromkeys(candidates)), key=score)

    for f in candidates:
        df_full = _read_any(f)
        if df_full is None:
            continue
        if "ingresos_de_explotacion" in df_full.columns:
            return df_full, f

    return None, None

def _ensure_ingresos(df_view: pd.DataFrame, base_path: str | None, diagnostic: bool = False) -> pd.DataFrame:
    if "ingresos_de_explotacion" in df_view.columns:
        return df_view

    df_full, _src = _find_base_with_ingresos(base_path)
    if df_full is None:
        if diagnostic:
            st.warning(
                "No encuentro ninguna base con 'ingresos_de_explotacion' "
                "(busqué en base_path, carpeta de DATA_PATH y ./data)."
            )
        return df_view

    merge_key = None
    if "codigo_nif" in df_view.columns and "codigo_nif" in df_full.columns:
        merge_key = "codigo_nif"
    elif "nombre" in df_view.columns and "nombre" in df_full.columns:
        merge_key = "nombre"

    if merge_key is None:
        if diagnostic:
            st.warning("No puedo hacer merge: no hay clave común (codigo_nif o nombre).")
        return df_view

    df_full2 = df_full[[merge_key, "ingresos_de_explotacion"]].copy()
    df_full2["ingresos_de_explotacion"] = pd.to_numeric(df_full2["ingresos_de_explotacion"], errors="coerce")
    out = df_view.merge(df_full2, on=merge_key, how="left")

    if diagnostic:
        st.caption(f"Base ingresos usada: {_src}")
        st.caption(f"Merge key: {merge_key} · ingresos no-nulo: {out['ingresos_de_explotacion'].notna().sum()}")

    return out

# ============================================================
# Helpers: UMAP (2D)
# ============================================================
@st.cache_data(show_spinner=False)
def _compute_umap_embedding(
    df_in: pd.DataFrame,
    vars_model: list[str],
    n_neighbors: int = 15,
    min_dist: float = 0.10,
    metric: str = "euclidean",
    random_state: int = 42,
) -> pd.DataFrame:
    try:
        import umap  # umap-learn
    except Exception as e:
        raise ImportError(
            "Falta la librería 'umap-learn'. Añádela a requirements.txt (umap-learn) "
            "y reinicia la app en Streamlit Cloud."
        ) from e

    df = df_in.copy()
    vars_ok = [v for v in vars_model if v in df.columns]
    if len(vars_ok) < 2:
        return pd.DataFrame()

    X = df[vars_ok].apply(pd.to_numeric, errors="coerce")
    mask = X.notna().all(axis=1)
    if mask.sum() < 10:
        return pd.DataFrame()

    X2 = X.loc[mask].to_numpy()
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X2)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=metric,
        random_state=int(random_state),
    )
    emb = reducer.fit_transform(Xs)

    out = df.loc[mask].copy()
    out["__umap1"] = emb[:, 0]
    out["__umap2"] = emb[:, 1]

    keep = [c for c in ["__umap1", "__umap2", "cluster_label", "nombre", "codigo_nif"] if c in out.columns]
    return out[keep].copy()

# ============================================================
# Helpers UI / hover
# ============================================================
def _normalize_cluster_label(x) -> str:
    m = {"1": "C1", "1.0": "C1", "2": "C2", "2.0": "C2", "3": "C3", "3.0": "C3"}
    s = str(x).strip()
    return m.get(s, s)

def _hovertemplate_basic() -> str:
    return (
        "<b>%{customdata[0]}</b>"
        "<br>NIF: %{customdata[1]}"
        "<br>Clúster: %{customdata[2]}"
        "<extra></extra>"
    )

def _customdata(df_: pd.DataFrame) -> np.ndarray:
    nombre = df_["nombre"].astype(str) if "nombre" in df_.columns else pd.Series(["—"] * len(df_))
    nif = df_["codigo_nif"].astype(str) if "codigo_nif" in df_.columns else pd.Series(["—"] * len(df_))
    cl = df_["cluster_label"].astype(str) if "cluster_label" in df_.columns else pd.Series(["—"] * len(df_))
    return np.stack([nombre.to_numpy(), nif.to_numpy(), cl.to_numpy()], axis=1)

def _selected_marker_style(color_hex: str) -> dict:
    return dict(size=16, symbol="circle", color=color_hex, line=dict(width=3, color="white"))

def _find_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _label(col: str) -> str:
    return LABELS.get(col, col)


def _make_label_maps(cols: list[str]) -> tuple[list[str], dict[str, str]]:
    """
    Evita colisiones de labels. Si dos columnas tienen el mismo label,
    añade [nombre_columna].
    """
    labels = [_label(c) for c in cols]
    seen: dict[str, int] = {}
    out = []

    for c, lab in zip(cols, labels):
        if lab in seen:
            seen[lab] += 1
            out.append(f"{lab} [{c}]")
        else:
            seen[lab] = 1
            out.append(lab)

    lab_to_col = {lab: c for lab, c in zip(out, cols)}
    return out, lab_to_col


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Convierte a numérico soportando formatos tipo:
      - 1.234,56
      - 5,9E+05
      - 5.9E+05
    """
    if s is None:
        return pd.Series([], dtype=float)

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    x = s.astype(str).str.strip()
    x = x.str.replace("\u00a0", "", regex=False)

    mask_sci = x.str.contains(r"[eE]", na=False)
    if mask_sci.any():
        xs = x.where(mask_sci, "")
        xs = xs.str.replace(",", ".", regex=False)
        x = x.where(~mask_sci, xs)

    mask_es = (~mask_sci) & x.str.contains(r"\.", na=False) & x.str.contains(r",", na=False)
    if mask_es.any():
        xe = x.where(mask_es, "")
        xe = xe.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        x = x.where(~mask_es, xe)

    mask_comma = (~mask_sci) & (~mask_es) & x.str.contains(",", na=False)
    if mask_comma.any():
        xc = x.where(mask_comma, "")
        xc = xc.str.replace(",", ".", regex=False)
        x = x.where(~mask_comma, xc)

    return pd.to_numeric(x, errors="coerce")

def _compute_interaction_surface(
    df_in: pd.DataFrame,
    x_var: str,
    y_var: str,
    z_var: str,
    grid_n: int = 35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame] | None:
    """
    Ajusta una superficie simple con interacción:
        z ~ x + y + x*y
    y devuelve una malla X, Y, Zhat para representar en 3D.

    También devuelve el dataframe limpio usado en el ajuste.
    """
    cols = [x_var, y_var, z_var]
    if any(c not in df_in.columns for c in cols):
        return None

    d = df_in[cols].copy()
    for c in cols:
        d[c] = _coerce_numeric_series(d[c])
    d = d.dropna().copy()

    if len(d) < 20:
        return None

    X = pd.DataFrame({
        x_var: d[x_var].astype(float),
        y_var: d[y_var].astype(float),
        "xy": (d[x_var] * d[y_var]).astype(float),
    })
    y = d[z_var].astype(float)

    model = LinearRegression()
    model.fit(X, y)

    x_grid = np.linspace(float(d[x_var].min()), float(d[x_var].max()), int(grid_n))
    y_grid = np.linspace(float(d[y_var].min()), float(d[y_var].max()), int(grid_n))
    XX, YY = np.meshgrid(x_grid, y_grid)

    X_pred = pd.DataFrame({
        x_var: XX.ravel(),
        y_var: YY.ravel(),
        "xy": (XX * YY).ravel(),
    })
    ZZ = model.predict(X_pred).reshape(XX.shape)

    return XX, YY, ZZ, d


def _predict_surface_point(
    x_val: float,
    y_val: float,
    df_fit: pd.DataFrame,
    x_var: str,
    y_var: str,
    z_var: str,
) -> float | None:
    """
    Predicción puntual usando el mismo modelo:
        z ~ x + y + x*y
    """
    d = df_fit[[x_var, y_var, z_var]].copy()
    d[x_var] = _coerce_numeric_series(d[x_var])
    d[y_var] = _coerce_numeric_series(d[y_var])
    d[z_var] = _coerce_numeric_series(d[z_var])
    d = d.dropna().copy()

    if len(d) < 20 or pd.isna(x_val) or pd.isna(y_val):
        return None

    X = pd.DataFrame({
        x_var: d[x_var].astype(float),
        y_var: d[y_var].astype(float),
        "xy": (d[x_var] * d[y_var]).astype(float),
    })
    y = d[z_var].astype(float)

    model = LinearRegression()
    model.fit(X, y)

    pt = pd.DataFrame({
        x_var: [float(x_val)],
        y_var: [float(y_val)],
        "xy": [float(x_val) * float(y_val)],
    })

    return float(model.predict(pt)[0])

# ============================================================
# Vista Resumen
# ============================================================
def render_resumen(
    df: pd.DataFrame,
    comparar_con: str,
    zoom: bool,
    base_path: str | None = None,
    show_subgroup_interpretation: bool = False,
    story_map_override: dict | None = None,
    story_title: str = "Interpretación del clúster",
    normalize_cluster_labels: bool = True,
):
    df = df.copy()

    # ======================
    # SELECTOR EMPRESA
    # ======================
    if "codigo_nif" in df.columns and "nombre" in df.columns:
        df["empresa_key"] = df["nombre"].astype(str) + "  —  " + df["codigo_nif"].astype(str)
        selector_col = "empresa_key"
    else:
        selector_col = "nombre" if "nombre" in df.columns else df.columns[0]

    empresa_sel = st.selectbox("Busca/selecciona empresa", sorted(df[selector_col].dropna().unique()))
    row = df.loc[df[selector_col] == empresa_sel].iloc[0]

    # ======================
    # Normaliza cluster_label
    # ======================
    _raw_cluster = row.get("cluster_label", "—")
    cluster_sel = _normalize_cluster_label(_raw_cluster) if normalize_cluster_labels else str(_raw_cluster).strip()

    # ======================
    # Referencia (cluster o total)
    # ======================
    if comparar_con == "Solo su cluster" and "cluster_label" in df.columns and "cluster_label" in row.index:
        df_ref = df[df["cluster_label"] == _raw_cluster].copy()
        if df_ref.empty and cluster_sel in {"C1", "C2", "C3"}:
            df_ref = df[df["cluster_label"].astype(str) == cluster_sel].copy()
    else:
        df_ref = df.copy()

    CLUSTER_N = df["cluster_label"].value_counts(dropna=True).to_dict() if "cluster_label" in df.columns else {}
    n_sel = int(CLUSTER_N.get(_raw_cluster, CLUSTER_N.get(cluster_sel, 0)))
    pct_sel = (n_sel / max(1, len(df))) * 100.0

    # ======================
    # Story (igual que lo tenías)
    # ======================
    CLUSTER_STORY = {
        "C1": {
            "titulo": "C1: modelo intensivo en circulante y eficiencia media-baja",
            "rasgos_estructurales": [
                "Rotación de stocks: la más baja del conjunto (ciclo operativo más lento).",
                "Productividad VA/pax: la más reducida (menor eficiencia en generación de valor).",
                "Inmovilizado/empleado: nivel intermedio (estructura productiva moderada).",
                "NOFS/Ventas: el más elevado (mayor presión de circulante y financiación operativa).",
            ],
            "rasgos_economicos": [
                "Resultados y rentabilidad: peor desempeño (resultado del ejercicio, explotación, EBITDA y cash flow).",
                "Estructura: mayor % de personal y márgenes contenidos, coherente con un modelo más intensivo en operativa y menos eficiente.",
            ],
            "lectura_economica": [
                "Empresas con ciclo operativo más lento y mayor necesidad de financiar existencias y clientes.",
                "Menor eficiencia productiva por trabajador y rentabilidad limitada.",
                "Modelo tradicional y operativo, con menor dinamismo comercial.",
            ],
            "implicaciones": [
                "Mejoras típicas: acelerar rotación, optimizar existencias/cobros/pagos y reducir tensión de circulante.",
                "Prioridad: elevar productividad (procesos/organización/eficiencia).",
                "Modelo más vulnerable: un shock de demanda o costes se traslada rápido a resultados.",
            ],
        },
        "C2": {
            "titulo": "C2: modelo ágil y eficiente en circulante (perfil competitivo)",
            "rasgos_estructurales": [
                "Rotación de stocks: la más alta del conjunto (máxima agilidad operativa/comercial).",
                "NOFS/Ventas: el más bajo (modelo más “ligero” en circulante, menor necesidad de financiación operativa).",
                "Inmovilizado/empleado: relativamente bajo (estructura menos rígida).",
                "Productividad VA/pax: nivel intermedio (mejora más por rotación/ejecución que por capital).",
            ],
            "rasgos_economicos": [
                "Competitividad: niveles más altos (perfil dinámico y orientado a ejecución).",
                "Resultados: moderados/positivos, típicamente mejores que C1 pero sin el perfil capital-intensivo de C3.",
            ],
            "lectura_economica": [
                "Empresas con ciclo operativo rápido, buena disciplina de circulante y alto dinamismo comercial.",
                "Modelo eficiente en capital circulante: aguanta mejor tensiones de liquidez.",
                "Compite por velocidad/ejecución más que por intensidad de capital.",
            ],
            "implicaciones": [
                "Modelo resiliente en términos de liquidez: menos exposición a tensiones por circulante.",
                "Palanca de mejora: subir margen/eficiencia sin perder rotación.",
                "Sensibilidad: necesita sostener volumen/rotación; caídas de demanda pueden impactar rápido.",
            ],
        },
        "C3": {
            "titulo": "C3: modelo intensivo en capital y alta productividad",
            "rasgos_estructurales": [
                "Inmovilizado/empleado: el más alto (modelo intensivo en capital/activos).",
                "Productividad VA/pax: la más alta (alto valor generado por unidad de trabajo).",
                "Rotación de stocks: intermedia (no compite por velocidad sino por estructura/productividad).",
            ],
            "rasgos_economicos": [
                "Resultados: superiores (mayor generación de EBITDA, cash flow y resultado del ejercicio).",
                "Escala: niveles de activo y fondos propios más altos, coherente con estructura intensiva en capital.",
            ],
            "lectura_economica": [
                "Empresas con estructura productiva fuerte, inversión relevante en activos y alta productividad.",
                "Modelo más sofisticado (posible perfil industrial/tecnológico o con economías de escala).",
                "Rinde bien si mantiene utilización eficiente de la capacidad instalada.",
            ],
            "implicaciones": [
                "Modelo robusto: mayor capacidad de inversión y resistencia ante shocks moderados.",
                "Riesgo típico: rigidez (costes fijos/capex) si cae demanda o baja utilización de activos.",
                "Foco: asegurar eficiencia del capital y disciplina de inversión/mantenimiento.",
            ],
        },
    }

    _story_map = story_map_override if isinstance(story_map_override, dict) else CLUSTER_STORY
    story = _story_map.get(
        str(cluster_sel),
        {
            "titulo": "Interpretación no definida",
            "rasgos_estructurales": ["Define aquí el texto del clúster."],
            "rasgos_economicos": [],
            "lectura_economica": [],
            "implicaciones": [],
        },
    )

    # ======================
    # Funciones
    # ======================
    def empirical_percentile(s: pd.Series, x: float) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0 or pd.isna(x):
            return np.nan
        return float((s <= x).mean() * 100.0)

    def robust_z_iqr(s_disp: pd.Series, x_disp: float) -> float:
        s_disp = pd.to_numeric(s_disp, errors="coerce").dropna()
        if len(s_disp) == 0 or pd.isna(x_disp):
            return np.nan
        med = float(s_disp.quantile(0.50))
        q1 = float(s_disp.quantile(0.25))
        q3 = float(s_disp.quantile(0.75))
        iqr = (q3 - q1) if (q3 - q1) != 0 else 1e-9
        z = (float(x_disp) - med) / iqr
        z = max(-2.0, min(2.0, z))
        return float(z)

    def overall_index(scores: list[float]) -> float:
        vals = [v for v in scores if pd.notna(v)]
        if not vals:
            return np.nan
        m = float(np.mean(vals))
        return (m + 2.0) / 4.0 * 100.0

    # ======================
    # LAYOUT PRINCIPAL
    # ======================
    left, right = st.columns([2.2, 1.3], gap="large")

    # ============================================================
    # LEFT
    # ============================================================
    with left:
        anchor("pca")

        # ======================
        # PCA
        # ======================
        with st.expander("PCA", expanded=True):
            if ("PC1" not in df.columns) or ("PC2" not in df.columns) or ("PC1" not in row.index) or ("PC2" not in row.index):
                st.warning("No puedo mostrar el gráfico PCA: faltan columnas PC1 y/o PC2.")
            else:
                color_map = {
                    "C1": "#1f3b73", "C2": "#d97706", "C3": "#15803d",
                    "1": "#1f3b73", "2": "#d97706", "3": "#15803d",
                    "1.0": "#1f3b73", "2.0": "#d97706", "3.0": "#15803d",
                }

                df_plot = df.copy()
                if "cluster_label" in df_plot.columns:
                    df_plot["cluster_label"] = df_plot["cluster_label"].astype(str).map(_normalize_cluster_label)

                fig = px.scatter(
                    df_plot,
                    x="PC1",
                    y="PC2",
                    color="cluster_label" if "cluster_label" in df_plot.columns else None,
                    opacity=0.75,
                    color_discrete_map=color_map,
                    labels={
                        "PC1": "Componente principal 1",
                        "PC2": "Componente principal 2",
                        "cluster_label": "Modelo de negocio",
                    },
                    custom_data=["nombre", "codigo_nif", "cluster_label"] if {"nombre", "codigo_nif", "cluster_label"}.issubset(df_plot.columns) else None,
                )
                
                fig.update_traces(
                    marker=dict(size=8, line=dict(width=0.5, color="white")),
                    hovertemplate=_hovertemplate_basic(),
                )

                plotly_layout_base(fig, height=520, margin=dict(l=30, r=30, t=60, b=30))
                fig.update_layout(legend=dict(orientation="h", y=-0.18))

                sel_color = color_map.get(cluster_sel, "#111111")
                fig.add_trace(
                    go.Scatter(
                        x=[float(row["PC1"])],
                        y=[float(row["PC2"])],
                        mode="markers",
                        marker=_selected_marker_style(sel_color),
                        showlegend=False,
                        customdata=np.array([[str(row.get("nombre", "—")), str(row.get("codigo_nif", "—")), str(cluster_sel)]]),
                        hovertemplate=_hovertemplate_basic(),
                    )
                )

                if zoom:
                    fig.update_xaxes(range=[float(row["PC1"]) - 1.0, float(row["PC1"]) + 1.0])
                    fig.update_yaxes(range=[float(row["PC2"]) - 1.0, float(row["PC2"]) + 1.0])

                st.plotly_chart(fig, use_container_width=True)

        # ======================
        # UMAP
        # ======================
        with st.expander("UMAP — proyección no lineal (sobre variables del clustering)", expanded=False):
            vars_model_umap = [v for v in VARS_CLUSTER if v in df.columns]
            if len(vars_model_umap) < 2:
                st.warning("No puedo calcular UMAP: faltan variables de VARS_CLUSTER.")
            else:
                c1, c2, c3 = st.columns([1.0, 1.0, 1.0])
                with c1:
                    umap_neighbors = st.slider("n_neighbors", 5, 80, 15, 1)
                with c2:
                    umap_min_dist = st.slider("min_dist", 0.0, 0.99, 0.10, 0.01)
                with c3:
                    umap_seed = st.number_input("random_state", min_value=0, max_value=10_000, value=42, step=1)
        
                try:
                    df_umap = _compute_umap_embedding(
                        df_in=df,
                        vars_model=vars_model_umap,
                        n_neighbors=umap_neighbors,
                        min_dist=umap_min_dist,
                        random_state=umap_seed,
                    )
                except ImportError as e:
                    st.error(str(e))
                    df_umap = pd.DataFrame()
        
                if df_umap.empty:
                    st.warning("UMAP no disponible (muy pocos casos válidos o faltan datos).")
                else:
                    df_umap = df_umap.copy()
                    df_umap["cluster_label"] = df_umap["cluster_label"].astype(str).map(_normalize_cluster_label)
        
                    color_map_umap = {"C1": "#1f3b73", "C2": "#d97706", "C3": "#15803d"}
        
                    use_custom = {"nombre", "codigo_nif", "cluster_label"}.issubset(df_umap.columns)
        
                    fig_u = px.scatter(
                        df_umap,
                        x="__umap1",
                        y="__umap2",
                        color="cluster_label",
                        opacity=0.85,
                        color_discrete_map=color_map_umap,
                        labels={
                            "__umap1": "UMAP 1",
                            "__umap2": "UMAP 2",
                            "cluster_label": "Modelo de negocio",
                        },
                        custom_data=["nombre", "codigo_nif", "cluster_label"] if use_custom else None,
                    )
        
                    fig_u.update_traces(
                        marker=dict(size=8, line=dict(width=0.5, color="white")),
                        hovertemplate=_hovertemplate_basic(),
                    )
        
                    plotly_layout_base(fig_u, height=520, margin=dict(l=30, r=30, t=60, b=30))
                    fig_u.update_layout(legend=dict(orientation="h", y=-0.18))
        
                    sel_name = str(row.get("nombre", "")).strip()
                    sel_nif = str(row.get("codigo_nif", "")).strip()
        
                    sub = pd.DataFrame()
                    if sel_nif and "codigo_nif" in df_umap.columns:
                        sub = df_umap[df_umap["codigo_nif"].astype(str).str.strip() == sel_nif].copy()
                    if sub.empty and sel_name and "nombre" in df_umap.columns:
                        sub = df_umap[df_umap["nombre"].astype(str).str.strip() == sel_name].copy()
        
                    if not sub.empty:
                        sel_color = color_map_umap.get(cluster_sel, "#111111")
                        fig_u.add_trace(
                            go.Scatter(
                                x=[float(sub.iloc[0]["__umap1"])],
                                y=[float(sub.iloc[0]["__umap2"])],
                                mode="markers",
                                marker=_selected_marker_style(sel_color),
                                showlegend=False,
                                customdata=np.array([[sel_name or "—", sel_nif or "—", str(cluster_sel)]]),
                                hovertemplate=_hovertemplate_basic(),
                            )
                        )
        
                    st.plotly_chart(fig_u, use_container_width=True)
                    st.caption("UMAP conserva vecindarios locales, pero las distancias globales no son directamente comparables.")

        # ======================
        # 3D
        # ======================
        with st.expander("3D — espacio del clustering (variables escaladas)", expanded=False):
            vars_model_3d = [v for v in ["rotacion_stocks", "productividad_va_pax", "inmovilizado_empleado"] if v in df.columns]
            if len(vars_model_3d) < 3:
                st.warning("No puedo mostrar 3D: necesito rotacion_stocks, productividad_va_pax e inmovilizado_empleado.")
            else:
                cA, cB, cC, cD = st.columns([1.0, 1.0, 1.0, 1.2])
                with cA:
                    rest_opacity = st.slider("Opacidad resto", 0.05, 0.90, 0.35, 0.05)
                with cB:
                    rest_size = st.slider("Tamaño resto", 2.0, 8.0, 4.0, 0.5)
                with cC:
                    sel_size = st.slider("Tamaño seleccionada", 4.0, 16.0, 10.0, 0.5)
                with cD:
                    show_selected = st.checkbox("Mostrar empresa seleccionada", value=True)

                zoom_3d = st.checkbox("Zoom 3D a la seleccionada", value=False)
                zoom_radius = st.slider("Radio zoom (z-score)", 0.5, 4.0, 2.2, 0.1) if zoom_3d else None

                df_3d = df.copy()
                for v in vars_model_3d:
                    df_3d[v] = pd.to_numeric(df_3d[v], errors="coerce")
                df_3d["cluster_label"] = df_3d["cluster_label"].astype(str).map(_normalize_cluster_label)

                need_cols = vars_model_3d + ["cluster_label"]
                df_3d = df_3d.dropna(subset=need_cols).copy()
                if df_3d.empty:
                    st.warning("No hay casos completos para construir el 3D.")
                else:
                    scaler = StandardScaler(with_mean=True, with_std=True)
                    Xs = scaler.fit_transform(df_3d[vars_model_3d].to_numpy())

                    df_3d["__z1"] = Xs[:, 0]
                    df_3d["__z2"] = Xs[:, 1]
                    df_3d["__z3"] = Xs[:, 2]

                    sel_name = str(row.get("nombre", "")).strip()
                    sel_nif = str(row.get("codigo_nif", "")).strip()

                    df_sel = pd.DataFrame()
                    if sel_nif and "codigo_nif" in df_3d.columns:
                        df_sel = df_3d[df_3d["codigo_nif"].astype(str).str.strip() == sel_nif].copy()
                    if df_sel.empty and sel_name and "nombre" in df_3d.columns:
                        df_sel = df_3d[df_3d["nombre"].astype(str).str.strip() == sel_name].copy()

                    color_map_3d = {"C1": "#1f3b73", "C2": "#d97706", "C3": "#15803d"}

                    fig3d = go.Figure()

                    for cl in ["C1", "C2", "C3"]:
                        dcl = df_3d[df_3d["cluster_label"].astype(str) == cl]
                        if dcl.empty:
                            continue
                        fig3d.add_trace(
                            go.Scatter3d(
                                x=dcl["__z1"],
                                y=dcl["__z2"],
                                z=dcl["__z3"],
                                mode="markers",
                                name=cl,
                                marker=dict(size=float(rest_size), opacity=float(rest_opacity), color=color_map_3d.get(cl, "#888")),
                                customdata=_customdata(dcl),
                                hovertemplate=_hovertemplate_basic(),
                            )
                        )

                    if show_selected and (not df_sel.empty):
                        x0 = float(df_sel.iloc[0]["__z1"])
                        y0 = float(df_sel.iloc[0]["__z2"])
                        z0 = float(df_sel.iloc[0]["__z3"])
                        sel_color = color_map_3d.get(cluster_sel, "#111111")

                        fig3d.add_trace(
                            go.Scatter3d(
                                x=[x0],
                                y=[y0],
                                z=[z0],
                                mode="markers",
                                marker=dict(
                                    size=float(sel_size),
                                    color=sel_color,
                                    opacity=1.0,
                                    line=dict(width=4, color="white"),
                                ),
                                showlegend=False,
                                customdata=np.array([[sel_name or "—", sel_nif or "—", str(cluster_sel)]]),
                                hovertemplate=_hovertemplate_basic(),
                            )
                        )

                        if zoom_3d and zoom_radius is not None:
                            fig3d.update_layout(
                                scene=dict(
                                    xaxis=dict(range=[x0 - zoom_radius, x0 + zoom_radius]),
                                    yaxis=dict(range=[y0 - zoom_radius, y0 + zoom_radius]),
                                    zaxis=dict(range=[z0 - zoom_radius, z0 + zoom_radius]),
                                )
                            )
                    elif show_selected and df_sel.empty:
                        st.info("La empresa seleccionada no aparece en el 3D (NaN en alguna de las 3 variables).")

                    plotly_layout_base(fig3d, height=520, margin=dict(l=30, r=30, t=60, b=30))
                    fig3d.update_layout(
                        scene=dict(
                            xaxis_title=f"{LABELS.get(vars_model_3d[0], vars_model_3d[0])} (z)",
                            yaxis_title=f"{LABELS.get(vars_model_3d[1], vars_model_3d[1])} (z)",
                            zaxis_title=f"{LABELS.get(vars_model_3d[2], vars_model_3d[2])} (z)",
                        ),
                        scene_camera=dict(eye=dict(x=1.35, y=1.35, z=1.10)),
                    )
                    st.plotly_chart(fig3d, use_container_width=True)

        # ======================
        # 3D SUPERFICIE
        # ======================
        with st.expander("3D — superficie de interacción", expanded=False):
            try:
                x_default = _find_first_existing(df, ["edad", "Edad", "EDAD"])
                y_default = _find_first_existing(df, ["precaglobal", "PRECAGLOB", "precaglob", "precariedad_global"])
                z_default = _find_first_existing(df, ["bienestar", "Bienestar", "BIENESTAR"])
        
                numeric_candidates = []
                for c in df.columns:
                    s = _coerce_numeric_series(df[c])
                    if s.notna().sum() >= 20:
                        numeric_candidates.append(c)
        
                if len(numeric_candidates) < 3:
                    st.warning("No hay suficientes variables numéricas para construir la superficie 3D.")
                else:
                    lbls_3d, lbl_to_var_3d = _make_label_maps(sorted(numeric_candidates, key=lambda c: _label(c)))
        
                    x_idx = 0
                    y_idx = min(1, len(lbls_3d) - 1)
                    z_idx = min(2, len(lbls_3d) - 1)
        
                    if x_default and x_default in numeric_candidates:
                        x_lab_default = next((lab for lab, var in lbl_to_var_3d.items() if var == x_default), None)
                        if x_lab_default in lbls_3d:
                            x_idx = lbls_3d.index(x_lab_default)
        
                    if y_default and y_default in numeric_candidates:
                        y_lab_default = next((lab for lab, var in lbl_to_var_3d.items() if var == y_default), None)
                        if y_lab_default in lbls_3d:
                            y_idx = lbls_3d.index(y_lab_default)
        
                    if z_default and z_default in numeric_candidates:
                        z_lab_default = next((lab for lab, var in lbl_to_var_3d.items() if var == z_default), None)
                        if z_lab_default in lbls_3d:
                            z_idx = lbls_3d.index(z_lab_default)
        
                    cA, cB, cC = st.columns([1, 1, 1])
                    with cA:
                        x_lab = st.selectbox("Eje X", lbls_3d, index=x_idx, key="surf_x")
                    with cB:
                        y_lab = st.selectbox("Eje Y", lbls_3d, index=y_idx, key="surf_y")
                    with cC:
                        z_lab = st.selectbox("Eje Z", lbls_3d, index=z_idx, key="surf_z")
        
                    x_var = lbl_to_var_3d[x_lab]
                    y_var = lbl_to_var_3d[y_lab]
                    z_var = lbl_to_var_3d[z_lab]
        
                    if len({x_var, y_var, z_var}) < 3:
                        st.info("Selecciona tres variables distintas.")
                    else:
                        c1, c2, c3 = st.columns([1.0, 1.0, 1.2])
                        with c1:
                            grid_n = st.slider("Resolución malla", 20, 60, 35, 5, key="surf_grid_n")
                        with c2:
                            point_size = st.slider("Tamaño punto empresa", 6, 20, 12, 1, key="surf_pt_size")
                        with c3:
                            show_selected_surface = st.checkbox("Mostrar empresa seleccionada", value=True, key="surf_show_sel")
        
                        surface_out = _compute_interaction_surface(
                            df_in=df,
                            x_var=x_var,
                            y_var=y_var,
                            z_var=z_var,
                            grid_n=int(grid_n),
                        )
        
                        if surface_out is None:
                            st.warning("No he podido construir la superficie 3D con esas variables.")
                        else:
                            XX, YY, ZZ, dfit = surface_out
        
                            fig_surface = go.Figure()
        
                            fig_surface.add_trace(
                                go.Surface(
                                    x=XX,
                                    y=YY,
                                    z=ZZ,
                                    opacity=0.88,
                                    showscale=False,
                                    hovertemplate=(
                                        f"{_label(x_var)}=%{{x:.2f}}<br>"
                                        f"{_label(y_var)}=%{{y:.2f}}<br>"
                                        f"Predicción=%{{z:.2f}}<extra></extra>"
                                    ),
                                )
                            )
        
                            if show_selected_surface:
                                x0 = _coerce_numeric_series(pd.Series([row.get(x_var, np.nan)])).iloc[0]
                                y0 = _coerce_numeric_series(pd.Series([row.get(y_var, np.nan)])).iloc[0]
        
                                if pd.notna(x0) and pd.notna(y0):
                                    z0 = _predict_surface_point(
                                        x_val=float(x0),
                                        y_val=float(y0),
                                        df_fit=dfit,
                                        x_var=x_var,
                                        y_var=y_var,
                                        z_var=z_var,
                                    )
        
                                    if z0 is not None:
                                        fig_surface.add_trace(
                                            go.Scatter3d(
                                                x=[float(x0)],
                                                y=[float(y0)],
                                                z=[float(z0)],
                                                mode="markers",
                                                marker=dict(
                                                    size=float(point_size),
                                                    color="#111111",
                                                    opacity=1.0,
                                                    line=dict(width=4, color="white"),
                                                ),
                                                showlegend=False,
                                                customdata=np.array([[
                                                    str(row.get("nombre", "—")),
                                                    str(row.get("codigo_nif", "—")),
                                                    str(cluster_sel),
                                                ]]),
                                                hovertemplate=_hovertemplate_basic(),
                                            )
                                        )
                                else:
                                    st.info("La empresa seleccionada no tiene valores válidos en las variables X/Y elegidas.")
        
                            plotly_layout_base(fig_surface, height=620, margin=dict(l=20, r=20, t=60, b=20))
                            fig_surface.update_layout(
                                title=f"Interacción {_label(x_var)} × {_label(y_var)} (sobre {_label(z_var)})",
                                scene=dict(
                                    xaxis_title=_label(x_var),
                                    yaxis_title=_label(y_var),
                                    zaxis_title=_label(z_var),
                                ),
                                scene_camera=dict(eye=dict(x=1.45, y=1.35, z=1.0)),
                            )
        
                            st.plotly_chart(fig_surface, use_container_width=True)
                            st.caption("La superficie se estima con un modelo lineal con interacción: Z ~ X + Y + X·Y.")
            except Exception as e:
                st.error(f"No se pudo renderizar la superficie 3D: {e}")

        # ======================
        # Interpretación
        # ======================
        anchor("interpretacion")
        with st.expander(story_title, expanded=True):
            st.caption(f"Clúster: **{cluster_sel}** · n={n_sel} ({pct_sel:.1f}% de la muestra)")
            st.markdown(f"**{story.get('titulo', '—')}**")
            st.markdown("**Rasgos estructurales:**")
            st.markdown("- " + "\n- ".join(story.get("rasgos_estructurales", [])))
            st.markdown("**Rasgos económicos:**")
            st.markdown("- " + "\n- ".join(story.get("rasgos_economicos", [])))
            st.markdown("**Lectura económica:**")
            st.markdown("- " + "\n- ".join(story.get("lectura_economica", [])))
            st.markdown("**Implicaciones prácticas:**")
            st.markdown("- " + "\n- ".join(story.get("implicaciones", [])))

        # ======================
        # Radar
        # ======================
        anchor("perfil-radar")
        with st.expander("Perfil (radar)", expanded=False):
            cats, r_emp, r_ref = [], [], []

            for var in VARS_CLUSTER:
                if var not in df_ref.columns:
                    continue

                s_raw = pd.to_numeric(df_ref[var], errors="coerce").dropna()
                v_raw = pd.to_numeric(row.get(var, np.nan), errors="coerce")
                if len(s_raw) == 0 or pd.isna(v_raw):
                    continue

                s_disp = to_display_scale(var, s_raw)
                v_disp = float(to_display_scale(var, pd.Series([v_raw])).iloc[0])

                z_emp = robust_z_iqr(s_disp, v_disp)
                if pd.isna(z_emp):
                    continue
                emp01 = (z_emp + 2.0) / 4.0
                ref01 = 0.5

                cats.append(LABELS.get(var, var))
                r_emp.append(emp01)
                r_ref.append(ref01)

            if len(cats) < 3:
                st.info("No hay suficientes variables válidas para el radar (necesito al menos 3).")
            else:
                cats_closed = cats + [cats[0]]
                emp_closed = r_emp + [r_emp[0]]
                ref_closed = r_ref + [r_ref[0]]

                fig_r = go.Figure()
                fig_r.add_trace(
                    go.Scatterpolar(
                        r=ref_closed,
                        theta=cats_closed,
                        mode="lines",
                        line=dict(width=2),
                        name="Referencia (mediana)",
                        hovertemplate="%{theta}: mediana<extra></extra>",
                    )
                )
                fig_r.add_trace(
                    go.Scatterpolar(
                        r=emp_closed,
                        theta=cats_closed,
                        fill="toself",
                        opacity=0.55,
                        name="Empresa",
                        hovertemplate="%{theta}: %{r:.0%}<extra></extra>",
                    )
                )

                plotly_layout_base(fig_r, height=520, margin=dict(l=30, r=30, t=60, b=30))
                fig_r.update_layout(
                    legend=dict(orientation="h", y=-0.15),
                    title="Radar — posición robusta (IQR) vs referencia",
                )
                st.plotly_chart(fig_r, use_container_width=True)

    # ============================================================
    # RIGHT: indicadores
    # ============================================================
    with right:
        anchor("indicadores")

        st.subheader("Empresa seleccionada")
        st.write(f"**{row.get('nombre', '—')}**")
        st.write(f"Clúster: **{cluster_sel}**")
        st.caption(f"Comparación: {comparar_con} (n={len(df_ref)})")
        st.divider()

        st.subheader("Indicadores (valor, percentil y estadísticos)")

        rows = []
        score_list: list[float] = []

        for var in VARS_CLUSTER:
            if var not in df.columns or var not in df_ref.columns:
                continue

            s_raw = pd.to_numeric(df_ref[var], errors="coerce").dropna()
            if len(s_raw) == 0:
                continue

            v_raw = pd.to_numeric(row.get(var, np.nan), errors="coerce")
            if pd.isna(v_raw):
                continue

            s_disp = to_display_scale(var, s_raw)
            v_disp = float(to_display_scale(var, pd.Series([v_raw])).iloc[0])

            q1 = float(s_disp.quantile(0.25))
            med = float(s_disp.quantile(0.50))
            q3 = float(s_disp.quantile(0.75))
            mean = float(s_disp.mean())
            sd = float(s_disp.std(ddof=1))

            tol = 0.01 * (abs(med) + 1e-9)
            if v_disp > med + tol:
                flag = "↑"
            elif v_disp < med - tol:
                flag = "↓"
            else:
                flag = "≈"

            pctl = empirical_percentile(s_disp, v_disp)
            z_iqr = robust_z_iqr(s_disp, v_disp)
            score_list.append(z_iqr)

            rows.append(
                {
                    "Indicador": LABELS.get(var, var),
                    "Valor empresa": v_disp,
                    "Percentil": pctl,
                    "vs mediana": flag,
                    "Q1": q1,
                    "Mediana": med,
                    "Q3": q3,
                    "Media": mean,
                    "Desv. típica": sd,
                }
            )

        stats_df = pd.DataFrame(rows)
        idx_global = overall_index(score_list)

        colx, coly = st.columns([1.0, 1.0])
        with colx:
            st.metric("Índice global (0–100)", value="" if pd.isna(idx_global) else f"{idx_global:.1f}")
        with coly:
            st.metric("Tamaño del clúster", value=f"{n_sel}", delta=f"{pct_sel:.1f}% muestra")

        if not stats_df.empty:
            stats_show = stats_df.copy()
            stats_show["Valor empresa"] = stats_show["Valor empresa"].apply(fmt_num)
            for c in ["Q1", "Mediana", "Q3", "Media", "Desv. típica"]:
                stats_show[c] = stats_show[c].apply(fmt_num)
            stats_show["Percentil"] = stats_df["Percentil"].apply(lambda x: "" if pd.isna(x) else f"{x:.0f}")

            def color_flag(x):
                if x == "↑":
                    return "color: #1a7f37; font-weight: 700;"
                if x == "↓":
                    return "color: #b42318; font-weight: 700;"
                if x == "≈":
                    return "color: #6b7280; font-weight: 700;"
                return ""

            styled = stats_show.style.map(color_flag, subset=["vs mediana"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.warning("No hay indicadores para mostrar.")

    # ============================================================
    # TOP EMPRESAS (plegable)
    # ============================================================
    anchor("top-empresas")
    with st.expander("Top empresas (global o por localidad)", expanded=False):
        df_top = _ensure_ingresos(df_view=df, base_path=base_path, diagnostic=False)
        if "ingresos_de_explotacion" not in df_top.columns:
            st.warning("No puedo calcular el Top: falta ingresos_de_explotacion.")
        else:
            df_top["ingresos_rank"] = pd.to_numeric(df_top["ingresos_de_explotacion"], errors="coerce")

            clusters = sorted(df_top["cluster_label"].dropna().unique()) if "cluster_label" in df_top.columns else []
            if not clusters:
                st.info("No hay clusters disponibles.")
            else:
                colA, colB = st.columns([1.2, 1.0])
                with colA:
                    cluster_choice = st.selectbox("Cluster", clusters)
                with colB:
                    top_n = st.slider("Top N", 5, 30, 10, 1)

                df_c = df_top[df_top["cluster_label"] == cluster_choice].copy()

                loc_col = None
                if "localidad_grp" in df_c.columns:
                    loc_col = "localidad_grp"
                elif "localidad" in df_c.columns:
                    loc_col = "localidad"

                opciones = ["Global (todas)"]
                if loc_col:
                    opciones += sorted(df_c[loc_col].dropna().unique())

                scope_choice = st.selectbox("Ámbito", opciones, index=0)

                if scope_choice == "Global (todas)":
                    df_rank = df_c.copy()
                    titulo = f"Top {top_n} — {cluster_choice} (global)"
                else:
                    df_rank = df_c[df_c[loc_col] == scope_choice].copy()
                    titulo = f"Top {top_n} — {cluster_choice} · {scope_choice}"

                top_df = (
                    df_rank.dropna(subset=["ingresos_rank"])
                    .sort_values("ingresos_rank", ascending=False)
                    .head(top_n)
                    .loc[:, ["nombre", "ingresos_rank"]]
                    .rename(columns={"ingresos_rank": "Ingresos de explotación"})
                )

                st.markdown(f"### {titulo}")
                if top_df.empty:
                    st.info("No hay datos suficientes para este filtro.")
                else:
                    top_df = top_df.reset_index(drop=True)
                    top_df.index += 1
                    top_df.index.name = "Ranking"
                    st.dataframe(top_df, use_container_width=True)

    # ============================================================
    # DESCARGAS (plegable)
    # ============================================================
    anchor("descargas")
    with st.expander("Descargas", expanded=False):
        if not stats_df.empty:
            export_df = stats_df.copy()
            export_df["Empresa"] = row.get("nombre", "—")
            export_df["Cluster"] = cluster_sel
            export_df = export_df[["Empresa", "Cluster"] + [c for c in export_df.columns if c not in ["Empresa", "Cluster"]]]

            buf = io.StringIO()
            export_df.to_csv(buf, index=False)
            st.download_button(
                label="⬇️ Descargar ficha de empresa (CSV)",
                data=buf.getvalue().encode("utf-8"),
                file_name=f"ficha_empresa_{cluster_sel}_{row.get('nombre', 'empresa')}.csv",
                mime="text/csv",
            )

        mini_cols = [c for c in ["nombre", "cluster_label", "PC1", "PC2"] + VARS_CLUSTER if c in df_ref.columns]
        buf2 = io.StringIO()
        df_ref[mini_cols].to_csv(buf2, index=False)
        st.download_button(
            label="⬇️ Descargar datos de referencia (CSV)",
            data=buf2.getvalue().encode("utf-8"),
            file_name=f"datos_referencia_{'cluster' if comparar_con == 'Solo su cluster' else 'total'}.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("Descargas — base completa por clúster")

        if "cluster_label" not in df.columns:
            st.warning("No puedo crear la descarga por clúster: falta `cluster_label`.")
        else:
            vars_ok = [v for v in VARS_CLUSTER if v in df.columns]
            if len(vars_ok) == 0:
                st.warning("No puedo crear la descarga por clúster: no encuentro `VARS_CLUSTER`.")
            else:
                df_full_valid = df.dropna(subset=vars_ok + ["cluster_label"]).copy()

                export_cols = [c for c in ["nombre", "codigo_nif", "cluster_label"] if c in df_full_valid.columns]
                if len(export_cols) < 2:
                    st.warning("No puedo crear la descarga: faltan columnas (nombre/codigo_nif/cluster_label).")
                else:
                    df_export_base = df_full_valid[export_cols].copy()
                    st.caption(f"Casos válidos (complete cases en VARS_CLUSTER): n={len(df_export_base)}")

                    def _download_df(_df: pd.DataFrame, fname: str, label: str):
                        buf = io.StringIO()
                        _df.to_csv(buf, index=False)
                        st.download_button(
                            label=label,
                            data=buf.getvalue().encode("utf-8"),
                            file_name=fname,
                            mime="text/csv",
                        )

                    cA, cB, cC, cD = st.columns(4)
                    with cA:
                        _download_df(df_export_base, "base_completa_clusters_total.csv", "⬇️ Total (CSV)")

                    for cl, col in [("C1", cB), ("C2", cC), ("C3", cD)]:
                        with col:
                            df_cl = df_export_base[df_full_valid["cluster_label"].astype(str).map(_normalize_cluster_label) == cl].copy()
                            _download_df(df_cl, f"base_completa_clusters_{cl}.csv", f"⬇️ {cl} (CSV)")
