# utils/views/resumen.py
import io
import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.config import VARS_CLUSTER, LABELS, DATA_PATH
from utils.fmt import to_display_scale, fmt_num

from sklearn.preprocessing import StandardScaler


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
    """Devuelve archivos candidatos (parquet/csv/xlsx/xls) bajo root (si root es carpeta) o root si es fichero."""
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
    """
    Busca una base que contenga ingresos_de_explotacion, en este orden:
      1) base_path (si es fichero o carpeta)
      2) carpeta de DATA_PATH
      3) carpeta de base_path (si era fichero)
      4) carpeta ./data (si existe)
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
    """
    Garantiza que df_view tenga ingresos_de_explotacion.
    Si no, busca una base con ingresos y hace merge por codigo_nif o nombre.
    """
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
    """
    Calcula UMAP sobre vars_model (escalado) y devuelve un DF con:
      - __umap1, __umap2
      - cluster_label (si existe)
      - nombre, codigo_nif (si existen)
    Solo usa complete cases en vars_model.
    """
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
    if normalize_cluster_labels:
        _cluster_map = {"1": "C1", "1.0": "C1", "2": "C2", "2.0": "C2", "3": "C3", "3.0": "C3"}
        cluster_sel = _cluster_map.get(str(_raw_cluster).strip(), str(_raw_cluster).strip())
    else:
        cluster_sel = str(_raw_cluster).strip()

    # ======================
    # Referencia (cluster o total)
    # ======================
    if comparar_con == "Solo su cluster" and "cluster_label" in df.columns and "cluster_label" in row.index:
        df_ref = df[df["cluster_label"] == _raw_cluster].copy()
        if df_ref.empty and cluster_sel in {"C1", "C2", "C3"}:
            df_ref = df[df["cluster_label"] == cluster_sel].copy()
    else:
        df_ref = df.copy()

    CLUSTER_N = df["cluster_label"].value_counts(dropna=True).to_dict() if "cluster_label" in df.columns else {}
    n_sel = int(CLUSTER_N.get(_raw_cluster, CLUSTER_N.get(cluster_sel, 0)))
    pct_sel = (n_sel / max(1, len(df))) * 100.0

    # ======================
    # Textos: clúster general
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
    # RADAR
    # ======================
    def make_radar(df_ref_: pd.DataFrame, row_: pd.Series) -> go.Figure:
        categories: list[str] = []
        empresa_vals: list[float] = []

        for var in VARS_CLUSTER:
            if var not in df_ref_.columns:
                continue
            s = pd.to_numeric(df_ref_[var], errors="coerce").dropna()
            v = pd.to_numeric(row_.get(var, np.nan), errors="coerce")
            if len(s) == 0 or pd.isna(v):
                continue

            v_disp = float(to_display_scale(var, pd.Series([v])).iloc[0])
            categories.append(LABELS.get(var, var))
            empresa_vals.append(v_disp)

        if len(categories) < 3:
            fig0 = go.Figure()
            fig0.update_layout(height=340, margin=dict(l=30, r=30, t=30, b=30), title="Radar (perfil de la empresa)")
            return fig0

        emp_norm: list[float] = []
        vars_in = [v for v in VARS_CLUSTER if LABELS.get(v, v) in categories]

        for i, var in enumerate(vars_in):
            s = pd.to_numeric(df_ref_[var], errors="coerce").dropna()
            if len(s) == 0:
                continue
            s_disp = to_display_scale(var, s)

            q1 = float(s_disp.quantile(0.25))
            q3 = float(s_disp.quantile(0.75))
            iqr = (q3 - q1) if (q3 - q1) != 0 else 1e-9

            e = (empresa_vals[i] - q1) / iqr
            e = max(0.0, min(2.0, e)) / 2.0
            emp_norm.append(e)

        if len(emp_norm) != len(categories):
            fig0 = go.Figure()
            fig0.update_layout(height=340, margin=dict(l=30, r=30, t=30, b=30), title="Radar (perfil de la empresa)")
            return fig0

        categories_closed = categories + [categories[0]]
        emp_closed = emp_norm + [emp_norm[0]]

        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatterpolar(
                r=emp_closed,
                theta=categories_closed,
                fill="toself",
                name="Empresa",
                opacity=0.6,
                hovertemplate="%{theta}: %{r:.0%}<extra></extra>",
            )
        )
        fig1.update_layout(
            height=420,
            margin=dict(l=30, r=30, t=50, b=30),
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%")),
            showlegend=False,
            title="Radar — perfil de la empresa (normalizado por IQR)",
        )
        return fig1

    # ======================
    # FUNCIONES: percentil + score global
    # ======================
    def empirical_percentile(s: pd.Series, x: float) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0 or pd.isna(x):
            return np.nan
        return float((s <= x).mean() * 100.0)

    def robust_score_iqr(s_disp: pd.Series, x_disp: float) -> float:
        s_disp = pd.to_numeric(s_disp, errors="coerce").dropna()
        if len(s_disp) == 0 or pd.isna(x_disp):
            return np.nan
        q1 = float(s_disp.quantile(0.25))
        med = float(s_disp.quantile(0.50))
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

    with left:
        st.markdown('<div id="pca"></div>', unsafe_allow_html=True)

        # ======================
        # PCA
        # ======================
        if ("PC1" not in df.columns) or ("PC2" not in df.columns) or ("PC1" not in row.index) or ("PC2" not in row.index):
            st.warning("No puedo mostrar el gráfico PCA: faltan columnas PC1 y/o PC2.")
        else:
            COLOR_MAP_PCA = {
                "C1": "#1f3b73", "C2": "#d97706", "C3": "#15803d",
                "1.0": "#1f3b73", "2.0": "#d97706", "3.0": "#15803d",
                "1": "#1f3b73", "2": "#d97706", "3": "#15803d",
            }

            fig = px.scatter(
                df,
                x="PC1",
                y="PC2",
                color="cluster_label" if "cluster_label" in df.columns else None,
                hover_name="nombre" if "nombre" in df.columns else None,
                opacity=0.75,
                color_discrete_map=COLOR_MAP_PCA,
                labels={"PC1": "Componente principal 1", "PC2": "Componente principal 2", "cluster_label": "Modelo de negocio"},
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
            fig.update_layout(legend=dict(orientation="h", y=-0.2))

            fig.add_trace(
                go.Scatter(
                    x=[row["PC1"]],
                    y=[row["PC2"]],
                    mode="markers",
                    marker=dict(size=18, symbol="circle-open", line=dict(width=4)),
                    showlegend=False,
                )
            )

            if zoom:
                fig.update_xaxes(range=[row["PC1"] - 1.0, row["PC1"] + 1.0])
                fig.update_yaxes(range=[row["PC2"] - 1.0, row["PC2"] + 1.0])

            st.plotly_chart(fig, use_container_width=True)

        # ======================
        # UMAP
        # ======================
        st.divider()
        st.subheader("UMAP — proyección no lineal (sobre variables del clustering)")

        show_umap = st.checkbox("Mostrar UMAP", value=True)
        if show_umap:
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
                    COLOR_MAP_UMAP = {
                        "C1": "#6FA1D9", "C2": "#BC523B", "C3": "#2F475A",
                        "1": "#6FA1D9", "2": "#BC523B", "3": "#2F475A",
                        "1.0": "#6FA1D9", "2.0": "#BC523B", "3.0": "#2F475A",
                    }

                    fig_u = px.scatter(
                        df_umap,
                        x="__umap1",
                        y="__umap2",
                        color="cluster_label",
                        color_discrete_map=COLOR_MAP_UMAP,
                        hover_name="nombre" if "nombre" in df_umap.columns else None,
                        opacity=0.85,
                        labels={"__umap1": "UMAP 1", "__umap2": "UMAP 2", "cluster_label": "Modelo de negocio"},
                    )
                    fig_u.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
                    fig_u.update_layout(legend=dict(orientation="h", y=-0.2))

                    if "nombre" in df_umap.columns and row.get("nombre", None) is not None:
                        nm = str(row.get("nombre"))
                        sub = df_umap[df_umap["nombre"].astype(str) == nm]
                        if len(sub) > 0:
                            fig_u.add_trace(
                                go.Scatter(
                                    x=[sub.iloc[0]["__umap1"]],
                                    y=[sub.iloc[0]["__umap2"]],
                                    mode="markers",
                                    marker=dict(size=18, symbol="circle-open", line=dict(width=4)),
                                    showlegend=False,
                                )
                            )

                    st.plotly_chart(fig_u, use_container_width=True)
                    st.caption("UMAP conserva vecindarios locales, pero las distancias globales no son directamente comparables.")

        # =========================
        # 3D (espacio REAL del clustering) — MEJORADO
        # =========================
        st.divider()
        st.subheader("3D — espacio del clustering (variables escaladas)")

        vars_model_3d = [v for v in ["rotacion_stocks", "productividad_va_pax", "inmovilizado_empleado"] if v in df.columns]

        if len(vars_model_3d) < 3:
            st.warning("No puedo mostrar 3D: necesito rotacion_stocks, productividad_va_pax e inmovilizado_empleado.")
        else:
            # Controles de visual
            cA, cB, cC, cD = st.columns([1.2, 1.2, 1.2, 1.4])
            with cA:
                rest_opacity = st.slider("Opacidad resto", 0.05, 0.90, 0.35, 0.05)
            with cB:
                rest_size = st.slider("Tamaño resto", 2.0, 8.0, 4.0, 0.5)
            with cC:
                sel_size = st.slider("Tamaño seleccionada", 4.0, 14.0, 7.0, 0.5)
            with cD:
                show_selected = st.checkbox("Mostrar empresa seleccionada", value=True)

            cE, cF, cG = st.columns([1.2, 1.2, 1.6])
            with cE:
                show_halo = st.checkbox("Halo (ayuda a localizar)", value=True)
            with cF:
                halo_size = st.slider("Tamaño halo", 10.0, 30.0, 16.0, 1.0)
            with cG:
                zoom_3d = st.checkbox("Zoom 3D a la seleccionada", value=False)

            zoom_radius = st.slider("Radio zoom (z-score)", 0.5, 4.0, 2.2, 0.1) if zoom_3d else None

            # 1) DF completo-case
            df_3d = df.copy()
            for v in vars_model_3d:
                df_3d[v] = pd.to_numeric(df_3d[v], errors="coerce")

            need_cols = vars_model_3d + (["cluster_label"] if "cluster_label" in df_3d.columns else [])
            df_3d = df_3d.dropna(subset=need_cols).copy()

            if df_3d.empty:
                st.warning("No hay casos completos para construir el 3D.")
            else:
                # 2) Escalado
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xs = scaler.fit_transform(df_3d[vars_model_3d].to_numpy())

                df_3d["__z1"] = Xs[:, 0]
                df_3d["__z2"] = Xs[:, 1]
                df_3d["__z3"] = Xs[:, 2]

                # 3) localizar empresa seleccionada dentro de df_3d
                sel_nombre = str(row.get("nombre", "")).strip()
                sel_nif = str(row.get("codigo_nif", "")).strip()

                df_sel = pd.DataFrame()
                if sel_nif and "codigo_nif" in df_3d.columns:
                    df_sel = df_3d[df_3d["codigo_nif"].astype(str).str.strip() == sel_nif].copy()
                if df_sel.empty and sel_nombre and "nombre" in df_3d.columns:
                    df_sel = df_3d[df_3d["nombre"].astype(str).str.strip() == sel_nombre].copy()

                # 4) figura
                COLOR_MAP = {
                    "C1": "#1f3b73", "C2": "#d97706", "C3": "#15803d",
                    "1": "#1f3b73",  "2": "#d97706",  "3": "#15803d",
                    "1.0": "#1f3b73","2.0": "#d97706","3.0": "#15803d",
                }

                fig3d = go.Figure()

                # Resto (más visible, NO tan difuminado)
                if "cluster_label" in df_3d.columns:
                    for cl in sorted(df_3d["cluster_label"].dropna().astype(str).unique()):
                        dcl = df_3d[df_3d["cluster_label"].astype(str) == cl]
                        fig3d.add_trace(
                            go.Scatter3d(
                                x=dcl["__z1"], y=dcl["__z2"], z=dcl["__z3"],
                                mode="markers",
                                name=str(cl),
                                marker=dict(
                                    size=float(rest_size),
                                    opacity=float(rest_opacity),
                                    color=COLOR_MAP.get(str(cl), "#888888"),
                                ),
                                hovertext=dcl["nombre"] if "nombre" in dcl.columns else None,
                                hovertemplate="%{hovertext}<extra></extra>" if "nombre" in dcl.columns else None,
                            )
                        )
                else:
                    fig3d.add_trace(
                        go.Scatter3d(
                            x=df_3d["__z1"], y=df_3d["__z2"], z=df_3d["__z3"],
                            mode="markers",
                            name="Empresas",
                            marker=dict(size=float(rest_size), opacity=float(rest_opacity), color="#888888"),
                        )
                    )

                # Seleccionada (integrada) + halo opcional
                if show_selected:
                    if not df_sel.empty:
                        x0 = float(df_sel.iloc[0]["__z1"])
                        y0 = float(df_sel.iloc[0]["__z2"])
                        z0 = float(df_sel.iloc[0]["__z3"])
                        nombre_sel = str(df_sel.iloc[0].get("nombre", "Seleccionada"))

                        if show_halo:
                            fig3d.add_trace(
                                go.Scatter3d(
                                    x=[x0], y=[y0], z=[z0],
                                    mode="markers",
                                    marker=dict(size=float(halo_size), color="white", opacity=0.70),
                                    showlegend=False,
                                    hoverinfo="skip",
                                )
                            )

                        # punto principal (más pequeño y "normal")
                        fig3d.add_trace(
                            go.Scatter3d(
                                x=[x0], y=[y0], z=[z0],
                                mode="markers",
                                marker=dict(
                                    size=float(sel_size),
                                    color="#ff2d55",
                                    opacity=1.0,
                                    line=dict(width=1.5, color="black"),
                                ),
                                showlegend=False,
                                hovertemplate=f"<b>{nombre_sel}</b><br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>",
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
                    else:
                        st.info("La empresa seleccionada no aparece en el 3D (tiene NaN en alguna de las 3 variables).")

                fig3d.update_layout(
                    height=650,
                    legend=dict(orientation="h", y=-0.12),
                    scene=dict(
                        xaxis_title=f"{LABELS.get(vars_model_3d[0], vars_model_3d[0])} (z)",
                        yaxis_title=f"{LABELS.get(vars_model_3d[1], vars_model_3d[1])} (z)",
                        zaxis_title=f"{LABELS.get(vars_model_3d[2], vars_model_3d[2])} (z)",
                    ),
                    scene_camera=dict(eye=dict(x=1.35, y=1.35, z=1.10)),
                )

                st.plotly_chart(fig3d, use_container_width=True)

        # ======================
        # Interpretación
        # ======================
        st.markdown('<div id="interpretacion"></div>', unsafe_allow_html=True)
        st.divider()

        st.subheader(story_title)
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

        st.markdown('<div id="perfil-radar"></div>', unsafe_allow_html=True)
        st.divider()
        st.subheader("Perfil (radar)")
        radar_fig = make_radar(df_ref_=df_ref, row_=row)
        st.plotly_chart(radar_fig, use_container_width=True)

    with right:
        st.markdown('<div id="indicadores"></div>', unsafe_allow_html=True)

        st.subheader("Empresa seleccionada")
        st.write(f"**{row.get('nombre', '—')}**")
        st.write(f"Cluster: **{cluster_sel}**")
        st.caption(f"Comparación: {comparar_con} (n={len(df_ref)})")
        st.divider()

        st.subheader("Indicadores (valor, percentil y estadísticos)")

        rows = []
        EPS = 1e-9
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

            if pd.isna(v_disp) or pd.isna(med):
                flag = "—"
            else:
                tol = 0.01 * (abs(med) + EPS)
                if v_disp > med + tol:
                    flag = "↑"
                elif v_disp < med - tol:
                    flag = "↓"
                else:
                    flag = "≈"

            pctl = empirical_percentile(s_disp, v_disp)
            z_iqr = robust_score_iqr(s_disp, v_disp)
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

            styled = stats_show.style.applymap(color_flag, subset=["vs mediana"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.warning("No hay indicadores para mostrar.")

    # ======================
    # TOP EMPRESAS (por ingresos)
    # ======================
    st.markdown('<div id="top-empresas"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("Top empresas (global o por localidad)")

    df_top = _ensure_ingresos(df_view=df, base_path=base_path, diagnostic=False)

    if "ingresos_de_explotacion" not in df_top.columns:
        st.warning("No puedo calcular el Top: falta ingresos_de_explotacion.")
        return

    df_top["ingresos_rank"] = pd.to_numeric(df_top["ingresos_de_explotacion"], errors="coerce")

    clusters = sorted(df_top["cluster_label"].dropna().unique()) if "cluster_label" in df_top.columns else []
    if not clusters:
        st.info("No hay clusters disponibles.")
        return

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

    # ======================
    # DESCARGAS
    # ======================
    st.markdown('<div id="descargas"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("Descargas")

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

    # ======================
    # DESCARGAS (extra: base completa por cluster)
    # ======================
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
                        df_cl = df_export_base[df_full_valid["cluster_label"].astype(str) == cl].copy()
                        _download_df(df_cl, f"base_completa_clusters_{cl}.csv", f"⬇️ {cl} (CSV)")
