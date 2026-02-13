# utils/views/jerarquico.py
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from sklearn.preprocessing import StandardScaler, RobustScaler

from utils.config import EXCLUDE_VARS, LABELS, VARS_CLUSTER, LOGGED_IN_MODEL
from utils.data_io import load_base_with_clusters


# ======================
# Labels helpers (UI)
# ======================
def _label_of(var: str) -> str:
    return LABELS.get(var, var)

def _make_label_maps(vars_list: list[str]):
    labels = [_label_of(v) for v in vars_list]
    seen = {}
    out_labels = []
    for v, lab in zip(vars_list, labels):
        if lab in seen:
            seen[lab] += 1
            out_labels.append(f"{lab} [{v}]")
        else:
            seen[lab] = 1
            out_labels.append(lab)
    lab_to_var = {lab: v for lab, v in zip(out_labels, vars_list)}
    return out_labels, lab_to_var


# ======================
# Core compute (cache)
# ======================
@st.cache_data(show_spinner=False)
def _compute_linkage(
    X: np.ndarray,
    method: str,
    metric: str,
):
    """
    Devuelve Z (linkage) de scipy para dendrograma.
    Cacheado: OJO, X debe ser np.ndarray (hashable por contenido).
    """
    from scipy.cluster.hierarchy import linkage
    Z = linkage(X, method=method, metric=metric)
    return Z

@st.cache_data(show_spinner=False)
def _cut_clusters(
    Z: np.ndarray,
    cut_mode: str,
    k: int,
    height: float,
):
    """
    Devuelve labels 1..k (o según corte) con fcluster.
    """
    from scipy.cluster.hierarchy import fcluster
    if cut_mode == "k":
        lab = fcluster(Z, t=int(k), criterion="maxclust")
    else:
        lab = fcluster(Z, t=float(height), criterion="distance")
    return lab


# ======================
# MAIN VIEW
# ======================
def render_jerarquico(df_app: pd.DataFrame, base_path: str):
    st.header("Clustering jerárquico")

    # Cargar base completa (por consistencia con el resto de vistas)
    try:
        df_full = load_base_with_clusters(base_path, df_app)
        try:
            from utils.recodes import apply_recodes
            df_full = apply_recodes(df_full)
        except Exception:
            pass
    except Exception as e:
        st.error(f"No he podido cargar la base completa o hacer el merge con clusters: {e}")
        st.stop()

    if "nombre" not in df_full.columns:
        st.warning("No veo columna `nombre`. La vista funciona igual, pero no podré etiquetar hojas con nombres.")
    if "cluster_label" not in df_full.columns:
        st.warning("No veo `cluster_label`. La vista funciona igual, pero no podré comparar con el clustering original.")

    # ======================
    # Selección de variables candidatas
    # ======================
    forbidden = {"codigo_nif", "nombre", "cluster_label", "PC1", "PC2", "empresa_key"} | set(EXCLUDE_VARS)

    num_cols = []
    for c in df_full.columns:
        if c in forbidden:
            continue
        if pd.api.types.is_numeric_dtype(df_full[c]):
            num_cols.append(c)
    num_cols = sorted(num_cols)

    if not num_cols:
        st.info("No encuentro variables numéricas candidatas para el clustering jerárquico.")
        st.stop()

    num_labels, num_lab_to_var = _make_label_maps(num_cols)

    # Defaults: variables del clustering original si existen
    default_vars = [v for v in VARS_CLUSTER if v in num_cols]
    default_labels = []
    for v in default_vars:
        lab = _label_of(v)
        cand2 = f"{lab} [{v}]"
        if cand2 in num_labels:
            default_labels.append(cand2)
        elif lab in num_labels:
            default_labels.append(lab)

    with st.sidebar:
        st.subheader("Configuración")

        vars_sel_labels = st.multiselect(
            "Variables para clustering",
            options=num_labels,
            default=default_labels if default_labels else num_labels[:8],
            key="hc_vars",
        )

        transform_mode = st.selectbox(
            "Transformación de variables del modelo",
            [
                "Usar tal cual (si estaban en log, se quedan en log)",
                "Desloggear variables del modelo (expm1 en LOGGED_IN_MODEL)",
            ],
            index=1,
            key="hc_transform_mode",
        )

        scale_mode = st.selectbox(
            "Escalado",
            ["StandardScaler (media=0, sd=1)", "RobustScaler (mediana/IQR)", "Sin escalado"],
            index=0,
            key="hc_scale_mode",
        )

        st.markdown("---")
        st.subheader("Linkage / Distancia")

        linkage_method = st.selectbox(
            "Método de enlace (linkage)",
            ["ward", "average", "complete", "single"],
            index=0,
            key="hc_linkage",
        )

        # Ward exige euclidean (en scipy linkage)
        metric_options = ["euclidean", "cityblock", "cosine", "chebyshev", "correlation"]
        if linkage_method == "ward":
            metric = "euclidean"
            st.caption("Nota: `ward` requiere `euclidean` (forzado).")
        else:
            metric = st.selectbox(
                "Métrica de distancia",
                metric_options,
                index=0,
                key="hc_metric",
            )

        st.markdown("---")
        st.subheader("Corte del dendrograma")

        cut_mode_ui = st.radio(
            "Tipo de corte",
            ["Número de clusters (k)", "Altura (distancia)"],
            horizontal=True,
            key="hc_cut_mode_ui",
        )
        cut_mode = "k" if cut_mode_ui.startswith("Número") else "h"

        k = st.slider("k (nº clusters)", 2, 12, 3, 1, key="hc_k")
        height = st.number_input("altura (distancia)", min_value=0.0, value=10.0, step=0.5, key="hc_h")

        st.markdown("---")
        show_labels = st.checkbox("Mostrar etiquetas (puede ir lento con muchas empresas)", value=False, key="hc_show_labels")
        max_labels = st.slider("Máx. etiquetas a mostrar", 50, 500, 150, 50, key="hc_max_labels")

        show_heatmap = st.checkbox("Mostrar heatmap (promedios por cluster)", value=True, key="hc_show_heatmap")

    if not vars_sel_labels:
        st.info("Selecciona al menos una variable.")
        st.stop()

    X_vars = [num_lab_to_var[x] for x in vars_sel_labels]

    # ======================
    # Preparar datos
    # ======================
    keep_cols = X_vars.copy()
    if "nombre" in df_full.columns:
        keep_cols.append("nombre")
    if "cluster_label" in df_full.columns:
        keep_cols.append("cluster_label")

    df_work = df_full[keep_cols].copy()

    for c in X_vars:
        df_work[c] = pd.to_numeric(df_work[c], errors="coerce")

    # drop NA en variables
    df_work = df_work.dropna(subset=X_vars).copy()
    if df_work.shape[0] < 5:
        st.info("Muy pocos casos tras eliminar NA. Revisa variables.")
        st.stop()

    # Transformación (deslog si procede)
    if "Desloggear" in transform_mode:
        for c in X_vars:
            if c in LOGGED_IN_MODEL:
                # ojo: si hay valores raros puede overflow -> lo controlamos
                s = df_work[c].to_numpy(dtype=float)
                s = np.clip(s, -100, 100)  # evita expm1 gigantesca por si hubiera algo muy extremo
                df_work[c] = np.expm1(s)

    # Escalado
    X = df_work[X_vars].to_numpy(dtype=float)

    if scale_mode.startswith("StandardScaler"):
        X = StandardScaler().fit_transform(X)
    elif scale_mode.startswith("RobustScaler"):
        X = RobustScaler().fit_transform(X)
    else:
        # sin escalado
        pass

    # ======================
    # Linkage + dendrograma
    # ======================
    st.caption(f"Casos usados: **{df_work.shape[0]}** · Variables: **{len(X_vars)}**")

    with st.spinner("Calculando dendrograma (linkage)…"):
        Z = _compute_linkage(X=X, method=linkage_method, metric=metric)

    # Etiquetas
    labels = None
    if show_labels:
        if "nombre" in df_work.columns:
            labels = df_work["nombre"].astype(str).tolist()
        else:
            labels = [str(i) for i in range(df_work.shape[0])]

        if len(labels) > int(max_labels):
            st.warning(f"Demasiadas etiquetas ({len(labels)}). Muestro solo hasta {max_labels} (sin etiquetas en el resto).")
            labels = None  # para no petar render

    st.subheader("Dendrograma")
    try:
        fig = ff.create_dendrogram(
            X,
            labels=labels,
            linkagefun=lambda _: Z,  # reutiliza linkage precomputado
            orientation="left",
        )
        fig.update_layout(height=max(500, min(1200, 18 * df_work.shape[0])), margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("No he podido renderizar el dendrograma con Plotly. Te dejo error y seguimos con el corte + tablas.")
        st.exception(e)

    # ======================
    # Corte => clusters
    # ======================
    with st.spinner("Aplicando corte y asignando clusters…"):
        lab = _cut_clusters(Z=Z, cut_mode=cut_mode, k=int(k), height=float(height))

    df_out = df_work.copy()
    df_out["cluster_hc"] = pd.Series(lab, index=df_out.index).astype(int)

    # renumerar clusters por tamaño (para que sea más “bonito”)
    sizes = df_out["cluster_hc"].value_counts().sort_values(ascending=False)
    remap = {old: new for new, old in enumerate(sizes.index.tolist(), start=1)}
    df_out["cluster_hc"] = df_out["cluster_hc"].map(remap).astype(int)

    st.subheader("Resultado del corte")
    ctab = df_out["cluster_hc"].value_counts().sort_index().reset_index()
    ctab.columns = ["cluster_hc", "N"]
    ctab["%"] = (100 * ctab["N"] / ctab["N"].sum()).round(1)
    st.dataframe(ctab, use_container_width=True, hide_index=True)

    # Comparación con cluster original (si existe)
    if "cluster_label" in df_out.columns:
        st.markdown("#### Comparación con clustering original (tabla cruzada)")
        cross = pd.crosstab(df_out["cluster_hc"], df_out["cluster_label"])
        st.dataframe(cross, use_container_width=True)

    # ======================
    # Heatmap de promedios por cluster
    # ======================
    if show_heatmap:
        st.subheader("Heatmap (promedios por cluster)")
        agg = df_out.groupby("cluster_hc")[X_vars].mean(numeric_only=True)

        # Convertimos a labels para UI
        agg_disp = agg.copy()
        agg_disp.columns = [_label_of(c) for c in agg_disp.columns]

        # normalizar para heatmap (z-score por variable, solo visual)
        mat = agg_disp.to_numpy(dtype=float)
        mu = np.nanmean(mat, axis=0)
        sd = np.nanstd(mat, axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        z = (mat - mu) / sd

        z_df = pd.DataFrame(z, index=[f"C{c}" for c in agg_disp.index], columns=agg_disp.columns)

        fig_hm = px.imshow(
            z_df,
            aspect="auto",
            labels=dict(x="Variable", y="Cluster HC", color="z"),
        )
        fig_hm.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_hm, use_container_width=True)

        st.caption("Nota: heatmap normalizado (z-score por variable) solo para facilitar lectura visual.")

    # ======================
    # Listado por cluster (expander)
    # ======================
    st.subheader("Empresas por cluster (jerárquico)")
    clusters_sorted = sorted(df_out["cluster_hc"].unique().tolist())

    for c in clusters_sorted:
        sub = df_out[df_out["cluster_hc"] == c].copy()
        with st.expander(f"Cluster HC {c} · N={len(sub)}"):
            show_cols = []
            if "nombre" in sub.columns:
                show_cols.append("nombre")
            if "codigo_nif" in sub.columns:
                show_cols.append("codigo_nif")
            if "cluster_label" in sub.columns:
                show_cols.append("cluster_label")
            show_cols.append("cluster_hc")
            st.dataframe(sub[show_cols] if show_cols else sub.head(30), use_container_width=True)

    # ======================
    # Descarga
    # ======================
    st.divider()
    st.subheader("Descargas")

    buf = io.StringIO()
    cols_export = []
    if "nombre" in df_out.columns:
        cols_export.append("nombre")
    if "codigo_nif" in df_out.columns:
        cols_export.append("codigo_nif")
    if "cluster_label" in df_out.columns:
        cols_export.append("cluster_label")
    cols_export.append("cluster_hc")
    df_out[cols_export].to_csv(buf, index=False)

    st.download_button(
        "Descargar asignación de clusters (CSV)",
        data=buf.getvalue().encode("utf-8"),
        file_name="clusters_jerarquico.csv",
        mime="text/csv",
        key="dl_hc",
    )