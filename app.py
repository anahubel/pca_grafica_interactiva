# app.py
import streamlit as st

# ✅ SIEMPRE lo primero y SOLO una vez
st.set_page_config(page_title="PCA + Clustering", layout="wide")

import pandas as pd
import plotly.io as pio

from utils.config import DATA_PATH, BASE_PATH
from utils.data_io import load_app_dataset, load_base_with_clusters

from utils.views.resumen import render_resumen
from utils.views.estadistica import render_estadistica
from utils.views.arbol_decision import render_arbol_decision
from utils.views.subgrupos import render_subgrupos
from utils.views.cluster_textil import render_cluster_textil
from utils.views.cuartiles import render_cuartiles

from utils.ui import load_css  # ✅ SOLO load_css (nada de inject_responsive_css)


# ============================================================
# BRANDING: CSS + Plotly template
# ============================================================
def apply_plotly_brand_template() -> None:
    brand_colors = ["#6FA1D9","#415F7F","#2F475A","#534A50","#8E868A","#D9AA84","#BC6A3B","#BC523B"]

    # ✅ Copia segura del template
    pio.templates["economicamente"] = pio.templates["plotly_white"]

    # ✅ Modifica SOLO layout
    pio.templates["economicamente"].layout.update(
        font=dict(family="Avenir Next, Avenir, Helvetica Neue, Arial, sans-serif", color="#000000"),
        title=dict(font=dict(family="Avenir Next, Avenir, Helvetica Neue, Arial, sans-serif")),
        colorway=brand_colors,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(gridcolor="rgba(47,71,90,0.12)", zerolinecolor="rgba(47,71,90,0.18)"),
        yaxis=dict(gridcolor="rgba(47,71,90,0.12)", zerolinecolor="rgba(47,71,90,0.18)"),
        legend=dict(font=dict(family="Avenir Next, Avenir, Helvetica Neue, Arial, sans-serif")),
    )
    pio.templates.default = "economicamente"


# ✅ Cargar CSS (si existe) pero SIN tocar layout responsive agresivo
load_css()
apply_plotly_brand_template()

st.title("Modelos de negocio: clustering")

vista = st.radio(
    "Vista",
    options=[
        "Resumen",
        "Estadística del modelo",
        "Árbol de decisión",
        "Subgrupos C1",
        "Sector textil",
        "Cuartiles",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

# ----------------------
# Carga DF principal
# ----------------------
try:
    df = load_app_dataset(DATA_PATH)
except Exception as e:
    st.error(f"No pude cargar DATA_PATH={DATA_PATH}. Error: {e}")
    st.stop()

required_cols = ["PC1", "PC2", "cluster_label", "nombre"]
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    st.error(f"Faltan columnas en {DATA_PATH}: {missing_required}")
    st.stop()

# ----------------------
# (Opcional) Carga base completa
# ----------------------
df_full: pd.DataFrame | None = None
try:
    df_full = load_base_with_clusters(base_path=BASE_PATH, df_app=df)
except Exception as e:
    df_full = None
    st.warning(f"No pude cargar la base completa desde BASE_PATH (se seguirá sin ella): {e}")

with st.sidebar:
    st.header("Controles")

    comparar_con = st.radio("Comparar contra", ["Solo su cluster", "Total (todas)"], index=0)
    zoom = st.checkbox("Zoom al punto seleccionado", value=False)

    if st.button("🔄 Forzar recarga (limpiar caché)", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if vista in ["Resumen", "Subgrupos C1", "Sector textil"]:
        st.markdown("### Índice")
        st.markdown(
            """
            <style>
              html { scroll-behavior: smooth; }
              .nav-index a, .nav-index a:visited { color: #111 !important; text-decoration: none !important; font-weight: 500; }
              .nav-index a:hover { color: #111 !important; opacity: 0.75; text-decoration: none !important; }
              .nav-index .item { margin: 8px 0; line-height: 1.2; }
            </style>

            <div class="nav-index">
              <div class="item"><a href="#pca">PCA</a></div>
              <div class="item"><a href="#interpretacion">Interpretación</a></div>
              <div class="item"><a href="#perfil-radar">Perfil (radar)</a></div>
              <div class="item"><a href="#indicadores">Indicadores</a></div>
              <div class="item"><a href="#comparacion-visual">Comparación visual</a></div>
              <div class="item"><a href="#casos-tipo">Casos tipo</a></div>
              <div class="item"><a href="#top-empresas">Top empresas</a></div>
              <div class="item"><a href="#descargas">Descargas</a></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------
# Base para cuartiles:
# mismas empresas del clustering + todas las variables disponibles
# ----------------------
df_cuartiles: pd.DataFrame | None = None

if df_full is not None and not df_full.empty:
    merge_key = None
    for k in ["empresa_key", "codigo_nif", "nif", "cif", "id_empresa", "id", "codigo", "nombre"]:
        if k in df.columns and k in df_full.columns:
            merge_key = k
            break

    if merge_key is not None:
        # nos quedamos SOLO con las empresas que están en la muestra del clustering
        ids_cluster = df[[merge_key]].drop_duplicates().copy()
        df_cuartiles = df_full.merge(ids_cluster, on=merge_key, how="inner")

        # por seguridad, si cluster_label no estuviera bien en df_full, lo rehacemos desde df
        if "cluster_label" not in df_cuartiles.columns or df_cuartiles["cluster_label"].isna().all():
            df_clusters = df[[merge_key, "cluster_label"]].drop_duplicates(subset=[merge_key])
            df_cuartiles = df_cuartiles.drop(columns=["cluster_label"], errors="ignore").merge(
                df_clusters, on=merge_key, how="left"
            )

        st.caption(
            f"Base cuartiles: {len(df_cuartiles)} empresas "
            f"(misma muestra que clustering) · {len(df_cuartiles.columns)} columnas"
        )
    else:
        df_cuartiles = df.copy()
else:
    df_cuartiles = df.copy()


# ----------------------
# Routing
# ----------------------
if vista == "Resumen":
    render_resumen(df=df, comparar_con=comparar_con, zoom=zoom, base_path=BASE_PATH)

elif vista == "Estadística del modelo":
    render_estadistica(df=df, base_path=BASE_PATH)

elif vista == "Árbol de decisión":
    render_arbol_decision(df_app=df, base_path=BASE_PATH)

elif vista == "Subgrupos C1":
    render_subgrupos(df=df, base_path=BASE_PATH, cluster_general="C1", k=3, comparar_con=comparar_con, zoom=zoom)

elif vista == "Sector textil":
    render_cluster_textil(df=df, base_path=BASE_PATH, comparar_con=comparar_con, zoom=zoom, k=3)

elif vista == "Cuartiles":
    render_cuartiles(df_cuartiles, base_path=BASE_PATH)