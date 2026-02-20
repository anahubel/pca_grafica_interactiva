# app.py
import os
import streamlit as st
import pandas as pd
import plotly.io as pio

from utils.config import DATA_PATH, BASE_PATH
from utils.data_io import load_data
from utils.data_io import load_base_with_clusters

from utils.views.resumen import render_resumen
from utils.views.estadistica import render_estadistica
from utils.views.arbol_decision import render_arbol_decision
from utils.views.subgrupos import render_subgrupos
from utils.views.cluster_textil import render_cluster_textil


# ============================================================
# BRANDING: CSS + Plotly template
# ============================================================
def apply_brand_css(path: str = "assets/brand.css") -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # no paramos la app por esto
        pass


def apply_plotly_brand_template() -> None:
    brand_colors = [
        "#6FA1D9",
        "#415F7F",
        "#2F475A",
        "#534A50",
        "#8E868A",
        "#D9AA84",
        "#BC6A3B",
        "#BC523B",
    ]
    base = pio.templates["plotly_white"]
    pio.templates["economicamente"] = base.update(
        {
            "layout": {
                "font": {
                    "family": "Avenir Next, Avenir, Helvetica Neue, Arial, sans-serif",
                    "color": "#000000",
                },
                "title": {"font": {"family": "Avenir Next, Avenir, Helvetica Neue, Arial, sans-serif"}},
                "colorway": brand_colors,
                "paper_bgcolor": "#FFFFFF",
                "plot_bgcolor": "#FFFFFF",
                "xaxis": {"gridcolor": "rgba(47,71,90,0.12)", "zerolinecolor": "rgba(47,71,90,0.18)"},
                "yaxis": {"gridcolor": "rgba(47,71,90,0.12)", "zerolinecolor": "rgba(47,71,90,0.18)"},
                "legend": {"font": {"family": "Avenir Next, Avenir, Helvetica Neue, Arial, sans-serif"}},
            }
        }
    )
    pio.templates.default = "economicamente"


# ============================================================
# APP
# ============================================================
st.set_page_config(page_title="PCA + Clustering", layout="wide")

apply_brand_css()
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
    ],
    horizontal=True,
    label_visibility="collapsed",
)

# ----------------------
# Carga DF principal
# ----------------------
DATA_MTIME = os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else 0.0
df = load_data(DATA_PATH, DATA_MTIME)

required_cols = ["PC1", "PC2", "cluster_label", "nombre"]
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    st.error(f"Faltan columnas en {DATA_PATH}: {missing_required}")
    st.stop()

# ----------------------
# Carga base completa (si la necesitas en vistas)
# ----------------------
if "df_base_full" not in st.session_state:
    try:
        df_full = load_base_with_clusters(base_path=BASE_PATH, df_app=df)
        if not isinstance(df_full, pd.DataFrame):
            raise TypeError(f"load_base_with_clusters no devolvió DataFrame (devolvió: {type(df_full)})")
        st.session_state["df_base_full"] = df_full
    except Exception as e:
        st.session_state["df_base_full"] = None
        st.error(f"No pude cargar la base completa desde BASE_PATH: {e}")

with st.sidebar:
    st.header("Controles")

    comparar_con = st.radio("Comparar contra", ["Solo su cluster", "Total (todas)"], index=0)
    zoom = st.checkbox("Zoom al punto seleccionado", value=False)

    # Índice: SOLO para las vistas que tienen anclas (incluimos Sector textil si replica subgrupos)
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
    render_cluster_textil(df=df, base_path=BASE_PATH, comparar_con=comparar_con, zoom=zoom)