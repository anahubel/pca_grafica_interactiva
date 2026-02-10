# app.py
import os
import streamlit as st

from utils.config import DATA_PATH, BASE_PATH
from utils.data_io import load_data
# ✅ Recomendado: imports al principio del app.py (arriba del todo)
from utils.views.resumen import render_resumen
from utils.views.estadistica import render_estadistica
from utils.views.arbol_decision import render_arbol_decision
from utils.views.jerarquico import render_jerarquico  # <-- tu nuevo .py

st.set_page_config(page_title="PCA + Clustering", layout="wide")
st.title("Modelos de negocio: clustering")

vista = st.radio(
    "Vista",
    options=["Resumen", "Estadística del modelo", "Árbol de decisión"],
    horizontal=True,
    label_visibility="collapsed",
)

DATA_MTIME = os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else 0.0
df = load_data(DATA_PATH, DATA_MTIME)

required_cols = ["PC1", "PC2", "cluster_label", "nombre"]
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    st.error(f"Faltan columnas en {DATA_PATH}: {missing_required}")
    st.stop()

with st.sidebar:
    st.header("Controles")
    comparar_con = st.radio("Comparar contra", ["Solo su cluster", "Total (todas)"], index=0)
    zoom = st.checkbox("Zoom al punto seleccionado", value=False)

    if vista == "Resumen":
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

# ✅ Routing de vistas (sin else-trampa)
if vista == "Resumen":
    render_resumen(df=df, comparar_con=comparar_con, zoom=zoom)

elif vista == "Estadística del modelo":
    render_estadistica(df=df, base_path=BASE_PATH)

elif vista == "Árbol de decisión":
    from utils.views.arbol_decision import render_arbol_decision
    render_arbol_decision(df_app=df, base_path=BASE_PATH)