# app.py
import os
import streamlit as st
import pandas as pd

from utils.config import DATA_PATH, BASE_PATH
from utils.data_io import load_data  # para el DF principal (CSV)
# Para la base completa, usamos el loader específico (parquet/csv + merge clusters)
from utils.data_io import load_base_with_clusters

from utils.views.resumen import render_resumen
from utils.views.estadistica import render_estadistica
from utils.views.arbol_decision import render_arbol_decision
from utils.views.subgrupos import render_subgrupos
from utils.views.jerarquico import render_jerarquico  # si lo usas


st.set_page_config(page_title="PCA + Clustering", layout="wide")
st.title("Modelos de negocio: clustering")

vista = st.radio(
    "Vista",
    options=[
        "Resumen",
        "Estadística del modelo",
        "Árbol de decisión",
        "Subgrupos C1",
        "Subgrupos C2",
        "Subgrupos C3",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

# ----------------------
# Carga DF principal (el que usas en la app)
# ----------------------
DATA_MTIME = os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else 0.0
df = load_data(DATA_PATH, DATA_MTIME)

required_cols = ["PC1", "PC2", "cluster_label", "nombre"]
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    st.error(f"Faltan columnas en {DATA_PATH}: {missing_required}")
    st.stop()

# ----------------------
# Carga base completa (para ingresos_de_explotacion, etc.)
# IMPORTANTE: aquí NO usamos load_data (que está pensado para el CSV de la app),
# usamos load_base_with_clusters para traer la base completa y añadir cluster_label.
# ----------------------
if "df_base_full" not in st.session_state:
    try:
        # Esto intentará parquet y si falla hará fallback a CSV (según tu data_io.py)
        # y además añadirá cluster_label desde df usando codigo_nif
        df_full = load_base_with_clusters(base_path=BASE_PATH, df_app=df)

        # seguridad: garantizamos DataFrame
        if not isinstance(df_full, pd.DataFrame):
            raise TypeError(f"load_base_with_clusters no devolvió DataFrame (devolvió: {type(df_full)})")

        st.session_state["df_base_full"] = df_full
    except Exception as e:
        st.session_state["df_base_full"] = None
        # OJO: todavía no hemos entrado al sidebar, así que mostramos error arriba
        st.error(f"No pude cargar la base completa desde BASE_PATH: {e}")

with st.sidebar:
    st.header("Controles")
    comparar_con = st.radio("Comparar contra", ["Solo su cluster", "Total (todas)"], index=0)
    zoom = st.checkbox("Zoom al punto seleccionado", value=False)

    # Mini diagnóstico seguro
    df_full = st.session_state.get("df_base_full", None)
    if df_full is None or not hasattr(df_full, "columns"):
        st.caption("Base completa: ❌ (no cargada desde BASE_PATH)")
    else:
        ok_ing = "ingresos_de_explotacion" in df_full.columns
        st.caption(f"Base completa: ✅ | ingresos_de_explotacion: {'✅' if ok_ing else '❌'}")

    if vista in ["Resumen", "Subgrupos C1", "Subgrupos C2", "Subgrupos C3"]:
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
# Routing (reordena aquí también)
# ----------------------
if vista == "Resumen":
    render_resumen(df=df, comparar_con=comparar_con, zoom=zoom, base_path=BASE_PATH)

elif vista == "Estadística del modelo":
    render_estadistica(df=df, base_path=BASE_PATH)

elif vista == "Árbol de decisión":   # 👈 aquí
    render_arbol_decision(df_app=df, base_path=BASE_PATH)

elif vista == "Subgrupos C1":
    render_subgrupos(df=df, base_path=BASE_PATH, cluster_general="C1", k=3, comparar_con=comparar_con, zoom=zoom)

elif vista == "Subgrupos C2":
    render_subgrupos(df=df, base_path=BASE_PATH, cluster_general="C2", k=3, comparar_con=comparar_con, zoom=zoom)

elif vista == "Subgrupos C3":
    render_subgrupos(df=df, base_path=BASE_PATH, cluster_general="C3", k=3, comparar_con=comparar_con, zoom=zoom)