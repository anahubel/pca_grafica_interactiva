import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ===== CONFIG =====
DATA_PATH = "data/processed/pca_plot_df.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["nombre"] = df["nombre"].astype(str)
    df["cluster_label"] = df["cluster_label"].astype(str)
    return df

df = load_data()

# ===== SIDEBAR =====
with st.sidebar:
    st.header("Controles")
    zoom = st.checkbox("Zoom al punto seleccionado", value=False)
    padx = st.slider("PC1 ±", 0.5, 10.0, 1.0, 0.1)
    pady = st.slider("PC2 ±", 0.5, 10.0, 1.0, 0.1)

# ===== SELECTOR =====
empresa = st.selectbox(
    "Busca o selecciona una empresa",
    sorted(df["nombre"].unique())
)

row = df[df["nombre"] == empresa].iloc[0]

# ===== GRÁFICO =====
fig = px.scatter(
    df,
    x="PC1",
    y="PC2",
    color="cluster_label",
    hover_name="nombre",
    opacity=0.65,
    labels={
        "PC1": "Componente principal 1",
        "PC2": "Componente principal 2",
        "cluster_label": "Modelo de negocio"
    }
)

fig.update_traces(marker=dict(size=7))
fig.update_layout(legend=dict(orientation="h", y=-0.25))

# Punto resaltado
fig.add_trace(go.Scatter(
    x=[row["PC1"]],
    y=[row["PC2"]],
    mode="markers+text",
    text=[empresa],
    textposition="top center",
    marker=dict(size=18, symbol="circle-open", line=dict(width=4)),
    showlegend=False
))

if zoom:
    fig.update_xaxes(range=[row["PC1"] - padx, row["PC1"] + padx])
    fig.update_yaxes(range=[row["PC2"] - pady, row["PC2"] + pady])

st.plotly_chart(fig, use_container_width=True)