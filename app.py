# app.py
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ======================
# CONFIG
# ======================
DATA_PATH = "data/processed/app_dataset.csv"

VARS_CLUSTER = [
    "rotacion_stocks",
    "productividad_ventas_pax",
    "margen",
    "nofs_ventas",
    "productividad_va_pax",
    "inmovilizado_empleado",
    "ebitda_porc",
    "ingresos_de_explotacion",
]

LABELS = {
    "rotacion_stocks": "Rotación de stocks",
    "productividad_ventas_pax": "Productividad ventas/pax",
    "margen": "Margen",
    "nofs_ventas": "NOFs / Ventas",
    "productividad_va_pax": "Productividad VA/pax",
    "inmovilizado_empleado": "Inmovilizado/empleado",
    "ebitda_porc": "EBITDA (%)",
    "ingresos_de_explotacion": "Ingresos de explotación",
}

# Estas dos variables VIENEN en log en tu pipeline nuevo (mismo nombre de columna)
LOGGED_IN_MODEL = {"rotacion_stocks", "ingresos_de_explotacion"}

# ======================
# APP
# ======================
st.set_page_config(page_title="PCA + Clustering", layout="wide")
st.title("Visualización del clustering mediante PCA")

# ✅ Para romper caché cuando cambia el CSV
DATA_MTIME = os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else 0.0


@st.cache_data(show_spinner=True)
def load_data(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Tipos básicos
    for c in ["nombre", "cluster_label", "codigo_nif"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # Numéricos
    for c in VARS_CLUSTER:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


df = load_data(DATA_PATH, DATA_MTIME)

# ======================
# CHECKS
# ======================
required_cols = ["PC1", "PC2", "cluster_label", "nombre"]
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    st.error(f"Faltan columnas en {DATA_PATH}: {missing_required}")
    st.stop()

# ======================
# SIDEBAR CONTROLS + ÍNDICE
# ======================
with st.sidebar:
    st.header("Controles")
    comparar_con = st.radio("Comparar contra", ["Solo su cluster", "Total (todas)"], index=0)
    zoom = st.checkbox("Zoom al punto seleccionado", value=False)

    st.markdown("### Índice")
    st.markdown(
        """
        <style>
          html { scroll-behavior: smooth; }

          /* Quitar subrayado + poner negro + mismo estilo en visitados */
          .nav-index a,
          .nav-index a:visited {
            color: #111 !important;
            text-decoration: none !important;
            font-weight: 500;
          }

          /* Hover un pelín más suave (opcional) */
          .nav-index a:hover {
            color: #111 !important;
            opacity: 0.75;
            text-decoration: none !important;
          }

          /* Espaciado entre items */
          .nav-index .item {
            margin: 8px 0;
            line-height: 1.2;
          }
        </style>

        <div class="nav-index">
          <div class="item"><a href="#pca">📌 PCA</a></div>
          <div class="item"><a href="#perfil-radar">🕸️ Perfil (radar)</a></div>
          <div class="item"><a href="#indicadores">📊 Indicadores</a></div>
          <div class="item"><a href="#comparacion-visual">📈 Comparación visual</a></div>
          <div class="item"><a href="#top-empresas">🏆 Top empresas</a></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ======================
# SELECTOR EMPRESA
# ======================
if "codigo_nif" in df.columns:
    df["empresa_key"] = df["nombre"] + "  —  " + df["codigo_nif"]
    selector_col = "empresa_key"
else:
    selector_col = "nombre"

empresa_sel = st.selectbox("Busca/selecciona empresa", sorted(df[selector_col].unique()))
row = df.loc[df[selector_col] == empresa_sel].iloc[0]

# Referencia (cluster o total)
if comparar_con == "Solo su cluster":
    df_ref = df[df["cluster_label"] == row["cluster_label"]].copy()
else:
    df_ref = df.copy()

# ======================
# RADAR (solo empresa)
# ======================
def make_radar(df_ref: pd.DataFrame, row: pd.Series) -> go.Figure:
    categories = []
    empresa_vals = []

    # Valores en escala ORIGINAL (deslogando donde toque)
    for var in VARS_CLUSTER:
        if var not in df_ref.columns:
            continue

        s = pd.to_numeric(df_ref[var], errors="coerce").dropna()
        v = pd.to_numeric(row.get(var, np.nan), errors="coerce")

        if len(s) == 0 or pd.isna(v):
            continue

        if var in LOGGED_IN_MODEL:
            s_disp = np.expm1(s)
            v_disp = np.expm1(v)
        else:
            s_disp = s
            v_disp = v

        categories.append(LABELS.get(var, var))
        empresa_vals.append(float(v_disp))

    if len(categories) < 3:
        fig = go.Figure()
        fig.update_layout(
            height=340,
            margin=dict(l=30, r=30, t=30, b=30),
            title="Radar (perfil de la empresa)",
        )
        return fig

    # Normalización robusta por IQR (Q1–Q3 del grupo de referencia)
    emp_norm = []

    for i, var in enumerate([v for v in VARS_CLUSTER if LABELS.get(v, v) in categories]):
        if var not in df_ref.columns:
            continue

        s = pd.to_numeric(df_ref[var], errors="coerce").dropna()
        if len(s) == 0:
            continue

        if var in LOGGED_IN_MODEL:
            s_disp = np.expm1(s)
        else:
            s_disp = s

        q1 = float(s_disp.quantile(0.25))
        q3 = float(s_disp.quantile(0.75))
        iqr = (q3 - q1) if (q3 - q1) != 0 else 1e-9

        e = (empresa_vals[i] - q1) / iqr
        e = max(0.0, min(2.0, e)) / 2.0  # recorte y re-escala a [0,1]
        emp_norm.append(e)

    categories_closed = categories + [categories[0]]
    emp_closed = emp_norm + [emp_norm[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=emp_closed,
            theta=categories_closed,
            fill="toself",
            name="Empresa",
            opacity=0.6,
            hovertemplate="%{theta}: %{r:.0%}<extra></extra>",
        )
    )

    fig.update_layout(
        height=420,
        margin=dict(l=30, r=30, t=50, b=30),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%")),
        showlegend=False,
        title="Radar — perfil de la empresa",
    )
    return fig

# ======================
# LAYOUT PRINCIPAL
# ======================
left, right = st.columns([2.2, 1.3], gap="large")

# ----------------------
# LEFT: PCA + RADAR
# ----------------------
with left:
    st.markdown('<div id="pca"></div>', unsafe_allow_html=True)

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
            "cluster_label": "Modelo de negocio",
        },
    )
    fig.update_traces(marker=dict(size=7))
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

    # (opcional) zoom simple centrando sin sliders
    if zoom:
        fig.update_xaxes(range=[row["PC1"] - 1.0, row["PC1"] + 1.0])
        fig.update_yaxes(range=[row["PC2"] - 1.0, row["PC2"] + 1.0])

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div id="perfil-radar"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("Perfil (radar)")
    radar_fig = make_radar(df_ref=df_ref, row=row)
    st.plotly_chart(radar_fig, use_container_width=True)

# ----------------------
# RIGHT: INDICADORES
# ----------------------
with right:
    st.markdown('<div id="indicadores"></div>', unsafe_allow_html=True)

    st.subheader("Empresa seleccionada")
    st.write(f"**{row['nombre']}**")
    st.write(f"Cluster: **{row['cluster_label']}**")
    st.caption(f"Comparación: {comparar_con} (n={len(df_ref)})")
    st.divider()

    st.subheader("Indicadores (valor y estadísticos)")

    rows = []
    EPS = 1e-9

    for var in VARS_CLUSTER:
        if var not in df.columns:
            continue

        s = df_ref[var].dropna()
        if len(s) == 0:
            continue

        val = row.get(var, np.nan)

        # Mostrar en escala ORIGINAL cuando la variable está logada en el modelo
        if var in LOGGED_IN_MODEL:
            s_disp = np.expm1(s)
            val_disp = np.expm1(val) if pd.notna(val) else np.nan
        else:
            s_disp = s
            val_disp = val

        q1 = s_disp.quantile(0.25)
        med = s_disp.quantile(0.50)
        q3 = s_disp.quantile(0.75)
        mean = s_disp.mean()
        sd = s_disp.std(ddof=1)

        if pd.isna(val_disp) or pd.isna(med):
            flag = "—"
        else:
            tol = 0.01 * (abs(med) + EPS)
            if val_disp > med + tol:
                flag = "↑"
            elif val_disp < med - tol:
                flag = "↓"
            else:
                flag = "≈"

        rows.append(
            {
                "Indicador": LABELS.get(var, var),
                "Valor empresa": val_disp,
                "vs mediana": flag,
                "Q1": q1,
                "Mediana": med,
                "Q3": q3,
                "Media": mean,
                "Desv. típica": sd,
            }
        )

    stats_df = pd.DataFrame(rows)

    if not stats_df.empty:
        num_cols = ["Valor empresa", "Q1", "Mediana", "Q3", "Media", "Desv. típica"]
        for c in num_cols:
            stats_df[c] = pd.to_numeric(stats_df[c], errors="coerce").round(3)

        def color_flag(x):
            if x == "↑":
                return "color: #1a7f37; font-weight: 700;"
            if x == "↓":
                return "color: #b42318; font-weight: 700;"
            if x == "≈":
                return "color: #6b7280; font-weight: 700;"
            return ""

        styled = stats_df.style.applymap(color_flag, subset=["vs mediana"]).format("{:.3f}", subset=num_cols)

        try:
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.write(styled)
    else:
        st.warning("No hay indicadores para mostrar.")

# ======================
# NIVEL 3: COMPARACIÓN VISUAL
# ======================
st.markdown('<div id="comparacion-visual"></div>', unsafe_allow_html=True)
st.divider()
st.subheader("Comparación visual (empresa vs mediana)")

if stats_df.empty:
    st.info("No hay suficientes datos para el gráfico.")
else:
    viz = stats_df.dropna(subset=["Valor empresa", "Mediana", "Q1", "Q3"]).copy()
    viz.columns = viz.columns.str.strip()

    viz["abs_gap"] = (viz["Valor empresa"] - viz["Mediana"]).abs()
    viz = viz.sort_values("abs_gap", ascending=False)

    max_n = max(4, min(12, len(viz)))
    top_n = st.slider("Nº de indicadores a mostrar", 4, max_n, min(8, max_n))
    viz = viz.head(top_n)

    use_log = st.checkbox("Escala log (recomendado si hay magnitudes muy distintas)", value=True)

    if viz.empty:
        st.info("No hay suficientes datos para mostrar el gráfico.")
    else:
        viz = viz.iloc[::-1].copy()
        ycats = viz["Indicador"].tolist()

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=viz["Valor empresa"],
                y=viz["Indicador"],
                mode="markers",
                name="Empresa",
                marker=dict(size=10),
                hovertemplate="Empresa: %{x:.3f}<extra></extra>",
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=viz["Mediana"],
                y=viz["Indicador"],
                mode="markers",
                name="Mediana",
                marker=dict(size=10, symbol="line-ns-open"),
                hovertemplate="Mediana: %{x:.3f}<extra></extra>",
            )
        )

        shapes = []
        for _, r in viz.iterrows():
            shapes.append(
                dict(
                    type="line",
                    x0=r["Q1"],
                    x1=r["Q3"],
                    y0=r["Indicador"],
                    y1=r["Indicador"],
                    line=dict(width=10),
                    opacity=0.25,
                )
            )

        fig2.update_layout(
            shapes=shapes,
            height=260 + 35 * len(viz),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Valor",
            yaxis_title="",
            yaxis=dict(categoryorder="array", categoryarray=ycats),
            legend=dict(orientation="h", y=-0.2),
        )

        if use_log:
            fig2.update_xaxes(type="log")

        st.plotly_chart(fig2, use_container_width=True)

# ======================
# TOP EMPRESAS (GLOBAL / POR LOCALIDAD)
# ======================
st.markdown('<div id="top-empresas"></div>', unsafe_allow_html=True)
st.divider()
st.subheader("Top empresas (global o por localidad)")

# métrica ranking (por detrás, no se muestra)
if "ingresos_de_explotacion" in df.columns:
    df["ingresos_rank"] = np.expm1(pd.to_numeric(df["ingresos_de_explotacion"], errors="coerce"))
else:
    df["ingresos_rank"] = np.nan

clusters = sorted(df["cluster_label"].dropna().unique())
if not clusters:
    st.info("No hay clusters disponibles.")
else:
    colA, colB = st.columns([1.2, 1.0])

    with colA:
        cluster_choice = st.selectbox("Cluster", clusters)

    with colB:
        top_n = st.slider("Top N", 5, 30, 10, 1)

    df_c = df[df["cluster_label"] == cluster_choice].copy()

    loc_col = None
    if "localidad_grp" in df_c.columns:
        loc_col = "localidad_grp"
        loc_label = "Localidad (grupo)"
    elif "localidad" in df_c.columns:
        loc_col = "localidad"
        loc_label = "Localidad"

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
              .loc[:, ["nombre"]]
    )

    st.markdown(f"### {titulo}")

    if top_df.empty:
        st.info("No hay datos suficientes para este filtro.")
    else:
        top_df = top_df.reset_index(drop=True)
        top_df.index += 1
        top_df.index.name = "Ranking"
        st.dataframe(top_df, use_container_width=True)
