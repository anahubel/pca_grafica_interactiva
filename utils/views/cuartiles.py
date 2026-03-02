# utils/views/cuartiles.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from utils.config import VARS_CLUSTER, LABELS
from utils.fmt import fmt_num

# Columnas numéricas "técnicas" que NO quieres en el desplegable
EXCLUDE_NUMERIC = {
    "id",
    "codigo_nif",
    "PC1", "PC2",
    "cluster_modelo_negocio",
    "cluster_label",
    "textil_cluster",
}

def _label(col: str) -> str:
    return LABELS.get(col, col)

def get_numeric_indicators(df: pd.DataFrame, min_non_null: int = 30) -> list[str]:
    """Devuelve columnas numéricas 'usables' para el desplegable."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in EXCLUDE_NUMERIC]
    num_cols = [c for c in num_cols if df[c].notna().sum() >= min_non_null]
    num_cols = sorted(num_cols, key=lambda c: _label(c))
    return num_cols

def _quartile_summary(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return {"N": 0, "Min": np.nan, "Q1": np.nan, "Mediana": np.nan, "Q3": np.nan, "Max": np.nan, "IQR": np.nan}

    q1 = float(x.quantile(0.25))
    q2 = float(x.quantile(0.50))
    q3 = float(x.quantile(0.75))
    return {
        "N": int(len(x)),
        "Min": float(x.min()),
        "Q1": q1,
        "Mediana": q2,
        "Q3": q3,
        "Max": float(x.max()),
        "IQR": float(q3 - q1),
    }

def quartile_table_by_cluster(df: pd.DataFrame, var: str, cluster_col: str = "cluster_label") -> tuple[pd.DataFrame, dict]:
    """
    Tabla % por clúster en Q1..Q4 usando CUARTILES GLOBALES del indicador (sobre todo el df filtrado).
    % es dentro de cada clúster (cada fila suma 100).
    """
    if cluster_col not in df.columns:
        return pd.DataFrame(), {"n_valid": 0}

    tmp = df[[cluster_col, var]].copy()
    tmp[var] = pd.to_numeric(tmp[var], errors="coerce")
    tmp = tmp.dropna(subset=[cluster_col, var]).copy()

    n_valid = len(tmp)
    if n_valid == 0:
        return pd.DataFrame(), {"n_valid": 0}

    q1, q2, q3 = tmp[var].quantile([0.25, 0.50, 0.75]).tolist()

    bins = [-np.inf, q1, q2, q3, np.inf]
    qlabels = ["Q1", "Q2", "Q3", "Q4"]
    tmp["Cuartil"] = pd.cut(tmp[var], bins=bins, labels=qlabels, include_lowest=True)

    counts = (
        tmp.groupby([cluster_col, "Cuartil"])
           .size()
           .unstack("Cuartil")
           .fillna(0)
           .astype(int)
    )

    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    # asegurar columnas Q1..Q4
    for c in qlabels:
        if c not in pct.columns:
            pct[c] = 0.0
    pct = pct[qlabels]

    pct.insert(0, "N", counts.sum(axis=1).astype(int))

    # formateo %
    pct_fmt = pct.copy()
    for c in qlabels:
        pct_fmt[c] = pct_fmt[c].apply(lambda x: f"{x:.1f}%")

    pct_fmt = pct_fmt.reset_index().rename(columns={cluster_col: "Clúster"})
    info = {"q1": q1, "q2": q2, "q3": q3, "n_valid": n_valid}
    return pct_fmt, info

def render_cuartiles(df: pd.DataFrame, base_path: str | None = None):
    st.header("Cuartiles (muestra general)")

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.error("No hay datos cargados para calcular cuartiles.")
        return

    # -----------------------------------
    # 1) Elige universo: toda la muestra vs muestra KMeans (complete cases en VARS_CLUSTER)
    # -----------------------------------
    st.subheader("Configuración")

    use_kmeans_sample = st.checkbox(
        "Usar solo la muestra 'KMeans' (complete cases en VARS_CLUSTER)",
        value=True,
        help="Actívalo si quieres que los cuartiles se calculen sobre EXACTAMENTE la misma muestra que entra al KMeans.",
    )

    vars_ok = [v for v in VARS_CLUSTER if v in df.columns]
    if use_kmeans_sample:
        if len(vars_ok) == 0:
            st.warning("No encuentro VARS_CLUSTER en el df, así que no puedo filtrar a la muestra KMeans.")
            df_use = df.copy()
        else:
            df_use = df.dropna(subset=vars_ok).copy()
    else:
        df_use = df.copy()

    st.caption(f"Filas válidas en el universo seleccionado: n={len(df_use)}")

    # -----------------------------------
    # 2) Desplegable con TODAS las numéricas
    # -----------------------------------
    numeric_cols = get_numeric_indicators(df_use, min_non_null=30)
    if not numeric_cols:
        st.warning("No he encontrado indicadores numéricos para mostrar (tras filtros).")
        return

    # Para mantener label bonito sin perder la col real
    label_to_col = { _label(c): c for c in numeric_cols }
    sel_label = st.selectbox("Selecciona indicador", list(label_to_col.keys()))
    var = label_to_col[sel_label]

    # -----------------------------------
    # 3) Cuartiles globales del indicador
    # -----------------------------------
    st.subheader(f"Cuartiles globales — {sel_label}")

    q = _quartile_summary(df_use[var])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N", f"{q['N']}")
    c2.metric("Q1 (25%)", "" if pd.isna(q["Q1"]) else fmt_num(q["Q1"]))
    c3.metric("Mediana (50%)", "" if pd.isna(q["Mediana"]) else fmt_num(q["Mediana"]))
    c4.metric("Q3 (75%)", "" if pd.isna(q["Q3"]) else fmt_num(q["Q3"]))

    show_tbl = pd.DataFrame([{"Indicador": sel_label, **q}])
    for col in ["Min", "Q1", "Mediana", "Q3", "Max", "IQR"]:
        show_tbl[col] = show_tbl[col].apply(fmt_num)
    st.dataframe(show_tbl, use_container_width=True, hide_index=True)

    # -----------------------------------
    # 4) Gráficas
    # -----------------------------------
    st.subheader("Gráficas")

    x = pd.to_numeric(df_use[var], errors="coerce").dropna()
    if len(x) > 0:
        q1, q2, q3 = x.quantile([0.25, 0.50, 0.75]).tolist()
        fig = px.histogram(
            x,
            nbins=30,
            opacity=0.85,
            title=f"Distribución — {sel_label}",
            labels={"value": sel_label},
        )
        for qv, name in [(q1, "Q1"), (q2, "Mediana"), (q3, "Q3")]:
            fig.add_vline(x=qv, line_width=2, annotation_text=name, annotation_position="top")
        st.plotly_chart(fig, use_container_width=True)

    if "cluster_label" in df_use.columns:
        df_box = df_use[["cluster_label", var]].copy()
        df_box[var] = pd.to_numeric(df_box[var], errors="coerce")
        df_box = df_box.dropna(subset=["cluster_label", var])
        if not df_box.empty:
            fig2 = px.box(
                df_box,
                x="cluster_label",
                y=var,
                points="outliers",
                title=f"{sel_label} por clúster",
                labels={"cluster_label": "Clúster", var: sel_label},
            )
            st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------
    # 5) ✅ Tabla: % clúster en Q1..Q4 (cuartiles globales)
    # -----------------------------------
    st.divider()
    st.subheader("% de cada clúster en Q1/Q2/Q3/Q4 (cuartiles globales)")

    if "cluster_label" not in df_use.columns:
        st.info("No puedo calcularlo: falta `cluster_label` en el dataset.")
        table_q = None
        info = None
    else:
        table_q, info = quartile_table_by_cluster(df_use, var, cluster_col="cluster_label")
        if info.get("n_valid", 0) == 0:
            st.info("No hay datos suficientes (variable o clúster con demasiados nulos).")
        else:
            st.caption(
                f"Cuartiles globales sobre n={info['n_valid']} · "
                f"Q1={info['q1']:.3f} · Mediana={info['q2']:.3f} · Q3={info['q3']:.3f}"
            )
            st.dataframe(table_q, use_container_width=True, hide_index=True)

    # -----------------------------------
    # 6) Descarga
    # -----------------------------------
    st.divider()
    st.subheader("Descarga")

    export_parts = []
    export_parts.append(pd.DataFrame([{
        "Nivel": "Global",
        "Indicador": var,
        "Indicador_label": sel_label,
        **q
    }]))

    if table_q is not None and isinstance(table_q, pd.DataFrame) and not table_q.empty:
        # guardamos también tabla Q por clúster (ya está formateada en %)
        tmp = table_q.copy()
        tmp.insert(0, "Indicador", var)
        tmp.insert(1, "Indicador_label", sel_label)
        tmp.insert(0, "Nivel", "Por_clúster_Q_global")
        export_parts.append(tmp)

    export_df = pd.concat(export_parts, ignore_index=True)

    buf = io.StringIO()
    export_df.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Descargar cuartiles (CSV)",
        data=buf.getvalue().encode("utf-8"),
        file_name=f"cuartiles_{var}.csv",
        mime="text/csv",
    )