# utils/views/cuartiles.py
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from utils.config import VARS_CLUSTER, LABELS
from utils.fmt import fmt_num
from utils.ui import plotly_layout_base

# Columnas numéricas "técnicas" que NO quieres en el desplegable
EXCLUDE_NUMERIC = {
    "id",
    "codigo_nif",
    "PC1", "PC2",
    "cluster_modelo_negocio",
    "cluster_label",
    "textil_cluster",
}

# ============================================================
# Helpers labels
# ============================================================
def _label(col: str) -> str:
    return LABELS.get(col, col)

def _make_label_maps(cols: list[str]) -> tuple[list[str], dict[str, str]]:
    """Evita colisiones de labels (si se repite el label, añade [col])."""
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

def get_numeric_indicators(df: pd.DataFrame, min_non_null: int = 30) -> list[str]:
    """Devuelve columnas numéricas usables."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in EXCLUDE_NUMERIC]
    num_cols = [c for c in num_cols if df[c].notna().sum() >= int(min_non_null)]
    # Ordena por label bonito
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

def _safe_quantiles(x: pd.Series) -> tuple[float, float, float]:
    """Cuantiles robustos: si hay poca varianza y se igualan, evita bins degenerados."""
    q1, q2, q3 = x.quantile([0.25, 0.50, 0.75]).tolist()
    # Si se empatan, mete un epsilon mínimo para que cut no reviente
    eps = 1e-12
    if not np.isfinite(q1): q1 = np.nan
    if not np.isfinite(q2): q2 = np.nan
    if not np.isfinite(q3): q3 = np.nan
    # solo si están finitos
    if np.isfinite(q1) and np.isfinite(q2) and q2 <= q1:
        q2 = q1 + eps
    if np.isfinite(q2) and np.isfinite(q3) and q3 <= q2:
        q3 = q2 + eps
    return float(q1), float(q2), float(q3)

def _winsorize(x: pd.Series, lo_q: float, hi_q: float) -> pd.Series:
    """Clip por cuantiles (solo para gráficas / percentiles si lo activas)."""
    x = pd.to_numeric(x, errors="coerce")
    lo = float(x.quantile(lo_q)) if x.notna().any() else np.nan
    hi = float(x.quantile(hi_q)) if x.notna().any() else np.nan
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return x
    return x.clip(lower=lo, upper=hi)

def _empirical_percentile(x: pd.Series, v: float) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0 or pd.isna(v):
        return np.nan
    return float((x <= v).mean() * 100.0)

def quartile_table_by_cluster(
    df: pd.DataFrame,
    var: str,
    cluster_col: str = "cluster_label",
) -> tuple[pd.DataFrame, dict]:
    """
    Tabla por clúster en Q1..Q4 usando cuartiles GLOBALES del indicador.
    Devuelve:
      - pct_fmt: % por clúster (filas suman ~100)
      - info: q1,q2,q3,n_valid + counts (sin formato)
    """
    if cluster_col not in df.columns:
        return pd.DataFrame(), {"n_valid": 0}

    tmp = df[[cluster_col, var]].copy()
    tmp[var] = pd.to_numeric(tmp[var], errors="coerce")
    tmp = tmp.dropna(subset=[cluster_col, var]).copy()

    n_valid = int(len(tmp))
    if n_valid == 0:
        return pd.DataFrame(), {"n_valid": 0}

    q1, q2, q3 = _safe_quantiles(tmp[var])

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

    # asegurar columnas Q1..Q4
    for c in qlabels:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[qlabels]

    pct = counts.div(np.maximum(1, counts.sum(axis=1)), axis=0) * 100.0

    # tabla formateada
    pct_fmt = pct.copy()
    pct_fmt.insert(0, "N", counts.sum(axis=1).astype(int))
    for c in qlabels:
        pct_fmt[c] = pct_fmt[c].map(lambda v: f"{float(v):.1f}%")

    pct_fmt = pct_fmt.reset_index().rename(columns={cluster_col: "Clúster"})

    info = {"q1": q1, "q2": q2, "q3": q3, "n_valid": n_valid, "counts": counts, "pct": pct}
    return pct_fmt, info

# ============================================================
# MAIN
# ============================================================
def render_cuartiles(df: pd.DataFrame, base_path: str | None = None):
    st.header("Cuartiles (muestra general)")

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.error("No hay datos cargados para calcular cuartiles.")
        return

    # ---------------------------------------------------------
    # 1) Universo
    # ---------------------------------------------------------
    st.subheader("Configuración")

    cA, cB, cC = st.columns([1.4, 1.2, 1.4], gap="large")

    with cA:
        use_kmeans_sample = st.checkbox(
            "Usar solo la muestra KMeans (complete cases en VARS_CLUSTER)",
            value=True,
            help="Actívalo si quieres que los cuartiles se calculen sobre EXACTAMENTE la misma muestra que entra al KMeans.",
            key="q_use_kmeans",
        )

    with cB:
        min_non_null = st.number_input(
            "Mín. no nulos para listar indicador",
            min_value=5,
            max_value=500,
            value=30,
            step=5,
            key="q_min_nonnull",
        )

    with cC:
        winsor = st.checkbox(
            "Clip extremos (winsorize) para gráficas",
            value=False,
            help="Solo afecta a las gráficas/percentiles (no modifica el df). Útil si hay outliers muy bestias.",
            key="q_winsor",
        )

    vars_ok = [v for v in VARS_CLUSTER if v in df.columns]
    if use_kmeans_sample and len(vars_ok) > 0:
        df_use = df.dropna(subset=vars_ok).copy()
    else:
        df_use = df.copy()

    if len(df_use) == 0:
        st.warning("Con estos filtros no queda ninguna fila.")
        return

    st.caption(f"Universo seleccionado: n={len(df_use)} ({100*len(df_use)/max(1,len(df)):.1f}% del total)")

    # ---------------------------------------------------------
    # 2) Indicador
    # ---------------------------------------------------------
    numeric_cols = get_numeric_indicators(df_use, min_non_null=int(min_non_null))
    if not numeric_cols:
        st.warning("No he encontrado indicadores numéricos para mostrar (tras filtros).")
        return

    labels, lab_to_col = _make_label_maps(numeric_cols)

    sel_label = st.selectbox(
        "Selecciona indicador",
        options=labels,
        index=0,
        key="q_indicator",
    )
    var = lab_to_col[sel_label]

    # serie base
    x_raw = pd.to_numeric(df_use[var], errors="coerce").dropna()
    if len(x_raw) == 0:
        st.warning("Este indicador no tiene valores válidos en el universo seleccionado.")
        return

    # winsor (solo para graficar / percentil)
    x_plot = _winsorize(x_raw, 0.01, 0.99) if winsor else x_raw

    # ---------------------------------------------------------
    # 3) Cuartiles globales
    # ---------------------------------------------------------
    st.subheader(f"Cuartiles globales — {sel_label}")

    q = _quartile_summary(x_raw)

    c1, c2, c3, c4, c5 = st.columns([1, 1.2, 1.2, 1.2, 1.2])
    c1.metric("N", f"{q['N']}")
    c2.metric("Q1 (25%)", "" if pd.isna(q["Q1"]) else fmt_num(q["Q1"]))
    c3.metric("Mediana (50%)", "" if pd.isna(q["Mediana"]) else fmt_num(q["Mediana"]))
    c4.metric("Q3 (75%)", "" if pd.isna(q["Q3"]) else fmt_num(q["Q3"]))
    c5.metric("IQR", "" if pd.isna(q["IQR"]) else fmt_num(q["IQR"]))

    show_tbl = pd.DataFrame([{"Indicador": sel_label, **q}])
    for col in ["Min", "Q1", "Mediana", "Q3", "Max", "IQR"]:
        show_tbl[col] = show_tbl[col].apply(fmt_num)
    st.dataframe(show_tbl, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # 4) “Dónde cae” empresa seleccionada (si existe)
    # ---------------------------------------------------------
    if "nombre" in df_use.columns:
        with st.expander("Empresa seleccionada: percentil y cuartil", expanded=False):
            # si en otras vistas creas empresa_key, intentamos usarlo también
            if "empresa_key" in df_use.columns:
                opts = sorted(df_use["empresa_key"].dropna().astype(str).unique())
                pick = st.selectbox("Empresa", opts, index=0, key="q_pick_empresa_key")
                row = df_use[df_use["empresa_key"].astype(str) == str(pick)].iloc[0]
            else:
                opts = sorted(df_use["nombre"].dropna().astype(str).unique())
                pick = st.selectbox("Empresa", opts, index=0, key="q_pick_nombre")
                row = df_use[df_use["nombre"].astype(str) == str(pick)].iloc[0]

            v = pd.to_numeric(row.get(var, np.nan), errors="coerce")
            if pd.isna(v):
                st.info("La empresa seleccionada no tiene valor válido en este indicador.")
            else:
                q1, q2, q3 = _safe_quantiles(x_raw)
                if v <= q1:
                    qtile = "Q1"
                elif v <= q2:
                    qtile = "Q2"
                elif v <= q3:
                    qtile = "Q3"
                else:
                    qtile = "Q4"

                pctl = _empirical_percentile(x_plot if winsor else x_raw, float(v))
                cA, cB = st.columns(2)
                cA.metric("Valor", fmt_num(v))
                cB.metric("Percentil", "" if pd.isna(pctl) else f"{pctl:.0f}")
                st.caption(f"Cuartil: **{qtile}** (cuartiles globales)")

    # ---------------------------------------------------------
    # 5) Gráficas
    # ---------------------------------------------------------
    st.subheader("Gráficas")

    cG1, cG2, cG3 = st.columns([1.2, 1.2, 1.2])
    with cG1:
        nbins = st.slider("Bins histograma", 10, 80, 30, 5, key="q_nbins")
    with cG2:
        show_kde_like = st.checkbox("Mostrar rug (densidad visual)", value=True, key="q_rug")
    with cG3:
        log_x = st.checkbox("Eje X log (si >0)", value=False, key="q_logx")

    df_hist = pd.DataFrame({"Valor": x_plot})
    fig = px.histogram(
        df_hist,
        x="Valor",
        nbins=int(nbins),
        opacity=0.88,
        labels={"Valor": sel_label},
        title=f"Distribución — {sel_label}" + (" (winsor)" if winsor else ""),
    )

    q1, q2, q3 = _safe_quantiles(x_raw)
    for qv, name in [(q1, "Q1"), (q2, "Mediana"), (q3, "Q3")]:
        if np.isfinite(qv):
            fig.add_vline(x=qv, line_width=2, annotation_text=name, annotation_position="top")

    if show_kde_like:
        fig.update_traces(marker_line_width=0)
        fig.add_trace(
            px.strip(df_hist, x="Valor").update_traces(jitter=0.35, opacity=0.25).data[0]
        )

    if log_x:
        # solo si todos > 0 (si no, rompe)
        if (df_hist["Valor"] > 0).all():
            fig.update_xaxes(type="log")
        else:
            st.info("No puedo poner log-x: hay valores <= 0.")

    plotly_layout_base(fig)
    fig.update_traces(hovertemplate=f"{sel_label}: %{{x}}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

    # Box por clúster
    if "cluster_label" in df_use.columns:
        df_box = df_use[["cluster_label", var]].copy()
        df_box[var] = pd.to_numeric(df_box[var], errors="coerce")
        df_box = df_box.dropna(subset=["cluster_label", var]).copy()
        if len(df_box) > 0:
            order = [c for c in ["C1", "C2", "C3"] if c in df_box["cluster_label"].astype(str).unique()]
            rest = [c for c in sorted(df_box["cluster_label"].astype(str).unique()) if c not in order]
            order = order + rest

            fig2 = px.box(
                df_box,
                x="cluster_label",
                y=var,
                points="suspectedoutliers",
                category_orders={"cluster_label": order},
                labels={"cluster_label": "Clúster", var: sel_label},
                title=f"{sel_label} por clúster",
            )
            plotly_layout_base(fig2, height=420)
            fig2.update_traces(hovertemplate=f"Clúster=%{{x}}<br>{sel_label}=%{{y}}<extra></extra>")
            st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------------------------------------
    # 6) Tabla % por clúster en cuartiles + heatmap + opcional conteos
    # ---------------------------------------------------------
    st.divider()
    st.subheader("% de cada clúster en Q1/Q2/Q3/Q4 (cuartiles globales)")

    if "cluster_label" not in df_use.columns:
        st.info("No puedo calcularlo: falta `cluster_label` en el dataset.")
        table_q, info = None, {"n_valid": 0}
    else:
        table_q, info = quartile_table_by_cluster(df_use, var, cluster_col="cluster_label")

    if info.get("n_valid", 0) == 0:
        st.info("No hay datos suficientes (variable o clúster con demasiados nulos).")
    else:
        st.caption(
            f"Cuartiles globales sobre n={info['n_valid']} · "
            f"Q1={info['q1']:.3f} · Mediana={info['q2']:.3f} · Q3={info['q3']:.3f}"
        )

        cT1, cT2 = st.columns([1.4, 1.0], gap="large")
        with cT1:
            st.dataframe(table_q, use_container_width=True, hide_index=True)

        with cT2:
            # Heatmap de % (sin el formateo con %)
            pct = info["pct"].copy()
            pct = pct.reset_index().rename(columns={"cluster_label": "Clúster"})
            pct_long = pct.melt(id_vars="Clúster", var_name="Cuartil", value_name="Pct")
            fig_hm = px.density_heatmap(
                pct_long,
                x="Cuartil",
                y="Clúster",
                z="Pct",
                histfunc="avg",
                text_auto=".0f",
                title="Heatmap (%)",
                labels={"Pct": "%"},
            )
            plotly_layout_base(fig_hm, height=320)
            fig_hm.update_traces(hovertemplate="Clúster=%{y}<br>%{x}=%{z:.1f}%<extra></extra>")
            st.plotly_chart(fig_hm, use_container_width=True)

        show_counts = st.checkbox("Mostrar conteos (N) por cuartil y clúster", value=False, key="q_show_counts")
        if show_counts:
            counts = info["counts"].copy()
            counts.insert(0, "N", counts.sum(axis=1).astype(int))
            counts = counts.reset_index().rename(columns={"cluster_label": "Clúster"})
            st.markdown("#### Conteos (N) por clúster y cuartil")
            st.dataframe(counts, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # 7) Descargas
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Descarga")

    # resumen global
    export_summary = pd.DataFrame([{
        "Nivel": "Global",
        "Indicador": var,
        "Indicador_label": sel_label,
        **q
    }])

    # tabla por clúster (%)
    export_pct = pd.DataFrame()
    export_counts = pd.DataFrame()

    if info.get("n_valid", 0) > 0 and table_q is not None and isinstance(table_q, pd.DataFrame) and not table_q.empty:
        export_pct = table_q.copy()
        export_pct.insert(0, "Indicador", var)
        export_pct.insert(1, "Indicador_label", sel_label)
        export_pct.insert(0, "Nivel", "Por_clúster_Q_global_%")

        # counts
        counts = info["counts"].copy()
        counts.insert(0, "N", counts.sum(axis=1).astype(int))
        export_counts = counts.reset_index().rename(columns={"cluster_label": "Clúster"})
        export_counts.insert(0, "Indicador", var)
        export_counts.insert(1, "Indicador_label", sel_label)
        export_counts.insert(0, "Nivel", "Por_clúster_Q_global_N")

    # download resumen+% (bonito para informe)
    export_main = pd.concat([export_summary, export_pct], ignore_index=True)
    buf1 = io.StringIO()
    export_main.to_csv(buf1, index=False)
    st.download_button(
        "⬇️ Descargar cuartiles (resumen + % por clúster) (CSV)",
        data=buf1.getvalue().encode("utf-8"),
        file_name=f"cuartiles_{var}_resumen_pct.csv",
        mime="text/csv",
        key="dl_quart_main",
    )

    # download detalle (incluye conteos si existen)
    export_detail_parts = [export_summary]
    if not export_pct.empty:
        export_detail_parts.append(export_pct)
    if not export_counts.empty:
        export_detail_parts.append(export_counts)

    export_detail = pd.concat(export_detail_parts, ignore_index=True)
    buf2 = io.StringIO()
    export_detail.to_csv(buf2, index=False)
    st.download_button(
        "⬇️ Descargar cuartiles (detalle, incluye conteos) (CSV)",
        data=buf2.getvalue().encode("utf-8"),
        file_name=f"cuartiles_{var}_detalle.csv",
        mime="text/csv",
        key="dl_quart_detail",
    )
