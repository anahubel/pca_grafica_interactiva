# utils/views/estadistica.py
import io
import os
import glob

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from itertools import combinations
from math import erf, sqrt

from utils.config import EXCLUDE_VARS, LABELS, VARS_CLUSTER
from utils.fmt import fmt_p, sig_stars
from utils.stats import (
    eps2_kw,
    magnitude_eps2,
    magnitude_cramers_v,
    compute_posthoc_mwu,
)
from utils.ui import plotly_layout_base, enforce_no_extra

# ============================================================
# DEBUG / UI MESSAGES
# ============================================================
DEBUG_UI = False

def _ui_info(msg: str):
    if DEBUG_UI:
        st.info(msg)

def _ui_warning(msg: str):
    if DEBUG_UI:
        st.warning(msg)

def _ui_caption(msg: str):
    if DEBUG_UI:
        st.caption(msg)

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
# Numeric parsing (robusto)
# ======================
def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Convierte a numérico sin romper notación científica (E+05) y soporta formatos ES:
      - "1.234,56"  -> 1234.56
      - "5,9E+05"   -> 590000.0
      - "5.9E+05"   -> 590000.0
    """
    if s is None:
        return pd.Series([], dtype=float)

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    x = s.astype(str).str.strip()
    x = x.str.replace("\u00a0", "", regex=False)

    mask_sci = x.str.contains(r"[eE]", na=False)
    if mask_sci.any():
        xs = x.where(mask_sci, "")
        xs = xs.str.replace(",", ".", regex=False)
        x = x.where(~mask_sci, xs)

    mask_es = (~mask_sci) & x.str.contains(r"\.", na=False) & x.str.contains(r",", na=False)
    if mask_es.any():
        xe = x.where(mask_es, "")
        xe = xe.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        x = x.where(~mask_es, xe)

    mask_comma = (~mask_sci) & (~mask_es) & x.str.contains(",", na=False)
    if mask_comma.any():
        xc = x.where(mask_comma, "")
        xc = xc.str.replace(",", ".", regex=False)
        x = x.where(~mask_comma, xc)

    return pd.to_numeric(x, errors="coerce")

def _to_display(var: str, s: pd.Series) -> pd.Series:
    """MODO SEGURO: no aplico escalados aquí para que no distorsione."""
    return s

# ======================
# Read parquet ONLY (duckdb opcional)
# ======================
def _has_duckdb() -> bool:
    try:
        import duckdb  # noqa: F401
        return True
    except Exception:
        return False

def _read_full_base(base_path: str) -> pd.DataFrame:
    """
    Lee base completa SOLO desde parquet.
    Si pyarrow falla, intenta fastparquet.
    Si duckdb está instalado, también intenta duckdb y repara a data_clean_fixed.parquet.
    NO usa CSV.
    """
    def _try_pyarrow(p: str):
        try:
            return pd.read_parquet(p, engine="pyarrow"), None
        except Exception as e:
            return None, e

    def _try_fastparquet(p: str):
        try:
            return pd.read_parquet(p, engine="fastparquet"), None
        except Exception as e:
            return None, e

    def _try_duckdb(p: str):
        try:
            import duckdb
            con = duckdb.connect(database=":memory:")
            df = con.execute("SELECT * FROM read_parquet(?)", [p]).df()
            con.close()
            return df, None
        except Exception as e:
            return None, e

    def _repair_with_duckdb(in_p: str, out_p: str) -> bool:
        try:
            import duckdb
            con = duckdb.connect(database=":memory:")
            con.execute(
                "COPY (SELECT * FROM read_parquet(?)) TO ? (FORMAT 'parquet')",
                [in_p, out_p],
            )
            con.close()
            return True
        except Exception:
            return False

    if os.path.isdir(base_path):
        cand = [
            os.path.join(base_path, "data_clean_fixed.parquet"),
            os.path.join(base_path, "data_clean.parquet"),
            os.path.join(base_path, "data.parquet"),
        ]
        cand += sorted(glob.glob(os.path.join(base_path, "*.parquet")))
    else:
        cand = [base_path] if str(base_path).lower().endswith(".parquet") else []
        if not cand:
            raise FileNotFoundError("base_path no es directorio ni archivo parquet.")

    last_err = None
    last_path = None

    for p in cand:
        if not p or not os.path.exists(p):
            continue
        last_path = p

        df, err = _try_pyarrow(p)
        if df is not None:
            return df
        last_err = err

        df, err = _try_fastparquet(p)
        if df is not None:
            _ui_warning(f"Leí el parquet con fastparquet: {os.path.basename(p)}")
            return df
        last_err = err

        if _has_duckdb():
            df, err = _try_duckdb(p)
            if df is not None:
                out_p = (
                    os.path.join(base_path, "data_clean_fixed.parquet")
                    if os.path.isdir(base_path)
                    else (os.path.splitext(p)[0] + "_fixed.parquet")
                )
                ok = _repair_with_duckdb(p, out_p)
                if ok and os.path.exists(out_p):
                    _ui_warning(f"Parquet reparado y guardado: {os.path.basename(out_p)}")
                    df2, _ = _try_pyarrow(out_p)
                    return df2 if df2 is not None else df
                _ui_warning("DuckDB leyó el parquet pero no pudo escribir el fixed. Uso DF leído por DuckDB.")
                return df
            last_err = err

    if not _has_duckdb():
        _ui_info("Nota: duckdb no está instalado. Si tienes parquet 'raros', instala duckdb para repararlos.")

    raise RuntimeError(f"No pude leer ningún parquet. Último archivo: {last_path}. Último error: {last_err}")

def _best_merge_key(df_base: pd.DataFrame, df_clusters: pd.DataFrame) -> str | None:
    candidates = ["empresa_key", "codigo_nif", "nif", "cif", "id_empresa", "id", "codigo"]
    for k in candidates:
        if k in df_base.columns and k in df_clusters.columns:
            return k

    common = [c for c in df_base.columns.intersection(df_clusters.columns)]
    bad = {"cluster_label", "cluster_modelo_negocio", "nombre", "Name"}
    common = [c for c in common if c not in bad]
    if not common:
        return None

    best = None
    best_nu = -1
    for c in common:
        nu = df_clusters[c].nunique(dropna=True)
        if nu > best_nu:
            best_nu = nu
            best = c
    return best

def _normalize_key(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.str.replace(r"\.0$", "", regex=True)
    x = x.str.replace(r"\s+", "", regex=True)
    x = x.replace({"nan": np.nan, "None": np.nan})
    return x

@st.cache_data(show_spinner=True)
def _load_full_with_clusters_cached(base_path: str, df_clusters_key: pd.DataFrame) -> pd.DataFrame:
    df_base = _read_full_base(base_path)

    dfc = df_clusters_key.copy()
    if "cluster_label" not in dfc.columns:
        raise ValueError("df_clusters no contiene 'cluster_label'.")

    key = _best_merge_key(df_base, dfc)
    if key is None:
        raise ValueError(
            "No encuentro una clave común para el merge. "
            "Asegúrate de compartir 'empresa_key' o 'codigo_nif' (u otra ID común)."
        )

    df_base[key] = _normalize_key(df_base[key])
    dfc[key] = _normalize_key(dfc[key])

    dfc_keep = dfc[[key, "cluster_label"]].drop_duplicates(subset=[key], keep="first")

    df_full = df_base.merge(dfc_keep, on=key, how="inner", validate="m:1")
    df_full = df_full[df_full["cluster_label"].notna()].copy()
    return df_full

# ======================
# Post-hoc helpers categóricas
# ======================
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

def two_prop_ztest(x1: int, n1: int, x2: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return np.nan
    p_pool = (x1 + x2) / (n1 + n2)
    denom = sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if p_pool not in (0, 1) else 0.0
    if denom == 0:
        return np.nan
    z = (x1 / n1 - x2 / n2) / denom
    return float(2 * (1 - _norm_cdf(abs(z))))

def p_adjust_holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        val = min(1.0, max(val, prev))
        adj[idx] = val
        prev = val
    return adj.tolist()

def p_adjust_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    return [min(1.0, float(p) * m) for p in pvals]

def cramers_v_from_ct(ct: pd.DataFrame, chi2: float) -> float:
    if pd.isna(chi2):
        return np.nan
    n = float(ct.to_numpy().sum())
    r, c = ct.shape
    denom = n * (min(r - 1, c - 1) if min(r - 1, c - 1) > 0 else 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan

# ======================
# Caches estadísticos
# ======================
@st.cache_data(show_spinner=False)
def _kw_table_cached(df_small: pd.DataFrame, vars_selected: tuple[str, ...], clusters: tuple[str, ...]) -> pd.DataFrame:
    from scipy.stats import kruskal

    out_rows = []
    for var in vars_selected:
        disp_by_cluster = {}
        groups = []

        for cl in clusters:
            s = _coerce_numeric_series(df_small.loc[df_small["cluster_label"] == cl, var]).dropna()
            s_disp = _to_display(var, s).dropna()
            disp_by_cluster[cl] = s_disp
            if len(s_disp) > 0:
                groups.append(s_disp.values)

        if len(groups) < 2:
            continue

        try:
            H, p = kruskal(*groups)
        except Exception:
            H, p = np.nan, np.nan

        n_total = int(sum(len(disp_by_cluster[cl]) for cl in clusters))
        k = int(sum(1 for cl in clusters if len(disp_by_cluster[cl]) > 0))
        e2 = eps2_kw(H, n_total, k)

        row_dict = {"Indicador": LABELS.get(var, var)}
        for cl in clusters:
            s_disp = disp_by_cluster.get(cl, pd.Series([], dtype=float))
            if len(s_disp) == 0:
                row_dict[cl] = ""
            else:
                med = float(s_disp.quantile(0.50))
                q1 = float(s_disp.quantile(0.25))
                q3 = float(s_disp.quantile(0.75))
                iqr = q3 - q1
                row_dict[cl] = f"{med:.3f} ({iqr:.3f})"

        row_dict["Estadístico (K-W)"] = H
        row_dict["P-valor"] = p
        row_dict["Sig."] = sig_stars(p)
        row_dict["ε²"] = e2
        row_dict["Magnitud efecto"] = magnitude_eps2(e2)
        row_dict["_var"] = var
        out_rows.append(row_dict)

    return pd.DataFrame(out_rows)

@st.cache_data(show_spinner=False)
def _posthoc_mwu_cached(df_small: pd.DataFrame, selected_var: str, clusters: tuple[str, ...], adj_method: str) -> pd.DataFrame:
    from scipy.stats import mannwhitneyu

    def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x)
        y = np.asarray(y)
        n1, n2 = len(x), len(y)
        if n1 == 0 or n2 == 0:
            return np.nan
        U = mannwhitneyu(x, y, alternative="two-sided").statistic
        return (2.0 * U) / (n1 * n2) - 1.0

    data_by = {}
    for cl in clusters:
        s = _coerce_numeric_series(df_small.loc[df_small["cluster_label"] == cl, selected_var]).dropna()
        s = _to_display(selected_var, s).dropna()
        data_by[cl] = s.values

    posthoc_df = compute_posthoc_mwu(
        data_by=data_by,
        mannwhitneyu_fn=mannwhitneyu,
        cliffs_delta_fn=cliffs_delta,
        sig_stars_fn=sig_stars,
        adjust=adj_method,
    )
    return posthoc_df

@st.cache_data(show_spinner=False)
def _chi2_table_cached(df_small_cat: pd.DataFrame, vars_cat: tuple[str, ...], agrupar_raras: bool, umbral_raras: int) -> tuple[pd.DataFrame, list]:
    from scipy.stats import chi2_contingency

    def format_contingency_like_report(ct: pd.DataFrame):
        clusters_order = [c for c in ["C1", "C2", "C3"] if c in ct.index]
        rest = [c for c in ct.index if c not in clusters_order]
        ct = ct.reindex(clusters_order + rest)

        row_totals = ct.sum(axis=1).replace(0, np.nan)
        pct_wide = (ct.div(row_totals, axis=0) * 100).fillna(0)

        out = pd.DataFrame(index=ct.columns)
        for cl in ct.index:
            n_vals = ct.loc[cl]
            p_vals = pct_wide.loc[cl]
            out[cl] = ["" if pd.isna(n) else f"{int(n)} ({p:.1f}%)" for n, p in zip(n_vals.values, p_vals.values)]

        report_df = out.reset_index().rename(columns={"index": "Categoría"})

        pct_long = (
            pct_wide.reset_index()
            .melt(id_vars="cluster_label", var_name="Categoría", value_name="Pct")
            .rename(columns={"cluster_label": "Cluster"})
        )
        return report_df, pct_long, pct_wide

    res_rows = []
    contingencias = []

    for var in vars_cat:
        sub = df_small_cat[["cluster_label", var]].dropna().copy()
        sub[var] = sub[var].astype(str)

        if agrupar_raras:
            vc = sub[var].value_counts(dropna=False)
            rare_levels = vc[vc < int(umbral_raras)].index
            if len(rare_levels) > 0:
                sub.loc[sub[var].isin(rare_levels), var] = "OTROS"

        ct = pd.crosstab(sub["cluster_label"], sub[var])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue

        try:
            chi2, p, dof, _ = chi2_contingency(ct.values)
        except Exception:
            chi2, p, dof = np.nan, np.nan, np.nan

        v = cramers_v_from_ct(ct, chi2)
        n = int(ct.values.sum())

        res_rows.append({
            "Variable": LABELS.get(var, var),
            "Chi2": chi2,
            "gl": dof,
            "P-valor": p,
            "Sig.": sig_stars(p),
            "V de Cramér": v,
            "Magnitud efecto": magnitude_cramers_v(v),
            "Categorías": int(ct.shape[1]),
            "N": n,
            "_var": var,
        })

        ct_report, pct_long, pct_wide = format_contingency_like_report(ct)

        dominant = {}
        for cl in pct_wide.index:
            if pct_wide.loc[cl].sum() <= 0:
                continue
            dom_cat = pct_wide.loc[cl].idxmax()
            dom_pct = float(pct_wide.loc[cl, dom_cat])
            dominant[cl] = (dom_cat, dom_pct)

        contingencias.append((var, ct, ct_report, pct_long, pct_wide, dominant))

    return pd.DataFrame(res_rows), contingencias

# ======================
# MAIN VIEW
# ======================
def render_estadistica(df: pd.DataFrame, base_path: str, group_col: str = "cluster_label", title: str | None = None):
    if title:
        st.header(title)

    st.header("Estadística del modelo")

    if group_col != "cluster_label":
        df_app_for_base = df.copy()
        df_app_for_base["cluster_label"] = df_app_for_base[group_col]
    else:
        df_app_for_base = df

    key_candidates = [c for c in ["empresa_key", "codigo_nif", "nif", "cif", "id_empresa", "id", "codigo"] if c in df_app_for_base.columns]
    if not key_candidates:
        st.error("No encuentro ninguna columna ID para mergear (empresa_key/codigo_nif/etc.) en el df de la app.")
        st.stop()

    df_clusters_key = df_app_for_base[key_candidates + ["cluster_label"]].copy()

    try:
        df_full = _load_full_with_clusters_cached(base_path=base_path, df_clusters_key=df_clusters_key)
        try:
            from utils.recodes import apply_recodes
            df_full = apply_recodes(df_full)
        except Exception:
            pass
    except Exception as e:
        st.error(f"No he podido cargar la base completa y mergear clusters: {e}")
        st.stop()

    st.session_state["df_base_full"] = df_full

    EXCLUDE_LOCAL = set(EXCLUDE_VARS) | {"streamlitApp", "streamlit_app", "StreamlitApp"}

    if DEBUG_UI:
        with st.expander("🧪 Auditoría de datos", expanded=False):
            st.write("Shape df_full:", df_full.shape)
            st.write("Clusters:", df_full["cluster_label"].value_counts(dropna=False))

    try:
        from scipy.stats import chi2_contingency  # noqa: F401
    except Exception:
        st.error("Necesitas `scipy`. Instala: pip install scipy")
        st.stop()

    tab_num, tab_cat = st.tabs(["Análisis de variables numéricas", "Análisis de variables categóricas"])

    # ======================
    # NUMÉRICAS
    # ======================
    with tab_num:
        st.subheader("Variables numéricas")
        st.caption("Mediana (IQR) por clúster + Kruskal–Wallis + ε² + Post-hoc (MWU)")

        ALL_OPT_NUM = "TODAS"

        numeric_cols = []
        for c in df_full.columns:
            if c in {"codigo_nif", "nombre", "cluster_label", "PC1", "PC2", "empresa_key"}:
                continue
            if c in EXCLUDE_LOCAL:
                continue

            col = df_full[c]
            if pd.api.types.is_numeric_dtype(col):
                numeric_cols.append(c)
                continue

            conv = _coerce_numeric_series(col)
            non_na_ratio = float(conv.notna().mean()) if len(conv) else 0.0
            if non_na_ratio >= 0.70:
                numeric_cols.append(c)

        numeric_cols = sorted(numeric_cols)
        if not numeric_cols:
            st.info("No encuentro variables numéricas analizables.")
            st.stop()

        num_labels, num_lab_to_var = _make_label_maps(numeric_cols)

        vars_selected_raw = st.multiselect(
            "Variables numéricas a analizar",
            options=[ALL_OPT_NUM] + num_labels,
            default=[ALL_OPT_NUM],
            key="num_vars_select",
        )

        if (ALL_OPT_NUM in vars_selected_raw) or (len(vars_selected_raw) == 0):
            vars_selected = numeric_cols
        else:
            vars_selected = [num_lab_to_var[x] for x in vars_selected_raw]

        clusters = tuple(sorted(df_full["cluster_label"].dropna().astype(str).unique()))
        vars_selected_t = tuple(vars_selected)

        df_small = df_full[["cluster_label"] + list(vars_selected_t)].copy()

        stats_num = _kw_table_cached(df_small=df_small, vars_selected=vars_selected_t, clusters=clusters)
        if stats_num.empty:
            st.info("No hay resultados.")
            st.stop()

        stats_num["P-valor"] = pd.to_numeric(stats_num["P-valor"], errors="coerce")
        stats_num = stats_num.sort_values("P-valor", ascending=True).reset_index(drop=True)
        show_df = stats_num.drop(columns=["_var"]).copy()

        with st.expander("Top variables por tamaño de efecto (ε²)", expanded=True):
            tmp_eff = stats_num[["Indicador", "ε²"]].copy()
            tmp_eff["ε²"] = pd.to_numeric(tmp_eff["ε²"], errors="coerce")
            tmp_eff = tmp_eff.dropna().sort_values("ε²", ascending=False).head(15)

            if tmp_eff.empty:
                st.info("Sin datos para top ε².")
            else:
                fig_eff = px.bar(
                    tmp_eff.iloc[::-1],
                    x="ε²",
                    y="Indicador",
                    orientation="h",
                    labels={"ε²": "ε² (Kruskal–Wallis)", "Indicador": ""},
                )
                plotly_layout_base(fig_eff, height=420, margin=dict(l=10, r=10, t=10, b=10))
                fig_eff.update_traces(hovertemplate="%{y}<br>ε²=%{x:.3f}<extra></extra>")
                enforce_no_extra(fig_eff)
                st.plotly_chart(fig_eff, use_container_width=True)

        st.markdown("### Tabla resumen (click fila → post-hoc)")
        selected_var = None

        def bold_p(val):
            try:
                if pd.isna(val):
                    return ""
                return "font-weight: 800;" if float(val) < 0.05 else ""
            except Exception:
                return ""

        sty = show_df.style.format({
            "Estadístico (K-W)": lambda x: "" if pd.isna(x) else f"{float(x):.2f}",
            "P-valor": fmt_p,
            "ε²": lambda x: "" if pd.isna(x) else f"{float(x):.3f}",
        }).map(bold_p, subset=["P-valor"])

        try:
            ev = st.dataframe(
                sty,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="kw_table_select",
            )
            if ev and hasattr(ev, "selection") and ev.selection.rows:
                selected_var = stats_num.iloc[int(ev.selection.rows[0])]["_var"]
        except Exception:
            st.info("Tu Streamlit no soporta selección por click. Uso selector alternativo.")
            var_list = stats_num["_var"].tolist()
            opts_labels, back_map = _make_label_maps(var_list)
            pick = st.selectbox("Variable para ver post-hoc", options=opts_labels, index=0, key="posthoc_pick")
            selected_var = back_map.get(pick)

        buf = io.StringIO()
        show_df.to_csv(buf, index=False)
        st.download_button(
            "Descargar tabla numéricas (CSV)",
            data=buf.getvalue().encode("utf-8"),
            file_name="estadistica_modelo_numericas.csv",
            mime="text/csv",
            key="dl_num",
        )

        st.divider()
        st.subheader("Post-hoc (comparaciones por pares)")

        if not selected_var:
            st.info("Selecciona una variable para ver el post-hoc.")
        else:
            with st.expander(f"Post-hoc de: {LABELS.get(selected_var, selected_var)}", expanded=True):
                plot_df = df_full[["cluster_label", selected_var]].copy()
                plot_df[selected_var] = _coerce_numeric_series(plot_df[selected_var])
                plot_df = plot_df.dropna(subset=["cluster_label", selected_var]).copy()
                plot_df["Valor"] = _to_display(selected_var, plot_df[selected_var])

                order = [c for c in ["C1", "C2", "C3"] if c in plot_df["cluster_label"].unique()]
                rest = [c for c in sorted(plot_df["cluster_label"].unique()) if c not in order]
                order = order + rest

                fig_dist = px.box(
                    plot_df,
                    x="cluster_label",
                    y="Valor",
                    points="all",
                    category_orders={"cluster_label": order},
                    labels={"cluster_label": "Clúster", "Valor": _label_of(selected_var)},
                )
                plotly_layout_base(fig_dist, height=420, margin=dict(l=10, r=10, t=10, b=10))
                fig_dist.update_traces(hovertemplate="Clúster=%{x}<br>Valor=%{y}<extra></extra>")
                enforce_no_extra(fig_dist)
                st.plotly_chart(fig_dist, use_container_width=True)

                sum_rows = []
                for cl in order:
                    s = plot_df.loc[plot_df["cluster_label"] == cl, "Valor"].dropna()
                    if len(s) == 0:
                        continue
                    q1, med, q3 = float(s.quantile(0.25)), float(s.quantile(0.50)), float(s.quantile(0.75))
                    sum_rows.append({"cluster": cl, "q1": q1, "med": med, "q3": q3})

                sum_df = pd.DataFrame(sum_rows)
                if not sum_df.empty:
                    fig_iqr = go.Figure()
                    fig_iqr.add_trace(go.Scatter(
                        x=sum_df["med"],
                        y=sum_df["cluster"],
                        mode="markers",
                        name="Mediana",
                        hovertemplate="Clúster=%{y}<br>Mediana=%{x:.3f}<extra></extra>",
                    ))
                    for _, r in sum_df.iterrows():
                        fig_iqr.add_shape(
                            type="line",
                            x0=r["q1"], x1=r["q3"],
                            y0=r["cluster"], y1=r["cluster"],
                            line=dict(width=10),
                            opacity=0.25,
                        )

                    plotly_layout_base(
                        fig_iqr,
                        height=280,
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", y=-0.25),
                        xaxis_title=_label_of(selected_var),
                        yaxis_title="",
                        showlegend=False,
                    )
                    enforce_no_extra(fig_iqr)
                    st.plotly_chart(fig_iqr, use_container_width=True)

                adj_method = st.radio(
                    "Ajuste por comparaciones múltiples",
                    options=["Holm", "Bonferroni"],
                    horizontal=True,
                    index=0,
                    key="posthoc_adj",
                )

                df_small_ph = df_full[["cluster_label", selected_var]].copy()
                posthoc_df = _posthoc_mwu_cached(
                    df_small=df_small_ph,
                    selected_var=selected_var,
                    clusters=clusters,
                    adj_method=adj_method,
                )

                desired = ["C3-C2", "C3-C1", "C2-C1"]
                if "Comparación" in posthoc_df.columns and all(x in posthoc_df["Comparación"].values for x in desired):
                    posthoc_df["__ord"] = posthoc_df["Comparación"].map({v: i for i, v in enumerate(desired)})
                    posthoc_df = posthoc_df.sort_values("__ord").drop(columns="__ord")

                sty_ph = posthoc_df.style.format({
                    "P-valor": fmt_p,
                    "P-ajustada": fmt_p,
                    "δ (Cliff)": lambda x: "" if pd.isna(x) else f"{float(x):.3f}",
                }).applymap(
                    lambda v: "font-weight:800;" if (not pd.isna(v) and float(v) < 0.05) else "",
                    subset=["P-ajustada"],
                )

                st.dataframe(sty_ph, use_container_width=True, hide_index=True)

                buf2 = io.StringIO()
                posthoc_df.to_csv(buf2, index=False)
                st.download_button(
                    "Descargar post-hoc (CSV)",
                    data=buf2.getvalue().encode("utf-8"),
                    file_name=f"posthoc_{selected_var}.csv",
                    mime="text/csv",
                    key="dl_posthoc",
                )

    # ======================
    # CATEGÓRICAS
    # ======================
    with tab_cat:
        st.subheader("Variables categóricas")
        st.caption("Chi-cuadrado + V de Cramér + Post-hoc por pares + contingencias")

        highlight_dominant = st.checkbox("Resaltar categoría dominante por clúster", value=True, key="ct_hi_dom")
        show_stacked = st.checkbox("Mostrar barras apiladas (%)", value=True, key="ct_show_bar")
        show_text = st.checkbox("Mostrar texto (categorías mayoritarias)", value=True, key="ct_show_text")

        ALL_OPT_CAT = "TODAS"

        cat_cols = []
        for c in df_full.columns:
            if c in {"codigo_nif", "nombre", "cluster_label", "PC1", "PC2", "cluster_modelo_negocio", "empresa_key"}:
                continue
            if c in EXCLUDE_LOCAL:
                continue
            if (
                pd.api.types.is_object_dtype(df_full[c])
                or pd.api.types.is_categorical_dtype(df_full[c])
                or pd.api.types.is_bool_dtype(df_full[c])
            ):
                cat_cols.append(c)

        cat_cols = sorted(cat_cols)
        if not cat_cols:
            st.info("No veo variables categóricas en la base completa.")
            st.stop()

        cat_labels, cat_lab_to_var = _make_label_maps(cat_cols)

        vars_cat_raw = st.multiselect(
            "Variables categóricas a analizar",
            options=[ALL_OPT_CAT] + cat_labels,
            default=[ALL_OPT_CAT],
            key="cat_vars_select",
        )

        if (ALL_OPT_CAT in vars_cat_raw) or (len(vars_cat_raw) == 0):
            vars_cat = cat_cols
        else:
            vars_cat = [cat_lab_to_var[x] for x in vars_cat_raw]

        agrupar_raras = st.checkbox("Agrupar categorías raras en Otros", value=True, key="cat_group_rare")
        umbral_raras = st.number_input(
            "Umbral otros (nº mínimo de casos)",
            min_value=1, max_value=50, value=5, step=1,
            key="cat_rare_thr"
        )

        df_small_cat = df_full[["cluster_label"] + vars_cat].copy()
        res_cat, contingencias = _chi2_table_cached(
            df_small_cat=df_small_cat,
            vars_cat=tuple(vars_cat),
            agrupar_raras=bool(agrupar_raras),
            umbral_raras=int(umbral_raras),
        )

        if res_cat.empty:
            st.info("No hay resultados.")
            st.stop()

        res_cat["P-valor"] = pd.to_numeric(res_cat["P-valor"], errors="coerce")
        res_cat = res_cat.sort_values("P-valor", ascending=True).reset_index(drop=True)

        with st.expander("Top variables por tamaño de efecto (V de Cramér)", expanded=True):
            tmp_v = res_cat[["Variable", "V de Cramér"]].copy()
            tmp_v["V de Cramér"] = pd.to_numeric(tmp_v["V de Cramér"], errors="coerce")
            tmp_v = tmp_v.dropna().sort_values("V de Cramér", ascending=False).head(15)
            if tmp_v.empty:
                st.info("Sin datos para top V.")
            else:
                fig_v = px.bar(
                    tmp_v.iloc[::-1],
                    x="V de Cramér",
                    y="Variable",
                    orientation="h",
                    labels={"V de Cramér": "V de Cramér", "Variable": ""},
                )
                plotly_layout_base(fig_v, height=420, margin=dict(l=10, r=10, t=10, b=10))
                fig_v.update_traces(hovertemplate="%{y}<br>V=%{x:.3f}<extra></extra>")
                enforce_no_extra(fig_v)
                st.plotly_chart(fig_v, use_container_width=True)

        st.markdown("### Tabla resumen (click fila → post-hoc)")
        selected_cat_var = None
        show_cat_df = res_cat.drop(columns=["_var"]).copy()

        try:
            ev = st.dataframe(
                show_cat_df.style.format({
                    "Chi2": lambda x: "" if pd.isna(x) else f"{float(x):.2f}",
                    "P-valor": fmt_p,
                    "V de Cramér": lambda x: "" if pd.isna(x) else f"{float(x):.3f}",
                }).applymap(
                    lambda v: "font-weight:800;" if (not pd.isna(v) and float(v) < 0.05) else "",
                    subset=["P-valor"],
                ),
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="chi2_table_select",
            )
            if ev and hasattr(ev, "selection") and ev.selection.rows:
                selected_cat_var = res_cat.iloc[int(ev.selection.rows[0])]["_var"]
        except Exception:
            st.info("Tu Streamlit no soporta selección por click. Uso selector alternativo.")
            var_list = res_cat["_var"].tolist()
            opts_labels, back_map = _make_label_maps(var_list)
            pick = st.selectbox("Variable para ver post-hoc", options=opts_labels, index=0, key="posthoc_cat_pick")
            selected_cat_var = back_map.get(pick)

        buf = io.StringIO()
        show_cat_df.to_csv(buf, index=False)
        st.download_button(
            "Descargar tabla categóricas (CSV)",
            data=buf.getvalue().encode("utf-8"),
            file_name="estadistica_modelo_categoricas.csv",
            mime="text/csv",
            key="dl_cat",
        )

        st.divider()
        st.subheader("Post-hoc categóricas (comparaciones por pares)")

        if not selected_cat_var:
            st.info("Selecciona una variable categórica para ver el post-hoc.")
        else:
            with st.expander(f"Post-hoc de: {LABELS.get(selected_cat_var, selected_cat_var)}", expanded=True):
                adj_method = st.radio(
                    "Ajuste por comparaciones múltiples",
                    options=["Holm", "Bonferroni"],
                    horizontal=True,
                    index=0,
                    key="posthoc_cat_adj",
                )

                sub_ph = df_full[["cluster_label", selected_cat_var]].dropna().copy()
                sub_ph[selected_cat_var] = sub_ph[selected_cat_var].astype(str)

                if agrupar_raras:
                    vc = sub_ph[selected_cat_var].value_counts(dropna=False)
                    rare_levels = vc[vc < int(umbral_raras)].index
                    if len(rare_levels) > 0:
                        sub_ph.loc[sub_ph[selected_cat_var].isin(rare_levels), selected_cat_var] = "OTROS"

                ct_full = pd.crosstab(sub_ph["cluster_label"], sub_ph[selected_cat_var])
                clusters_ph = ct_full.index.tolist()
                pairs = list(combinations(clusters_ph, 2))

                from scipy.stats import chi2_contingency

                rows_pairs, pvals = [], []
                for a, b in pairs:
                    ct2 = ct_full.loc[[a, b], :]
                    ct2 = ct2.loc[:, ct2.sum(axis=0) > 0]

                    try:
                        chi2, p, dof, _ = chi2_contingency(ct2.values)
                    except Exception:
                        chi2, p, dof = np.nan, np.nan, np.nan

                    v = cramers_v_from_ct(ct2, chi2)
                    rows_pairs.append({
                        "Comparación": f"{a}-{b}",
                        "Chi2": chi2,
                        "gl": dof,
                        "P-valor": p,
                        "V de Cramér": v,
                        "Magnitud efecto": magnitude_cramers_v(v),
                    })
                    pvals.append(1.0 if pd.isna(p) else float(p))

                padj = p_adjust_holm(pvals) if adj_method == "Holm" else p_adjust_bonferroni(pvals)
                for i in range(len(rows_pairs)):
                    rows_pairs[i]["P-ajustada"] = padj[i]
                    rows_pairs[i]["Sig."] = sig_stars(padj[i])

                posthoc_pairs = pd.DataFrame(rows_pairs)

                desired = ["C3-C2", "C3-C1", "C2-C1"]
                if "Comparación" in posthoc_pairs.columns and all(x in posthoc_pairs["Comparación"].values for x in desired):
                    posthoc_pairs["__ord"] = posthoc_pairs["Comparación"].map({v: i for i, v in enumerate(desired)})
                    posthoc_pairs = posthoc_pairs.sort_values("__ord").drop(columns="__ord")

                st.markdown("#### Diferencia global por pares")
                st.dataframe(
                    posthoc_pairs.style.format({
                        "Chi2": lambda x: "" if pd.isna(x) else f"{float(x):.2f}",
                        "P-valor": fmt_p,
                        "P-ajustada": fmt_p,
                        "V de Cramér": lambda x: "" if pd.isna(x) else f"{float(x):.3f}",
                    }).applymap(
                        lambda v: "font-weight:800;" if (not pd.isna(v) and float(v) < 0.05) else "",
                        subset=["P-ajustada"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("#### Categorías que explican la diferencia")
                top_k = st.slider("Top categorías por comparación", 3, 15, 8, 1, key="posthoc_cat_topk")

                pair_labels = [f"{a} vs {b}" for a, b in pairs]
                pair_pick = st.radio("Elige comparación", options=pair_labels, horizontal=True, key="pair_pick_cat")

                label_to_pair = {f"{a} vs {b}": (a, b) for a, b in pairs}
                a, b = label_to_pair[pair_pick]

                ct2 = ct_full.loc[[a, b], :]
                n1, n2 = int(ct2.loc[a].sum()), int(ct2.loc[b].sum())

                rows_cat, pvals_cat = [], []
                for cat in ct2.columns:
                    x1, x2 = int(ct2.loc[a, cat]), int(ct2.loc[b, cat])
                    p = two_prop_ztest(x1, n1, x2, n2)
                    pvals_cat.append(1.0 if pd.isna(p) else float(p))

                    p1 = 100 * x1 / n1 if n1 > 0 else np.nan
                    p2 = 100 * x2 / n2 if n2 > 0 else np.nan

                    rows_cat.append({
                        "Categoría": cat,
                        f"{a} %": p1,
                        f"{b} %": p2,
                        "Δ pp": (p1 - p2) if (not pd.isna(p1) and not pd.isna(p2)) else np.nan,
                        "P-valor": p,
                    })

                padj_cat = p_adjust_holm(pvals_cat) if adj_method == "Holm" else p_adjust_bonferroni(pvals_cat)
                for i in range(len(rows_cat)):
                    rows_cat[i]["P-ajustada"] = padj_cat[i]
                    rows_cat[i]["Sig."] = sig_stars(padj_cat[i])

                df_cat = (
                    pd.DataFrame(rows_cat)
                    .assign(absΔ=lambda d: d["Δ pp"].abs())
                    .sort_values("absΔ", ascending=False)
                    .head(top_k)
                    .drop(columns="absΔ")
                )

                st.dataframe(
                    df_cat.style.format({
                        f"{a} %": "{:.1f}%",
                        f"{b} %": "{:.1f}%",
                        "Δ pp": "{:+.1f}",
                        "P-valor": fmt_p,
                        "P-ajustada": fmt_p,
                    }).applymap(
                        lambda v: "font-weight:800;" if (not pd.isna(v) and float(v) < 0.05) else "",
                        subset=["P-ajustada"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                buf_phc = io.StringIO()
                posthoc_pairs.to_csv(buf_phc, index=False)
                st.download_button(
                    "Descargar post-hoc global categóricas (CSV)",
                    data=buf_phc.getvalue().encode("utf-8"),
                    file_name=f"posthoc_categoricas_{selected_cat_var}.csv",
                    mime="text/csv",
                    key="dl_posthoc_cat",
                )

        st.divider()
        st.subheader("Tablas de contingencia")

        vars_cont = [v for (v, *_rest) in contingencias]
        cont_labels, cont_lab_to_var = _make_label_maps(vars_cont)

        sel_cont = st.selectbox(
            "Mostrar contingencia de",
            options=["(todas)"] + cont_labels,
            index=0,
            key="ct_sel",
        )

        sel_cont_var = "(todas)" if sel_cont == "(todas)" else cont_lab_to_var.get(sel_cont)

        def style_report_df(report_df: pd.DataFrame, pct_wide: pd.DataFrame):
            if not highlight_dominant:
                return report_df

            df_show = report_df.copy()
            cat_col = None
            for cand in ["Categoría", "Categoria", "category", "Category"]:
                if cand in df_show.columns:
                    cat_col = cand
                    break
            if cat_col is None:
                df_show = df_show.reset_index().rename(columns={"index": "Categoría"})
                cat_col = "Categoría"

            dom = {}
            for cl in pct_wide.index:
                if pct_wide.loc[cl].sum() > 0:
                    dom[cl] = pct_wide.loc[cl].idxmax()

            def apply_row(row):
                cat = row.get(cat_col, None)
                styles = []
                for col in df_show.columns:
                    if col == cat_col:
                        styles.append("")
                    else:
                        if col in dom and cat == dom[col]:
                            styles.append("background-color: #fff3b0; font-weight: 800;")
                        else:
                            styles.append("")
                return styles

            return df_show.style.apply(apply_row, axis=1)

        for (var, ct_raw, ct_report, pct_long, pct_wide, dominant) in contingencias:
            if sel_cont_var != "(todas)" and var != sel_cont_var:
                continue

            with st.expander(f"{LABELS.get(var, var)}  ·  {ct_raw.shape[0]} clusters × {ct_raw.shape[1]} categorías"):
                styled_or_df = style_report_df(ct_report, pct_wide)
                st.dataframe(styled_or_df, use_container_width=True, hide_index=True)

                if show_stacked and not pct_long.empty:
                    fig = px.bar(
                        pct_long,
                        x="Pct",
                        y="Cluster",
                        color="Categoría",
                        orientation="h",
                        barmode="stack",
                        text=pct_long["Pct"].map(lambda x: "" if x < 6 else f"{x:.0f}%"),
                    )
                    plotly_layout_base(
                        fig,
                        height=260,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="%",
                        yaxis_title="",
                        legend=dict(orientation="h", y=-0.25),
                    )
                    fig.update_xaxes(range=[0, 100])
                    fig.update_traces(hovertemplate="Cluster=%{y}<br>%=%{x:.1f}<extra></extra>")
                    enforce_no_extra(fig)
                    st.plotly_chart(fig, use_container_width=True)

                if show_text:
                    st.markdown("**Categorías mayoritarias por clúster:**")
                    for cl in ["C1", "C2", "C3"]:
                        if cl in dominant:
                            cat, pctv = dominant[cl]
                            st.markdown(f"- **{cl}**: {cat} ({pctv:.1f}%)")
