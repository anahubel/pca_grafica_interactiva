# utils/views/resumen.py
import io
import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.config import VARS_CLUSTER, LABELS, DATA_PATH
from utils.fmt import to_display_scale, fmt_num


# ============================================================
# Helpers: encontrar y mergear ingresos_de_explotacion
# ============================================================
def _read_any(path: str) -> pd.DataFrame | None:
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(path)
        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        if ext == ".parquet":
            return pd.read_parquet(path)
    except Exception:
        return None
    return None


def _list_candidate_files(root: str) -> list[str]:
    """Devuelve archivos candidatos (parquet/csv/xlsx/xls) bajo root (si root es carpeta) o root si es fichero."""
    if not root:
        return []
    if os.path.isfile(root):
        return [root]
    if os.path.isdir(root):
        patterns = ["**/*.parquet", "**/*.csv", "**/*.xlsx", "**/*.xls"]
        files: list[str] = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(root, p), recursive=True))
        return files
    return []


@st.cache_data(show_spinner=False)
def _find_base_with_ingresos(base_path: str | None) -> tuple[pd.DataFrame | None, str | None]:
    """
    Busca una base que contenga ingresos_de_explotacion, en este orden:
      1) base_path (si es fichero o carpeta)
      2) carpeta de DATA_PATH
      3) carpeta de base_path (si era fichero)
      4) carpeta ./data (si existe)
    """
    search_roots: list[str] = []

    if base_path:
        search_roots.append(base_path)
        if os.path.isfile(base_path):
            search_roots.append(os.path.dirname(base_path))

    if DATA_PATH:
        search_roots.append(os.path.dirname(DATA_PATH))

    if os.path.isdir("data"):
        search_roots.append("data")

    candidates: list[str] = []
    for r in search_roots:
        candidates.extend(_list_candidate_files(r))

    # heurística: priorizar nombres típicos de base original/completa
    def score(p: str) -> int:
        name = os.path.basename(p).lower()
        s = 0
        if any(k in name for k in ["original", "completa", "full", "raw", "base"]):
            s += 10
        if "clean" in name:
            s -= 2
        if "interim" in p.lower():
            s -= 1
        return -s  # para sort asc

    candidates = sorted(list(dict.fromkeys(candidates)), key=score)

    for f in candidates:
        df_full = _read_any(f)
        if df_full is None:
            continue
        if "ingresos_de_explotacion" in df_full.columns:
            return df_full, f

    return None, None


def _ensure_ingresos(df_view: pd.DataFrame, base_path: str | None, diagnostic: bool = False) -> pd.DataFrame:
    """
    Garantiza que df_view tenga ingresos_de_explotacion.
    Si no, busca una base con ingresos y hace merge por codigo_nif o nombre.
    """
    if "ingresos_de_explotacion" in df_view.columns:
        return df_view

    df_full, _src = _find_base_with_ingresos(base_path)
    if df_full is None:
        # Sin diagnóstico visible (no checkbox), pero si quisieras activarlo en dev:
        if diagnostic:
            st.warning(
                "No encuentro ninguna base con 'ingresos_de_explotacion' "
                "(busqué en base_path, carpeta de DATA_PATH y ./data)."
            )
        return df_view

    merge_key = None
    if "codigo_nif" in df_view.columns and "codigo_nif" in df_full.columns:
        merge_key = "codigo_nif"
    elif "nombre" in df_view.columns and "nombre" in df_full.columns:
        merge_key = "nombre"

    if merge_key is None:
        if diagnostic:
            st.warning("No puedo hacer merge: no hay clave común (codigo_nif o nombre).")
        return df_view

    df_full2 = df_full[[merge_key, "ingresos_de_explotacion"]].copy()
    df_full2["ingresos_de_explotacion"] = pd.to_numeric(df_full2["ingresos_de_explotacion"], errors="coerce")

    out = df_view.merge(df_full2, on=merge_key, how="left")

    # Nada visible: sin caption verde, sin botones
    if diagnostic:
        st.caption(f"Base ingresos usada: {_src}")
        st.caption(f"Merge key: {merge_key} · ingresos no-nulo: {out['ingresos_de_explotacion'].notna().sum()}")

    return out


# ============================================================
# Vista Resumen
# ============================================================
def render_resumen(df: pd.DataFrame, comparar_con: str, zoom: bool, base_path: str | None = None):
    df = df.copy()

    # ======================
    # SELECTOR EMPRESA
    # ======================
    if "codigo_nif" in df.columns:
        df["empresa_key"] = df["nombre"].astype(str) + "  —  " + df["codigo_nif"].astype(str)
        selector_col = "empresa_key"
    else:
        selector_col = "nombre"

    empresa_sel = st.selectbox("Busca/selecciona empresa", sorted(df[selector_col].dropna().unique()))
    row = df.loc[df[selector_col] == empresa_sel].iloc[0]

    # Referencia (cluster o total)
    if comparar_con == "Solo su cluster":
        df_ref = df[df["cluster_label"] == row["cluster_label"]].copy()
    else:
        df_ref = df.copy()

    # ======================
    # INTERPRETACIÓN CLÚSTER + IMPLICACIONES
    # ======================
    CLUSTER_N = df["cluster_label"].value_counts(dropna=True).to_dict()
    cluster_sel = row["cluster_label"]
    n_sel = int(CLUSTER_N.get(cluster_sel, 0))
    pct_sel = (n_sel / max(1, len(df))) * 100

    CLUSTER_STORY = {
        "C1": {
            "titulo": "C1: ciclo largo y VA bajo",
            "bullets": [
                "La rotación de stocks es muy baja, el valor de NOFS/Ventas muy altas muestra una fuerte necesidad de financiación.",
                "Productividad baja similar o inferior a C1, resultados y EBITDA reducidos.",
                "El valor de personal (%) es el más alto del conjunto, escala pequeña-media algo mayor que C3 pero muy lejos de C2 y la rotación de activos es baja.",
            ],
            "implicaciones": [
                "Empresas menos dinámicas, pero estables.",
                "Son empresas con ciclos largos, mayor inmovilización y menor capacidad de convertir recursos en resultados.",
                "Pueden ser empresas que suelen responder a nichos, especialización o rigidez estructural con menor margen de maniobra.",
            ],
        },
        "C2": {
            "titulo": "C2: modelo eficiente, escalable y dominante",
            "bullets": [
                "Se puede observar que se da una máxima productividad VA y ventas por empleado, resultados EBITDA y cash flow muy superiores.",
                "Se da una escala enorme de ingresos, activo, fondos propios, inmovilizado, existencias y materiales.",
                "El valor de las NOFS/Ventas es intermedio con una estructura financiera estable, personal (%) bajo con modelo capital-intensivo.",
                "El valor de EBITDA (%) y rendimiento más altos, inmovilizado/empleado muy elevado con inversión tecnológica/industrial.",
            ],
            "implicaciones": [
                "Tenemos empresas grandes, productivas y financieramente sólidas.",
                "Este es el modelo “ganador” clásico donde se da una eficiencia + escala + productividad, alta capacidad de generación de valor, una estructura madura y robusta.",
                "Pueden ser empresas industriales consolidadas con modelos de negocio difíciles de replicar.",
            ],
        },
        "C3": {
            "titulo": "C3: modelo intensivo en rotación y margen con baja escala",
            "bullets": [
                "La rotación de stocks es muy alta con ciclo operativo rápido, alta competitividad.",
                "NOFS/Ventas bajas con poca financiación del circulante, productividad de ventas y VA por empleado baja/media.",
                "Los resultados económicos son positivos pero moderados, el personal (%) es elevado indicándonos una estructura intensiva en trabajo.",
                "La escala es reducida con ingresos, activo, fondos propios e inmovilizado bajos. La alta rotación de activos nos indica que exprimen bien lo que tienen.",
            ],
            "implicaciones": [
                "En este grupo podemos estar englobando empresas pequeñas/medianas muy dinámicas operativamente.",
                "Empresas que son ágiles, de volumen contenido y que compensan su menor tamaño con rotación, margen operativo y eficiencia comercial.",
                "No crecen tanto por escala como por velocidad y control operativo.",
            ],
        },
    }

    story = CLUSTER_STORY.get(
        str(cluster_sel),
        {
            "titulo": "Interpretación no definida",
            "bullets": ["Define aquí el texto del clúster para tu memoria/defensa."],
            "implicaciones": ["Añade recomendaciones específicas por clúster."],
        },
    )

    # ======================
    # RADAR
    # ======================
    def make_radar(df_ref_: pd.DataFrame, row_: pd.Series) -> go.Figure:
        categories: list[str] = []
        empresa_vals: list[float] = []

        for var in VARS_CLUSTER:
            if var not in df_ref_.columns:
                continue
            s = pd.to_numeric(df_ref_[var], errors="coerce").dropna()
            v = pd.to_numeric(row_.get(var, np.nan), errors="coerce")
            if len(s) == 0 or pd.isna(v):
                continue

            _s_disp = to_display_scale(var, s)
            v_disp = float(to_display_scale(var, pd.Series([v])).iloc[0])

            categories.append(LABELS.get(var, var))
            empresa_vals.append(v_disp)

        if len(categories) < 3:
            fig0 = go.Figure()
            fig0.update_layout(height=340, margin=dict(l=30, r=30, t=30, b=30), title="Radar (perfil de la empresa)")
            return fig0

        emp_norm: list[float] = []
        vars_in = [v for v in VARS_CLUSTER if LABELS.get(v, v) in categories]

        for i, var in enumerate(vars_in):
            s = pd.to_numeric(df_ref_[var], errors="coerce").dropna()
            if len(s) == 0:
                continue
            s_disp = to_display_scale(var, s)

            q1 = float(s_disp.quantile(0.25))
            q3 = float(s_disp.quantile(0.75))
            iqr = (q3 - q1) if (q3 - q1) != 0 else 1e-9

            e = (empresa_vals[i] - q1) / iqr
            e = max(0.0, min(2.0, e)) / 2.0
            emp_norm.append(e)

        if len(emp_norm) != len(categories):
            # fallback seguro si hubiera algún descuadre por datos faltantes
            fig0 = go.Figure()
            fig0.update_layout(height=340, margin=dict(l=30, r=30, t=30, b=30), title="Radar (perfil de la empresa)")
            return fig0

        categories_closed = categories + [categories[0]]
        emp_closed = emp_norm + [emp_norm[0]]

        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatterpolar(
                r=emp_closed,
                theta=categories_closed,
                fill="toself",
                name="Empresa",
                opacity=0.6,
                hovertemplate="%{theta}: %{r:.0%}<extra></extra>",
            )
        )
        fig1.update_layout(
            height=420,
            margin=dict(l=30, r=30, t=50, b=30),
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%")),
            showlegend=False,
            title="Radar — perfil de la empresa (normalizado por IQR)",
        )
        return fig1

    # ======================
    # FUNCIONES: percentil + score global
    # ======================
    def empirical_percentile(s: pd.Series, x: float) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0 or pd.isna(x):
            return np.nan
        return float((s <= x).mean() * 100.0)

    def robust_score_iqr(s_disp: pd.Series, x_disp: float) -> float:
        s_disp = pd.to_numeric(s_disp, errors="coerce").dropna()
        if len(s_disp) == 0 or pd.isna(x_disp):
            return np.nan
        q1 = float(s_disp.quantile(0.25))
        med = float(s_disp.quantile(0.50))
        q3 = float(s_disp.quantile(0.75))
        iqr = (q3 - q1) if (q3 - q1) != 0 else 1e-9
        z = (float(x_disp) - med) / iqr
        z = max(-2.0, min(2.0, z))
        return float(z)

    def overall_index(scores: list[float]) -> float:
        vals = [v for v in scores if pd.notna(v)]
        if not vals:
            return np.nan
        m = float(np.mean(vals))
        return (m + 2.0) / 4.0 * 100.0

    # ======================
    # LAYOUT PRINCIPAL
    # ======================
    left, right = st.columns([2.2, 1.3], gap="large")

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

        if zoom:
            fig.update_xaxes(range=[row["PC1"] - 1.0, row["PC1"] + 1.0])
            fig.update_yaxes(range=[row["PC2"] - 1.0, row["PC2"] + 1.0])

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div id="interpretacion"></div>', unsafe_allow_html=True)
        st.divider()
        st.subheader("Interpretación del clúster")
        st.caption(f"Clúster: **{cluster_sel}** · n={n_sel} ({pct_sel:.1f}% de la muestra)")
        st.markdown(f"**{story['titulo']}**")
        st.markdown("- " + "\n- ".join(story["bullets"]))
        st.markdown("**Implicaciones prácticas:**")
        st.markdown("- " + "\n- ".join(story["implicaciones"]))

        st.markdown('<div id="perfil-radar"></div>', unsafe_allow_html=True)
        st.divider()
        st.subheader("Perfil (radar)")
        radar_fig = make_radar(df_ref_=df_ref, row_=row)
        st.plotly_chart(radar_fig, use_container_width=True)

    with right:
        st.markdown('<div id="indicadores"></div>', unsafe_allow_html=True)

        st.subheader("Empresa seleccionada")
        st.write(f"**{row['nombre']}**")
        st.write(f"Cluster: **{row['cluster_label']}**")
        st.caption(f"Comparación: {comparar_con} (n={len(df_ref)})")
        st.divider()

        st.subheader("Indicadores (valor, percentil y estadísticos)")

        rows = []
        EPS = 1e-9
        score_list: list[float] = []

        for var in VARS_CLUSTER:
            if var not in df.columns or var not in df_ref.columns:
                continue

            s_raw = pd.to_numeric(df_ref[var], errors="coerce").dropna()
            if len(s_raw) == 0:
                continue

            v_raw = pd.to_numeric(row.get(var, np.nan), errors="coerce")
            if pd.isna(v_raw):
                continue

            s_disp = to_display_scale(var, s_raw)
            v_disp = float(to_display_scale(var, pd.Series([v_raw])).iloc[0])

            q1 = float(s_disp.quantile(0.25))
            med = float(s_disp.quantile(0.50))
            q3 = float(s_disp.quantile(0.75))
            mean = float(s_disp.mean())
            sd = float(s_disp.std(ddof=1))

            if pd.isna(v_disp) or pd.isna(med):
                flag = "—"
            else:
                tol = 0.01 * (abs(med) + EPS)
                if v_disp > med + tol:
                    flag = "↑"
                elif v_disp < med - tol:
                    flag = "↓"
                else:
                    flag = "≈"

            pctl = empirical_percentile(s_disp, v_disp)
            z_iqr = robust_score_iqr(s_disp, v_disp)
            score_list.append(z_iqr)

            rows.append(
                {
                    "Indicador": LABELS.get(var, var),
                    "Valor empresa": v_disp,
                    "Percentil": pctl,
                    "vs mediana": flag,
                    "Q1": q1,
                    "Mediana": med,
                    "Q3": q3,
                    "Media": mean,
                    "Desv. típica": sd,
                }
            )

        stats_df = pd.DataFrame(rows)
        idx_global = overall_index(score_list)

        colx, coly = st.columns([1.0, 1.0])
        with colx:
            st.metric("Índice global (0–100)", value="" if pd.isna(idx_global) else f"{idx_global:.1f}")
        with coly:
            st.metric("Tamaño del clúster", value=f"{n_sel}", delta=f"{pct_sel:.1f}% muestra")

        if not stats_df.empty:
            num_cols = ["Valor empresa", "Q1", "Mediana", "Q3", "Media", "Desv. típica", "Percentil"]
            for c in num_cols:
                stats_df[c] = pd.to_numeric(stats_df[c], errors="coerce")

            stats_show = stats_df.copy()
            stats_show["Valor empresa"] = stats_show["Valor empresa"].apply(fmt_num)
            for c in ["Q1", "Mediana", "Q3", "Media", "Desv. típica"]:
                stats_show[c] = stats_show[c].apply(fmt_num)
            stats_show["Percentil"] = stats_df["Percentil"].apply(lambda x: "" if pd.isna(x) else f"{x:.0f}")

            def color_flag(x):
                if x == "↑":
                    return "color: #1a7f37; font-weight: 700;"
                if x == "↓":
                    return "color: #b42318; font-weight: 700;"
                if x == "≈":
                    return "color: #6b7280; font-weight: 700;"
                return ""

            styled = stats_show.style.applymap(color_flag, subset=["vs mediana"])
            try:
                st.dataframe(styled, use_container_width=True, hide_index=True)
            except Exception:
                st.write(stats_show)
        else:
            st.warning("No hay indicadores para mostrar.")

    # ======================
    # TOP EMPRESAS (por ingresos)
    # ======================
    st.markdown('<div id="top-empresas"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("Top empresas (global o por localidad)")

    # IMPORTANT: diagnostic oculto (sin checkbox)
    df_top = _ensure_ingresos(df_view=df, base_path=base_path, diagnostic=False)

    if "ingresos_de_explotacion" not in df_top.columns:
        st.warning("No puedo calcular el Top: falta ingresos_de_explotacion.")
        return

    df_top["ingresos_rank"] = pd.to_numeric(df_top["ingresos_de_explotacion"], errors="coerce")

    clusters = sorted(df_top["cluster_label"].dropna().unique())
    if not clusters:
        st.info("No hay clusters disponibles.")
        return

    colA, colB = st.columns([1.2, 1.0])
    with colA:
        cluster_choice = st.selectbox("Cluster", clusters)
    with colB:
        top_n = st.slider("Top N", 5, 30, 10, 1)

    df_c = df_top[df_top["cluster_label"] == cluster_choice].copy()

    loc_col = None
    if "localidad_grp" in df_c.columns:
        loc_col = "localidad_grp"
    elif "localidad" in df_c.columns:
        loc_col = "localidad"

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
        .loc[:, ["nombre", "ingresos_rank"]]
        .rename(columns={"ingresos_rank": "Ingresos de explotación"})
    )

    st.markdown(f"### {titulo}")

    if top_df.empty:
        st.info("No hay datos suficientes para este filtro.")
    else:
        top_df = top_df.reset_index(drop=True)
        top_df.index += 1
        top_df.index.name = "Ranking"
        st.dataframe(top_df, use_container_width=True)

    # ======================
    # DESCARGAS
    # ======================
    st.markdown('<div id="descargas"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("Descargas")

    if "stats_df" in locals() and not stats_df.empty:
        export_df = stats_df.copy()
        export_df["Empresa"] = row["nombre"]
        export_df["Cluster"] = row["cluster_label"]
        export_df = export_df[["Empresa", "Cluster"] + [c for c in export_df.columns if c not in ["Empresa", "Cluster"]]]

        buf = io.StringIO()
        export_df.to_csv(buf, index=False)
        st.download_button(
            label="⬇️ Descargar ficha de empresa (CSV)",
            data=buf.getvalue().encode("utf-8"),
            file_name=f"ficha_empresa_{row['cluster_label']}_{row['nombre']}.csv",
            mime="text/csv",
        )

    mini_cols = [c for c in ["nombre", "cluster_label", "PC1", "PC2"] + VARS_CLUSTER if c in df_ref.columns]
    buf2 = io.StringIO()
    df_ref[mini_cols].to_csv(buf2, index=False)
    st.download_button(
        label="⬇️ Descargar datos de referencia (CSV)",
        data=buf2.getvalue().encode("utf-8"),
        file_name=f"datos_referencia_{'cluster' if comparar_con == 'Solo su cluster' else 'total'}.csv",
        mime="text/csv",
    )