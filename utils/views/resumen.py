# utils/views/resumen.py
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.config import VARS_CLUSTER, LABELS
from utils.fmt import to_display_scale, fmt_num


def render_resumen(df: pd.DataFrame, comparar_con: str, zoom: bool):
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
    def make_radar(df_ref: pd.DataFrame, row: pd.Series) -> go.Figure:
        categories = []
        empresa_vals = []

        for var in VARS_CLUSTER:
            if var not in df_ref.columns:
                continue

            s = pd.to_numeric(df_ref[var], errors="coerce").dropna()
            v = pd.to_numeric(row.get(var, np.nan), errors="coerce")
            if len(s) == 0 or pd.isna(v):
                continue

            s_disp = to_display_scale(var, s)
            v_disp = float(to_display_scale(var, pd.Series([v])).iloc[0])

            categories.append(LABELS.get(var, var))
            empresa_vals.append(v_disp)

        if len(categories) < 3:
            fig = go.Figure()
            fig.update_layout(height=340, margin=dict(l=30, r=30, t=30, b=30), title="Radar (perfil de la empresa)")
            return fig

        emp_norm = []
        for i, var in enumerate([v for v in VARS_CLUSTER if LABELS.get(v, v) in categories]):
            s = pd.to_numeric(df_ref[var], errors="coerce").dropna()
            if len(s) == 0:
                continue
            s_disp = to_display_scale(var, s)

            q1 = float(s_disp.quantile(0.25))
            q3 = float(s_disp.quantile(0.75))
            iqr = (q3 - q1) if (q3 - q1) != 0 else 1e-9

            e = (empresa_vals[i] - q1) / iqr
            e = max(0.0, min(2.0, e)) / 2.0
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
            title="Radar — perfil de la empresa (normalizado por IQR)",
        )
        return fig

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

    # ----------------------
    # LEFT: PCA + EXPLICACIÓN + RADAR
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
            labels={"PC1": "Componente principal 1", "PC2": "Componente principal 2", "cluster_label": "Modelo de negocio"},
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

        with st.expander("ℹ️ ¿Qué representa este PCA? (explicación + contribución de indicadores)", expanded=False):
            st.markdown(
                """
**Cómo leer el gráfico:**
- Cada punto es una empresa.
- Empresas cercanas → perfiles de indicadores similares (en el espacio estandarizado del modelo).
- Los colores corresponden al clúster asignado por K-means.
- PC1 y PC2 son combinaciones lineales de los indicadores (resumen de la variabilidad).

**Nota práctica:** como en la app no re-entrenamos el PCA, estimamos la “influencia” de cada indicador con su **correlación** con PC1 y PC2.
"""
            )
            corr_rows = []
            for var in VARS_CLUSTER:
                if var not in df.columns:
                    continue
                x = pd.to_numeric(df[var], errors="coerce")
                corr1 = x.corr(df["PC1"])
                corr2 = x.corr(df["PC2"])
                corr_rows.append({"Indicador": LABELS.get(var, var), "corr(PC1)": corr1, "corr(PC2)": corr2})
            corr_df = pd.DataFrame(corr_rows)
            if not corr_df.empty:
                corr_df["|corr(PC1)|"] = corr_df["corr(PC1)"].abs()
                corr_df["|corr(PC2)|"] = corr_df["corr(PC2)"].abs()
                corr_df = corr_df.sort_values(["|corr(PC1)|", "|corr(PC2)|"], ascending=False).drop(
                    columns=["|corr(PC1)|", "|corr(PC2)|"]
                )
                st.dataframe(
                    corr_df.style.format({"corr(PC1)": "{:.3f}", "corr(PC2)": "{:.3f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No hay datos suficientes para calcular correlaciones con PC1/PC2.")

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
        radar_fig = make_radar(df_ref=df_ref, row=row)
        st.plotly_chart(radar_fig, use_container_width=True)

    # ----------------------
    # RIGHT: INDICADORES + PERCENTILES + SCORE GLOBAL
    # ----------------------
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
        score_list = []

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
    # COMPARACIÓN VISUAL
    # ======================
    st.markdown('<div id="comparacion-visual"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("Comparación visual (empresa vs mediana)")

    if stats_df.empty:
        st.info("No hay suficientes datos para el gráfico.")
    else:
        viz = stats_df.dropna(subset=["Valor empresa", "Mediana", "Q1", "Q3"]).copy()
        viz["abs_gap"] = (viz["Valor empresa"] - viz["Mediana"]).abs()
        viz = viz.sort_values("abs_gap", ascending=False)

        max_n = max(4, min(12, len(viz)))
        top_n = st.slider("Nº de indicadores a mostrar", 4, max_n, min(8, max_n))
        viz = viz.head(top_n)

        use_log = st.checkbox("Escala log (recomendado si hay magnitudes muy distintas)", value=True)

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
                hovertemplate="Empresa: %{x}<extra></extra>",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=viz["Mediana"],
                y=viz["Indicador"],
                mode="markers",
                name="Mediana",
                marker=dict(size=10, symbol="line-ns-open"),
                hovertemplate="Mediana: %{x}<extra></extra>",
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
    # CASOS TIPO
    # ======================
    st.markdown('<div id="casos-tipo"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("Casos tipo (por clúster)")

    rep_n = 3
    out_n = 1

    tmp = df[["nombre", "cluster_label", "PC1", "PC2"]].dropna().copy()
    if tmp.empty:
        st.info("No hay datos suficientes para construir casos tipo.")
    else:
        cases_rows = []
        for cl, g in tmp.groupby("cluster_label"):
            c1 = float(g["PC1"].mean())
            c2 = float(g["PC2"].mean())
            d = np.sqrt((g["PC1"] - c1) ** 2 + (g["PC2"] - c2) ** 2)
            gg = g.copy()
            gg["dist"] = d

            reps = gg.sort_values("dist", ascending=True).head(rep_n)
            outs = gg.sort_values("dist", ascending=False).head(out_n)

            for _, r in reps.iterrows():
                cases_rows.append({"Cluster": cl, "Tipo": "Representativa", "Empresa": r["nombre"]})
            for _, r in outs.iterrows():
                cases_rows.append({"Cluster": cl, "Tipo": "Outlier", "Empresa": r["nombre"]})

        cases_df = pd.DataFrame(cases_rows)
        st.dataframe(cases_df, use_container_width=True, hide_index=True)

    # ======================
    # TOP EMPRESAS
    # ======================
    st.markdown('<div id="top-empresas"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("Top empresas (global o por localidad)")

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

    mini_cols = [c for c in ["nombre", "cluster_label", "PC1", "PC2"] + VARS_CLUSTER if c in df.columns]
    buf2 = io.StringIO()
    df_ref[mini_cols].to_csv(buf2, index=False)
    st.download_button(
        label="⬇️ Descargar datos de referencia (CSV)",
        data=buf2.getvalue().encode("utf-8"),
        file_name=f"datos_referencia_{'cluster' if comparar_con=='Solo su cluster' else 'total'}.csv",
        mime="text/csv",
    )