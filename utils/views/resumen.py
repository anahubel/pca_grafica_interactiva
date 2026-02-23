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

    def score(p: str) -> int:
        name = os.path.basename(p).lower()
        s = 0
        if any(k in name for k in ["original", "completa", "full", "raw", "base"]):
            s += 10
        if "clean" in name:
            s -= 2
        if "interim" in p.lower():
            s -= 1
        return -s  # sort asc

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

    if diagnostic:
        st.caption(f"Base ingresos usada: {_src}")
        st.caption(f"Merge key: {merge_key} · ingresos no-nulo: {out['ingresos_de_explotacion'].notna().sum()}")

    return out


# ============================================================
# Vista Resumen
# ============================================================
def render_resumen(
    df: pd.DataFrame,
    comparar_con: str,
    zoom: bool,
    base_path: str | None = None,
    show_subgroup_interpretation: bool = False,   # 👈 por defecto NO
    story_map_override: dict | None = None,        # 👈 opcional (Textil)
    story_title: str = "Interpretación del clúster",  # 👈 opcional (Textil)
    normalize_cluster_labels: bool = True,   # 👈 NUEVO
):
    df = df.copy()

    # ======================
    # SELECTOR EMPRESA
    # ======================
    if "codigo_nif" in df.columns and "nombre" in df.columns:
        df["empresa_key"] = df["nombre"].astype(str) + "  —  " + df["codigo_nif"].astype(str)
        selector_col = "empresa_key"
    else:
        selector_col = "nombre" if "nombre" in df.columns else df.columns[0]

    empresa_sel = st.selectbox("Busca/selecciona empresa", sorted(df[selector_col].dropna().unique()))
    row = df.loc[df[selector_col] == empresa_sel].iloc[0]

    # ======================
    # Normaliza cluster_label (por si viene como 1.0/2.0/3.0)
    # ======================
    _raw_cluster = row.get("cluster_label", "—")
    if normalize_cluster_labels:
        _cluster_map = {"1": "C1", "1.0": "C1", "2": "C2", "2.0": "C2", "3": "C3", "3.0": "C3"}
        cluster_sel = _cluster_map.get(str(_raw_cluster).strip(), str(_raw_cluster).strip())
    else:
        cluster_sel = str(_raw_cluster).strip()

    # ======================
    # Referencia (cluster o total)
    # ======================
    if comparar_con == "Solo su cluster" and "cluster_label" in df.columns and "cluster_label" in row.index:
        df_ref = df[df["cluster_label"] == _raw_cluster].copy()
        if df_ref.empty and cluster_sel in {"C1", "C2", "C3"}:
            df_ref = df[df["cluster_label"] == cluster_sel].copy()
    else:
        df_ref = df.copy()

    # ======================
    # Tamaño cluster seleccionado
    # ======================
    CLUSTER_N = df["cluster_label"].value_counts(dropna=True).to_dict() if "cluster_label" in df.columns else {}
    n_sel = int(CLUSTER_N.get(_raw_cluster, CLUSTER_N.get(cluster_sel, 0)))
    pct_sel = (n_sel / max(1, len(df))) * 100.0

    # ======================
    # Textos: clúster general
    # ======================
    CLUSTER_STORY = {
        "C1": {
            "titulo": "C1: modelo intensivo en circulante y eficiencia media-baja",
            "rasgos_estructurales": [
                "Rotación de stocks: la más baja del conjunto (ciclo operativo más lento).",
                "Productividad VA/pax: la más reducida (menor eficiencia en generación de valor).",
                "Inmovilizado/empleado: nivel intermedio (estructura productiva moderada).",
                "NOFS/Ventas: el más elevado (mayor presión de circulante y financiación operativa).",
            ],
            "rasgos_economicos": [
                "Resultados y rentabilidad: peor desempeño (resultado del ejercicio, explotación, EBITDA y cash flow).",
                "Estructura: mayor % de personal y márgenes contenidos, coherente con un modelo más intensivo en operativa y menos eficiente.",
            ],
            "lectura_economica": [
                "Empresas con ciclo operativo más lento y mayor necesidad de financiar existencias y clientes.",
                "Menor eficiencia productiva por trabajador y rentabilidad limitada.",
                "Modelo tradicional y operativo, con menor dinamismo comercial.",
            ],
            "implicaciones": [
                "Mejoras típicas: acelerar rotación, optimizar existencias/cobros/pagos y reducir tensión de circulante.",
                "Prioridad: elevar productividad (procesos/organización/eficiencia).",
                "Modelo más vulnerable: un shock de demanda o costes se traslada rápido a resultados.",
            ],
        },
        "C2": {
            "titulo": "C2: modelo ágil y eficiente en circulante (perfil competitivo)",
            "rasgos_estructurales": [
                "Rotación de stocks: la más alta del conjunto (máxima agilidad operativa/comercial).",
                "NOFS/Ventas: el más bajo (modelo más “ligero” en circulante, menor necesidad de financiación operativa).",
                "Inmovilizado/empleado: relativamente bajo (estructura menos rígida).",
                "Productividad VA/pax: nivel intermedio (mejora más por rotación/ejecución que por capital).",
            ],
            "rasgos_economicos": [
                "Competitividad: niveles más altos (perfil dinámico y orientado a ejecución).",
                "Resultados: moderados/positivos, típicamente mejores que C1 pero sin el perfil capital-intensivo de C3.",
            ],
            "lectura_economica": [
                "Empresas con ciclo operativo rápido, buena disciplina de circulante y alto dinamismo comercial.",
                "Modelo eficiente en capital circulante: aguanta mejor tensiones de liquidez.",
                "Compite por velocidad/ejecución más que por intensidad de capital.",
            ],
            "implicaciones": [
                "Modelo resiliente en términos de liquidez: menos exposición a tensiones por circulante.",
                "Palanca de mejora: subir margen/eficiencia sin perder rotación.",
                "Sensibilidad: necesita sostener volumen/rotación; caídas de demanda pueden impactar rápido.",
            ],
        },
        "C3": {
            "titulo": "C3: modelo intensivo en capital y alta productividad",
            "rasgos_estructurales": [
                "Inmovilizado/empleado: el más alto (modelo intensivo en capital/activos).",
                "Productividad VA/pax: la más alta (alto valor generado por unidad de trabajo).",
                "Rotación de stocks: intermedia (no compite por velocidad sino por estructura/productividad).",
            ],
            "rasgos_economicos": [
                "Resultados: superiores (mayor generación de EBITDA, cash flow y resultado del ejercicio).",
                "Escala: niveles de activo y fondos propios más altos, coherente con estructura intensiva en capital.",
            ],
            "lectura_economica": [
                "Empresas con estructura productiva fuerte, inversión relevante en activos y alta productividad.",
                "Modelo más sofisticado (posible perfil industrial/tecnológico o con economías de escala).",
                "Rinde bien si mantiene utilización eficiente de la capacidad instalada.",
            ],
            "implicaciones": [
                "Modelo robusto: mayor capacidad de inversión y resistencia ante shocks moderados.",
                "Riesgo típico: rigidez (costes fijos/capex) si cae demanda o baja utilización de activos.",
                "Foco: asegurar eficiencia del capital y disciplina de inversión/mantenimiento.",
            ],
        },
    }

    # ✅ story_map override (Textil u otras vistas)
    _story_map = story_map_override if isinstance(story_map_override, dict) else CLUSTER_STORY
    story = _story_map.get(
        str(cluster_sel),
        {
            "titulo": "Interpretación no definida",
            "rasgos_estructurales": ["Define aquí el texto del clúster."],
            "rasgos_economicos": [],
            "lectura_economica": [],
            "implicaciones": [],
        },
    )

    # ======================
    # SUBGRUPO (SOLO si show_subgroup_interpretation=True)
    # ======================
    subgroup_col = None
    subgroup_sel = None
    sub_story = None

    SUBGROUP_STORY_C1 = {
        "1.0": {
            "titulo": "C1.1: capital-intensivo y productivo",
            "rasgos_estructurales": [
                "Inmovilizado/empleado: el más alto del subgrupo.",
                "Productividad VA/pax: la más alta del subgrupo.",
                "Rotación de stocks: intermedia.",
            ],
            "rasgos_economicos": [],
            "lectura_economica": [
                "Perfil basado en estructura/capacidad y productividad, más que en velocidad comercial.",
                "Modelo con mayor rigidez operativa (costes fijos/activos) si cae la demanda.",
            ],
            "implicaciones": [
                "Asegurar utilización eficiente de capacidad y disciplina de inversión.",
                "Mejoras de eficiencia operativa para sostener productividad.",
            ],
        },
        "2.0": {
            "titulo": "C1.2: ligero pero menos eficiente",
            "rasgos_estructurales": [
                "Rotación de stocks: la más baja del subgrupo.",
                "Productividad VA/pax: la más baja del subgrupo.",
                "Inmovilizado/empleado: el más bajo del subgrupo.",
            ],
            "rasgos_economicos": [],
            "lectura_economica": [
                "Perfil con menos estructura y también menor capacidad de convertir en valor.",
                "Potencialmente el subgrupo más vulnerable dentro de C1.",
            ],
            "implicaciones": [
                "Prioridad: elevar productividad (procesos/organización/mix) y eficiencia.",
                "Revisar disciplina de circulante y costes si los resultados acompañan este perfil.",
            ],
        },
        "3.0": {
            "titulo": "C1.3: ágil en rotación, productividad intermedia",
            "rasgos_estructurales": [
                "Rotación de stocks: la más alta del subgrupo.",
                "Productividad VA/pax: intermedia.",
                "Inmovilizado/empleado: intermedio.",
            ],
            "rasgos_economicos": [],
            "lectura_economica": [
                "Perfil que compite más por agilidad/ejecución que por intensidad de capital.",
                "Puede escalar si convierte rotación en margen y eficiencia.",
            ],
            "implicaciones": [
                "Palanca clave: mejorar margen/eficiencia sin perder rotación.",
                "Vigilar sensibilidad a caídas de volumen (necesita mantener rotación).",
            ],
        },
    }

    if show_subgroup_interpretation and cluster_sel == "C1":
        # detecta columna (solo C1)
        ALLOWED_C1_SUBGROUP_COLS = ["cluster_modelo_negocio", "cluster_k3", "cluster_k3_label"]
        subgroup_col = next((c for c in ALLOWED_C1_SUBGROUP_COLS if c in df.columns), None)

        if subgroup_col is not None:
            v = row.get(subgroup_col, None)
            if pd.notna(v):
                subgroup_sel = str(v).strip()

        # normaliza 1/2/3 -> 1.0/2.0/3.0
        if subgroup_sel in {"1", "2", "3"}:
            subgroup_sel = {"1": "1.0", "2": "2.0", "3": "3.0"}[subgroup_sel]

        if subgroup_sel in {"1.0", "2.0", "3.0"}:
            sub_story = SUBGROUP_STORY_C1.get(subgroup_sel)

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
    # LAYOUT PRINCIPAL (PCA + interpretación + radar + tabla indicadores)
    # ======================
    left, right = st.columns([2.2, 1.3], gap="large")

    with left:
        st.markdown('<div id="pca"></div>', unsafe_allow_html=True)

        if ("PC1" not in df.columns) or ("PC2" not in df.columns) or ("PC1" not in row.index) or ("PC2" not in row.index):
            st.warning("No puedo mostrar el gráfico PCA: faltan columnas PC1 y/o PC2 en el dataset cargado.")
        else:
            COLOR_MAP = {
            "C1": "#1f3b73",   # azul oscuro
            "C2": "#d97706",   # naranja fuerte
            "C3": "#15803d",   # verde intenso
            "1.0": "#1f3b73",
            "2.0": "#d97706",
            "3.0": "#15803d",
        }

            
            fig = px.scatter(
                df,
                x="PC1",
                y="PC2",
                color="cluster_label" if "cluster_label" in df.columns else None,
                hover_name="nombre" if "nombre" in df.columns else None,
                opacity=0.75,
                color_discrete_map=COLOR_MAP,
                labels={
                    "PC1": "Componente principal 1",
                    "PC2": "Componente principal 2",
                    "cluster_label": "Modelo de negocio",
                },
            )
            fig.update_traces(
                marker=dict(
                    size=8,
                    line=dict(width=0.5, color="white")
                )
            )
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

        # 👇 título configurable (Textil)
        st.subheader(story_title)

        st.caption(f"Clúster: **{cluster_sel}** · n={n_sel} ({pct_sel:.1f}% de la muestra)")
        st.markdown(f"**{story.get('titulo', '—')}**")

        st.markdown("**Rasgos estructurales:**")
        st.markdown("- " + "\n- ".join(story.get("rasgos_estructurales", [])))

        st.markdown("**Rasgos económicos:**")
        st.markdown("- " + "\n- ".join(story.get("rasgos_economicos", [])))

        st.markdown("**Lectura económica:**")
        st.markdown("- " + "\n- ".join(story.get("lectura_economica", [])))

        st.markdown("**Implicaciones prácticas:**")
        st.markdown("- " + "\n- ".join(story.get("implicaciones", [])))

        # ✅ Subgrupo SOLO si lo activas explícitamente
        if show_subgroup_interpretation and sub_story is not None:
            st.divider()
            st.subheader("Interpretación del subgrupo (solo C1)")

            pretty = {"1.0": "C1.1", "2.0": "C1.2", "3.0": "C1.3"}.get(str(subgroup_sel), str(subgroup_sel))
            st.caption(f"Subgrupo: **{pretty}** · columna={subgroup_col}")

            st.markdown(f"**{sub_story.get('titulo', '—')}**")

            st.markdown("**Rasgos estructurales:**")
            st.markdown("- " + "\n- ".join(sub_story.get("rasgos_estructurales", [])))

            if sub_story.get("rasgos_economicos"):
                st.markdown("**Rasgos económicos:**")
                st.markdown("- " + "\n- ".join(sub_story.get("rasgos_economicos", [])))

            st.markdown("**Lectura económica:**")
            st.markdown("- " + "\n- ".join(sub_story.get("lectura_economica", [])))

            st.markdown("**Implicaciones prácticas:**")
            st.markdown("- " + "\n- ".join(sub_story.get("implicaciones", [])))

        st.markdown('<div id="perfil-radar"></div>', unsafe_allow_html=True)
        st.divider()
        st.subheader("Perfil (radar)")
        radar_fig = make_radar(df_ref_=df_ref, row_=row)
        st.plotly_chart(radar_fig, use_container_width=True)

    with right:
        st.markdown('<div id="indicadores"></div>', unsafe_allow_html=True)

        st.subheader("Empresa seleccionada")
        st.write(f"**{row.get('nombre', '—')}**")
        st.write(f"Cluster: **{cluster_sel}**")
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
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.warning("No hay indicadores para mostrar.")

    # ======================
    # PERFIL SINTÉTICO POR CLÚSTER (AL FINAL)
    # ======================
    st.markdown('<div id="casos-tipo"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("Perfil sintético por clúster")

    if "cluster_label" not in df.columns:
        st.warning("No puedo construir el perfil sintético: falta `cluster_label`.")
    else:
        clusters_all = [c for c in ["C1", "C2", "C3"] if c in df["cluster_label"].dropna().unique()]
        clusters_rest = sorted([c for c in df["cluster_label"].dropna().unique() if c not in clusters_all])
        clusters_order = clusters_all + clusters_rest

        counts = (
            df["cluster_label"]
            .value_counts(dropna=True)
            .reindex(clusters_order)
            .fillna(0)
            .astype(int)
        )
        counts_df = pd.DataFrame({"Clúster": counts.index, "Nº casos": counts.values})
        st.dataframe(counts_df, use_container_width=True, hide_index=True)

        def _median_by_cluster(df_in: pd.DataFrame, var: str) -> pd.Series:
            s = pd.to_numeric(df_in[var], errors="coerce")
            s_disp = to_display_scale(var, s)
            tmp = df_in[["cluster_label"]].copy()
            tmp["_val"] = s_disp
            med = tmp.groupby("cluster_label")["_val"].median()
            return med.reindex(clusters_order)

        rows_profile = []
        rows_medians = []

        for var in VARS_CLUSTER:
            if var not in df.columns:
                continue

            med = _median_by_cluster(df, var)
            valid = med.dropna()
            if len(valid) < 2:
                continue

            row_prof = {"Indicador": LABELS.get(var, var)}

            if len(valid) == 3:
                order = valid.sort_values()
                low, mid, high = order.index[0], order.index[1], order.index[2]
                for cl in clusters_order:
                    if cl == high:
                        row_prof[cl] = "↑"
                    elif cl == low:
                        row_prof[cl] = "↓"
                    elif cl == mid:
                        row_prof[cl] = "~~"
                    else:
                        row_prof[cl] = ""
            else:
                order = valid.sort_values()
                low = order.index[0]
                high = order.index[-1]
                for cl in clusters_order:
                    if cl == high:
                        row_prof[cl] = "↑"
                    elif cl == low:
                        row_prof[cl] = "↓"
                    elif cl in valid.index:
                        row_prof[cl] = "~~"
                    else:
                        row_prof[cl] = ""

            rows_profile.append(row_prof)

            row_m = {"Indicador": LABELS.get(var, var)}
            for cl in clusters_order:
                v = med.get(cl, np.nan)
                row_m[cl] = np.nan if pd.isna(v) else float(v)
            rows_medians.append(row_m)

        profile_df = pd.DataFrame(rows_profile)
        if profile_df.empty:
            st.info("No hay suficientes datos para construir el perfil sintético.")
        else:
            cols_show = ["Indicador"] + clusters_order
            profile_df = profile_df.reindex(columns=[c for c in cols_show if c in profile_df.columns])

            st.caption("↑ = valor más alto (mediana), ~~ = valor intermedio, ↓ = valor más bajo.")
            st.dataframe(profile_df, use_container_width=True, hide_index=True)

            with st.expander("Ver medianas (para validación)", expanded=False):
                med_df = pd.DataFrame(rows_medians)
                med_df = med_df.reindex(columns=[c for c in cols_show if c in med_df.columns])
                for c in clusters_order:
                    if c in med_df.columns:
                        med_df[c] = pd.to_numeric(med_df[c], errors="coerce").apply(
                            lambda x: "" if pd.isna(x) else fmt_num(x)
                        )
                st.dataframe(med_df, use_container_width=True, hide_index=True)

    # ======================
    # TOP EMPRESAS (por ingresos)
    # ======================
    st.markdown('<div id="top-empresas"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("Top empresas (global o por localidad)")

    df_top = _ensure_ingresos(df_view=df, base_path=base_path, diagnostic=False)

    if "ingresos_de_explotacion" not in df_top.columns:
        st.warning("No puedo calcular el Top: falta ingresos_de_explotacion.")
        return

    df_top["ingresos_rank"] = pd.to_numeric(df_top["ingresos_de_explotacion"], errors="coerce")

    clusters = sorted(df_top["cluster_label"].dropna().unique()) if "cluster_label" in df_top.columns else []
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

    if not stats_df.empty:
        export_df = stats_df.copy()
        export_df["Empresa"] = row.get("nombre", "—")
        export_df["Cluster"] = cluster_sel
        export_df = export_df[["Empresa", "Cluster"] + [c for c in export_df.columns if c not in ["Empresa", "Cluster"]]]

        buf = io.StringIO()
        export_df.to_csv(buf, index=False)
        st.download_button(
            label="⬇️ Descargar ficha de empresa (CSV)",
            data=buf.getvalue().encode("utf-8"),
            file_name=f"ficha_empresa_{cluster_sel}_{row.get('nombre', 'empresa')}.csv",
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
