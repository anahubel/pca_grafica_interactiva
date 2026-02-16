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
def render_resumen(df: pd.DataFrame, comparar_con: str, zoom: bool, base_path: str | None = None):
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
    _cluster_map = {"1": "C1", "1.0": "C1", "2": "C2", "2.0": "C2", "3": "C3", "3.0": "C3"}
    cluster_sel = _cluster_map.get(str(_raw_cluster).strip(), str(_raw_cluster).strip())

    # Referencia (cluster o total)
    if comparar_con == "Solo su cluster" and "cluster_label" in df.columns and "cluster_label" in row.index:
        # ojo: filtramos por el valor real que haya en df (puede ser "C1" o "1.0")
        df_ref = df[df["cluster_label"] == _raw_cluster].copy()
        # fallback por si df está ya normalizado en texto
        if df_ref.empty and cluster_sel in {"C1", "C2", "C3"}:
            df_ref = df[df["cluster_label"] == cluster_sel].copy()
    else:
        df_ref = df.copy()

    # ======================
    # INTERPRETACIÓN CLÚSTER + SUBGRUPO (si existe) + IMPLICACIONES
    # ======================
    CLUSTER_N = df["cluster_label"].value_counts(dropna=True).to_dict() if "cluster_label" in df.columns else {}
    n_sel = int(CLUSTER_N.get(_raw_cluster, CLUSTER_N.get(cluster_sel, 0)))
    pct_sel = (n_sel / max(1, len(df))) * 100.0

    # ---- detectar columna de subgrupo (si existe)
    SUBGROUP_COL_CANDIDATES = [
        "subgrupo", "subgrupo_label", "subgrupo_cluster",
        "cluster_subgrupo", "cluster_subgrupo_label",
        "subcluster", "subcluster_label",
        "cluster_k3", "cluster_k3_label",
        "cluster_modelo_negocio",
    ]
    subgroup_col = next((c for c in SUBGROUP_COL_CANDIDATES if c in df.columns), None)
    subgroup_sel = None
    if subgroup_col is not None:
        v = row.get(subgroup_col, None)
        if pd.notna(v):
            subgroup_sel = str(v).strip()  # suele venir como "1.0", "2.0", "3.0"

    CLUSTER_STORY = {
        "C1": {
            "titulo": "C1: modelo lento, exigente en circulante y poco rentable",
            "bullets": [
                "Presenta la rotación de stocks más baja y el nivel de competitividad más reducido, lo que sugiere un ciclo operativo más lento.",
                "Es el grupo con mayor NOFS/Ventas, indicando una mayor necesidad de financiar el circulante (mayor dependencia de recursos para sostener la operativa).",
                "Muestra las productividades más bajas (VA/empleado y ventas/empleado) y también los peores resultados (resultado del ejercicio, explotación, EBITDA y cash flow).",
                "El porcentaje de personal es el más alto del conjunto, apuntando a una estructura más intensiva en trabajo y menos eficiente.",
            ],
            "implicaciones": [
                "Empresas con menor dinamismo comercial/operativo y con más presión financiera por el circulante.",
                "El foco de mejora suele estar en acelerar rotación, optimizar gestión de existencias/cobros/pagos y elevar productividad.",
                "Modelo más vulnerable: si no mejora eficiencia, cualquier shock de demanda o de costes se traslada rápido a resultados.",
            ],
        },
        "C2": {
            "titulo": "C2: modelo líder en productividad y generación de resultados",
            "bullets": [
                "Es el grupo con mayor productividad (VA/empleado y ventas/empleado), muy por encima de C1 y C3.",
                "Registra los niveles más altos de resultado del ejercicio, resultado de explotación, EBITDA y cash flow, mostrando fuerte capacidad de generar valor.",
                "Tiene el porcentaje de personal más bajo, lo que encaja con un modelo más eficiente/capital-intensivo y escalable.",
                "Mantiene NOFS/Ventas en un nivel intermedio: estructura de circulante más equilibrada que C1, sin llegar al nivel “ligero” de C3.",
            ],
            "implicaciones": [
                "Empresas más sólidas y eficientes: suelen sostener ventajas competitivas por escala, procesos y productividad.",
                "Modelo “ganador”: combina productividad + resultados + eficiencia de costes laborales.",
                "El reto típico no es sobrevivir, sino mantener la ventaja: innovación, inversión selectiva y control de complejidad al crecer.",
            ],
        },
        "C3": {
            "titulo": "C3: modelo dinámico y competitivo, ligero en circulante",
            "bullets": [
                "Es el grupo con mayor rotación de stocks y mayor competitividad, reflejando un ciclo operativo más rápido y presión comercial elevada.",
                "Presenta el NOFS/Ventas más bajo, lo que sugiere una necesidad menor de financiar el circulante (modelo más “ligero” y ágil).",
                "Sus productividades son intermedias: claramente por debajo de C2, pero en general por encima de C1 (especialmente en ventas/empleado).",
                "Los resultados (ejercicio, explotación, EBITDA y cash flow) son moderados: mejores que C1 pero lejos del nivel de C2. El personal (%) es intermedio-alto.",
            ],
            "implicaciones": [
                "Empresas operativamente ágiles: compiten por velocidad, rotación y ejecución.",
                "El crecimiento suele venir de optimizar margen/eficiencia y profesionalizar procesos para acercarse al desempeño de C2.",
                "Modelo con buena flexibilidad financiera (bajo NOFS/Ventas), pero sensible a caídas de demanda porque necesita mantener volumen/rotación.",
            ],
        },
    }

    # ======================
    # SUBGRUPOS (k=3) — textos basados en tus perfiles sintéticos (medianas)
    # OJO: aquí asumo subgrupo "1.0/2.0/3.0" dentro de cada C1/C2/C3
    # ======================
    SUBGROUP_STORY = {
        "C1": {
            "1.0": {
                "titulo": "Subgrupo 1 (C1.1): más intensivo en VA y activos, pero con menor eficiencia comercial",
                "bullets": [
                    "Productividad VA/pax e inmovilizado/empleado altos: perfil más intensivo en estructura/capacidad.",
                    "Rotación de stocks y NOFS/Ventas en zona intermedia: no es el más ágil, pero tampoco el más tensionado dentro de C1.",
                    "Ventas/pax y margen más bajos del clúster: convierte peor la estructura en rendimiento comercial.",
                ],
                "implicaciones": [
                    "Prioridad: mejorar margen y ejecución comercial (pricing/mix/eficiencia) para “monetizar” la estructura productiva.",
                    "Optimizar rotación y procesos para reducir rigidez y mejorar caja.",
                ],
            },
            "2.0": {
                "titulo": "Subgrupo 2 (C1.2): el C1 más ágil y mejor ejecutor",
                "bullets": [
                    "Mayor rotación de stocks, mayor ventas/pax y mayor margen: es el subgrupo con mejor desempeño operativo-comercial dentro de C1.",
                    "NOFS/Ventas e inmovilizado/empleado bajos: estructura más ligera y menos exigente en financiación del circulante.",
                    "Productividad VA/pax intermedia: mejora sobre todo por rotación y margen.",
                ],
                "implicaciones": [
                    "Candidato natural a converger hacia un modelo más eficiente si consolida productividad y profesionaliza procesos.",
                    "Mantener disciplina de circulante (cobros/pagos/stock) para sostener la agilidad.",
                ],
            },
            "3.0": {
                "titulo": "Subgrupo 3 (C1.3): el más tensionado en circulante y el menos productivo",
                "bullets": [
                    "Rotación de stocks baja y NOFS/Ventas alta: ciclo más lento y mayor necesidad de financiación del circulante.",
                    "Productividad VA/pax más baja del clúster; desempeño comercial intermedio.",
                    "Inmovilizado/empleado intermedio: no compensa la rigidez con productividad.",
                ],
                "implicaciones": [
                    "Prioridad: gestión del circulante y rotación (stock, cobros, plazos) para reducir tensión financiera.",
                    "Plan de eficiencia/productividad (procesos, control de costes/tiempos) para estabilizar resultados.",
                ],
            },
        },

        "C2": {
            "1.0": {
                "titulo": "Subgrupo 1 (C2.1): muy rentable y comercial, pero más exigente en circulante",
                "bullets": [
                    "Ventas/pax y margen más altos del clúster: foco en rendimiento comercial y rentabilidad.",
                    "NOFS/Ventas alto y rotación de stocks más baja dentro de C2: modelo menos “ligero” en circulante.",
                    "Productividad VA/pax e inmovilizado/empleado intermedios: no es el más capital-intensivo de C2.",
                ],
                "implicaciones": [
                    "Mejora clara en rotación y circulante: liberar caja sin perder margen.",
                    "Revisar inventarios y condiciones con clientes/proveedores para sostener escalabilidad.",
                ],
            },
            "2.0": {
                "titulo": "Subgrupo 2 (C2.2): capital-intensivo y muy eficiente (perfil industrial/tecnológico)",
                "bullets": [
                    "Mayor rotación de stocks y mayor productividad VA/pax: eficiencia operativa con alta generación de valor.",
                    "Inmovilizado/empleado alto: inversión/capacidad productiva relevante.",
                    "NOFS/Ventas bajo: circulante muy bien controlado pese a la intensidad de activos.",
                ],
                "implicaciones": [
                    "Modelo robusto: sostener ventaja con inversión selectiva y excelencia operativa.",
                    "Evitar complejidad improductiva y cuidar mantenimiento/renovación de activos.",
                ],
            },
            "3.0": {
                "titulo": "Subgrupo 3 (C2.3): el C2 más templado (menor diferencial)",
                "bullets": [
                    "Ventas/pax, margen, VA/pax e inmovilizado/empleado más bajos dentro de C2: pierde parte del diferencial del clúster líder.",
                    "Rotación de stocks y NOFS/Ventas intermedios: sin extremos claros.",
                    "Sigue siendo un modelo sólido, pero con menor intensidad de ventaja competitiva.",
                ],
                "implicaciones": [
                    "Oportunidad: converger a C2.2 (eficiencia/VA) o C2.1 (margen/ventas) según palancas internas.",
                    "Revisar procesos y estructura de costes para recuperar diferencial.",
                ],
            },
        },

        "C3": {
            "1.0": {
                "titulo": "Subgrupo 1 (C3.1): el C3 más “premium” (margen y productividad altos)",
                "bullets": [
                    "Ventas/pax, margen y VA/pax altos: mejor combinación de productividad y rentabilidad dentro de C3.",
                    "Inmovilizado/empleado alto: cierta estructura/capacidad que se está aprovechando bien.",
                    "Rotación y NOFS/Ventas intermedios: equilibrio entre agilidad y control.",
                ],
                "implicaciones": [
                    "Buen punto de partida para crecer: profesionalizar y escalar sin perder margen.",
                    "Mantener disciplina operativa para que el aumento de estructura no reduzca agilidad.",
                ],
            },
            "2.0": {
                "titulo": "Subgrupo 2 (C3.2): el más ágil (máxima rotación y circulante ligero), con margen más ajustado",
                "bullets": [
                    "Rotación de stocks más alta y NOFS/Ventas más bajo: máxima agilidad y menor necesidad de financiar el circulante.",
                    "Margen más bajo del clúster: compite por velocidad/volumen más que por rentabilidad unitaria.",
                    "Estructura ligera (inmovilizado/empleado bajo) con productividades intermedias.",
                ],
                "implicaciones": [
                    "Palanca clave: mejorar margen (pricing/mix/eficiencia) manteniendo la rotación.",
                    "Vigilar sensibilidad a caídas de demanda: necesita volumen para sostener resultados.",
                ],
            },
            "3.0": {
                "titulo": "Subgrupo 3 (C3.3): pequeño y más tensionado (baja rotación, alto NOFS/Ventas)",
                "bullets": [
                    "Rotación de stocks baja y NOFS/Ventas alto: pierde parte de la agilidad típica de C3 y requiere más financiación del circulante.",
                    "Ventas/pax y VA/pax bajos: menor productividad.",
                    "Margen intermedio e inmovilizado/empleado intermedio: no compensa con rentabilidad.",
                ],
                "implicaciones": [
                    "Prioridad: recuperar agilidad (rotación/circulante) y elevar productividad para estabilizar el modelo.",
                    "Al ser un subgrupo pequeño, revisar posibles casos atípicos o condicionantes sectoriales.",
                ],
            },
        },
    }

    # story principal (cluster)
    story = CLUSTER_STORY.get(
        str(cluster_sel),
        {
            "titulo": "Interpretación no definida",
            "bullets": ["Define aquí el texto del clúster para tu memoria/defensa."],
            "implicaciones": ["Añade recomendaciones específicas por clúster."],
        },
    )

    # story del subgrupo (si existe)
    sub_story = None
    if subgroup_sel is not None and str(cluster_sel) in SUBGROUP_STORY:
        sub_story = SUBGROUP_STORY[str(cluster_sel)].get(str(subgroup_sel), None)

    # ======================
    # (a partir de aquí, tu render sigue igual)
    # Solo recuerda que ahora tienes:
    #   - cluster_sel (normalizado a C1/C2/C3 si venía 1.0/2.0/3.0)
    #   - subgroup_col (nombre columna subgrupo o None)
    #   - subgroup_sel (valor subgrupo: "1.0"/"2.0"/"3.0" o None)
    #   - story (texto cluster)
    #   - sub_story (texto subgrupo o None)
    # ======================

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
            fig = px.scatter(
                df,
                x="PC1",
                y="PC2",
                color="cluster_label" if "cluster_label" in df.columns else None,
                hover_name="nombre" if "nombre" in df.columns else None,
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