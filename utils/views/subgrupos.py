# utils/views/subgrupos.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import kruskal

from utils.config import VARS_CLUSTER, LABELS
from utils.fmt import fmt_p, sig_stars
from utils.stats import eps2_kw, magnitude_eps2
from utils.subgroups import add_subclusters_within_general, build_subgroup_column


def _label(v: str) -> str:
    return LABELS.get(v, v)


def _mean_sd(s: pd.Series) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return ""
    return f"{x.mean():.3f} ({x.std(ddof=1):.3f})"


def _profile_table(df: pd.DataFrame, group_col: str, vars_num: list[str]) -> pd.DataFrame:
    df2 = df.dropna(subset=[group_col]).copy()
    groups = sorted(df2[group_col].unique())
    rows = []
    for v in vars_num:
        row = {"Indicador": _label(v)}
        for g in groups:
            row[str(g)] = _mean_sd(df2.loc[df2[group_col] == g, v])
        rows.append(row)
    # N
    n_row = {"Indicador": "N"}
    for g in groups:
        n_row[str(g)] = int((df2[group_col] == g).sum())
    return pd.concat([pd.DataFrame(rows), pd.DataFrame([n_row])], ignore_index=True)


def _kw_table(df: pd.DataFrame, group_col: str, vars_num: list[str]) -> pd.DataFrame:
    df2 = df.dropna(subset=[group_col]).copy()
    groups = sorted(df2[group_col].unique())
    k = len(groups)
    n = len(df2)

    out = []
    for v in vars_num:
        data = [
            pd.to_numeric(df2.loc[df2[group_col] == g, v], errors="coerce").dropna().values
            for g in groups
        ]
        if any(len(x) == 0 for x in data) or k < 2:
            out.append(
                {
                    "Indicador": _label(v),
                    "H (K-W)": np.nan,
                    "p": np.nan,
                    "Sig.": "",
                    "ε²": np.nan,
                    "Magnitud": "",
                }
            )
            continue
        H, p = kruskal(*data)
        e = eps2_kw(H, n=n, k=k)
        out.append(
            {
                "Indicador": _label(v),
                "H (K-W)": H,
                "p": p,
                "Sig.": sig_stars(p),
                "ε²": e,
                "Magnitud": magnitude_eps2(e),
            }
        )
    df_out = pd.DataFrame(out)
    df_out["p"] = df_out["p"].apply(fmt_p)
    return df_out


def render_subgrupos(
    df: pd.DataFrame,
    base_path: str,
    cluster_general: str,
    k: int = 3,
    comparar_con: str = "cluster",
    zoom: bool = False,
):
    """
    Vista: Subgrupos dentro de un cluster general (C1/C2/C3)

    OBJETIVO (como pides):
    - Mantener EXACTAMENTE la misma estructura del modelo general:
        * Resumen
        * Estadística del modelo
        * Árbol de decisión
      pero aplicado a subgrupos (S1/S2/S3) dentro del cluster_general.

    IMPLEMENTACIÓN:
    - Calcula subclustering k dentro del cluster indicado
    - Filtra al cluster_general y crea columna común 'subcluster'
    - Construye df_subapp (idéntico a df) pero con cluster_label = subcluster
      para poder reutilizar tus vistas existentes sin reescribirlas.
    """
    st.header(f"Subgrupos dentro de {cluster_general} (k={k})")

    # 1) Añade subclusters (cacheado en utils.subgroups)
    df_sc = add_subclusters_within_general(
        df,
        general_col="cluster_label",
        vars_cluster=VARS_CLUSTER,
        k=k
    )

    # 2) Filtra al cluster y crea columna común 'subcluster'
    df_c = build_subgroup_column(
        df_sc,
        general_cluster_value=cluster_general,
        general_col="cluster_label"
    )
    df_c = df_c.dropna(subset=["subcluster"]).copy()
    if len(df_c) == 0:
        st.warning("No hay casos suficientes o falta subcluster para este cluster.")
        return

    # 3) DF "para la app" (engancha Resumen/Estadística/Árbol sin tocar sus códigos)
    df_subapp = df_c.copy()

    # Importante: cluster_label pasa a ser el subgrupo (S1/S2/S3)
    # Mantengo tipo string para que se pinte bien y sea estable.
    df_subapp["cluster_label"] = df_subapp["subcluster"].astype(str)

    # ------------------
    # Tabs: igual que modelo general
    # ------------------
    tab_resumen, tab_estad, tab_arbol = st.tabs(
        ["Resumen", "Estadística del modelo", "Árbol de decisión"]
    )

    # -------- Resumen (idéntico al general) --------
    with tab_resumen:
        # Reutiliza tu vista del resumen general para que la gráfica sea EXACTAMENTE igual
        from utils.views.resumen import render_resumen

        from utils.views.resumen import render_resumen

        # Textos para subgrupos (C1) -> claves "1.0", "2.0", "3.0"
        SUBGROUP_STORY = {
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
                    "Mayor rigidez operativa (costes fijos/activos) si cae la demanda.",
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
                    "Menos estructura y menor capacidad de convertir en valor.",
                    "Subgrupo potencialmente más vulnerable dentro de C1.",
                ],
                "implicaciones": [
                    "Prioridad: elevar productividad (procesos/organización/mix) y eficiencia.",
                    "Revisar disciplina de circulante y costes.",
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
                    "Compite más por agilidad/ejecución que por intensidad de capital.",
                    "Puede escalar si convierte rotación en margen y eficiencia.",
                ],
                "implicaciones": [
                    "Mejorar margen/eficiencia sin perder rotación.",
                    "Vigilar sensibilidad a caídas de volumen.",
                ],
            },
        }

        render_resumen(
            df=df_subapp,
            comparar_con=comparar_con,
            zoom=zoom,
            base_path=base_path,
            normalize_cluster_labels=False,           # 👈 CLAVE: NO convertir 1.0 -> C1
            story_map_override=SUBGROUP_STORY,        # 👈 CLAVE: textos por subgrupo
            story_title="Interpretación del subgrupo",
            show_subgroup_interpretation=False,       # 👈 para que NO salga el bloque extra
        )

        # (Opcional) Si quieres además dejar estas tablas del subgrupo debajo
        # sin romper la estética, lo dejo plegado:
        with st.expander("Ver perfil y contrastes de subgrupos (opcional)", expanded=False):
            st.subheader("Perfil por subgrupo (media (sd) + N)")
            prof = _profile_table(df_c, group_col="subcluster", vars_num=VARS_CLUSTER)
            st.dataframe(prof, use_container_width=True)

            st.subheader("Contrastes entre subgrupos (Kruskal–Wallis + ε²)")
            kw = _kw_table(df_c, group_col="subcluster", vars_num=VARS_CLUSTER)
            st.dataframe(kw, use_container_width=True)

    # -------- Estadística del modelo (reutilizada) --------
    with tab_estad:
        from utils.views.estadistica import render_estadistica

        # Si tu render_estadistica ya admite group_col, perfecto.
        # Si no, seguirá usando cluster_label por defecto (que ya es subgrupo).
        try:
            render_estadistica(
                df=df_subapp,
                base_path=base_path,
                group_col="cluster_label",
                title=None,
            )
        except TypeError:
            render_estadistica(
                df=df_subapp,
                base_path=base_path,
            )

    # -------- Árbol de decisión (reutilizado) --------
    with tab_arbol:
        from utils.views.arbol_decision import render_arbol_decision

        # Igual: usamos df_subapp con cluster_label=subgrupo
        # Nota: tu arbol_decision.py a veces recibe df_app=...
        try:
            render_arbol_decision(df_app=df_subapp, base_path=base_path)
        except TypeError:
            render_arbol_decision(df=df_subapp, base_path=base_path)

    # ------------------
    # Descarga
    # ------------------
    st.divider()
    st.subheader("Descargas")
    st.download_button(
        "Descargar datos (solo este cluster + subgrupos)",
        data=df_c.to_csv(index=False).encode("utf-8"),
        file_name=f"subgrupos_{cluster_general}.csv",
        mime="text/csv",
    )