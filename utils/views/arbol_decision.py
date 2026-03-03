# utils/views/arbol_decision.py
from __future__ import annotations

import io
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from utils.config import EXCLUDE_VARS, LABELS, VARS_CLUSTER, LOGGED_IN_MODEL

# ============================================================
# Labels helpers
# ============================================================
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


# ============================================================
# Graphviz DOT builder (estilo rpart)
# ============================================================
def _rgba(hex_color: str, alpha: float = 0.25) -> str:
    """Convierte #RRGGBB a #RRGGBBAA (Graphviz soporta alpha en hex largo)."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    a = int(max(0, min(1, alpha)) * 255)
    return f"#{h}{a:02x}"


def build_rpart_like_dot(clf: DecisionTreeClassifier, feature_names: list[str], class_names: list[str]) -> str:
    tree = clf.tree_
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value.squeeze(axis=1)  # (nodes, n_classes)
    samples = tree.n_node_samples

    total = float(samples[0]) if samples[0] > 0 else 1.0

    # Paleta coherente con tu app
    class_color = {"C1": "#f0a44c", "C2": "#9aa0a6", "C3": "#6ea8fe"}
    default_colors = ["#f0a44c", "#9aa0a6", "#6ea8fe", "#6bc98a", "#d58cff"]

    def node_fill(pred_class: str, pmax: float):
        base = class_color.get(pred_class, default_colors[class_names.index(pred_class) % len(default_colors)])
        alpha = 0.18 + 0.55 * float(pmax)  # 0.18..0.73
        return _rgba(base, alpha=alpha)

    def fmt_probs(probs: np.ndarray) -> str:
        pct = np.round(100 * probs).astype(int)
        return " ".join([f"{x:02d}" for x in pct])

    dot = []
    dot.append("digraph Tree {")
    dot.append('  graph [rankdir=TB, nodesep=0.20, ranksep=0.30];')
    dot.append('  node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, penwidth=1.2];')
    dot.append('  edge  [fontname="Helvetica", fontsize=11, color="#666666"];')

    for i in range(n_nodes):
        s = float(samples[i]) if samples[i] else 0.0
        vraw = np.asarray(value[i], dtype=float)
        vsum = float(np.nansum(vraw))
        if vsum > 0:
            probs = vraw / vsum
            pred_idx = int(np.nanargmax(vraw))
        else:
            probs = np.zeros_like(vraw, dtype=float)
            pred_idx = 0

        pred_class = class_names[pred_idx]
        pmax = float(np.nanmax(probs)) if probs.size > 0 else 0.0

        pct_node = 100.0 * s / total if total > 0 else 0.0
        probs_str = fmt_probs(probs)

        if children_left[i] != children_right[i]:
            f = feature_names[feature[i]]
            thr = threshold[i]
            split_txt = f"{_label_of(f)} ≥ {thr:.3g}"
            split_html = f'<TR><TD><FONT POINT-SIZE="11">{split_txt}</FONT></TD></TR>'
        else:
            split_html = ""

        label = (
            '<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2">'
            f'<TR><TD><B>{pred_class}</B></TD></TR>'
            f'<TR><TD>{probs_str}</TD></TR>'
            f'<TR><TD><FONT POINT-SIZE="11">{pct_node:.0f}%</FONT></TD></TR>'
            f"{split_html}"
            "</TABLE>>"
        )

        fill = node_fill(pred_class, pmax)
        dot.append(f'  {i} [label={label}, fillcolor="{fill}"];')

    for i in range(n_nodes):
        if children_left[i] == children_right[i]:
            continue
        left = children_left[i]
        right = children_right[i]
        dot.append(f'  {i} -> {left} [label="no"];')
        dot.append(f'  {i} -> {right} [label="yes"];')

    dot.append("}")
    return "\n".join(dot)


# ============================================================
# Sanitizado robusto
# ============================================================
def _sanitize_X(X: pd.DataFrame, cap_q: float = 0.995, hard_cap: float = 1e12) -> pd.DataFrame:
    """
    - Convierte a numérico
    - Reemplaza inf/-inf por NaN
    - Clip por percentil (cap_q)
    - Clip por hard_cap (valor absoluto)
    - Imputa mediana
    - Fuerza float64
    """
    X = X.copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    caps = X.quantile(cap_q, numeric_only=True)
    for c in X.columns:
        if c in caps.index and pd.notna(caps[c]):
            X[c] = X[c].clip(upper=float(caps[c]))

    X = X.clip(lower=-hard_cap, upper=hard_cap)
    X = X.fillna(X.median(numeric_only=True))
    X = X.astype(np.float64)

    if not np.isfinite(X.to_numpy()).all():
        bad_cols = [col for col in X.columns if not np.isfinite(X[col].to_numpy()).all()]
        raise ValueError(f"Siguen quedando valores no finitos en columnas: {bad_cols}")

    return X


# ============================================================
# Reglas de nodos
# ============================================================
def _build_parent_map(clf: DecisionTreeClassifier):
    tree = clf.tree_
    parent = {0: -1}
    is_left = {}
    for i in range(tree.node_count):
        l = tree.children_left[i]
        r = tree.children_right[i]
        if l != r:
            parent[l] = i
            parent[r] = i
            is_left[l] = True
            is_left[r] = False
    return parent, is_left


def _node_rule_text(clf: DecisionTreeClassifier, node_id: int, feature_names: list[str]) -> str:
    tree = clf.tree_
    parent, is_left = _build_parent_map(clf)

    parts = []
    cur = node_id
    while cur != 0 and cur in parent and parent[cur] != -1:
        p = parent[cur]
        f_idx = tree.feature[p]
        thr = tree.threshold[p]
        f = feature_names[f_idx]

        if is_left.get(cur, True):
            parts.append(f"{_label_of(f)} < {thr:.3g}")
        else:
            parts.append(f"{_label_of(f)} ≥ {thr:.3g}")

        cur = p

    parts.reverse()
    return " AND ".join(parts) if parts else "(raíz)"


def _node_pred_and_probs(clf: DecisionTreeClassifier, node_id: int, class_names: list[str]):
    tree = clf.tree_
    v = tree.value[node_id].squeeze()
    tot = float(v.sum()) if float(v.sum()) > 0 else 0.0
    probs = (v / tot) if tot > 0 else np.zeros_like(v, dtype=float)
    pred = class_names[int(np.argmax(v))] if tot > 0 else class_names[0]
    pct = np.round(100 * probs).astype(int)
    return pred, probs, pct, int(tot)


# ============================================================
# Cache de entrenamiento (evita recalcular todo)
# ============================================================
@st.cache_resource(show_spinner=False)
def _fit_tree_cached(
    df_model: pd.DataFrame,
    X_vars: tuple[str, ...],
    max_depth: int,
    min_leaf: int,
    min_split: int,
    criterion: str,
    ccp_alpha: float,
    cap_q: float,
    hard_cap: float,
    class_weight: str | None,
):
    # y
    y = df_model["cluster_label"].astype(str)
    pref = ["C1", "C2", "C3"]
    class_names = [c for c in pref if c in y.unique()]
    class_names += [c for c in sorted(y.unique()) if c not in class_names]

    # X
    X = df_model[list(X_vars)].copy()

    # saneo -> deslog -> saneo
    X = _sanitize_X(X, cap_q=float(cap_q), hard_cap=float(hard_cap))
    for c in X_vars:
        if c in LOGGED_IN_MODEL:
            X[c] = np.expm1(X[c])
    X = _sanitize_X(X, cap_q=float(cap_q), hard_cap=float(hard_cap))

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        min_samples_split=min_split,
        criterion=criterion,
        ccp_alpha=float(ccp_alpha),
        class_weight=class_weight,
        random_state=42,
    )
    clf.fit(X, y)

    y_pred = clf.predict(X)

    dot = build_rpart_like_dot(clf=clf, feature_names=list(X_vars), class_names=class_names)

    return clf, X, y, y_pred, class_names, dot


# ============================================================
# MAIN VIEW
# ============================================================
def render_arbol_decision(df_app: pd.DataFrame, base_path: str):
    st.header("Árbol de decisión")

    # ============================================================
    # 1) Dataset: en tu caso estás entrenando con df_app (OK).
    #    (Si quieres, luego lo cambiamos para usar df_base_full de session_state)
    # ============================================================
    df_full = df_app.copy()
    try:
        from utils.recodes import apply_recodes
        df_full = apply_recodes(df_full)
    except Exception:
        pass

    if "cluster_label" not in df_full.columns:
        st.error("No existe `cluster_label` en el dataframe.")
        st.stop()

    df_full = df_full[df_full["cluster_label"].notna()].copy()

    forbidden = {"codigo_nif", "nombre", "cluster_label", "PC1", "PC2", "empresa_key"} | set(EXCLUDE_VARS)

    # detecta numéricas
    num_cols = []
    for c in df_full.columns:
        if c in forbidden:
            continue
        if pd.api.types.is_numeric_dtype(df_full[c]):
            num_cols.append(c)
    num_cols = sorted(num_cols)

    if not num_cols:
        st.error("No encuentro variables numéricas para entrenar el árbol.")
        st.stop()

    num_labels, num_lab_to_var = _make_label_maps(num_cols)

    # ============================================================
    # 2) Controles (arriba) + layout en tabs
    # ============================================================
    with st.container():
        st.subheader("Configuración")

        c1, c2, c3 = st.columns([2.2, 1.2, 1.2], gap="large")

        with c1:
            default_vars = [v for v in VARS_CLUSTER if v in num_cols]
            default_labels = []
            for v in default_vars:
                lab = _label_of(v)
                cand2 = f"{lab} [{v}]"
                if cand2 in num_labels:
                    default_labels.append(cand2)
                elif lab in num_labels:
                    default_labels.append(lab)

            vars_sel_labels = st.multiselect(
                "Variables predictoras",
                options=num_labels,
                default=default_labels if default_labels else num_labels[:8],
                key="tree_vars",
            )

        with c2:
            max_depth = st.slider("max_depth", 1, 10, 4, 1, key="tree_max_depth")
            min_leaf = st.slider("min_samples_leaf", 1, 50, 10, 1, key="tree_min_leaf")
            min_split = st.slider("min_samples_split", 2, 100, 20, 1, key="tree_min_split")

        with c3:
            criterion = st.selectbox("criterion", ["gini", "entropy", "log_loss"], index=0, key="tree_criterion")
            ccp_alpha = st.slider("ccp_alpha (poda)", 0.0, 0.05, 0.0, 0.001, key="tree_ccp")
            class_weight = st.selectbox(
                "class_weight",
                options=["None", "balanced"],
                index=0,
                key="tree_class_weight",
            )
            class_weight = None if class_weight == "None" else "balanced"

        with st.expander("Limpieza numérica (saneado)", expanded=False):
            cap_q = st.slider("Clip percentil", 0.90, 0.999, 0.995, 0.001, key="tree_cap_q")
            hard_cap = st.number_input(
                "Clip absoluto (|x|)",
                min_value=1e6,
                max_value=1e15,
                value=1e12,
                step=1e6,
                format="%.0f",
                key="tree_hard_cap",
            )

    if not vars_sel_labels:
        st.info("Selecciona al menos una variable predictora.")
        st.stop()

    X_vars = tuple(num_lab_to_var[x] for x in vars_sel_labels)

    # ============================================================
    # 3) df_model
    # ============================================================
    keep_cols = ["cluster_label"] + list(X_vars)
    for extra in ["nombre", "codigo_nif"]:
        if extra in df_full.columns:
            keep_cols.append(extra)

    df_model = df_full[keep_cols].copy()
    for c in X_vars:
        df_model[c] = pd.to_numeric(df_model[c], errors="coerce")

    df_model = df_model.dropna(subset=["cluster_label"]).copy()
    if df_model.empty or df_model["cluster_label"].nunique() < 2:
        st.info("No hay datos suficientes (o solo hay un clúster).")
        st.stop()

    # ============================================================
    # 4) Entrenar (cacheado)
    # ============================================================
    try:
        clf, X, y, y_pred, class_names, dot = _fit_tree_cached(
            df_model=df_model,
            X_vars=X_vars,
            max_depth=int(max_depth),
            min_leaf=int(min_leaf),
            min_split=int(min_split),
            criterion=str(criterion),
            ccp_alpha=float(ccp_alpha),
            cap_q=float(cap_q),
            hard_cap=float(hard_cap),
            class_weight=class_weight,
        )
    except Exception as e:
        st.error(f"Error entrenando el árbol: {e}")
        st.stop()

    # métricas
    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro")

    # ============================================================
    # 5) Tabs (sin expanders anidados)
    # ============================================================
    tab_tree, tab_perf, tab_imp, tab_nodes, tab_dl = st.tabs(
        ["Árbol", "Calidad", "Importancias", "Empresas por nodo", "Descargas"]
    )

    # -------------------
    # TAB: Árbol
    # -------------------
    with tab_tree:
        st.subheader("Árbol (Graphviz estilo rpart)")
        st.graphviz_chart(dot, use_container_width=True)

        st.caption("Nota: este árbol se entrena con las variables seleccionadas y saneadas (clipping + imputación mediana).")

    # -------------------
    # TAB: Calidad
    # -------------------
    with tab_perf:
        st.subheader("Métricas")
        cA, cB, cC = st.columns(3)
        cA.metric("Accuracy", f"{acc:.3f}")
        cB.metric("Balanced accuracy", f"{bacc:.3f}")
        cC.metric("Macro F1", f"{f1m:.3f}")

        st.subheader("Matriz de confusión")

        cm = confusion_matrix(y, y_pred, labels=class_names)
        cm_df = pd.DataFrame(cm, index=[f"Real {c}" for c in class_names], columns=[f"Pred {c}" for c in class_names])
        st.markdown("**Recuentos**")
        st.dataframe(cm_df, use_container_width=True)

        # normalizada por fila
        cm_norm = cm / np.maximum(1, cm.sum(axis=1, keepdims=True))
        cmn_df = pd.DataFrame(
            cm_norm,
            index=[f"Real {c}" for c in class_names],
            columns=[f"Pred {c}" for c in class_names],
        )
        st.markdown("**Normalizada por fila (%)**")
        show = (cmn_df * 100.0).round(1).astype(str) + "%"
        st.dataframe(show, use_container_width=True)

        st.caption("Ojo: estas métricas están calculadas *sobre los mismos datos de entrenamiento* (sirve para explicación, no para validar generalización).")

    # -------------------
    # TAB: Importancias (Plotly, no st.bar_chart)
    # -------------------
    with tab_imp:
        st.subheader("Importancia de variables")
        import plotly.express as px
    
        imp = pd.DataFrame({"var": list(X_vars), "imp": clf.feature_importances_})
        imp["label"] = imp["var"].map(_label_of)
        imp = imp.sort_values("imp", ascending=False)
    
        max_k = int(min(30, len(imp)))
    
        if max_k == 0:
            st.info("No hay importancias que mostrar.")
            st.stop()
    
        if max_k <= 2:
            st.caption(f"Solo hay {max_k} variables: muestro todas (sin slider).")
            impk = imp.copy()
        else:
            # ✅ aquí nunca habrá min==max
            k = st.slider(
                "Top K",
                min_value=2,
                max_value=max_k,
                value=min(10, max_k),
                step=1,
                key="tree_imp_topk",
            )
            impk = imp.head(k)
    
        impk_plot = impk.iloc[::-1]  # para que el top quede arriba
        fig = px.bar(
            impk_plot,
            x="imp",
            y="label",
            orientation="h",
            labels={"imp": "Importancia", "label": ""},
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Importancia=%{x:.3f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------
    # TAB: Empresas por nodo
    # -------------------
    with tab_nodes:
        st.subheader("Empresas por nodo")

        # Path
        leaf_id = clf.apply(X)
        path = clf.decision_path(X)

        node_to_rows_leaf = defaultdict(list)
        for i, nid in enumerate(leaf_id):
            node_to_rows_leaf[int(nid)].append(i)

        node_to_rows_all = defaultdict(list)
        coo = path.tocoo()
        for r, c in zip(coo.row, coo.col):
            node_to_rows_all[int(c)].append(int(r))

        mode = st.radio(
            "Seleccionar",
            ["Hojas (grupos finales)", "Cualquier nodo (incluye internos)"],
            horizontal=True,
            key="tree_node_mode",
        )

        if mode == "Hojas (grupos finales)":
            candidate_nodes = sorted(node_to_rows_leaf.keys())
            map_rows = node_to_rows_leaf
        else:
            candidate_nodes = sorted(node_to_rows_all.keys())
            map_rows = node_to_rows_all

        if not candidate_nodes:
            st.info("No hay nodos candidatos.")
            st.stop()

        # selector compacto + filtro
        q = st.text_input("Filtrar nodos (por predicción o regla)", value="", key="tree_node_filter").strip().lower()

        node_rows = []
        node_meta = {}
        for nid in candidate_nodes:
            pred, probs, pct, ncases = _node_pred_and_probs(clf, nid, class_names)
            rule = _node_rule_text(clf, nid, list(X_vars))

            purity = int(np.max(pct)) if len(pct) else 0
            label = f"Nodo {nid} · Pred={pred} · pureza={purity}% · n={ncases}"
            node_meta[label] = {"nid": nid, "pred": pred, "rule": rule, "pct": pct, "ncases": ncases}

            if q:
                hay = (label + " " + rule).lower()
                if q not in hay:
                    continue
            node_rows.append(label)

        if not node_rows:
            st.info("No hay nodos que coincidan con el filtro.")
            st.stop()

        pick = st.selectbox("Nodo", options=node_rows, index=0, key="tree_node_pick")
        meta = node_meta[pick]
        nid_sel = meta["nid"]

        st.caption(f"**Regla del nodo:** {meta['rule']}")
        st.caption(
            "**% por clase:** " + " / ".join([f"{c}={p:02d}%" for c, p in zip(class_names, meta["pct"])])
        )

        rows = map_rows.get(nid_sel, [])
        if not rows:
            st.info("No hay empresas en este nodo.")
            st.stop()

        df_list = df_model.iloc[rows].copy()

        show_cols = []
        if "nombre" in df_list.columns:
            show_cols.append("nombre")
        if "codigo_nif" in df_list.columns:
            show_cols.append("codigo_nif")
        show_cols.append("cluster_label")

        st.markdown(f"### Empresas en este nodo: {len(df_list)}")
        st.dataframe(df_list[show_cols], use_container_width=True)

        buf = io.StringIO()
        df_list[show_cols].to_csv(buf, index=False)
        st.download_button(
            "Descargar empresas de este nodo (CSV)",
            data=buf.getvalue().encode("utf-8"),
            file_name=f"empresas_nodo_{nid_sel}.csv",
            mime="text/csv",
            key=f"dl_node_{nid_sel}",
        )

    # -------------------
    # TAB: Descargas
    # -------------------
    with tab_dl:
        st.subheader("Descargas")

        # DOT
        st.download_button(
            "⬇️ Descargar árbol (DOT)",
            data=dot.encode("utf-8"),
            file_name="arbol_decision.dot",
            mime="text/plain",
            key="dl_tree_dot",
        )

        # Predicciones
        out = df_model.copy()
        out["pred_cluster"] = y_pred
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Descargar dataset con predicción (CSV)",
            data=buf.getvalue().encode("utf-8"),
            file_name="arbol_dataset_con_pred.csv",
            mime="text/csv",
            key="dl_tree_pred",
        )
