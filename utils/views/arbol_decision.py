# utils/views/arbol_decision.py
import numpy as np
import pandas as pd
import streamlit as st
import io

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from utils.config import EXCLUDE_VARS, LABELS, VARS_CLUSTER, LOGGED_IN_MODEL
from utils.data_io import load_base_with_clusters


# ======================
# Helpers labels
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
# Graphviz DOT builder (estilo rpart)
# ======================
def _rgba(hex_color: str, alpha: float = 0.25) -> str:
    """Convierte #RRGGBB a #RRGGBBAA (Graphviz soporta alpha en hex largo)."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    a = int(max(0, min(1, alpha)) * 255)
    return f"#{h}{a:02x}"

def build_rpart_like_dot(clf: DecisionTreeClassifier, feature_names: list[str], class_names: list[str]):
    tree = clf.tree_
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value.squeeze(axis=1)  # shape (nodes, n_classes)
    samples = tree.n_node_samples

    total = float(samples[0]) if samples[0] > 0 else 1.0

    # Paleta similar a tu app (C1 naranja, C2 gris, C3 azul)
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
        # value[i] puede venir como "conteos" o como "pesos"
        vraw = np.asarray(value[i], dtype = float)
        vsum = float(np.nansum(vraw))
        if vsum > 0:
            probs = vraw / vsum
            pred_idx = int(np.nanargmax(vraw))
        else:
            probs = np.zeros_like(vraw, dtype = float)
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


# ======================
# Sanitizado robusto
# ======================
def _sanitize_X(X: pd.DataFrame, cap_q: float = 0.995, hard_cap: float = 1e12) -> pd.DataFrame:
    """
    - Convierte a numérico
    - Reemplaza inf/-inf por NaN
    - Clip por percentil (cap_q)
    - Clip por hard_cap (valor absoluto) para evitar float32 overflow
    - Imputa mediana
    - Fuerza float64
    """
    X = X.copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # clip por percentil (más agresivo que 0.999)
    caps = X.quantile(cap_q, numeric_only=True)
    for c in X.columns:
        if c in caps.index and pd.notna(caps[c]):
            X[c] = X[c].clip(upper=float(caps[c]))

    # clip duro por valor absoluto
    X = X.clip(lower=-hard_cap, upper=hard_cap)

    # imputación
    X = X.fillna(X.median(numeric_only=True))

    X = X.astype(np.float64)

    # check final
    if not np.isfinite(X.to_numpy()).all():
        bad_cols = [col for col in X.columns if not np.isfinite(X[col].to_numpy()).all()]
        raise ValueError(f"Siguen quedando valores no finitos en columnas: {bad_cols}")

    return X


# ======================
# MAIN VIEW
# ======================
from collections import defaultdict

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

def _node_rule_text(clf: DecisionTreeClassifier, node_id: int, feature_names: list[str]):
    """
    Devuelve una regla tipo 'X < thr AND Y >= thr ...' desde la raíz hasta node_id.
    """
    tree = clf.tree_
    parent, is_left = _build_parent_map(clf)

    parts = []
    cur = node_id
    while cur != 0 and cur in parent and parent[cur] != -1:
        p = parent[cur]
        f_idx = tree.feature[p]
        thr = tree.threshold[p]
        f = feature_names[f_idx]

        # Si cur es hijo izquierdo => condición "no" del split (x < thr si mostramos split como >=)
        if is_left.get(cur, True):
            parts.append(f"{_label_of(f)} < {thr:.3g}")
        else:
            parts.append(f"{_label_of(f)} ≥ {thr:.3g}")

        cur = p

    parts.reverse()
    return " AND ".join(parts) if parts else "(raíz)"

def _node_pred_and_probs(clf: DecisionTreeClassifier, node_id: int, class_names: list[str]):
    """
    Predicción y % por clase en el nodo (según tree.value).
    """
    tree = clf.tree_
    v = tree.value[node_id].squeeze()
    tot = float(v.sum()) if float(v.sum()) > 0 else 0.0
    probs = (v / tot) if tot > 0 else np.zeros_like(v, dtype=float)
    pred = class_names[int(np.argmax(v))] if tot > 0 else class_names[0]
    pct = np.round(100 * probs).astype(int)
    return pred, probs, pct, int(tot)


def render_arbol_decision(df_app: pd.DataFrame, base_path: str):
    st.header("Árbol de decisión (estilo rpart)")

    # Base completa + clusters
    try:
        df_full = load_base_with_clusters(base_path, df_app)

        try:
            from utils.recodes import apply_recodes
            df_full = apply_recodes(df_full)
        except Exception:
            pass

    except Exception as e:
        st.error(f"No he podido cargar la base completa o hacer el merge con clusters: {e}")
        st.stop()

    if "cluster_label" not in df_full.columns:
        st.error("No existe `cluster_label` en la base completa tras el merge.")
        st.stop()

    df_full = df_full[df_full["cluster_label"].notna()].copy()

    forbidden = {"codigo_nif", "nombre", "cluster_label", "PC1", "PC2", "empresa_key"} | set(EXCLUDE_VARS)

    num_cols = []
    for c in df_full.columns:
        if c in forbidden:
            continue
        if pd.api.types.is_numeric_dtype(df_full[c]):
            num_cols.append(c)
    num_cols = sorted(num_cols)

    num_labels, num_lab_to_var = _make_label_maps(num_cols)

    with st.sidebar:
        st.subheader("Configuración del árbol")

        default_vars = [v for v in VARS_CLUSTER if v in num_cols]
        # defaults a labels (resolviendo duplicados)
        default_labels = []
        for v in default_vars:
            lab = _label_of(v)
            cand2 = f"{lab} [{v}]"
            if cand2 in num_labels:
                default_labels.append(cand2)
            elif lab in num_labels:
                default_labels.append(lab)

        vars_sel_labels = st.multiselect(
            "Variables (predictoras)",
            options=num_labels,
            default=default_labels if default_labels else num_labels[:8],
            key="tree_vars",
        )

        max_depth = st.slider("Profundidad máxima (max_depth)", 1, 10, 4, 1)
        min_leaf = st.slider("Mínimo por hoja (min_samples_leaf)", 1, 50, 10, 1)
        min_split = st.slider("Mínimo para split (min_samples_split)", 2, 100, 20, 1)

        criterion = st.selectbox("Criterio", ["gini", "entropy", "log_loss"], index=0)

        ccp_alpha = st.slider("Pruning (ccp_alpha ~ cp)", 0.0, 0.05, 0.0, 0.001)

        st.markdown("---")
        st.caption("Limpieza para entrenamiento")
        cap_q = st.slider("Clip percentil", 0.90, 0.999, 0.995, 0.001)
        hard_cap = st.number_input("Clip absoluto (|x|)", min_value=1e6, max_value=1e15, value=1e12, step=1e6, format="%.0f")

        show_top_imp = st.checkbox("Mostrar Top variables por importancia", value=True)
        top_k = st.slider("Top K", 5, 25, 10, 1)

    if not vars_sel_labels:
        st.info("Selecciona al menos una variable.")
        st.stop()

    X_vars = [num_lab_to_var[x] for x in vars_sel_labels]

    keep_cols = ["cluster_label"] + X_vars
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

    y = df_model["cluster_label"].astype(str)
    pref = ["C1", "C2", "C3"]
    class_names = [c for c in pref if c in y.unique()]
    class_names += [c for c in sorted(y.unique()) if c not in class_names]

    X = df_model[X_vars].copy()

    # ✅ SANEAR ANTES del expm1 (evita que expm1 cree inf)
    try:
        X = _sanitize_X(X, cap_q=float(cap_q), hard_cap=float(hard_cap))
    except Exception as e:
        st.error(f"Problema saneando X (antes de deslog): {e}")
        st.stop()

    # ✅ deslog seguro (y volver a sanear)
    for c in X_vars:
        if c in LOGGED_IN_MODEL:
            # si hay valores enormes, expm1 puede overflow -> inf; luego lo saneamos
            X[c] = np.expm1(X[c])

    try:
        X = _sanitize_X(X, cap_q=float(cap_q), hard_cap=float(hard_cap))
    except Exception as e:
        st.error(f"Problema saneando X (después de deslog): {e}")
        st.stop()

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        min_samples_split=min_split,
        criterion=criterion,
        ccp_alpha=ccp_alpha,
        random_state=42,
    )

    try:
        clf.fit(X, y)
    except Exception as e:
        st.error("Sigue fallando el fit. Te dejo diagnóstico de columnas más problemáticas:")
        diag = pd.DataFrame({
            "na": X.isna().sum(),
            "max_abs": X.abs().max(),
            "p99.9": X.quantile(0.999),
        }).sort_values("max_abs", ascending=False)
        st.dataframe(diag.head(30), use_container_width=True)
        st.exception(e)
        st.stop()

    dot = build_rpart_like_dot(clf=clf, feature_names=X_vars, class_names=class_names)

    st.subheader("Árbol de decisión (sin estandarizar: unidades reales)")
    try:
        st.graphviz_chart(dot, use_container_width=True)
    except Exception as e:
        st.error("No he podido renderizar Graphviz.")
        st.exception(e)

    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    st.caption(f"Exactitud (sobre los mismos datos usados para entrenar): **{acc:.3f}**")

    cm = confusion_matrix(y, y_pred, labels=class_names)
    cm_df = pd.DataFrame(cm, index=[f"Real {c}" for c in class_names], columns=[f"Pred {c}" for c in class_names])
    st.dataframe(cm_df, use_container_width=True)

    if show_top_imp:
        st.subheader("Top variables por importancia (Decision Tree)")
        imp = pd.DataFrame({"Variable": X_vars, "Importancia": clf.feature_importances_})
        imp["Variable"] = imp["Variable"].map(lambda v: _label_of(v))
        imp = imp.sort_values("Importancia", ascending=False).head(top_k)
        st.bar_chart(imp.set_index("Variable")["Importancia"])

    # =========================================================
    # Empresas por nodo del árbol  ✅ (DENTRO de la función)
    # =========================================================
    st.divider()
    st.subheader("Empresas por nodo del árbol")

    # Recalcular path / pertenencia a nodos (X es el mismo que has usado para clf.fit)
    leaf_id = clf.apply(X)                 # hoja por fila
    path = clf.decision_path(X)            # csr: filas=casos, cols=nodos

    tree = clf.tree_

    # Mapas: nodo -> índices (filas) que caen/pasan por ese nodo
    node_to_rows = defaultdict(list)
    for i, nid in enumerate(leaf_id):
        node_to_rows[int(nid)].append(i)

    node_to_rows_all = defaultdict(list)
    coo = path.tocoo()
    for r, c in zip(coo.row, coo.col):
        node_to_rows_all[int(c)].append(int(r))

    mode = st.radio(
        "¿Qué quieres seleccionar?",
        ["Hojas (grupos finales)", "Cualquier nodo (incluye internos)"],
        horizontal=True,
        key="tree_node_mode",
    )

    if mode == "Hojas (grupos finales)":
        candidate_nodes = sorted(node_to_rows.keys())
        map_rows = node_to_rows
    else:
        candidate_nodes = sorted(node_to_rows_all.keys())
        map_rows = node_to_rows_all

    if not candidate_nodes:
        st.info("No hay nodos candidatos para listar empresas.")
        return

    # Etiquetas bonitas
    class_names_sorted = class_names
    labels_nodes = []
    node_id_by_label = {}

    for nid in candidate_nodes:
        pred, probs, pct, ncases = _node_pred_and_probs(clf, nid, class_names_sorted)
        rule = _node_rule_text(clf, nid, X_vars)

        pct_str = " ".join([f"{x:02d}" for x in pct])
        txt = f"Nodo {nid} · {pred} · {pct_str} · n={ncases}"
        if rule and rule != "(raíz)":
            txt += f" · {rule}"

        labels_nodes.append(txt)
        node_id_by_label[txt] = nid

    pick = st.selectbox("Selecciona un nodo", options=labels_nodes, index=0, key="tree_node_pick")
    nid_sel = node_id_by_label[pick]

    rule_sel = _node_rule_text(clf, nid_sel, X_vars)
    pred, probs, pct, ncases = _node_pred_and_probs(clf, nid_sel, class_names_sorted)

    st.caption(f"**Regla del nodo:** {rule_sel}")
    st.caption(
        f"**Predicción:** {pred}  ·  **% por clase:** "
        + " / ".join([f"{c}={p:02d}%" for c, p in zip(class_names_sorted, pct)])
    )

    rows = map_rows.get(nid_sel, [])
    if not rows:
        st.info("No hay empresas en este nodo.")
        return

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