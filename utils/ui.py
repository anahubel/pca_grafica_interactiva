# utils/ui.py
from __future__ import annotations
from pathlib import Path
import streamlit as st

def load_css(path: str = "assets/brand.css") -> None:
    """Carga CSS una sola vez."""
    if st.session_state.get("__css_loaded__", False):
        return
    p = Path(path)
    if p.exists():
        st.markdown(f"<style>{p.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    st.session_state["__css_loaded__"] = True


def plotly_layout_base(fig=None, height: int | None = None):
    layout = dict(
        template="plotly_white",
        # ❌ QUITAR margin de aquí
        hoverlabel=dict(font_size=12),
    )
    if height is not None:
        layout["height"] = height

    if fig is None:
        return layout

    fig.update_layout(**layout)
    return fig


def remove_plotly_extra(fig):
    """
    Fuerza <extra></extra> para evitar textos tipo 'trace 0' / 'streamlitApp' en hover.
    (Útil si te queda algún trace sin hovertemplate.)
    """
    for tr in fig.data:
        ht = getattr(tr, "hovertemplate", None)
        if ht:
            if "<extra></extra>" not in ht:
                tr.hovertemplate = ht + "<extra></extra>"
        else:
            # Si no hay hovertemplate, le ponemos uno mínimo sin extra
            tr.hovertemplate = "%{x}, %{y}<extra></extra>"
    return fig


def anchor(id_: str):
    st.markdown(f'<div id="{id_}"></div>', unsafe_allow_html=True)