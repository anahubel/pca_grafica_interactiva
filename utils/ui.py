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


def plotly_layout_base(fig, height=None, **overrides):
    """
    Aplica un layout base de Plotly de forma segura.
    Siempre devuelve la figura (fig).
    """
    base = dict(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.15),
        hoverlabel=dict(font_size=12),
    )
    if height is not None:
        base["height"] = height
    base.update(overrides)
    fig.update_layout(**base)
    return fig


def anchor(id_: str):
    st.markdown(f'<div id="{id_}"></div>', unsafe_allow_html=True)


def remove_plotly_extra(fig):
    """
    Fuerza <extra></extra> para evitar textos tipo 'trace 0' en hover.
    """
    for tr in fig.data:
        ht = getattr(tr, "hovertemplate", None)
        if ht:
            if "<extra></extra>" not in ht:
                tr.hovertemplate = ht + "<extra></extra>"
        else:
            tr.hovertemplate = "%{x}, %{y}<extra></extra>"
    return fig


def enforce_no_extra(fig):
    """Alias de compatibilidad."""
    return remove_plotly_extra(fig)