from __future__ import annotations

import io

import pandas as pd
import plotly.express as px


# Palette inspired by viridis — colour-blind friendly, biologists' default.
_PALETTE = ["#440154", "#482878", "#3e4a89", "#31688e",
            "#26828e", "#1f9e89", "#35b779", "#6ece58"]


def boxplot_by_timepoint(df: pd.DataFrame, *, value: str = "area_px",
                         facet: str | None = "treatment") -> str:
    """Return a Plotly figure as standalone HTML."""
    use_facet = facet if facet and facet in df.columns else None
    fig = px.box(
        df, x="timepoint", y=value,
        color=use_facet,
        facet_col=use_facet,
        facet_col_wrap=4,
        points="outliers",
        color_discrete_sequence=_PALETTE,
    )
    fig.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=48, r=24, t=48, b=48),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#e6e9ee"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, title_text="timepoint",
                     tickfont=dict(size=10, color="#b3bac5"),
                     title_font=dict(color="#b3bac5"),
                     linecolor="#2f3845")
    fig.update_yaxes(gridcolor="#232a33", title_text=value,
                     tickfont=dict(size=10, color="#b3bac5"),
                     title_font=dict(color="#b3bac5"),
                     linecolor="#2f3845", zerolinecolor="#2f3845")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    buf = io.StringIO()
    fig.write_html(buf, include_plotlyjs=False, full_html=False,
                   config={"displaylogo": False})
    return buf.getvalue()
