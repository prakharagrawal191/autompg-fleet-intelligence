"""Plotly chart system: one theme, stable cluster identity, accessible defaults."""
import html

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import theme as T

# Keep zoom / pan / reset / PNG download; drop tools that do nothing useful here.
CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines",
                               "hoverCompareCartesian", "hoverClosestCartesian"],
    "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "auto_mpg_chart"},
}
# Diverging scale (correlations, z-scores): teal (negative) - neutral slate - gold (positive).
DIVERGING = [[0.0, T.TEAL], [0.5, T.BORDER], [1.0, T.GOLD]]
DEFAULT_HOVER = [("mpg", "MPG", ".1f"), ("horsepower", "HP", ".0f"),
                 ("weight", "Weight (lb)", ",.0f"), ("cylinders", "Cylinders", ".0f")]


def style(fig, height=380, title=None, subtitle=None, xtitle=None, ytitle=None, legend=True):
    """Apply the shared dark theme. Heights grow ~15% in presentation mode."""
    if st.session_state.get("presentation_mode"):
        height = int(height * 1.15)
    axis = dict(gridcolor=T.GRID, zeroline=False, linecolor=T.BORDER, showline=True,
                tickfont=dict(color=T.MUTED, size=11), title_font=dict(color=T.TEXT_2, size=12),
                automargin=True)
    ttl = None
    if title:
        ttl = dict(text=title, x=0.0, xanchor="left", y=0.98, yanchor="top",
                   font=dict(size=15, color=T.WHITE))
        if subtitle:
            ttl["subtitle"] = dict(text=subtitle, font=dict(size=12, color=T.MUTED))
    top = (72 if subtitle else 52) if title else 16
    fig.update_layout(
        template="none", height=height, title=ttl,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=T.FONT, color=T.TEXT_2, size=12),
        margin=dict(l=8, r=12, t=top + (26 if legend else 0), b=8),
        xaxis=dict(axis, title_text=xtitle) if xtitle is not None else axis,
        yaxis=dict(axis, title_text=ytitle) if ytitle is not None else axis,
        showlegend=bool(legend),
        # Legend above the plot area: it can never collide with the x-axis title.
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(color=T.TEXT_2, size=11), bgcolor="rgba(0,0,0,0)",
                    itemsizing="constant", title_text=""),
        hoverlabel=dict(bgcolor=T.SURFACE, bordercolor=T.BORDER_STRONG,
                        font=dict(family=T.FONT, color=T.WHITE, size=12)),
        hovermode="closest", dragmode="zoom",
        # Plotly derives modebar colours from the (transparent) paper: 2.4:1. Muted icons give ~7:1.
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=T.MUTED, activecolor=T.WHITE),
    )
    return fig


def plain(values):
    """Escape data for Plotly text/hover, which renders an HTML subset (<a>, <span style>, ...)."""
    return [html.escape(str(v), quote=False) for v in values]


def show(fig, alt, key=None):
    """Render with the factory theme (not Streamlit's) and an accessible description."""
    st.plotly_chart(fig, width="stretch", theme=None, config=CONFIG, alt=alt, key=key)


def ref_line(fig, x=None, y=None, text="", color=T.MUTED, dash="dot", position="top right"):
    kw = dict(line_color=color, line_dash=dash, line_width=1.5, annotation_text=text,
              annotation_position=position, annotation_font=dict(color=color, size=11))
    if x is not None:
        fig.add_vline(x=x, **kw)
    if y is not None:
        fig.add_hline(y=y, **kw)
    return fig


def _hover(cols, frame):
    cols = [c for c in cols if c[0] in frame.columns]
    data = frame[[c[0] for c in cols]].to_numpy() if cols else None
    lines = "<br>".join(f"{lab} %{{customdata[{j}]:{fmt}}}" for j, (_, lab, fmt) in enumerate(cols))
    return data, lines


def cluster_scatter(frame, x, y, labels, k, *, xtitle, ytitle, title=None, subtitle=None,
                    hover=DEFAULT_HOVER, centers=None, center_label="Centroid", size=8,
                    opacity=0.85, show_text=False, visible=None, height=440):
    """Scatter coloured AND shaped by cluster id (identity never relies on colour alone).

    ``centers``: k x 2 array already in the plotted coordinates (inverse-transformed
    original units or PCA scores). ``visible``: cluster ids to emphasise; the rest
    stay as faint context so a filter never silently drops the full picture.
    """
    labels = np.asarray(labels)
    names = np.asarray(plain(frame["car_name"])) if "car_name" in frame.columns else None
    data, lines = _hover(hover, frame)
    fig = go.Figure()
    for i in range(int(k)):
        m = labels == i
        on = visible is None or i in visible
        tpl = (("<b>%{hovertext}</b><br>" if names is not None else "")
               + f"Cluster {i}<br>{xtitle} %{{x:.2f}} · {ytitle} %{{y:.2f}}"
               + (f"<br>{lines}" if lines else "") + "<extra></extra>")
        fig.add_trace(go.Scatter(
            x=frame[x].to_numpy()[m], y=frame[y].to_numpy()[m], name=T.cluster_label(i),
            mode="markers+text" if (show_text and on) else "markers",
            text=names[m] if (show_text and on and names is not None) else None,
            textposition="top center", textfont=dict(size=9, color=T.TEXT_2),
            hovertext=names[m] if names is not None else None,
            customdata=data[m] if data is not None else None, hovertemplate=tpl,
            marker=dict(color=T.cluster_color(i) if on else T.BORDER_STRONG,
                        symbol=T.cluster_symbol(i), size=size, opacity=opacity if on else 0.18,
                        line=dict(width=0.6, color=T.BACKGROUND)),
            showlegend=on, legendgroup=f"c{i}"))
    if centers is not None:
        for i in range(int(k)):
            if visible is not None and i not in visible:
                continue
            fig.add_trace(go.Scatter(
                x=[float(centers[i][0])], y=[float(centers[i][1])], mode="markers",
                name=f"{center_label}s", legendgroup="centers", showlegend=i == min(visible or [0]),
                marker=dict(symbol="x", size=16, color=T.WHITE, line=dict(width=2, color=T.cluster_color(i))),
                hovertemplate=(f"<b>Cluster {i} {center_label.lower()}</b><br>{xtitle} %{{x:.2f}}"
                               f"<br>{ytitle} %{{y:.2f}}<extra></extra>")))
    return style(fig, height, title, subtitle, xtitle, ytitle)


def cluster_sizes(counts, title=None, subtitle=None):
    """Horizontal bars of vehicles per cluster in id order, labelled 'n · p%'."""
    ids = [int(i) for i in counts.index]
    n = np.asarray(counts.values, dtype=float)
    pct = 100 * n / max(n.sum(), 1)
    fig = go.Figure(go.Bar(
        x=n, y=[T.cluster_label(i) for i in ids], orientation="h",
        marker=dict(color=[T.cluster_color(i) for i in ids], line=dict(width=0)),
        text=[f"{int(a)} · {p:.1f}%" for a, p in zip(n, pct)], textposition="outside",
        textfont=dict(color=T.TEXT_2, size=12), cliponaxis=False,
        hovertemplate="%{y}: %{x} vehicles<extra></extra>"))
    fig = style(fig, 70 + 40 * len(ids), title, subtitle, "Vehicles", None, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, n.max() * 1.7 if len(n) else 1])  # room for the outside 'n · p%' labels
    return fig


def silhouette_by_k(kr, selected, upto=None, documented=None, height=330):
    """K-Means sweep silhouettes: gold = selected K, 'highest' marks the observed maximum.

    ``documented`` (a K) is annotated separately so the documented engineering choice
    never looks identical to the metric optimum.
    """
    d = kr if upto is None else kr[kr["K"] <= upto]
    best = int(kr.loc[kr["silhouette"].idxmax(), "K"])
    top = float(kr["silhouette"].max())
    fig = go.Figure(go.Bar(
        x=d["K"], y=d["silhouette"], marker=dict(color=[T.GOLD if k == selected else T.TEAL for k in d["K"]],
                                                 line=dict(width=0)),
        text=[f"{v:.4f}" for v in d["silhouette"]], textposition="outside", textfont=dict(color=T.TEXT_2, size=11),
        cliponaxis=False, name="Silhouette", hovertemplate="K=%{x}<br>Silhouette %{y:.4f}<extra></extra>"))
    fig = style(fig, height, "Silhouette by K", f"Gold = selected K ({selected}); higher is better.",
                "Number of clusters (K)", "Silhouette", legend=False)
    ks = [int(k) for k in kr["K"]]
    # Notes ride on the tick labels so they never overlap neighbouring bars.
    notes = {best: "highest"}
    if documented is not None and documented != best:
        notes[documented] = "documented"
    fig.update_xaxes(tickvals=ks, ticktext=[f"{k}<br>{notes[k]}" if k in notes else str(k) for k in ks],
                     range=[min(ks) - 0.6, max(ks) + 0.6])
    fig.update_yaxes(range=[0, top * 1.25])
    return fig


def _mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return [a[j] + (b[j] - a[j]) * t for j in range(3)]


def _lum(rgb):
    ch = [v / 255 for v in rgb]
    ch = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in ch]
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]


def _ink(value, zmin, zmax):
    """Label colour with the higher WCAG contrast on the diverging cell (black/white: >= 4.58:1)."""
    t = (value - zmin) / ((zmax - zmin) or 1)
    lum = _lum(_mix(T.TEAL, T.BORDER, t * 2) if t < 0.5 else _mix(T.BORDER, T.GOLD, (t - 0.5) * 2))
    return T.INK_DARK if (lum + 0.05) / 0.05 > 1.05 / (lum + 0.05) else T.INK_LIGHT


def diverging_heatmap(matrix, title=None, subtitle=None, zmin=-1.0, zmax=1.0, fmt=".2f",
                      colorbar="r", height=380, xtitle=None, ytitle=None):
    """Annotated heatmap on the shared diverging scale (rows/cols from the DataFrame)."""
    z = matrix.to_numpy(dtype=float)
    xs, ys = [str(c) for c in matrix.columns], [str(r) for r in matrix.index]
    fig = go.Figure(go.Heatmap(
        z=z, x=xs, y=ys, zmin=zmin, zmax=zmax, colorscale=DIVERGING, xgap=2, ygap=2,
        colorbar=dict(title=dict(text=colorbar, font=dict(color=T.MUTED)), thickness=10,
                      tickfont=dict(color=T.MUTED), outlinewidth=0),
        hovertemplate="%{y} × %{x}: %{z:" + fmt + "}<extra></extra>"))
    for r, yv in enumerate(ys):
        for c, xv in enumerate(xs):
            v = z[r, c]
            if np.isfinite(v):
                fig.add_annotation(x=xv, y=yv, text=format(v, fmt), showarrow=False,
                                   font=dict(color=_ink(v, zmin, zmax), size=13))
    fig = style(fig, height, title, subtitle, xtitle, ytitle, legend=False)
    fig.update_xaxes(showgrid=False, showline=False, side="bottom")
    fig.update_yaxes(showgrid=False, showline=False, autorange="reversed")
    return fig


def mpl_dark(fig, ax):
    """Theme a Matplotlib figure (used by the validated SciPy dendrogram)."""
    fig.patch.set_facecolor(T.BACKGROUND)
    ax.set_facecolor(T.BACKGROUND)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T.BORDER)
    ax.tick_params(colors=T.MUTED, labelsize=9)
    ax.yaxis.label.set_color(T.TEXT_2)
    ax.xaxis.label.set_color(T.TEXT_2)
    ax.title.set_color(T.WHITE)
    ax.grid(axis="y", color=T.GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax
