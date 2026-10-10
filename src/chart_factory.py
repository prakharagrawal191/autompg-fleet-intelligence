"""Central Plotly factory: professional dark theme for every chart."""
from typing import Optional, List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from src.theme import T, cluster_colors


# Chart constants
FONT = T.FONT_FAMILY
FONT_MONO = T.FONT_MONO
WHITE = T.TEXT_PRIMARY
MUTED = T.TEXT_MUTED
GRID = T.BORDER
SURFACE = T.SURFACE
BORDER = T.BORDER

# Default chart heights
HEIGHT_SMALL = 300
HEIGHT_MEDIUM = 400
HEIGHT_LARGE = 500
HEIGHT_XL = 600

# Semantic color constants (defined early: used as function defaults below)
TEXT_SECONDARY = T.TEXT_SECONDARY
GOLD = T.GOLD
TEAL = T.TEAL
SUCCESS = T.SUCCESS
WARNING = T.WARNING
DANGER = T.DANGER
INFO = T.INFO

# Cluster color sequence
CLUSTER_COLORS_6 = cluster_colors(6)


def base_layout(
    height: int = HEIGHT_MEDIUM,
    title: str = "",
    show_legend: bool = True,
    margin: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Base layout configuration for all charts."""
    if margin is None:
        margin = dict(l=56, r=18, t=56, b=52)

    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family=FONT, size=13),
        margin=margin,
        height=height,
        title=dict(
            text=title,
            font=dict(size=16, color=WHITE, family=FONT, weight=600),
            x=0.02,
            xanchor="left",
        ),
        xaxis=dict(
            gridcolor=GRID,
            zerolinecolor=GRID,
            zerolinewidth=1,
            tickfont=dict(color=MUTED, size=11, family=FONT),
            title_font=dict(color=TEXT_SECONDARY, size=12, family=FONT, weight=500),
            linecolor=BORDER,
            linewidth=1,
            mirror=True,
            showspikes=False,
            spikecolor=GOLD,
            spikethickness=1,
        ),
        yaxis=dict(
            gridcolor=GRID,
            zerolinecolor=GRID,
            zerolinewidth=1,
            tickfont=dict(color=MUTED, size=11, family=FONT),
            title_font=dict(color=TEXT_SECONDARY, size=12, family=FONT, weight=500),
            linecolor=BORDER,
            linewidth=1,
            mirror=True,
            showspikes=False,
            spikecolor=GOLD,
            spikethickness=1,
        ),
        legend=dict(
            font=dict(color=WHITE, size=11, family=FONT),
            bgcolor="rgba(22, 35, 59, 0.9)",
            bordercolor=BORDER,
            borderwidth=1,
            itemsizing="constant",
            itemwidth=30,
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0.0,
        ) if show_legend else dict(visible=False),
        hoverlabel=dict(
            bgcolor=SURFACE,
            font=dict(color=WHITE, size=12, family=FONT),
            bordercolor=BORDER,
            namelength=-1,
        ),
        colorway=CLUSTER_COLORS_6,
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color=MUTED,
            activecolor=GOLD,
            orientation="h",
            remove=["lasso2d", "select2d", "autoScale2d", "toggleSpikelines", "hoverCompareCartesian"],
        ),
        dragmode="zoom",
        hovermode="closest",
        spikedistance=20,
        transition=dict(duration=300, easing="cubic-in-out"),
    )


def style(fig: go.Figure, height: int = HEIGHT_MEDIUM, title: str = "", show_legend: bool = True) -> go.Figure:
    """Apply standard styling to a Plotly figure."""
    fig.update_layout(**base_layout(height, title, show_legend))
    return fig


def toolbar_off(fig: go.Figure) -> go.Figure:
    """Remove unnecessary toolbar buttons."""
    fig.update_layout(
        modebar_remove=[
            "lasso2d", "select2d", "autoScale2d",
            "toggleSpikelines", "hoverCompareCartesian",
            "zoom2d", "pan2d", "zoomIn2d", "zoomOut2d",
            "resetScale2d", "toImage"
        ]
    )
    return fig


def toolbar_minimal(fig: go.Figure) -> go.Figure:
    """Keep only essential toolbar buttons (zoom, pan, reset, download)."""
    fig.update_layout(
        modebar_remove=[
            "lasso2d", "select2d", "autoScale2d",
            "toggleSpikelines", "hoverCompareCartesian",
        ]
    )
    return fig


def apply_cluster_colors(fig: go.Figure, trace_names: Optional[List[str]] = None) -> go.Figure:
    """Apply consistent cluster color palette to traces."""
    colors = CLUSTER_COLORS_6
    for i, trace in enumerate(fig.data):
        if trace_names is None or trace.name in trace_names:
            color = colors[i % len(colors)]
            if hasattr(trace, 'marker'):
                trace.marker.color = color
            if hasattr(trace, 'line'):
                trace.line.color = color
    return fig


def style_scatter(
    fig: go.Figure,
    marker_size: int = 8,
    marker_opacity: float = 0.85,
    line_width: float = 0.5,
    line_color: str = "#0F172A",
    show_legend: bool = True,
) -> go.Figure:
    """Style scatter plot traces consistently."""
    for trace in fig.data:
        if trace.type == "scatter" or trace.type == "scattergl":
            trace.marker.size = marker_size
            trace.marker.opacity = marker_opacity
            trace.marker.line.width = line_width
            trace.marker.line.color = line_color
    return style(fig, show_legend=show_legend)


def style_bar(
    fig: go.Figure,
    opacity: float = 0.9,
    line_width: float = 0,
    show_legend: bool = True,
) -> go.Figure:
    """Style bar chart traces consistently."""
    for trace in fig.data:
        if trace.type == "bar":
            trace.marker.opacity = opacity
            trace.marker.line.width = line_width
    return style(fig, show_legend=show_legend)


def style_line(
    fig: go.Figure,
    line_width: int = 3,
    marker_size: int = 6,
    show_legend: bool = True,
) -> go.Figure:
    """Style line chart traces consistently."""
    for trace in fig.data:
        if trace.type == "scatter" and trace.mode in ("lines", "lines+markers", "markers+lines"):
            trace.line.width = line_width
            trace.marker.size = marker_size
    return style(fig, show_legend=show_legend)


def style_heatmap(
    fig: go.Figure,
    colorscale: str = "Tealgrn",
    show_scale: bool = False,
    show_legend: bool = False,
) -> go.Figure:
    """Style heatmap consistently."""
    fig.update_layout(**base_layout(show_legend=show_legend))
    for trace in fig.data:
        if trace.type == "heatmap":
            trace.colorscale = colorscale
            trace.showscale = show_scale
            trace.colorbar = dict(
                thickness=12,
                len=0.7,
                tickfont=dict(color=MUTED, size=10),
                outlinecolor=BORDER,
                outlinewidth=1,
            )
    return fig


def add_centroid_markers(
    fig: go.Figure,
    centroids_x: List[float],
    centroids_y: List[float],
    labels: Optional[List[str]] = None,
    size: int = 15,
    symbol: str = "x",
    color: str = WHITE,
    line_color: str = "#0F172A",
    line_width: int = 2,
    name: str = "Centroids",
) -> go.Figure:
    """Add centroid markers to a scatter plot."""
    if labels is None:
        labels = [f"Cluster {i}" for i in range(len(centroids_x))]

    fig.add_trace(go.Scatter(
        x=centroids_x,
        y=centroids_y,
        mode="markers",
        marker=dict(
            size=size,
            symbol=symbol,
            color=color,
            line=dict(width=line_width, color=line_color),
        ),
        name=name,
        hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        text=labels,
        showlegend=True,
    ))
    return fig


def add_reference_line(
    fig: go.Figure,
    x: Optional[float] = None,
    y: Optional[float] = None,
    color: str = GOLD,
    dash: str = "dash",
    width: int = 2,
    annotation_text: str = "",
    annotation_position: str = "top right",
) -> go.Figure:
    """Add a reference line with optional annotation."""
    if x is not None:
        fig.add_vline(
            x=x, line_color=color, line_dash=dash, line_width=width,
            annotation_text=annotation_text, annotation_position=annotation_position,
            annotation_font=dict(color=color, size=11, family=FONT),
        )
    if y is not None:
        fig.add_hline(
            y=y, line_color=color, line_dash=dash, line_width=width,
            annotation_text=annotation_text, annotation_position=annotation_position,
            annotation_font=dict(color=color, size=11, family=FONT),
        )
    return fig


def add_kde_trace(
    fig: go.Figure,
    x_vals,
    y_vals,
    color: str = GOLD,
    width: int = 3,
    name: str = "KDE",
    fill: str = "none",
) -> go.Figure:
    """Add a KDE line trace to a histogram."""
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        fill=fill,
        fillcolor=color + "22" if fill != "none" else None,
        hovertemplate="Density: %{y:.2f}<extra></extra>",
    ))
    return fig


def create_empty_state_figure(
    message: str = "No data to display",
    height: int = HEIGHT_MEDIUM,
) -> go.Figure:
    """Create an empty state figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color=MUTED, size=14, family=FONT),
    )
    fig.update_layout(**base_layout(height=height, title="", show_legend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def configure_axis_ranges(
    fig: go.Figure,
    x_range: Optional[List[float]] = None,
    y_range: Optional[List[float]] = None,
    pad_pct: float = 0.05,
) -> go.Figure:
    """Configure axis ranges with padding."""
    if x_range is not None:
        pad = (x_range[1] - x_range[0]) * pad_pct
        fig.update_xaxes(range=[x_range[0] - pad, x_range[1] + pad])
    if y_range is not None:
        pad = (y_range[1] - y_range[0]) * pad_pct
        fig.update_yaxes(range=[y_range[0] - pad, y_range[1] + pad])
    return fig