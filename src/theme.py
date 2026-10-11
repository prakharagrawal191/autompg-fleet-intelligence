"""Central design tokens: colours, type and the cluster palette (presentation only)."""

# Core palette (the product contract - reuse, never invent per section).
NAVY = "#1E293B"
BACKGROUND = "#0F172A"
GOLD = "#D4A72C"
TEAL = "#2A9D8F"
SUCCESS = "#3FB950"
WARNING = "#D9A520"
DANGER = "#E0483C"
WHITE = "#F8FAFC"
MUTED = "#94A3B8"

# Neutrals and text tints derived for legibility (WCAG contrast measured on SURFACE).
SURFACE = "#16233B"        # cards and panels
SURFACE_2 = "#1E2D45"      # raised controls, hover
BORDER = "#334155"
BORDER_STRONG = "#475569"  # borders only - fails text contrast (2.4:1)
TEXT_2 = "#CBD5E1"         # secondary text, 10.6:1
GRID = "#243247"           # chart gridlines
ON_GOLD = "#1A1207"        # text on gold fills, 8.3:1
ON_TEAL = BACKGROUND       # text on teal fills, 5.4:1 (white fails at 3.2:1)
TEAL_TEXT = "#5EC4B6"      # teal as small text, 7.5:1 (base teal is 4.7:1)
DANGER_TEXT = "#F07B71"    # danger as small text, 5.8:1 (base red is 3.9:1)
INK_DARK, INK_LIGHT = "#000000", "#FFFFFF"  # heatmap cell labels: max contrast on any diverging step

FONT = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
MONO = "ui-monospace, 'Cascadia Code', 'JetBrains Mono', Consolas, monospace"

# Cluster identity. Validated with the dataviz six checks on BACKGROUND and SURFACE
# (dark mode): adjacent CVD dE >= 8.4, normal-vision dE >= 19.3, contrast >= 3:1.
# Only the first three slots pass the all-pairs check, so scatter plots with K >= 4
# must also encode the cluster with CLUSTER_SYMBOLS (never colour alone).
CLUSTER_PALETTE = ("#3987E5", "#D95926", "#199E70", "#C98500", "#D55181", "#9085E9")
CLUSTER_SYMBOLS = ("circle", "square", "diamond", "triangle-up", "cross", "pentagon")

TONES = {  # semantic tone -> colour; status tones always ship with a text label
    "neutral": MUTED, "gold": GOLD, "teal": TEAL,
    "success": SUCCESS, "warning": WARNING, "danger": DANGER,
}


def cluster_color(i):
    return CLUSTER_PALETTE[int(i) % len(CLUSTER_PALETTE)]


def cluster_symbol(i):
    return CLUSTER_SYMBOLS[int(i) % len(CLUSTER_SYMBOLS)]


def cluster_colors(k):
    return [cluster_color(i) for i in range(int(k))]


def cluster_label(i):
    return f"Cluster {int(i)}"
