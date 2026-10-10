"""Central theme: colours, spacing, typography, and cluster palette."""
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ThemeTokens:
    """Design tokens for consistent theming across the application."""

    # Core palette
    NAVY: str = "#1E293B"
    BACKGROUND: str = "#0F172A"
    SURFACE: str = "#16233B"
    SURFACE_ELEVATED: str = "#1E2D45"
    BORDER: str = "#334155"
    BORDER_STRONG: str = "#475569"

    # Accent colors
    GOLD: str = "#D4A72C"
    GOLD_HOVER: str = "#E0B83E"
    GOLD_MUTED: str = "#D4A72C33"
    TEAL: str = "#2A9D8F"
    TEAL_HOVER: str = "#3BC9A6"
    TEAL_MUTED: str = "#2A9D8F33"

    # Semantic colors
    SUCCESS: str = "#3FB950"
    SUCCESS_MUTED: str = "#3FB95033"
    WARNING: str = "#D9A520"
    WARNING_MUTED: str = "#D9A52033"
    DANGER: str = "#E0483C"
    DANGER_MUTED: str = "#E0483C33"
    INFO: str = "#38BDF8"
    INFO_MUTED: str = "#38BDF833"

    # Text colors
    TEXT_PRIMARY: str = "#F8FAFC"
    TEXT_SECONDARY: str = "#CBD5E1"
    TEXT_MUTED: str = "#94A3B8"
    TEXT_INVERSE: str = "#0F172A"
    TEXT_ON_GOLD: str = "#1A1207"

    # Typography
    FONT_FAMILY: str = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"
    FONT_MONO: str = "'JetBrains Mono', 'Fira Code', 'Consolas', monospace"

    # Font sizes (rem-based scale)
    FS_XS: str = "0.7rem"      # 11px
    FS_SM: str = "0.75rem"     # 12px
    FS_BASE: str = "0.875rem"  # 14px
    FS_LG: str = "1rem"        # 16px
    FS_XL: str = "1.125rem"    # 18px
    FS_2XL: str = "1.25rem"    # 20px
    FS_3XL: str = "1.5rem"     # 24px
    FS_4XL: str = "2rem"       # 32px
    FS_5XL: str = "2.5rem"     # 40px

    # Font weights
    FW_NORMAL: int = 400
    FW_MEDIUM: int = 500
    FW_SEMIBOLD: int = 600
    FW_BOLD: int = 700
    FW_EXTRABOLD: int = 800

    # Line heights
    LH_TIGHT: float = 1.1
    LH_NORMAL: float = 1.5
    LH_RELAXED: float = 1.625

    # Spacing (rem-based)
    SPACE_0: str = "0"
    SPACE_1: str = "0.25rem"   # 4px
    SPACE_2: str = "0.5rem"    # 8px
    SPACE_3: str = "0.75rem"   # 12px
    SPACE_4: str = "1rem"      # 16px
    SPACE_5: str = "1.25rem"   # 20px
    SPACE_6: str = "1.5rem"    # 24px
    SPACE_8: str = "2rem"      # 32px
    SPACE_10: str = "2.5rem"   # 40px
    SPACE_12: str = "3rem"     # 48px
    SPACE_16: str = "4rem"     # 64px

    # Border radius
    RADIUS_SM: str = "0.375rem"   # 6px
    RADIUS_MD: str = "0.5rem"     # 8px
    RADIUS_LG: str = "0.75rem"    # 12px
    RADIUS_XL: str = "1rem"       # 16px
    RADIUS_FULL: str = "9999px"

    # Shadows
    SHADOW_SM: str = "0 1px 2px 0 rgb(0 0 0 / 0.05)"
    SHADOW_MD: str = "0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)"
    SHADOW_LG: str = "0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)"
    SHADOW_GOLD: str = "0 0 0 1px #D4A72C, 0 4px 14px -2px #D4A72C40"

    # Transitions
    TRANSITION_FAST: str = "150ms ease"
    TRANSITION_NORMAL: str = "250ms ease"
    TRANSITION_SLOW: str = "350ms ease"

    # Layout
    MAX_WIDTH: str = "1280px"
    MAX_WIDTH_NARROW: str = "960px"
    SIDEBAR_WIDTH: str = "280px"

    # Z-index layers
    Z_DROPDOWN: int = 100
    Z_STICKY: int = 200
    Z_MODAL: int = 300
    Z_POPOVER: int = 400
    Z_TOAST: int = 500

    # Cluster palette (extended for up to 6 clusters)
    BASE_PALETTE: List[str] = None

    def __post_init__(self):
        if self.BASE_PALETTE is None:
            object.__setattr__(self, 'BASE_PALETTE', [
                "#D4A72C",  # Gold
                "#2A9D8F",  # Teal
                "#E0483C",  # Danger/Red
                "#3FB950",  # Success/Green
                "#E8823C",  # Orange
                "#8FA3B8",  # Muted Blue
            ])

    def cluster_colors(self, k: int) -> List[str]:
        """Return k cluster colors from the base palette."""
        k = int(k)
        return [self.BASE_PALETTE[i % len(self.BASE_PALETTE)] for i in range(k)]


# Global theme instance
T = ThemeTokens()

# Backward compatibility constants
NAVY = T.NAVY
BACKGROUND = T.BACKGROUND
GOLD = T.GOLD
TEAL = T.TEAL
SUCCESS = T.SUCCESS
WARNING = T.WARNING
DANGER = T.DANGER
WHITE = T.TEXT_PRIMARY
MUTED = T.TEXT_MUTED
PANEL = T.SURFACE
BORDER = T.BORDER
FONT = T.FONT_FAMILY
BASE_PALETTE = T.BASE_PALETTE


def cluster_colors(k: int) -> List[str]:
    """Backward compatible cluster color function."""
    return T.cluster_colors(k)