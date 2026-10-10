"""Reusable UI primitives (presentational only, no science)."""
import streamlit as st
from src.theme import T


def css_base(presentation: bool = False) -> str:
    """Generate the base CSS for the application."""
    max_w = T.MAX_WIDTH_NARROW if presentation else T.MAX_WIDTH
    bf = "18px" if presentation else "15px"

    css = f"""
<style>
:root {{
    --navy: {T.NAVY};
    --ink: {T.BACKGROUND};
    --surface: {T.SURFACE};
    --surface-elevated: {T.SURFACE_ELEVATED};
    --gold: {T.GOLD};
    --gold-hover: {T.GOLD_HOVER};
    --gold-muted: {T.GOLD_MUTED};
    --teal: {T.TEAL};
    --teal-hover: {T.TEAL_HOVER};
    --teal-muted: {T.TEAL_MUTED};
    --ok: {T.SUCCESS};
    --warn: {T.WARNING};
    --bad: {T.DANGER};
    --info: {T.INFO};
    --txt: {T.TEXT_PRIMARY};
    --txt-secondary: {T.TEXT_SECONDARY};
    --mut: {T.TEXT_MUTED};
    --border: {T.BORDER};
    --border-strong: {T.BORDER_STRONG};
    --radius-sm: {T.RADIUS_SM};
    --radius-md: {T.RADIUS_MD};
    --radius-lg: {T.RADIUS_LG};
    --radius-xl: {T.RADIUS_XL};
    --radius-full: {T.RADIUS_FULL};
    --shadow-sm: {T.SHADOW_SM};
    --shadow-md: {T.SHADOW_MD};
    --shadow-lg: {T.SHADOW_LG};
    --shadow-gold: {T.SHADOW_GOLD};
    --transition-fast: {T.TRANSITION_FAST};
    --transition-normal: {T.TRANSITION_NORMAL};
    --transition-slow: {T.TRANSITION_SLOW};
    --font: {T.FONT_FAMILY};
    --font-mono: {T.FONT_MONO};
}}

html, body, [class*="css"] {{
    font-family: var(--font);
}}

.stApp {{
    background: var(--ink);
    color: var(--txt);
    font-size: {bf};
}}

.block-container {{
    max-width: {max_w};
    padding-top: 0.75rem;
    padding-bottom: 2rem;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: #141f36;
    border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0.6rem;
}}

section[data-testid="stSidebar"] * {{
    color: var(--txt) !important;
}}

section[data-testid="stSidebar"] .navlbl {{
    font-size: {T.FS_XS};
    letter-spacing: 0.15em;
    color: var(--mut) !important;
    font-weight: {T.FW_BOLD};
    margin: 1.5rem 0 0.5rem;
    text-transform: uppercase;
}}
section[data-testid="stSidebar"] .brand-lockup {{
    padding: 0.35rem 0.2rem 0.75rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.35rem;
}}
section[data-testid="stSidebar"] .brand-lockup .b1 {{
    font-weight: {T.FW_EXTRABOLD};
    letter-spacing: 0.06em;
    font-size: {T.FS_BASE};
}}
section[data-testid="stSidebar"] .brand-lockup .b2 {{
    font-size: {T.FS_XS};
    color: var(--mut) !important;
    margin-top: 0.2rem;
}}

/* Sidebar nav: hide native radio dots, use clean list rows */
section[data-testid="stSidebar"] div[data-testid="stRadio"] svg {{
    display: none !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[class*="e1dm8mpf3"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[class*="e1dm8mpf4"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[class*="e1dm8mpf5"] {{
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-testid="stRadioOption"] {{
    border: 1px solid transparent;
    border-radius: var(--radius-md);
    padding: 0.42rem 0.65rem;
    margin: 0.12rem 0;
    transition: background var(--transition-fast), border-color var(--transition-fast);
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] p {{
    font-size: 0.86rem;
    margin: 0;
}}

section[data-testid="stSidebar"] div[data-testid="stRadio"] > label:hover {{
    background: var(--surface-elevated);
}}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {{
    background: var(--surface-elevated);
    border-left: 3px solid var(--gold);
}}

/* Animations */
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: none; }}
}}

@keyframes pulseGold {{
    from {{ box-shadow: 0 0 0 0 var(--gold-muted); }}
    to {{ box-shadow: 0 0 0 12px transparent; }}
}}

@keyframes slideIn {{
    from {{ opacity: 0; transform: translateX(-10px); }}
    to {{ opacity: 1; transform: none; }}
}}

/* Hero section */
.hero {{
    background: var(--navy);
    border: 1px solid var(--border);
    border-left: 4px solid var(--gold);
    border-radius: var(--radius-md);
    padding: 1.0rem 1.2rem;
    margin: 0.5rem 0 0.9rem;
}}

.hero h1 {{
    font-size: {T.FS_5XL};
    letter-spacing: 0.02em;
    margin: 0;
    font-weight: {T.FW_EXTRABOLD};
    line-height: {T.LH_TIGHT};
}}

.hero h1 span {{
    color: var(--gold);
}}

.hero .sub {{
    font-size: {T.FS_XL};
    margin-top: 0.75rem;
    font-weight: {T.FW_SEMIBOLD};
    color: var(--txt-secondary);
}}

.hero .meta {{
    font-size: {T.FS_SM};
    letter-spacing: 0.1em;
    color: var(--mut);
    margin-top: 0.75rem;
    font-weight: {T.FW_BOLD};
    text-transform: uppercase;
}}

.hero .desc {{
    color: var(--txt-secondary);
    font-size: {T.FS_BASE};
    margin-top: 1rem;
    max-width: 48rem;
    line-height: {T.LH_RELAXED};
}}

/* Top bar */
.topbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    background: var(--navy);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.75rem 1rem;
    margin: 0.25rem 0 1rem;
    font-size: {T.FS_SM};
}}

.topbar .brand {{
    font-weight: {T.FW_EXTRABOLD};
    letter-spacing: 0.06em;
    font-size: {T.FS_SM};
    white-space: nowrap;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.topbar .right {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    min-width: 0;
}}

.topbar .brand small {{
    display: block;
    font-weight: {T.FW_SEMIBOLD};
    color: var(--mut);
    font-size: {T.FS_XS};
}}

/* Breadcrumbs */
.crumb {{
    font-size: {T.FS_SM};
    color: var(--mut);
    margin: 0.25rem 0 1rem;
}}

.crumb b {{
    color: var(--txt);
}}

/* KPI Cards */
.kpirow {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 0.5rem 0 1.5rem;
}}

.kpi {{
    background: var(--surface);
    color: var(--txt);
    border: 1px solid var(--border);
    border-top: 4px solid var(--gold);
    border-radius: var(--radius-lg);
    padding: 0.9rem 0.9rem;
    animation: fadeUp 0.5s ease both;
    min-height: 96px;
    transition: transform var(--transition-normal), box-shadow var(--transition-normal);
}}

.kpi:hover {{
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}}

.kpi.teal {{ border-top-color: var(--teal); }}
.kpi.success {{ border-top-color: var(--ok); }}
.kpi.warning {{ border-top-color: var(--warn); }}
.kpi.danger {{ border-top-color: var(--bad); }}
.kpi.info {{ border-top-color: var(--info); }}

.kpi .v {{
    font-size: {T.FS_3XL};
    font-weight: {T.FW_EXTRABOLD};
    line-height: {T.LH_TIGHT};
}}

.kpi .l {{
    font-size: {T.FS_XS};
    font-weight: {T.FW_BOLD};
    letter-spacing: 0.1em;
    color: var(--mut);
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}}

.kpi .s {{
    font-size: {T.FS_XS};
    color: var(--mut);
    margin-top: 0.25rem;
}}

/* Dark Panel */
.darkpanel {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    animation: fadeUp 0.4s ease both;
}}

/* Callout */
.callout {{
    border-left: 4px solid var(--gold);
    background: var(--surface);
    border-radius: 0 var(--radius-lg) var(--radius-lg) 0;
    padding: 0.75rem 1rem;
    margin: 0.75rem 0;
    font-size: {T.FS_BASE};
}}

.callout.teal {{ border-left-color: var(--teal); }}
.callout.success {{ border-left-color: var(--ok); }}
.callout.warning {{ border-left-color: var(--warn); }}
.callout.danger {{ border-left-color: var(--bad); }}
.callout.info {{ border-left-color: var(--info); }}

/* Badges */
.badge {{
    display: inline-block;
    font-size: {T.FS_XS};
    font-weight: {T.FW_BOLD};
    letter-spacing: 0.05em;
    padding: 0.25rem 0.625rem;
    border-radius: var(--radius-full);
    border: 1px solid var(--border);
}}

.badge.ok {{
    background: var(--ok-muted);
    color: #7EE2A0;
    border-color: rgba(63, 185, 80, 0.4);
}}

.badge.warn {{
    background: var(--warn-muted);
    color: #F0C75E;
    border-color: rgba(217, 165, 32, 0.4);
}}

.badge.bad {{
    background: var(--bad-muted);
    color: #F1948A;
    border-color: rgba(224, 72, 60, 0.4);
}}

.badge.info {{
    background: var(--info-muted);
    color: #7DD3FC;
    border-color: rgba(56, 189, 248, 0.4);
}}

/* Pipeline Tracker */
.pipe {{
    display: grid;
    grid-template-columns: repeat(8, minmax(0, 1fr));
    gap: 0.45rem;
    margin: 0.6rem 0 0.9rem;
}}
@media (max-width: 1100px) {{
    .pipe {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
}}
@media (max-width: 640px) {{
    .pipe {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}

/* Page header + pipeline status strip */
.pagehead .eyebrow {{
    font-size: {T.FS_XS};
    letter-spacing: 0.18em;
    color: var(--mut);
    font-weight: {T.FW_BOLD};
}}
.pagehead h1 {{
    margin: 0.15rem 0 0;
    font-size: 1.55rem;
    line-height: 1.15;
}}
.pageblurb {{
    color: var(--txt-secondary);
    font-size: {T.FS_BASE};
    margin-top: 0.35rem;
    max-width: 70rem;
}}
.pageid {{
    color: var(--mut);
    font-size: {T.FS_XS};
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}}
.statline {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem 0.9rem;
    align-items: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.5rem 0.8rem;
    margin: 0.6rem 0 0.9rem;
    font-size: {T.FS_SM};
    color: var(--txt-secondary);
}}
.statline b {{ color: var(--txt); }}
.statline .gold {{ color: var(--gold); font-weight: {T.FW_BOLD}; }}

.pstep {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--border-strong);
    border-radius: var(--radius-md);
    padding: 0.55rem 0.4rem;
    text-align: center;
    font-size: {T.FS_SM};
    transition: all var(--transition-normal);
}}

.pstep.done {{
    border-top-color: var(--ok);
}}

.pstep.active {{
    border-top-color: var(--gold);
    animation: pulseGold 1.5s ease-out 1;
}}

.pstep .n {{
    font-weight: {T.FW_BOLD};
    letter-spacing: 0.05em;
    font-size: {T.FS_XS};
    color: var(--mut);
}}

.pstep .t {{
    font-weight: {T.FW_SEMIBOLD};
    margin-top: 0.25rem;
    color: var(--txt);
}}

/* Archetype Card */
.arch {{
    background: var(--surface);
    color: var(--txt);
    border: 1px solid var(--border);
    border-top: 5px solid var(--gold);
    border-radius: var(--radius-xl);
    padding: 1.5rem;
    animation: fadeUp 0.5s ease both;
}}

.arch.c1 {{ border-top-color: var(--teal); }}
.arch.c2 {{ border-top-color: var(--bad); }}
.arch.c3 {{ border-top-color: var(--ok); }}
.arch.c4 {{ border-top-color: var(--warn); }}
.arch.c5 {{ border-top-color: var(--info); }}

/* Progress Bar */
.probar {{
    height: 6px;
    background: var(--border);
    border-radius: var(--radius-full);
    overflow: hidden;
    margin: 0.25rem 0 0.75rem;
}}

.probar > div {{
    height: 100%;
    background: linear-gradient(90deg, var(--teal), var(--gold));
    border-radius: var(--radius-full);
    transition: width var(--transition-slow);
}}

/* Method Card */
.mcard {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.25rem;
    margin: 0.5rem 0;
    transition: all var(--transition-fast);
}}

.mcard.sel {{
    border-color: var(--gold);
    box-shadow: var(--shadow-gold);
}}

/* Takeaway */
.takeaway {{
    background: #122033;
    border: 1px dashed var(--border-strong);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: {T.FS_BASE};
}}

/* Viva Box */
.viva {{
    background: #0B1B2A;
    border: 1px solid var(--teal);
    border-radius: var(--radius-md);
    padding: 0.875rem 1.25rem;
    margin: 0.75rem 0;
    font-size: {T.FS_BASE};
}}

/* Status Bar */
.sbar {{
    background: var(--navy);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.75rem 1rem;
    font-size: {T.FS_SM};
    color: var(--txt-secondary);
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
}}

.sbar b {{
    color: var(--gold);
}}

/* Horizontal Rule */
.rule {{
    height: 3px;
    background: linear-gradient(90deg, var(--gold), var(--teal));
    border: none;
    margin: 1rem 0 1.5rem;
    border-radius: 2px;
}}

/* Pill */
.pill {{
    display: inline-block;
    font-size: {T.FS_XS};
    font-weight: {T.FW_BOLD};
    letter-spacing: 0.05em;
    padding: 0.25rem 0.75rem;
    border-radius: var(--radius-full);
    border: 1px solid var(--border);
}}

.pill.ok {{
    background: var(--ok-muted);
    color: #7EE2A0;
    border-color: rgba(63, 185, 80, 0.4);
}}

.pill.warn {{
    background: var(--warn-muted);
    color: #F0C75E;
    border-color: rgba(217, 165, 32, 0.4);
}}

/* Archetype Detail Card */
.acard {{
    background: var(--surface);
    color: var(--txt);
    border: 1px solid var(--border);
    border-top: 5px solid var(--gold);
    border-radius: var(--radius-xl);
    padding: 1.5rem;
    animation: fadeUp 0.5s ease both;
    min-height: 320px;
}}

.acard h3 {{
    font-size: {T.FS_XS};
    letter-spacing: 0.15em;
    color: var(--mut);
    margin: 0;
    text-transform: uppercase;
}}

.acard h2 {{
    font-size: {T.FS_XL};
    margin: 0.5rem 0 1rem;
    font-weight: {T.FW_BOLD};
}}

.acard .big {{
    font-size: {T.FS_3XL};
    font-weight: {T.FW_EXTRABOLD};
}}

.acard .row {{
    display: flex;
    justify-content: space-between;
    font-size: {T.FS_SM};
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid var(--border);
}}

/* Demo Banner */
.demobanner {{
    background: #0B1B2A;
    border: 1px solid var(--gold);
    border-radius: var(--radius-md);
    padding: 0.875rem 1.25rem;
    margin: 0.75rem 0;
    font-size: {T.FS_BASE};
}}

/* App Chrome */
header[data-testid="stHeader"] {{
    background: var(--ink);
}}

/* Buttons */
div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button,
.stButton > button,
.stDownloadButton > button {{
    background: var(--gold) !important;
    color: var(--txt-on-gold) !important;
    border: none !important;
    font-weight: {T.FW_BOLD} !important;
    border-radius: var(--radius-md) !important;
    padding: 0.625rem 1.25rem !important;
    font-size: {T.FS_SM} !important;
    transition: all var(--transition-fast) !important;
}}

div[data-testid="stButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover,
.stButton > button:hover,
.stDownloadButton > button:hover {{
    background: var(--gold-hover) !important;
    color: var(--txt-on-gold) !important;
    box-shadow: var(--shadow-gold) !important;
}}

div[data-testid="stButton"] button:focus-visible,
div[data-testid="stDownloadButton"] button:focus-visible,
.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible {{
    outline: 2px solid var(--gold) !important;
    outline-offset: 2px !important;
}}

/* Secondary button variant */
.stButton > button[kind="secondary"] {{
    background: var(--surface-elevated) !important;
    color: var(--txt) !important;
    border: 1px solid var(--border) !important;
}}

.stButton > button[kind="secondary"]:hover {{
    background: var(--border) !important;
    box-shadow: none !important;
}}

/* Sidebar buttons */
section[data-testid="stSidebar"] div[data-testid="stButton"] button {{
    width: 100%;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {{
    background: rgba(255,255,255,0.04) !important;
    color: var(--txt) !important;
    border: 1px solid var(--border) !important;
    font-weight: 500 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"]:hover {{
    background: rgba(255,255,255,0.08) !important;
    border-color: var(--border-strong) !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {{
    font-weight: 700 !important;
}}

/* Empty State */
.empty {{
    background: #122033;
    border: 1px dashed var(--border-strong);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    margin: 1rem 0;
    text-align: center;
    color: var(--txt-secondary);
    font-size: {T.FS_BASE};
}}

.empty b {{
    color: var(--txt);
    letter-spacing: 0.02em;
}}

/* Panel */
.panel {{
    background: var(--surface);
    color: var(--txt);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}}

.panel b {{
    color: var(--txt);
}}

/* Spec Grid */
.spec {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.75rem;
    margin: 1rem 0;
}}

.spec > div {{
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
}}

.spec .k {{
    background: var(--navy);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.375rem 0.75rem;
    font-size: {T.FS_XS};
    font-weight: {T.FW_BOLD};
    letter-spacing: 0.1em;
    color: var(--mut);
    text-transform: uppercase;
}}

.spec .w {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.75rem;
    font-size: {T.FS_LG};
    font-weight: {T.FW_BOLD};
    color: var(--txt);
}}

/* Focus visible */
:focus-visible {{
    outline: 2px solid var(--gold);
    outline-offset: 2px;
}}

/* Dataframe styling */
[data-testid="stDataFrame"] {{
    border-radius: var(--radius-lg);
    overflow: hidden;
    border: 1px solid var(--border);
}}

[data-testid="stDataFrame"] div[data-testid="stTable"] {{
    background: var(--surface);
}}

/* Metric cards in Streamlit */
[data-testid="stMetric"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1rem;
}}

/* Selectbox, multiselect */
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {{
    background: var(--surface-elevated);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-md);
}}

[data-testid="stSelectbox"] > div:hover,
[data-testid="stMultiSelect"] > div:hover {{
    border-color: var(--gold);
}}

/* File uploader: integrated panel, not a white box */
[data-testid="stFileUploader"] {{
    background: var(--surface);
    border: 1px dashed var(--border-strong);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
}}
[data-testid="stFileUploader"] section {{
    background: transparent;
}}
[data-testid="stFileUploader"] button {{
    background: var(--surface-elevated) !important;
    color: var(--txt) !important;
    border: 1px solid var(--border-strong) !important;
}}

/* Text/number inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {{
    background: var(--surface-elevated) !important;
    color: var(--txt) !important;
    border: 1px solid var(--border-strong) !important;
}}

/* Primary action buttons */
div[data-testid="stButton"] button[kind="primary"] {{
    background: var(--gold) !important;
    color: #1a1207 !important;
    border: 1px solid var(--gold) !important;
    font-weight: 700 !important;
}}
div[data-testid="stButton"] button[kind="primary"]:hover {{
    background: var(--gold-hover) !important;
}}

/* Responsive safety */
.block-container {{
    min-width: 0;
}}
img, video, canvas {{
    max-width: 100%;
}}
@media (max-width: 768px) {{
    .topbar {{ padding: 0.6rem 0.75rem; }}
    .pagehead h1 {{ font-size: 1.25rem; }}
    .kpirow {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.6rem; }}
    .kpi .v {{ font-size: 1.25rem; }}
    section[data-testid="stSidebar"] .brand-lockup .b1 {{ font-size: 0.82rem; }}
}}

/* Slider */
[data-testid="stSlider"] {{
    color: var(--gold);
}}

/* Checkbox */
[data-testid="stCheckbox"] {{
    color: var(--txt);
}}

/* Radio */
[data-testid="stRadio"] > label {{
    color: var(--txt);
}}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {{
    gap: 0.5rem;
}}

[data-testid="stTabs"] [role="tab"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.5rem 1rem;
    color: var(--txt-secondary);
    transition: all var(--transition-fast);
}}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    background: var(--surface-elevated);
    border-color: var(--gold);
    color: var(--gold);
    font-weight: {T.FW_SEMIBOLD};
}}

/* Expander */
[data-testid="stExpander"] {{
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--surface) !important;
}}

[data-testid="stExpander"] summary {{
    color: var(--txt) !important;
    font-weight: {T.FW_SEMIBOLD} !important;
}}

/* Alerts */
.stAlert {{
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
}}

.stAlert[data-baseweb="notification"] {{
    background: var(--surface) !important;
}}

/* Progress */
.stProgress > div > div {{
    background: var(--gold) !important;
}}

/* Mobile Responsive */
@media (max-width: 1024px) {{
    .block-container {{
        padding-left: 1rem;
        padding-right: 1rem;
    }}
    .hero h1 {{
        font-size: {T.FS_4XL};
    }}
    .kpirow {{
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }}
}}

@media (max-width: 768px) {{
    .hero h1 {{
        font-size: {T.FS_3XL};
    }}
    .hero .sub {{
        font-size: {T.FS_LG};
    }}
    .kpirow {{
        grid-template-columns: 1fr 1fr;
    }}
    .topbar {{
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
    }}
    .pipe {{
        flex-direction: column;
    }}
    .pstep {{
        min-width: 100%;
    }}
}}

@media (max-width: 480px) {{
    .kpirow {{
        grid-template-columns: 1fr;
    }}
    .spec {{
        grid-template-columns: 1fr;
    }}
    .block-container {{
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }}
    .hero {{
        padding: 1.5rem 1rem;
    }}
}}
</style>
"""
    return css


def render_topbar(ready: bool) -> None:
    """Render the top navigation bar."""
    live = (
        "<span class='badge ok'>● LIVE ANALYSIS</span>" if ready
        else "<span class='badge warn'>○ AWAITING DATA</span>"
    )
    st.markdown(
        f"""
<div class='topbar'>
    <div class='brand'>
        AUTO MPG ANALYTICS
        <small>Automotive Fleet Intelligence</small>
    </div>
    <div>{live} &nbsp; Assessment 10 · Question 4</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_crumbs(*parts: str) -> None:
    """Render breadcrumb navigation."""
    crumb_html = " / ".join(
        (f"<b style='color:{T.TEXT_PRIMARY}'>{p}</b>" if i == len(parts) - 1 else p)
        for i, p in enumerate(parts)
    )
    st.markdown(f"<div class='crumb'>{crumb_html}</div>", unsafe_allow_html=True)


def render_info_callout(text: str, teal: bool = False, variant: str = "default") -> None:
    """Render an informational callout box."""
    variants = {
        "default": "callout",
        "teal": "callout teal",
        "success": "callout success",
        "warning": "callout warning",
        "danger": "callout danger",
        "info": "callout info",
    }
    cls = variants.get(variant, variants["teal" if teal else "default"])
    st.markdown(f"<div class='{cls}'>{text}</div>", unsafe_allow_html=True)


def render_method_card(name: str, desc: str, selected: bool, note: str = "") -> None:
    """Render a method selection card."""
    cls = "mcard sel" if selected else "mcard"
    note_html = f"<br><span style='color:{T.TEXT_MUTED};font-size:{T.FS_XS}'>{note}</span>" if note else ""
    st.markdown(
        f"<div class='{cls}'><b>{name}</b><br>{desc}{note_html}</div>",
        unsafe_allow_html=True,
    )


def render_insight_card(title: str, text: str, evidence: str = "") -> None:
    """Render an insight card with evidence."""
    evidence_html = (
        f"<br><span style='color:{T.TEXT_MUTED};font-size:{T.FS_XS}'>{evidence}</span>"
        if evidence
        else ""
    )
    st.markdown(
        f"<div class='mcard'><b>{title}</b><br>{text}{evidence_html}</div>",
        unsafe_allow_html=True,
    )


def render_section_header(title: str, desc: str = "") -> None:
    """Render a section header with optional description."""
    st.markdown(f"**{title}**")
    if desc:
        st.caption(desc)


def render_takeaway(text: str) -> None:
    """Render a key takeaway box."""
    st.markdown(
        f"<div class='takeaway'><b>KEY TAKEAWAY</b><br>{text}</div>",
        unsafe_allow_html=True,
    )


def viva_box(text: str) -> None:
    """Render a viva-style explanation box."""
    st.markdown(
        f"<div class='viva'><b>Explain like a viva</b><br>{text}</div>",
        unsafe_allow_html=True,
    )


def render_kpi_row(cards: list) -> None:
    """Render a row of KPI cards."""
    html = ["<div class='kpirow'>"]
    for i, (lbl, val, sub, teal) in enumerate(cards):
        cls = "kpi teal" if teal else "kpi"
        html.append(
            f"<div class='{cls}' style='animation-delay:{i * 0.07}s'>"
            f"<div class='l'>{lbl}</div>"
            f"<div class='v'>{val}</div>"
            f"<div class='s'>{sub}</div>"
            f"</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def empty_state(title: str, body: str, hint: str = "") -> None:
    """Render an empty state."""
    hint_html = f"<br><span style='font-size:{T.FS_XS}'>{hint}</span>" if hint else ""
    st.markdown(
        f"<div class='empty'><b>{title}</b><br>{body}{hint_html}</div>",
        unsafe_allow_html=True,
    )


def render_pipeline_tracker(steps: list) -> None:
    """Render the pipeline tracker."""
    st.markdown(
        "<div class='pipe'>"
        + "".join(
            f"<div class='pstep {stt}'><div class='n'>{n}</div>"
            f"<div class='t'>{t}</div>"
            f"<div class='n'>{'DONE' if stt == 'done' else stt.upper()}</div></div>"
            for n, t, stt in steps
        )
        + "</div>",
        unsafe_allow_html=True,
    )