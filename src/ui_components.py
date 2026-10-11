"""Reusable presentation primitives (no science).

Text arguments are HTML-escaped by default; only parameters named ``*_html``
(or ``html=True``) accept trusted markup, so data such as uploaded vehicle
names can never inject markup into the page.
"""
import html as _html

import streamlit as st

from src import theme as T


def esc(value):
    return _html.escape(str(value), quote=True)


def md(markup):
    st.markdown(markup, unsafe_allow_html=True)


_CSS = """
:root{--bg:#0F172A;--navy:#1E293B;--surface:#16233B;--surface-2:#1E2D45;--border:#334155;
--border-strong:#475569;--text:#F8FAFC;--text-2:#CBD5E1;--muted:#94A3B8;--gold:#D4A72C;
--teal:#2A9D8F;--teal-text:#5EC4B6;--success:#3FB950;--warning:#D9A520;--danger:#E0483C;--danger-text:#F07B71;
--on-gold:#1A1207;--r-sm:6px;--r:10px;--fs-xs:.72rem;--fs-sm:.8rem;--fs-base:.95rem;
--fs-lg:1.05rem;--fs-xl:1.3rem;--fs-2xl:1.75rem;--mono:__MONO__}
.stApp{background:var(--bg)}
[data-testid="stHeader"]{background:var(--bg)}
[data-testid="stMainBlockContainer"],.block-container{max-width:1320px;padding-top:4rem;padding-bottom:3rem}
[data-testid="stMarkdownContainer"] p{max-width:78ch}
/* Streamlit pulls markdown blocks up by 1rem; custom components must not overlap the next element */
[data-testid="stMarkdownContainer"]:has(> [class*="ui-"]){margin-bottom:0}

/* Sidebar navigation: compact list rows; active row = raised surface + gold marker */
[data-testid="stSidebarUserContent"]{padding-top:.5rem}
.ui-navhead{font-size:var(--fs-xs);font-weight:700;letter-spacing:.14em;color:var(--muted);
text-transform:uppercase;margin:.9rem 0 .1rem .25rem}
.st-key-nav{gap:.15rem}
.st-key-nav button,.st-key-nav button[kind="secondary"]{border:0;background:transparent;color:var(--text-2);
min-height:2.1rem;padding:.25rem .65rem}
.st-key-nav button > div{justify-content:flex-start}
.st-key-nav button p{font-size:.9rem;font-weight:500}
.st-key-nav button:hover,.st-key-nav button[kind="secondary"]:hover{background:var(--surface-2);color:var(--text)}
.st-key-nav button[kind="primary"]{background:var(--surface-2);color:var(--text);box-shadow:inset 3px 0 0 var(--gold)}
.st-key-nav button[kind="primary"] p{font-weight:650}
.st-key-nav button[kind="primary"]:hover{background:var(--surface-2)}
.ui-brand{padding:.25rem .25rem .6rem;border-bottom:1px solid var(--border)}
.ui-brand-name{font-weight:800;letter-spacing:.08em;font-size:.82rem;color:var(--gold)}
.ui-brand-sub{font-size:var(--fs-sm);color:var(--text-2);margin-top:.1rem}

/* Buttons: gold primary with dark text (8.3:1); quiet secondary. Streamlit darkens gold on
   hover/focus (dark text drops to 4.3:1), so keep the gold and signal state with a light border. */
button[kind="primary"]{color:var(--on-gold)}
button[kind="primary"]:is(:hover,:focus-visible):not(:disabled){background-color:var(--gold);border-color:var(--text)}
button[kind="primary"] p{font-weight:650}
[data-testid="stMultiSelect"] [data-tag]{color:var(--on-gold)}
button[kind="secondary"]{background:var(--surface-2);border-color:var(--border)}
button:focus-visible,[role="tab"]:focus-visible,a:focus-visible{outline:2px solid var(--gold);outline-offset:2px}

/* Top bar */
.ui-topbar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:.5rem 1rem;
padding:.55rem .9rem;border:1px solid var(--border);border-radius:var(--r);background:var(--navy);margin-bottom:1rem}
.ui-topbar-brand{font-weight:800;letter-spacing:.08em;font-size:.78rem;color:var(--gold)}
.ui-topbar-sub{font-size:var(--fs-sm);color:var(--text-2)}
.ui-topbar-right{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem .75rem;font-size:var(--fs-sm);color:var(--muted)}

/* Pills: always text + glyph, never colour alone */
.ui-pill{display:inline-flex;align-items:center;gap:.4rem;font-size:var(--fs-xs);font-weight:700;letter-spacing:.06em;
text-transform:uppercase;padding:.2rem .6rem;border-radius:999px;border:1px solid var(--border);color:var(--text-2);
background:var(--surface);white-space:nowrap}
.ui-pill i{width:.5rem;height:.5rem;border-radius:50%;background:var(--muted);display:inline-block}
.ui-pill.t-success{color:var(--success);border-color:rgba(63,185,80,.45)}.ui-pill.t-success i{background:var(--success)}
.ui-pill.t-teal{color:var(--teal-text);border-color:rgba(42,157,143,.5)}.ui-pill.t-teal i{background:var(--teal)}
.ui-pill.t-gold{color:var(--gold);border-color:rgba(212,167,44,.5)}.ui-pill.t-gold i{background:var(--gold)}
.ui-pill.t-warning{color:var(--warning);border-color:rgba(217,165,32,.5)}.ui-pill.t-warning i{background:var(--warning)}
.ui-pill.t-danger{color:var(--danger-text);border-color:rgba(224,72,60,.55)}.ui-pill.t-danger i{background:var(--danger)}

/* Page and section headers */
.ui-crumbs{font-size:var(--fs-sm);color:var(--muted);margin-bottom:.35rem}
.ui-crumbs b{color:var(--text-2);font-weight:600}
.ui-title{font-size:var(--fs-2xl);font-weight:700;line-height:1.15;margin:0;color:var(--text);letter-spacing:-.01em}
.ui-sub{font-size:1rem;color:var(--text-2);margin-top:.35rem;max-width:72ch}
.ui-section{margin-top:.6rem}
.ui-eyebrow{font-size:var(--fs-xs);font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.ui-section-title{font-size:var(--fs-lg);font-weight:650;color:var(--text);margin-top:.1rem}
.ui-section-sub{font-size:.88rem;color:var(--muted);margin-top:.15rem;max-width:78ch}

/* Hero (Overview): a quiet technical grid, no imagery */
.ui-hero{border:1px solid var(--border);border-radius:var(--r);padding:1.4rem 1.5rem;
background-color:var(--navy);background-image:linear-gradient(rgba(148,163,184,.06) 1px,transparent 1px),
linear-gradient(90deg,rgba(148,163,184,.06) 1px,transparent 1px);background-size:28px 28px}
.ui-hero .ui-title{font-size:2rem;margin-top:.35rem}
.ui-hero-meta{font-size:var(--fs-sm);color:var(--gold);font-weight:600;letter-spacing:.04em;margin-top:.45rem}
.ui-hero-desc{color:var(--text-2);margin-top:.6rem;max-width:60ch;line-height:1.55}

/* KPI cards */
.ui-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,170px),1fr));gap:.75rem}
.ui-kpi{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:.8rem .95rem;min-width:0}
.ui-kpi.t-success{border-left:3px solid var(--success)}.ui-kpi.t-warning{border-left:3px solid var(--warning)}
.ui-kpi.t-danger{border-left:3px solid var(--danger)}.ui-kpi.t-gold{border-left:3px solid var(--gold)}
.ui-kpi.t-teal{border-left:3px solid var(--teal)}
.ui-kpi-label{font-size:var(--fs-xs);font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.ui-kpi-value{font-size:1.6rem;font-weight:700;line-height:1.2;margin-top:.2rem;color:var(--text);
font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.ui-kpi-sub{font-size:var(--fs-sm);color:var(--muted);margin-top:.15rem}

/* Generic surfaces */
.ui-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:.9rem 1rem}
.ui-callout{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--muted);
border-radius:var(--r-sm);padding:.7rem .95rem;font-size:.92rem;color:var(--text-2);line-height:1.5}
.ui-callout b,.ui-callout strong{color:var(--text)}
.ui-callout-title{font-size:var(--fs-xs);font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--text);margin-bottom:.25rem}
.ui-callout.t-teal{border-left-color:var(--teal)}.ui-callout.t-gold{border-left-color:var(--gold)}
.ui-callout.t-success{border-left-color:var(--success)}.ui-callout.t-warning{border-left-color:var(--warning)}
.ui-callout.t-danger{border-left-color:var(--danger)}
.ui-callout ul{margin:.3rem 0 0;padding-left:1.1rem}.ui-callout li{margin:.3rem 0}
.ui-callout .ui-next{color:var(--muted);font-size:var(--fs-sm)}
.ui-empty{border:1px dashed var(--border-strong);border-radius:var(--r);padding:1.25rem 1.1rem;text-align:center;background:var(--surface)}
.ui-empty-title{font-size:var(--fs-sm);font-weight:700;letter-spacing:.1em;color:var(--text)}
.ui-empty-body{color:var(--text-2);margin-top:.3rem}
.ui-empty-next{color:var(--muted);font-size:var(--fs-sm);margin-top:.35rem}
.ui-prov{font-size:var(--fs-sm);color:var(--muted);display:flex;flex-wrap:wrap;gap:.25rem .6rem;align-items:center}
.ui-prov b{color:var(--text-2);font-weight:600}
.ui-takeaway{border:1px solid var(--border);border-left:3px solid var(--gold);border-radius:var(--r-sm);
padding:.75rem .95rem;background:var(--surface);color:var(--text-2)}
.ui-takeaway-tag{font-size:var(--fs-xs);font-weight:700;letter-spacing:.12em;color:var(--gold);margin-bottom:.2rem}
.ui-viva{border:1px solid rgba(42,157,143,.5);border-radius:var(--r-sm);padding:.65rem .9rem;background:var(--surface);
color:var(--text-2);font-size:.9rem;line-height:1.5}
.ui-viva-tag{font-size:var(--fs-xs);font-weight:700;letter-spacing:.12em;color:var(--teal-text);margin-bottom:.15rem}

/* Pipeline stages */
.ui-pipe{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:.5rem}
.ui-stage{background:var(--surface);border:1px solid var(--border);border-top:3px solid var(--border-strong);
border-radius:var(--r-sm);padding:.55rem .6rem;min-width:0}
.ui-stage-top{display:flex;justify-content:space-between;gap:.25rem;font-size:var(--fs-xs);color:var(--muted);font-weight:700;letter-spacing:.05em}
.ui-stage-label{font-weight:650;color:var(--text);margin-top:.2rem;font-size:.92rem}
.ui-stage-detail{font-size:var(--fs-xs);color:var(--muted);margin-top:.1rem;overflow-wrap:anywhere}
.ui-stage.s-complete{border-top-color:var(--success)}.ui-stage.s-complete .ui-stage-state{color:var(--success)}
.ui-stage.s-ready{border-top-color:var(--gold)}.ui-stage.s-ready .ui-stage-state{color:var(--gold)}
.ui-stage.s-available{border-top-color:var(--teal)}.ui-stage.s-available .ui-stage-state{color:var(--teal-text)}
.ui-stage.s-blocked .ui-stage-label{color:var(--muted)}
@media (max-width:1100px){.ui-pipe{grid-template-columns:repeat(4,minmax(0,1fr))}}
@media (max-width:560px){.ui-pipe{grid-template-columns:repeat(2,minmax(0,1fr))}}

/* Step header (vertical process, e.g. preprocessing) */
.ui-step{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem .75rem}
.ui-step-n{width:1.7rem;height:1.7rem;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;
font-size:var(--fs-sm);font-weight:700;border:1px solid var(--border-strong);color:var(--text-2)}
.ui-step-title{font-weight:650;color:var(--text);letter-spacing:.02em}
.ui-step span{min-width:0;overflow-wrap:anywhere}
.ui-step-body{color:var(--text-2);font-size:.9rem;margin-top:.35rem}
.ui-arrow{text-align:center;color:var(--muted);font-size:1rem;line-height:1;margin:-.35rem 0}

/* Cluster identity */
.ui-swatch{display:inline-block;width:.7rem;height:.7rem;border-radius:2px;background:var(--c,var(--muted));flex:none}
.ui-chip{display:inline-flex;align-items:center;gap:.4rem;font-size:var(--fs-sm);font-weight:600;color:var(--text);
border:1px solid var(--border);border-radius:999px;padding:.12rem .55rem;background:var(--surface);max-width:100%}
.ui-ccard{background:var(--surface);border:1px solid var(--border);border-top:3px solid var(--c,var(--muted));
border-radius:var(--r);padding:.9rem 1rem;height:100%}
.ui-ccard-head{display:flex;justify-content:space-between;align-items:center;gap:.5rem;font-size:var(--fs-xs);
font-weight:700;letter-spacing:.08em;color:var(--muted)}
.ui-ccard-head span:first-child{display:inline-flex;align-items:center;gap:.4rem}
.ui-ccard-title{font-size:1.05rem;font-weight:700;color:var(--text);margin-top:.35rem}
.ui-ccard-headline{font-size:1.5rem;font-weight:700;color:var(--text);font-variant-numeric:tabular-nums;margin-top:.25rem}
.ui-ccard-headline small{font-size:.8rem;color:var(--muted);font-weight:600;margin-left:.2rem}
.ui-ccard dl{margin:.55rem 0 0;display:grid;gap:.1rem}
.ui-ccard dl div{display:flex;justify-content:space-between;gap:.5rem;border-top:1px solid var(--border);padding:.3rem 0;font-size:.86rem}
.ui-ccard dt{color:var(--muted)}.ui-ccard dd{margin:0;color:var(--text);font-variant-numeric:tabular-nums;text-align:right}
.ui-ccard-foot{font-size:var(--fs-sm);color:var(--text-2);margin-top:.5rem;line-height:1.45}

/* Spec grid, key-values, proportion bars */
.ui-spec{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,150px),1fr));gap:.5rem}
.ui-spec div{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-sm);padding:.5rem .7rem;min-width:0}
.ui-spec span{display:block;font-size:var(--fs-xs);font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.ui-spec b{display:block;font-size:1.05rem;color:var(--text);font-variant-numeric:tabular-nums;margin-top:.1rem;overflow-wrap:anywhere}
.ui-kv{display:grid;gap:0}
.ui-kv div{display:flex;justify-content:space-between;gap:.75rem;padding:.35rem 0;border-bottom:1px solid var(--border);font-size:.9rem}
.ui-kv div:last-child{border-bottom:0}
.ui-kv span{color:var(--muted)}.ui-kv b{color:var(--text);font-weight:600;font-variant-numeric:tabular-nums;text-align:right}
.ui-bars{display:grid;gap:.45rem}
.ui-bar{display:grid;grid-template-columns:minmax(4.5rem,auto) 1fr auto;align-items:center;gap:.6rem;font-size:.86rem}
.ui-bar-label{color:var(--text-2)}
.ui-bar-track{height:.55rem;background:var(--surface-2);border-radius:2px;overflow:hidden}
.ui-bar-fill{display:block;height:100%;border-radius:2px;background:var(--c,var(--teal))}
.ui-bar-value{color:var(--text);font-variant-numeric:tabular-nums;font-weight:600;min-width:3.5rem;text-align:right}

/* Insight cards */
.ui-insight{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:.85rem 1rem;height:100%}
.ui-insight-kind{font-size:var(--fs-xs);font-weight:700;letter-spacing:.12em;color:var(--teal)}
.ui-insight-kind.k-model{color:var(--gold)}
.ui-insight-title{font-weight:650;color:var(--text);margin-top:.2rem}
.ui-insight-text{color:var(--text-2);margin-top:.25rem;font-size:.92rem;line-height:1.5}
.ui-insight-evidence{color:var(--muted);font-size:var(--fs-sm);margin-top:.4rem;font-variant-numeric:tabular-nums}

/* Demo bar */
.ui-demo{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem .8rem;border:1px solid rgba(212,167,44,.55);
border-radius:var(--r-sm);padding:.55rem .85rem;background:var(--surface);color:var(--text-2);font-size:.9rem}
.ui-demo-tag{font-size:var(--fs-xs);font-weight:800;letter-spacing:.12em;color:var(--gold)}
.ui-demo b{color:var(--text)}

/* Streamlit surfaces */
[data-testid="stExpander"] details{border-color:var(--border);border-radius:var(--r)}
[data-testid="stExpander"] summary p{font-weight:600}
[data-testid="stTabs"] [role="tablist"]{gap:1.4rem;flex-wrap:wrap}
[data-testid="stTabs"] button[role="tab"] p{font-weight:600;font-size:.95rem}
[class*="st-key-mcard_on"]{border-color:var(--gold)!important}
[data-testid="stCaptionContainer"]{color:var(--muted)}

@media (max-width:640px){
 [data-testid="stMainBlockContainer"],.block-container{padding-left:1rem;padding-right:1rem;padding-top:3.75rem}
 .ui-title{font-size:1.4rem}.ui-hero .ui-title{font-size:1.5rem}.ui-hero{padding:1.1rem 1rem}
 .ui-kpi-value{font-size:1.35rem}.ui-topbar{padding:.5rem .7rem}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
"""

_PRESENTATION_CSS = """
.stApp{font-size:17px}
.ui-crumbs{display:none}
.ui-title{font-size:2.1rem}.ui-sub{font-size:1.1rem}
.ui-kpi-value{font-size:2rem}.ui-kpi-label{font-size:.8rem}
.ui-section-title{font-size:1.2rem}
[class*="st-key-tech_"]{display:none}
"""


def inject_css(presentation=False):
    css = _CSS.replace("__MONO__", T.MONO) + (_PRESENTATION_CSS if presentation else "")
    st.html(f"<style>{css}</style>")


# ---- shell -----------------------------------------------------------------
def pill(text, tone="neutral"):
    return f"<span class='ui-pill t-{esc(tone)}'><i aria-hidden='true'></i>{esc(text)}</span>"


def top_bar(status, extras=()):
    label, tone, detail = status
    chips = "".join(pill(x, "gold") for x in extras)
    md("<div class='ui-topbar' role='region' aria-label='Analysis status'>"
       "<div><div class='ui-topbar-brand'>AUTO MPG ANALYTICS</div>"
       "<div class='ui-topbar-sub'>Automotive Fleet Intelligence</div></div>"
       f"<div class='ui-topbar-right'>{chips}<span>Assessment 10 · Question 4</span>"
       f"<span title='{esc(detail)}'>{pill(label, tone)}</span>"
       f"<span class='ui-topbar-sub'>{esc(detail)}</span></div></div>")


def page_header(title, subtitle="", crumbs=()):
    trail = ""
    if crumbs:
        parts = [esc(c) for c in crumbs]
        parts[-1] = f"<b>{parts[-1]}</b>"
        trail = f"<div class='ui-crumbs'>{' / '.join(parts)}</div>"
    sub = f"<div class='ui-sub'>{esc(subtitle)}</div>" if subtitle else ""
    md(f"<header class='ui-head'>{trail}<div class='ui-title' role='heading' aria-level='1'>"
       f"{esc(title)}</div>{sub}</header>")


def section(title, subtitle="", eyebrow=""):
    eb = f"<div class='ui-eyebrow'>{esc(eyebrow)}</div>" if eyebrow else ""
    sub = f"<div class='ui-section-sub'>{esc(subtitle)}</div>" if subtitle else ""
    md(f"<div class='ui-section'>{eb}<div class='ui-section-title' role='heading' aria-level='2'>"
       f"{esc(title)}</div>{sub}</div>")


def hero(eyebrow, title, meta, description):
    md(f"<div class='ui-hero'><div class='ui-eyebrow'>{esc(eyebrow)}</div>"
       f"<div class='ui-title' role='heading' aria-level='1'>{esc(title)}</div>"
       f"<div class='ui-hero-meta'>{esc(meta)}</div><div class='ui-hero-desc'>{esc(description)}</div></div>")


# ---- content primitives ------------------------------------------------------
def kpis(cards):
    """cards: dicts with label, value, optional sub and tone. Keep definitions in visible text."""
    out = []
    for c in cards:
        tone = c.get("tone")
        cls = f"ui-kpi t-{esc(tone)}" if tone else "ui-kpi"
        sub = f"<div class='ui-kpi-sub'>{esc(c['sub'])}</div>" if c.get("sub") else ""
        out.append(f"<div class='{cls}'><div class='ui-kpi-label'>{esc(c['label'])}</div>"
                   f"<div class='ui-kpi-value'>{esc(c['value'])}</div>{sub}</div>")
    md(f"<div class='ui-kpis'>{''.join(out)}</div>")


def empty_state(title, body, next_step=""):
    nxt = f"<div class='ui-empty-next'>Next step: {esc(next_step)}</div>" if next_step else ""
    md(f"<div class='ui-empty' role='status'><div class='ui-empty-title'>{esc(title)}</div>"
       f"<div class='ui-empty-body'>{esc(body)}</div>{nxt}</div>")


def callout(text, title="", tone="neutral", html=False):
    body = text if html else esc(text)
    head = f"<div class='ui-callout-title'>{esc(title)}</div>" if title else ""
    md(f"<div class='ui-callout t-{esc(tone)}'>{head}{body}</div>")


def provenance(text, label="Showing"):
    md(f"<div class='ui-prov'><span aria-hidden='true'>◆</span><span>{esc(label)}</span><b>{esc(text)}</b></div>")


def takeaway(text, html=False):
    md(f"<div class='ui-takeaway'><div class='ui-takeaway-tag'>KEY TAKEAWAY</div>{text if html else esc(text)}</div>")


def viva(text):
    """Viva explanation, shown only while the sidebar 'Viva explanations' toggle is on."""
    if st.session_state.get("viva_mode"):
        md(f"<div class='ui-viva'><div class='ui-viva-tag'>VIVA NOTE</div>{esc(text)}</div>")


_STATE_TEXT = {"complete": "✓ Complete", "ready": "● Ready", "available": "○ Available", "blocked": "— Waiting"}


def pipeline(stages):
    items = "".join(
        f"<div class='ui-stage s-{esc(x['state'])}' role='listitem'>"
        f"<div class='ui-stage-top'><span>{esc(x['n'])}</span>"
        f"<span class='ui-stage-state'>{_STATE_TEXT.get(x['state'], esc(x['state']))}</span></div>"
        f"<div class='ui-stage-label'>{esc(x['label'])}</div>"
        f"<div class='ui-stage-detail'>{esc(x['detail'])}</div></div>" for x in stages)
    md(f"<div class='ui-pipe' role='list' aria-label='Analysis pipeline'>{items}</div>")


_STEP_PILL = {"complete": ("Complete", "success"), "ready": ("Ready", "gold"),
              "available": ("Available", "teal"), "blocked": ("Waiting", "neutral")}


def step_header(n, title, state, detail=""):
    text, tone = _STEP_PILL.get(state, (state, "neutral"))
    body = f"<div class='ui-step-body'>{esc(detail)}</div>" if detail else ""
    md(f"<div class='ui-step'><span class='ui-step-n'>{esc(n)}</span>"
       f"<span class='ui-step-title'>{esc(title)}</span>{pill(text, tone)}</div>{body}")


def arrow():
    md("<div class='ui-arrow' aria-hidden='true'>↓</div>")


def chip(i, name=""):
    tail = f" · {esc(name)}" if name else ""
    return (f"<span class='ui-chip'><span class='ui-swatch' style='--c:{T.cluster_color(i)}'></span>"
            f"Cluster {int(i)}{tail}</span>")


def cluster_card(i, title, headline, unit, rows, count="", foot=""):
    """rows: list of (label, value text). Top rule and swatch carry the cluster colour."""
    dl = "".join(f"<div><dt>{esc(a)}</dt><dd>{esc(b)}</dd></div>" for a, b in rows)
    ft = f"<div class='ui-ccard-foot'>{esc(foot)}</div>" if foot else ""
    md(f"<div class='ui-ccard' style='--c:{T.cluster_color(i)}'><div class='ui-ccard-head'>"
       f"<span><span class='ui-swatch'></span>CLUSTER {int(i)}</span><span>{esc(count)}</span></div>"
       f"<div class='ui-ccard-title'>{esc(title)}</div>"
       f"<div class='ui-ccard-headline'>{esc(headline)}<small>{esc(unit)}</small></div>"
       f"<dl>{dl}</dl>{ft}</div>")


def spec_grid(items):
    """items: list of (label, value text)."""
    md("<div class='ui-spec'>" + "".join(f"<div><span>{esc(a)}</span><b>{esc(b)}</b></div>"
                                         for a, b in items) + "</div>")


def keyvals(rows, card=True):
    inner = "".join(f"<div><span>{esc(a)}</span><b>{esc(b)}</b></div>" for a, b in rows)
    md(f"<div class='{'ui-card ' if card else ''}ui-kv'>{inner}</div>")


def bars(rows, label=""):
    """Inline proportion bars. rows: (label, fraction 0-1, value text, colour or None)."""
    out = []
    for name, frac, val, color in rows:
        pct = max(0.0, min(1.0, float(frac))) * 100
        c = f"--c:{color};" if color else ""
        out.append(f"<div class='ui-bar'><span class='ui-bar-label'>{esc(name)}</span>"
                   f"<span class='ui-bar-track'><span class='ui-bar-fill' style='{c}width:{pct:.2f}%'></span></span>"
                   f"<span class='ui-bar-value'>{esc(val)}</span></div>")
    aria = f" aria-label='{esc(label)}'" if label else ""
    md(f"<div class='ui-bars' role='group'{aria}>{''.join(out)}</div>")


def insight(title, text, evidence="", kind="DATA EVIDENCE"):
    k = " k-model" if kind.upper().startswith("MODEL") else ""
    ev = f"<div class='ui-insight-evidence'>{esc(evidence)}</div>" if evidence else ""
    md(f"<div class='ui-insight'><div class='ui-insight-kind{k}'>{esc(kind)}</div>"
       f"<div class='ui-insight-title'>{esc(title)}</div><div class='ui-insight-text'>{esc(text)}</div>{ev}</div>")


def demo_bar(step, total, title, instruction, done):
    state = "" if done is None else pill("done" if done else "not yet done", "success" if done else "neutral")
    md(f"<div class='ui-demo' role='status'><span class='ui-demo-tag'>DEMO MODE · STEP {step} OF {total}</span>"
       f"<b>{esc(title)}</b><span>{esc(instruction)}</span>{state}</div>")


def flash():
    """Show (and clear) a one-shot confirmation queued by a callback."""
    msg = st.session_state.get("flash")
    if msg:
        st.session_state["flash"] = None
        st.toast(msg[0] if isinstance(msg, tuple) else msg, icon=":material/check_circle:")
