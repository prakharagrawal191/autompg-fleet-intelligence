"""Automotive Fleet Profiling & Efficiency Clustering - Streamlit shell.

Navigation, global status and modes live here; each page lives in src/views/
and every number comes from src/auto_mpg_utils.py through src/state.py.
M.Tech ML Lab - Question 4, Assessment 10 - 26MML0003.
"""
import importlib

import streamlit as st

from src import state as S
from src import ui_components as UI

st.set_page_config(page_title="Auto MPG Fleet Intelligence", page_icon=":material/directions_car:",
                   layout="wide", initial_sidebar_state="auto")
S.init()
UI.inject_css(st.session_state["presentation_mode"])

NAV = (
    ("Analysis", (("Overview", ":material/space_dashboard:", "overview"),
                  ("Dataset", ":material/table_view:", "dataset"),
                  ("Preprocessing", ":material/tune:", "preprocessing"),
                  ("EDA Explorer", ":material/query_stats:", "eda"))),
    ("Modeling", (("Clustering Lab", ":material/bubble_chart:", "clustering"),
                  ("Cluster Quality", ":material/fact_check:", "evaluation"),
                  ("PCA Explorer", ":material/scatter_plot:", "pca"))),
    ("Interpretation", (("Fleet Archetypes", ":material/directions_car:", "archetypes"),
                        ("Vehicle Lookup", ":material/search:", "lookup"),
                        ("Insights", ":material/lightbulb:", "insights"))),
    ("System", (("Methodology", ":material/account_tree:", "methodology"),
                ("Exports", ":material/download:", "exports"),
                ("About", ":material/info:", "about"))),
)
MODULES = {name: mod for _, items in NAV for name, _, mod in items}
page = st.session_state["page"] if st.session_state["page"] in MODULES else "Overview"


def _demo_move(delta):
    s = st.session_state
    s["demo_step"] = max(0, min(len(S.DEMO_STEPS) - 1, int(s["demo_step"]) + delta))
    S.goto(S.DEMO_STEPS[s["demo_step"]][1])


with st.sidebar:
    UI.md("<div class='ui-brand'><div class='ui-brand-name'>NAVIGATION</div>"
          "<div class='ui-brand-sub'>M.Tech ML Lab · Q4 · 26MML0003</div></div>")
    with st.container(key="nav"):
        for group, items in NAV:
            UI.md(f"<div class='ui-navhead'>{group}</div>")
            for name, icon, _ in items:
                st.button(name, key=f"navbtn_{name}", icon=icon, width="stretch",
                          type="primary" if name == page else "secondary",
                          on_click=S.goto, args=(name,))
    UI.md("<div class='ui-navhead'>Modes</div>")
    st.toggle("Demo Mode", key="demo_mode",
              help="Guided step-by-step walkthrough. It never runs or changes any computation.")
    st.toggle("Presentation Mode", key="presentation_mode",
              help="Larger type and charts for projectors; hides technical-detail panels.")
    st.toggle("Viva explanations", key="viva_mode",
              help="Shows concise viva notes describing the actual implementation.")
    st.button("Reset Analysis", width="stretch", icon=":material/restart_alt:",
              on_click=S.reset_analysis,
              help="Clears the loaded data and every result. Project files are never modified.")

modes = [m for m, on in (("Demo", st.session_state["demo_mode"]),
                         ("Presentation", st.session_state["presentation_mode"])) if on]
UI.top_bar(S.status(), modes)
UI.flash()

if st.session_state["demo_mode"] and page != "Overview":
    i = int(st.session_state["demo_step"])
    title, target, instruction = S.DEMO_STEPS[i]
    UI.demo_bar(i + 1, len(S.DEMO_STEPS), title, instruction, S.demo_step_done(i))
    with st.container(horizontal=True):  # explicit keys: pages render gate buttons with the same labels
        st.button("Previous", key="demo_prev", icon=":material/chevron_left:", disabled=i == 0,
                  on_click=_demo_move, args=(-1,))
        st.button("Next", key="demo_next", icon=":material/chevron_right:", icon_position="right",
                  type="primary", disabled=i == len(S.DEMO_STEPS) - 1, on_click=_demo_move, args=(1,))
        if target != page:
            st.button(f"Open {target}", key="demo_open", on_click=S.goto, args=(target,))

importlib.import_module(f"src.views.{MODULES[page]}").render()
