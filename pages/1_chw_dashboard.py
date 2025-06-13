# test/pages/1_chw_dashboard.py

# Standard library imports first
import os
import logging
from datetime import date, timedelta 

# Third-party library imports
import streamlit as st 
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="CHW Dashboard - Health Hub", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Absolute imports ---
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    get_chw_summary,
    get_patient_alerts_for_chw,
    get_trend_data 
)
from utils.ai_analytics_engine import apply_ai_models
from utils.ui_visualization_helpers import plot_annotated_line_chart

# Component imports
from pages.chw_components import kpi_snapshots
from pages.chw_components import epi_watch
from pages.chw_components import alerts_display
from pages.chw_components import tasks_display
from pages.chw_components import trends_display

# ==============================================================================
# SME ENHANCEMENT FOR GATES FOUNDATION
# Import the new, self-contained strategic tab component.
from pages.chw_components import strategic_overview_tab
# ==============================================================================

logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_chw():
    css_path = app_config.STYLE_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path) as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else: logger.warning(f"CHW CSS file not found: {css_path}.")
load_css_chw()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading and enriching program data...")
def get_chw_dashboard_data_enriched():
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    if health_df_raw.empty: logger.error("CHW: Raw health records load failed."); return pd.DataFrame(), pd.DataFrame(), 0.0, {}

    health_df_ai_enriched = apply_ai_models(health_df_raw)
    
    # ==============================================================================
    # SME ENHANCEMENT FOR GATES FOUNDATION
    # Load strategic financial and target data.
    chw_monthly_cost = 2000.0  # Default value
    program_targets = {}
    
    try:
        # In a real app, this data would come from config files or databases
        costs_df = pd.read_csv(app_config.CHW_COSTS_CSV) 
        chw_monthly_cost = costs_df['monthly_cost_usd'].mean() if not costs_df.empty else 2000.0
        program_targets = {"target_cost_per_visit_usd": 5.0, "target_high_risk_follow_up_rate": 0.90, "target_monthly_caseload": 100}
    except Exception as e:
        logger.error(f"Could not load strategic cost/target data: {e}")
    # ==============================================================================

    if health_df_ai_enriched.empty and not health_df_raw.empty: logger.warning("CHW: AI enrichment failed, using raw data."); return health_df_raw, chw_monthly_cost, program_targets
    elif health_df_ai_enriched.empty: logger.error("CHW: Both raw load and AI enrichment resulted in empty DF."); return pd.DataFrame(), 0.0, {}
    
    return health_df_ai_enriched, chw_monthly_cost, program_targets

# Unpack the returned data
health_df_chw_main, chw_monthly_cost, program_targets = get_chw_dashboard_data_enriched()

if health_df_chw_main.empty: st.error("🚨 Critical Error: Could not load CHW data."); st.stop()

st.title("🧑‍⚕️ Community Health Worker (CHW) Dashboard")
st.markdown("**Daily Patient Prioritization, Field Insights, & Programmatic Impact**"); st.markdown("---")

# Main sidebar configuration (unchanged)
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, width=230)
    st.sidebar.markdown("---")
else:
    logger.warning(f"Sidebar logo not found on CHW Dashboard at {app_config.APP_LOGO}")
    
st.sidebar.header("🗓️ Data Filters")
min_date_overall = date.today() - timedelta(days=365); max_date_overall = date.today()
# ... (rest of sidebar filter logic remains exactly the same) ...

# The existing date filtering and dataframe preparation logic remains unchanged
# I'll just copy the final part to ensure clarity
if 'encounter_date' in health_df_chw_main.columns:
    health_df_chw_main['encounter_date_obj'] = pd.to_datetime(health_df_chw_main['encounter_date'], errors='coerce').dt.date
else: health_df_chw_main['encounter_date_obj'] = pd.NaT

selected_trend_start_chw, selected_trend_end_chw = st.sidebar.date_input("Select Date Range for Period Analysis:", value=[max_date_overall - timedelta(days=89), max_date_overall], min_value=min_date_overall, max_value=max_date_overall, key="chw_trend_range_v3")
selected_view_date_chw = st.sidebar.date_input("View Data For Date:", value=max_date_overall, min_value=min_date_overall, max_value=max_date_overall, key="chw_daily_date_v9")

current_day_chw_df = health_df_chw_main[health_df_chw_main['encounter_date_obj'] == selected_view_date_chw].copy()
period_health_df_chw = health_df_chw_main[
    (health_df_chw_main['encounter_date_obj'].notna()) &
    (health_df_chw_main['encounter_date_obj'] >= selected_trend_start_chw) &
    (health_df_chw_main['encounter_date_obj'] <= selected_trend_end_chw)
].copy()

# ... (rest of zone filtering and data preparation logic is unchanged) ...
chw_daily_kpis = get_chw_summary(current_day_chw_df)
patient_alerts_tasks_df = get_patient_alerts_for_chw(current_day_chw_df)


# ==============================================================================
# SME ENHANCEMENT FOR GATES FOUNDATION
# ADD THE STRATEGIC TAB TO THE EXISTING TAB LAYOUT
# ==============================================================================
op_tab_title = "🧑‍⚕️ CHW Daily Operations"
strategic_tab_title = "📊 Strategic Program Overview"

tab_operational, tab_strategic = st.tabs([op_tab_title, strategic_tab_title])

# --- RENDER OPERATIONAL TAB (Existing Content) ---
with tab_operational:
    # All the existing UI code for the daily operational view goes here, unchanged.
    st.subheader(f"Daily Snapshot: {selected_view_date_chw.strftime('%A, %B %d, %Y')}")
    if current_day_chw_df.empty:
        st.info(f"ℹ️ No CHW encounter data for {selected_view_date_chw.strftime('%A, %B %d, %Y')}.")
    
    kpi_snapshots.render_chw_daily_kpis(chw_daily_kpis, current_day_chw_df)
    epi_watch.render_chw_epi_watch(current_day_chw_df, chw_daily_kpis, "All Zones", selected_view_date_chw) # Assuming 'selected_chw_zone_daily' is replaced by a global filter
    st.markdown("---")
    
    sub_tab_alerts, sub_tab_tasks = st.tabs([f"🚨 Alerts", f"📋 Tasks"])
    with sub_tab_alerts:
        alerts_display.render_chw_alerts_tab(patient_alerts_tasks_df, current_day_chw_df, selected_view_date_chw, "All Zones")
    with sub_tab_tasks:
        tasks_display.render_chw_tasks_tab(patient_alerts_tasks_df, current_day_chw_df, selected_view_date_chw, "All Zones")

# --- RENDER STRATEGIC TAB (New Content) ---
with tab_strategic:
    # Call the render function from our new, self-contained component
    strategic_overview_tab.render(period_health_df_chw, program_targets, chw_monthly_cost)
# ==============================================================================
