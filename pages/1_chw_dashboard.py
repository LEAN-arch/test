# test/pages/1_chw_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
from datetime import date, timedelta
import numpy as np

from config import app_config
from utils.core_data_processing import (
    load_health_records,
    get_chw_summary,
    get_patient_alerts_for_chw,
    get_trend_data # Keep for Period Overview if needed directly here
)
from utils.ai_analytics_engine import apply_ai_models
# Import the component functions - Adjust path if your structure is different
# e.g. from .chw_components import ... if chw_components is a submodule of pages
from pages.chw_components import kpi_snapshots, epi_watch, alerts_display, tasks_display, trends_display
from utils.ui_visualization_helpers import plot_annotated_line_chart # For Period Overview chart

# --- Page Configuration and Styling ---
st.set_page_config(page_title="CHW Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_chw():
    css_path = app_config.STYLE_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path) as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        logger.info("CHW Dashboard: CSS loaded successfully.")
    else: logger.warning(f"CHW CSS file not found: {css_path}.")
load_css_chw()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading CHW data...")
def get_chw_dashboard_data_enriched():
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    if health_df_raw.empty:
        logger.error("CHW Dashboard: Raw health records failed to load.")
        return pd.DataFrame()
    health_df_ai_enriched = apply_ai_models(health_df_raw)
    if health_df_ai_enriched.empty and not health_df_raw.empty:
        logger.warning("CHW Dashboard: AI enrichment failed, using raw (but cleaned) data.")
        return health_df_raw # Return cleaned raw data if AI fails
    elif health_df_ai_enriched.empty: # Both raw and AI enrichment resulted in empty
        logger.error("CHW Dashboard: Both raw load and AI enrichment resulted in empty DataFrame.")
        return pd.DataFrame()
    logger.info(f"CHW Dashboard: Loaded and AI-enriched {len(health_df_ai_enriched)} records.")
    return health_df_ai_enriched
health_df_chw_main = get_chw_dashboard_data_enriched()

if health_df_chw_main.empty:
    st.error("🚨 Critical Error: Could not load or process CHW data. Dashboard cannot be rendered."); st.stop()

st.title("🧑‍⚕️ Community Health Worker (CHW) Dashboard")
st.markdown("**Daily Patient Prioritization, Field Insights, & Wellness Monitoring**")
st.markdown("---")

# --- Sidebar ---
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, use_column_width='auto'); st.sidebar.markdown("---")
st.sidebar.header("🗓️ CHW Filters")

min_date_overall = date.today() - timedelta(days=365) # Default fallback
max_date_overall = date.today() # Default fallback
if not health_df_chw_main.empty and 'encounter_date' in health_df_chw_main and health_df_chw_main['encounter_date'].notna().any():
    min_date_overall = health_df_chw_main['encounter_date'].dropna().min().date()
    max_date_overall = health_df_chw_main['encounter_date'].dropna().max().date()
if min_date_overall > max_date_overall: min_date_overall = max_date_overall

st.sidebar.markdown("#### Daily Snapshot View")
default_daily_date = max_date_overall
selected_view_date_chw = st.sidebar.date_input("View Data For Date:", value=default_daily_date, min_value=min_date_overall, max_value=max_date_overall, key="chw_daily_date_v9")

st.sidebar.markdown("---"); st.sidebar.markdown("#### Period Analysis View")
default_trend_end = selected_view_date_chw
default_trend_start = default_trend_end - timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if default_trend_start < min_date_overall: default_trend_start = min_date_overall
selected_trend_start_chw, selected_trend_end_chw = st.sidebar.date_input("Select Date Range for Period Analysis:", value=[default_trend_start, default_trend_end], min_value=min_date_overall, max_value=max_date_overall, key="chw_trend_range_v2")
if selected_trend_start_chw > selected_trend_end_chw: st.sidebar.error("Error: Start date must be before end date."); selected_trend_start_chw = selected_trend_end_chw

# Ensure encounter_date_obj column exists for filtering
if 'encounter_date' in health_df_chw_main.columns:
    health_df_chw_main['encounter_date_obj'] = pd.to_datetime(health_df_chw_main['encounter_date'], errors='coerce').dt.date
else: # Fallback if encounter_date is missing (shouldn't happen if load_health_records is robust)
    health_df_chw_main['encounter_date_obj'] = pd.NaT

current_day_chw_df = health_df_chw_main[health_df_chw_main['encounter_date_obj'] == selected_view_date_chw].copy()

chw_zones_today = sorted(current_day_chw_df['zone_id'].dropna().unique().tolist()) if not current_day_chw_df.empty and 'zone_id' in current_day_chw_df else []
selected_chw_zone_daily = "All Zones"
if chw_zones_today :
    selected_chw_zone_daily = st.sidebar.selectbox("Filter Daily Snapshot by Zone:", options=["All Zones"] + chw_zones_today, index=0, key="chw_zone_filter_daily_v2")
    if selected_chw_zone_daily != "All Zones":
        current_day_chw_df = current_day_chw_df[current_day_chw_df['zone_id'] == selected_chw_zone_daily]

# Prepare data for Daily Snapshot sections
if current_day_chw_df.empty:
    chw_daily_kpis = get_chw_summary(pd.DataFrame(columns=health_df_chw_main.columns)) # Pass empty df with schema
    patient_alerts_tasks_df = get_patient_alerts_for_chw(pd.DataFrame(columns=health_df_chw_main.columns))
else:
    chw_daily_kpis = get_chw_summary(current_day_chw_df)
    patient_alerts_tasks_df = get_patient_alerts_for_chw(current_day_chw_df)

# --- Daily Snapshot Section ---
zone_display_daily = f"({selected_chw_zone_daily})" if selected_chw_zone_daily != "All Zones" and chw_zones_today else "(All My Zones)"
st.subheader(f"Daily Snapshot: {selected_view_date_chw.strftime('%A, %B %d, %Y')} {zone_display_daily}")
kpi_snapshots.render_chw_daily_kpis(chw_daily_kpis, current_day_chw_df) # Pass current_day_chw_df
epi_watch.render_chw_epi_watch(current_day_chw_df, chw_daily_kpis, selected_chw_zone_daily, selected_view_date_chw)

# --- Period Overview Section ---
st.subheader(f"Period Overview: {selected_trend_start_chw.strftime('%d %b %Y')} - {selected_trend_end_chw.strftime('%d %b %Y')}")
# Data for period analysis
period_health_df_chw = health_df_chw_main[
    (health_df_chw_main['encounter_date_obj'].notna()) & # Ensure not NaT before comparison
    (health_df_chw_main['encounter_date_obj'] >= selected_trend_start_chw) &
    (health_df_chw_main['encounter_date_obj'] <= selected_trend_end_chw)
].copy()

# If a daily zone filter is active, allow it to also filter the period overview for consistency, if desired
if selected_chw_zone_daily != "All Zones" and chw_zones_today and 'zone_id' in period_health_df_chw.columns:
    period_health_df_chw = period_health_df_chw[period_health_df_chw['zone_id'] == selected_chw_zone_daily]

if period_health_df_chw.empty:
    zone_context_period_msg = f" in {selected_chw_zone_daily}" if selected_chw_zone_daily != "All Zones" and chw_zones_today else ""
    st.info(f"No data available for period analysis ({selected_trend_start_chw.strftime('%d %b')} - {selected_trend_end_chw.strftime('%d %b %Y')}{zone_context_period_msg}).")
else:
    period_kpi_cols = st.columns(3)
    with period_kpi_cols[0]:
        total_visits_period = period_health_df_chw['encounter_id'].nunique()
        num_days_period = (selected_trend_end_chw - selected_trend_start_chw).days + 1
        avg_daily_visits_period = total_visits_period / num_days_period if num_days_period > 0 else 0
        st.metric("Total Visits in Period", total_visits_period, f"{avg_daily_visits_period:.1f} avg/day")
    with period_kpi_cols[1]:
        unique_patients_period = period_health_df_chw['patient_id'].nunique()
        st.metric("Unique Patients Seen", unique_patients_period)
    with period_kpi_cols[2]:
        avg_risk_period = np.nan
        if 'ai_risk_score' in period_health_df_chw.columns and period_health_df_chw['ai_risk_score'].notna().any():
             avg_risk_period = period_health_df_chw['ai_risk_score'].mean()
        st.metric("Avg. Patient Risk (Period)", f"{avg_risk_period:.0f}" if pd.notna(avg_risk_period) else "N/A")
    
    if 'encounter_date' in period_health_df_chw.columns and 'patient_id' in period_health_df_chw.columns:
        # Ensure 'encounter_date' in period_health_df_chw is datetime64 for get_trend_data
        if not pd.api.types.is_datetime64_ns_dtype(period_health_df_chw['encounter_date']):
             period_health_df_chw.loc[:, 'encounter_date'] = pd.to_datetime(period_health_df_chw['encounter_date'], errors='coerce')
        
        # CORRECTED: Use the correct variable name
        daily_visits_trend_period_data = get_trend_data(period_health_df_chw, value_col='patient_id', date_col='encounter_date', period='D', agg_func='nunique')
        if not daily_visits_trend_period_data.empty:
            st.plotly_chart(plot_annotated_line_chart(daily_visits_trend_period_data, "Daily Patients Visited in Period", y_axis_title="# Patients", height=app_config.COMPACT_PLOT_HEIGHT, date_format="%d %b"), use_container_width=True)
        else: st.caption("No data for daily visits trend within selected period.")
st.markdown("---")

# Tabs for Daily Alerts, Daily Task List, and Activity Trends
tab_titles = [
    f"🚨 Alerts ({selected_view_date_chw.strftime('%d %b')})",
    f"📋 Tasks ({selected_view_date_chw.strftime('%d %b')})",
    f"📈 Trends ({selected_trend_start_chw.strftime('%d %b')} - {selected_trend_end_chw.strftime('%d %b')})"
]
tab_alerts, tab_tasks, tab_chw_activity_trends = st.tabs(tab_titles)

with tab_alerts:
    alerts_display.render_chw_alerts_tab(patient_alerts_tasks_df, current_day_chw_df, selected_view_date_chw, selected_chw_zone_daily)

with tab_tasks:
    tasks_display.render_chw_tasks_tab(patient_alerts_tasks_df, current_day_chw_df, selected_view_date_chw, selected_chw_zone_daily)

with tab_chw_activity_trends:
    trends_display.render_chw_activity_trends_tab(health_df_chw_main, selected_trend_start_chw, selected_trend_end_chw, selected_chw_zone_daily) # Pass main_df and filters
