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
    get_patient_alerts_for_chw
    # get_trend_data is now used within chw_components.trends_display
)
from utils.ai_analytics_engine import apply_ai_models
# Import the component functions
from pages.chw_components import kpi_snapshots, epi_watch, alerts_display, tasks_display, trends_display

# --- Page Configuration and Styling ---
st.set_page_config(page_title="CHW Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_chw():
    css_path = app_config.STYLE_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path) as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else: logger.warning(f"CHW CSS file not found: {css_path}.")
load_css_chw()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading CHW data...")
def get_chw_dashboard_data_enriched(): # Function definition unchanged
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    if health_df_raw.empty: return pd.DataFrame()
    health_df_ai_enriched = apply_ai_models(health_df_raw)
    return health_df_ai_enriched if not health_df_ai_enriched.empty else health_df_raw
health_df_chw_main = get_chw_dashboard_data_enriched()

if health_df_chw_main.empty:
    st.error("🚨 Critical Error: Could not load CHW data."); st.stop()

st.title("🧑‍⚕️ Community Health Worker (CHW) Dashboard")
st.markdown("**Daily Patient Prioritization, Field Insights, & Wellness Monitoring**")
st.markdown("---")

# --- Sidebar ---
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, use_column_width='auto'); st.sidebar.markdown("---")
st.sidebar.header("🗓️ CHW Filters")

min_date_overall = health_df_chw_main['encounter_date'].min().date() if not health_df_chw_main.empty and 'encounter_date' in health_df_chw_main and not health_df_chw_main['encounter_date'].dropna().empty else date.today() - timedelta(days=365)
max_date_overall = health_df_chw_main['encounter_date'].max().date() if not health_df_chw_main.empty and 'encounter_date' in health_df_chw_main and not health_df_chw_main['encounter_date'].dropna().empty else date.today()
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

# Filter data for selected SINGLE day
health_df_chw_main['encounter_date_obj'] = pd.to_datetime(health_df_chw_main['encounter_date'], errors='coerce').dt.date # ensure obj col always created
current_day_chw_df = health_df_chw_main[health_df_chw_main['encounter_date_obj'] == selected_view_date_chw].copy()

chw_zones_today = sorted(current_day_chw_df['zone_id'].unique().tolist()) if not current_day_chw_df.empty and 'zone_id' in current_day_chw_df else []
selected_chw_zone_daily = "All Zones" # Default
if chw_zones_today: # Only show if zones are present for the day
    selected_chw_zone_daily = st.sidebar.selectbox("Filter Daily Snapshot by Zone:", options=["All Zones"] + chw_zones_today, index=0, key="chw_zone_filter_daily_v2")
    if selected_chw_zone_daily != "All Zones":
        current_day_chw_df = current_day_chw_df[current_day_chw_df['zone_id'] == selected_chw_zone_daily]

if current_day_chw_df.empty:
    # Make columns match the main df if empty to avoid errors in get_chw_summary
    empty_df_for_summary = pd.DataFrame(columns=health_df_chw_main.columns)
    chw_daily_kpis = get_chw_summary(empty_df_for_summary)
    patient_alerts_tasks_df = get_patient_alerts_for_chw(empty_df_for_summary)
else:
    chw_daily_kpis = get_chw_summary(current_day_chw_df)
    patient_alerts_tasks_df = get_patient_alerts_for_chw(current_day_chw_df) # removed redundant thresholds

zone_display_daily = f"({selected_chw_zone_daily})" if selected_chw_zone_daily != "All Zones" and chw_zones_today else "(All My Zones)"
st.subheader(f"Daily Snapshot: {selected_view_date_chw.strftime('%A, %B %d, %Y')} {zone_display_daily}")

# Render KPIs and Epi Watch for the selected day
kpi_snapshots.render_chw_daily_kpis(chw_daily_kpis, current_day_chw_df)
epi_watch.render_chw_epi_watch(current_day_chw_df, chw_daily_kpis, selected_chw_zone_daily, selected_view_date_chw)


# --- Period Overview Section (Data filtered by selected_trend_date_range_chw) ---
# ... (This section from previous complete file, no major changes needed, it already uses period_health_df_chw)
st.subheader(f"Period Overview: {selected_trend_start_chw.strftime('%d %b %Y')} - {selected_trend_end_chw.strftime('%d %b %Y')}")
period_health_df_chw = health_df_chw_main[
    (health_df_chw_main['encounter_date_obj'] >= selected_trend_start_chw) &
    (health_df_chw_main['encounter_date_obj'] <= selected_trend_end_chw)
].copy()
if selected_chw_zone_daily != "All Zones" and chw_zones_today: # If a daily zone filter is active, also apply to period overview for context
    period_health_df_chw = period_health_df_chw[period_health_df_chw['zone_id'] == selected_chw_zone_daily]

if period_health_df_chw.empty:
    st.info(f"No data available for the period analysis ({selected_trend_start_chw.strftime('%d %b')} - {selected_trend_end_chw.strftime('%d %b %Y')}).")
else:
    # KPIs for the period
    # ... (Period KPI logic from previous example) ...
    period_kpi_cols = st.columns(3) # Example KPIs for period
    with period_kpi_cols[0]:
        total_visits_period = period_health_df_chw['encounter_id'].nunique()
        num_days_period = (selected_trend_end_chw - selected_trend_start_chw).days + 1
        avg_daily_visits_period = total_visits_period / num_days_period if num_days_period > 0 else 0
        st.metric("Total Visits in Period", total_visits_period, f"{avg_daily_visits_period:.1f} avg/day")
    with period_kpi_cols[1]:
        unique_patients_period = period_health_df_chw['patient_id'].nunique()
        st.metric("Unique Patients Seen", unique_patients_period)
    with period_kpi_cols[2]:
        avg_risk_period = period_health_df_chw['ai_risk_score'].mean() if 'ai_risk_score' in period_health_df_chw.columns and period_health_df_chw['ai_risk_score'].notna().any() else np.nan
        st.metric("Avg. Patient Risk (Period)", f"{avg_risk_period:.0f}" if pd.notna(avg_risk_period) else "N/A")
    # Example Trend Chart in Period Overview
    if 'encounter_date' in period_health_df_chw.columns and 'patient_id' in period_health_df_chw.columns:
        # Ensure 'encounter_date' in period_health_df_chw is datetime for get_trend_data
        daily_visits_trend_period = get_trend_data(period_health_df_chw, value_col='patient_id', date_col='encounter_date', period='D', agg_func='nunique')
        if not daily_visits_trend_period.empty:
            st.plotly_chart(plot_annotated_line_chart(daily_visits_trend_period, "Daily Patients Visited in Period", y_axis_title="# Patients", height=app_config.COMPACT_PLOT_HEIGHT, date_format="%d %b"), use_container_width=True)
st.markdown("---")

# Tabs for Daily Alerts, Daily Task List, and Activity Trends (now uses selected date range)
tab_alerts, tab_tasks, tab_chw_activity_trends = st.tabs([
    f"🚨 Alerts ({selected_view_date_chw.strftime('%d %b')})",
    f"📋 Tasks ({selected_view_date_chw.strftime('%d %b')})",
    f"📈 Activity Trends ({selected_trend_start_chw.strftime('%d %b')} - {selected_trend_end_chw.strftime('%d %b')})"
])

with tab_alerts:
    alerts_display.render_chw_alerts_tab(patient_alerts_tasks_df, current_day_chw_df, selected_view_date_chw, selected_chw_zone_daily)

with tab_tasks:
    tasks_display.render_chw_tasks_tab(patient_alerts_tasks_df, current_day_chw_df, selected_view_date_chw, selected_chw_zone_daily)

with tab_chw_activity_trends:
    trends_display.render_chw_activity_trends_tab(health_df_chw_main, selected_trend_start_chw, selected_trend_end_chw, selected_chw_zone_daily) # Pass main_df and let func filter
