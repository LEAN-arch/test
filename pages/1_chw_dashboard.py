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
    get_trend_data # Retained if period overview chart uses it directly here
)
from utils.ai_analytics_engine import apply_ai_models
# Assuming components are in a subfolder like 'pages/chw_components'
# Adjust import if your structure is flat or different.
# For a flat structure 'test/pages/chw_kpi_snapshots.py', it would be:
# from chw_kpi_snapshots import render_chw_daily_kpis
from pages.chw_components import kpi_snapshots, epi_watch, alerts_display, tasks_display, trends_display
from utils.ui_visualization_helpers import plot_annotated_line_chart

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
def get_chw_dashboard_data_enriched():
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    if health_df_raw.empty: logger.error("CHW: Raw health records load failed."); return pd.DataFrame()
    health_df_ai_enriched = apply_ai_models(health_df_raw)
    if health_df_ai_enriched.empty and not health_df_raw.empty: logger.warning("CHW: AI enrichment failed, using raw data."); return health_df_raw
    elif health_df_ai_enriched.empty: logger.error("CHW: Both raw load and AI enrichment resulted in empty DF."); return pd.DataFrame()
    return health_df_ai_enriched
health_df_chw_main = get_chw_dashboard_data_enriched()

if health_df_chw_main.empty: st.error("🚨 Critical Error: Could not load CHW data."); st.stop()

st.title("🧑‍⚕️ Community Health Worker (CHW) Dashboard")
st.markdown("**Daily Patient Prioritization, Field Insights, & Wellness Monitoring**"); st.markdown("---")

if os.path.exists(app_config.APP_LOGO): st.sidebar.image(app_config.APP_LOGO, use_column_width='auto'); st.sidebar.markdown("---")
st.sidebar.header("🗓️ CHW Filters")

min_date_overall = date.today() - timedelta(days=365); max_date_overall = date.today()
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
selected_trend_start_chw, selected_trend_end_chw = st.sidebar.date_input("Select Date Range for Period Analysis:", value=[default_trend_start, default_trend_end], min_value=min_date_overall, max_value=max_date_overall, key="chw_trend_range_v3") # Incremented
if selected_trend_start_chw > selected_trend_end_chw: st.sidebar.error("Error: Start date must be before end date."); selected_trend_start_chw = selected_trend_end_chw

if 'encounter_date' in health_df_chw_main.columns:
    health_df_chw_main['encounter_date_obj'] = pd.to_datetime(health_df_chw_main['encounter_date'], errors='coerce').dt.date
else: health_df_chw_main['encounter_date_obj'] = pd.NaT
current_day_chw_df = health_df_chw_main[health_df_chw_main['encounter_date_obj'] == selected_view_date_chw].copy()

chw_zones_today = []
if 'zone_id' in current_day_chw_df.columns: chw_zones_today = sorted(current_day_chw_df['zone_id'].dropna().unique().tolist())
selected_chw_zone_daily = "All Zones"
if chw_zones_today :
    selected_chw_zone_daily = st.sidebar.selectbox("Filter Daily Snapshot by Zone:", options=["All Zones"] + chw_zones_today, index=0, key="chw_zone_filter_daily_v2")
    if selected_chw_zone_daily != "All Zones": current_day_chw_df = current_day_chw_df[current_day_chw_df['zone_id'] == selected_chw_zone_daily]

# Ensure columns exist for empty DFs passed to summary/alert functions
empty_df_schema = pd.DataFrame(columns=health_df_chw_main.columns)
chw_daily_kpis = get_chw_summary(current_day_chw_df if not current_day_chw_df.empty else empty_df_schema)
patient_alerts_tasks_df = get_patient_alerts_for_chw(current_day_chw_df if not current_day_chw_df.empty else empty_df_schema)

zone_display_daily = f"({selected_chw_zone_daily})" if selected_chw_zone_daily != "All Zones" and chw_zones_today else "(All My Zones)"
st.subheader(f"Daily Snapshot: {selected_view_date_chw.strftime('%A, %B %d, %Y')} {zone_display_daily}")
if current_day_chw_df.empty : st.info(f"ℹ️ No CHW encounter data for {selected_view_date_chw.strftime('%A, %B %d, %Y')} {zone_display_daily}.")

kpi_snapshots.render_chw_daily_kpis(chw_daily_kpis, current_day_chw_df) # current_day_chw_df used for AI Prio task count
epi_watch.render_chw_epi_watch(current_day_chw_df, chw_daily_kpis, selected_chw_zone_daily, selected_view_date_chw)

st.subheader(f"Period Overview: {selected_trend_start_chw.strftime('%d %b %Y')} - {selected_trend_end_chw.strftime('%d %b %Y')}")
period_health_df_chw = health_df_chw_main[
    (health_df_chw_main['encounter_date_obj'].notna()) &
    (health_df_chw_main['encounter_date_obj'] >= selected_trend_start_chw) &
    (health_df_chw_main['encounter_date_obj'] <= selected_trend_end_chw)
].copy()
if selected_chw_zone_daily != "All Zones" and chw_zones_today and 'zone_id' in period_health_df_chw.columns:
    period_health_df_chw = period_health_df_chw[period_health_df_chw['zone_id'] == selected_chw_zone_daily]

if period_health_df_chw.empty:
    st.info(f"No data for period analysis ({selected_trend_start_chw.strftime('%d %b')} - {selected_trend_end_chw.strftime('%d %b %Y')}).")
else:
    period_kpi_cols = st.columns(3)
    with period_kpi_cols[0]:
        total_visits = period_health_df_chw['encounter_id'].nunique()
        num_days = (selected_trend_end_chw - selected_trend_start_chw).days + 1
        avg_daily = total_visits / num_days if num_days > 0 else 0
        st.metric("Total Visits in Period", total_visits, f"{avg_daily:.1f} avg/day")
    with period_kpi_cols[1]: st.metric("Unique Patients Seen", period_health_df_chw['patient_id'].nunique())
    with period_kpi_cols[2]:
        avg_risk = np.nan
        if 'ai_risk_score' in period_health_df_chw and period_health_df_chw['ai_risk_score'].notna().any() : avg_risk = period_health_df_chw['ai_risk_score'].mean()
        st.metric("Avg. Patient Risk (Period)", f"{avg_risk:.0f}" if pd.notna(avg_risk) else "N/A")
    
    if 'encounter_date' in period_health_df_chw.columns and 'patient_id' in period_health_df_chw.columns:
        if not pd.api.types.is_datetime64_ns_dtype(period_health_df_chw['encounter_date']):
             period_health_df_chw.loc[:, 'encounter_date'] = pd.to_datetime(period_health_df_chw['encounter_date'], errors='coerce')
        period_health_df_chw.dropna(subset=['encounter_date'], inplace=True)
        if not period_health_df_chw.empty:
            daily_visits_trend_data = get_trend_data(period_health_df_chw, 'patient_id', 'encounter_date', 'D', 'nunique')
            if not daily_visits_trend_data.empty: st.plotly_chart(plot_annotated_line_chart(daily_visits_trend_data, "Daily Patients Visited in Period", "# Patients", height=app_config.COMPACT_PLOT_HEIGHT, date_format="%d %b", y_is_count=True), use_container_width=True)
            else: st.caption("No trend data for daily visits in selected period.")
st.markdown("---")

tab_titles_chw = [f"🚨 Alerts ({selected_view_date_chw.strftime('%d %b')})", f"📋 Tasks ({selected_view_date_chw.strftime('%d %b')})", f"📈 Activity Trends"]
tab_alerts_disp, tab_tasks_disp, tab_chw_activity_trends_disp = st.tabs(tab_titles_chw)

with tab_alerts_disp: alerts_display.render_chw_alerts_tab(patient_alerts_tasks_df, current_day_chw_df, selected_view_date_chw, selected_chw_zone_daily)
with tab_tasks_disp: tasks_display.render_chw_tasks_tab(patient_alerts_tasks_df, current_day_chw_df, selected_view_date_chw, selected_chw_zone_daily)
with tab_chw_activity_trends_disp: trends_display.render_chw_activity_trends_tab(health_df_chw_main, selected_trend_start_chw, selected_trend_end_chw, selected_chw_zone_daily)
