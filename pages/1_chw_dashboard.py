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
    get_chw_summary, # Used for daily snapshot
    get_patient_alerts_for_chw, # Used for daily snapshot
    get_trend_data
)
from utils.ai_analytics_engine import apply_ai_models
from utils.ui_visualization_helpers import (
    render_kpi_card,
    render_traffic_light,
    plot_annotated_line_chart,
    plot_bar_chart
)

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

# --- Data Loading and AI Enrichment ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading CHW data...")
def get_chw_dashboard_data_enriched():
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    if health_df_raw.empty: return pd.DataFrame()
    health_df_ai_enriched = apply_ai_models(health_df_raw)
    return health_df_ai_enriched if not health_df_ai_enriched.empty else health_df_raw
health_df_chw_main = get_chw_dashboard_data_enriched()

# --- Main Page Rendering ---
if health_df_chw_main.empty:
    st.error("🚨 Critical Error: Could not load CHW data."); st.stop()

st.title("🧑‍⚕️ Community Health Worker (CHW) Dashboard")
st.markdown("**Daily Patient Prioritization, Field Insights, & Wellness Monitoring**")
st.markdown("---")

# --- Sidebar Filters ---
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, use_column_width='auto'); st.sidebar.markdown("---")
st.sidebar.header("🗓️ CHW Filters")

min_date_overall = health_df_chw_main['encounter_date'].min().date() if not health_df_chw_main.empty and 'encounter_date' in health_df_chw_main and not health_df_chw_main['encounter_date'].dropna().empty else date.today() - timedelta(days=365)
max_date_overall = health_df_chw_main['encounter_date'].max().date() if not health_df_chw_main.empty and 'encounter_date' in health_df_chw_main and not health_df_chw_main['encounter_date'].dropna().empty else date.today()
if min_date_overall > max_date_overall: min_date_overall = max_date_overall

# Filter 1: Single Date for Daily Snapshot
st.sidebar.markdown("#### Daily Snapshot View")
default_daily_date = max_date_overall
selected_view_date_chw = st.sidebar.date_input("View Data For Date:", value=default_daily_date, min_value=min_date_overall, max_value=max_date_overall, key="chw_daily_date_v9")

# Filter 2: Date Range for Period Overview & Trends
st.sidebar.markdown("---"); st.sidebar.markdown("#### Period Analysis View")
default_trend_end = selected_view_date_chw # Link trend end to daily view by default
default_trend_start = default_trend_end - timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if default_trend_start < min_date_overall: default_trend_start = min_date_overall

selected_trend_start_chw, selected_trend_end_chw = st.sidebar.date_input(
    "Select Date Range for Period Analysis:",
    value=[default_trend_start, default_trend_end],
    min_value=min_date_overall, max_value=max_date_overall,
    key="chw_trend_range_v2"
)
if selected_trend_start_chw > selected_trend_end_chw:
    st.sidebar.error("Error: Start date must be before end date for period analysis.")
    selected_trend_start_chw = selected_trend_end_chw # Default to single day range if error


# --- Data Filtering for selected SINGLE day (for Daily Snapshot sections) ---
health_df_chw_main['encounter_date_obj'] = pd.to_datetime(health_df_chw_main['encounter_date']).dt.date
current_day_chw_df = health_df_chw_main[health_df_chw_main['encounter_date_obj'] == selected_view_date_chw].copy()

chw_zones_today = sorted(current_day_chw_df['zone_id'].unique().tolist()) if not current_day_chw_df.empty else []
selected_chw_zone_daily = "All Zones"
if chw_zones_today and len(chw_zones_today) > 1 : # Only show filter if there are multiple zones FOR THAT DAY
    selected_chw_zone_daily = st.sidebar.selectbox("Filter Daily Snapshot by Zone:", options=["All Zones"] + chw_zones_today, index=0, key="chw_zone_filter_daily_v1")
    if selected_chw_zone_daily != "All Zones":
        current_day_chw_df = current_day_chw_df[current_day_chw_df['zone_id'] == selected_chw_zone_daily]

# Calculate daily KPIs and alerts from current_day_chw_df
if current_day_chw_df.empty:
    chw_daily_kpis = get_chw_summary(pd.DataFrame(columns=health_df_chw_main.columns))
    patient_alerts_tasks_df = get_patient_alerts_for_chw(pd.DataFrame(columns=health_df_chw_main.columns))
else:
    chw_daily_kpis = get_chw_summary(current_day_chw_df)
    patient_alerts_tasks_df = get_patient_alerts_for_chw(current_day_chw_df)

# --- Daily Snapshot Section (KPIs, Local Epi for the selected_view_date_chw) ---
zone_display_daily = f"({selected_chw_zone_daily})" if selected_chw_zone_daily != "All Zones" else "(All My Zones)"
st.subheader(f"Daily Snapshot: {selected_view_date_chw.strftime('%A, %B %d, %Y')} {zone_display_daily}")
# KPIs ... (As before, uses chw_daily_kpis which is based on single day)
# ... (Local Epi Snapshot for single day, as before) ...
kpi_cols_chw_overview = st.columns(4)
with kpi_cols_chw_overview[0]: render_kpi_card("Visits Today", str(chw_daily_kpis.get('visits_today', 0)), "🚶‍♀️", status="Good High" if chw_daily_kpis.get('visits_today', 0) >= 10 else "Low")
with kpi_cols_chw_overview[1]:
    high_prio_tasks_count = 0
    if not current_day_chw_df.empty and 'ai_followup_priority_score' in current_day_chw_df.columns: high_prio_tasks_count = current_day_chw_df[current_day_chw_df['ai_followup_priority_score'] >= 80]['patient_id'].nunique()
    render_kpi_card("AI High-Prio Follow-ups Today", str(high_prio_tasks_count), "🎯", status="High" if high_prio_tasks_count > 2 else "Low")
with kpi_cols_chw_overview[2]:
    avg_risk_visited = chw_daily_kpis.get('avg_patient_risk_visited_today', np.nan)
    render_kpi_card("Avg. Risk (Visited Today)", f"{avg_risk_visited:.0f}" if pd.notna(avg_risk_visited) else "N/A", "📈", status="High" if pd.notna(avg_risk_visited) and avg_risk_visited >= 70 else "Low")
with kpi_cols_chw_overview[3]: render_kpi_card("Fever Alerts (Today)", str(chw_daily_kpis.get('patients_fever_visited_today', 0)), "🔥", status="High" if chw_daily_kpis.get('patients_fever_visited_today', 0) > 0 else "Low")
st.markdown("---")


# --- NEW: Period Overview Section (based on selected_trend_date_range_chw) ---
st.subheader(f"Period Overview: {selected_trend_start_chw.strftime('%d %b %Y')} - {selected_trend_end_chw.strftime('%d %b %Y')}")
# Filter health_df_chw_main for the selected trend period
period_health_df_chw = health_df_chw_main[
    (health_df_chw_main['encounter_date_obj'] >= selected_trend_start_chw) &
    (health_df_chw_main['encounter_date_obj'] <= selected_trend_end_chw)
].copy()

# Optional: Filter period_health_df_chw by CHW's assigned zone(s) if that's desired for period summary
# For simplicity, let's assume period overview is for all zones unless a global CHW zone filter is set elsewhere.

if period_health_df_chw.empty:
    st.info(f"No data available for the selected period analysis ({selected_trend_start_chw.strftime('%d %b')} - {selected_trend_end_chw.strftime('%d %b %Y')}).")
else:
    period_kpi_cols = st.columns(3)
    with period_kpi_cols[0]:
        total_visits_period = period_health_df_chw['encounter_id'].nunique() # Total CHW encounters
        avg_daily_visits_period = total_visits_period / ((selected_trend_end_chw - selected_trend_start_chw).days + 1)
        render_kpi_card("Total Visits in Period", str(total_visits_period), "🗓️", delta=f"{avg_daily_visits_period:.1f} avg/day", delta_type="neutral", help_text="Total CHW patient encounters within the selected date range.")
    with period_kpi_cols[1]:
        unique_patients_period = period_health_df_chw['patient_id'].nunique()
        render_kpi_card("Unique Patients Seen", str(unique_patients_period), "🧑‍🤝‍🧑", help_text="Number of unique patients seen by CHW during the period.")
    with period_kpi_cols[2]:
        avg_risk_period = period_health_df_chw['ai_risk_score'].mean() if 'ai_risk_score' in period_health_df_chw.columns and period_health_df_chw['ai_risk_score'].notna().any() else np.nan
        render_kpi_card("Avg. Patient Risk (Period)", f"{avg_risk_period:.0f}" if pd.notna(avg_risk_period) else "N/A", "📉", help_text="Average AI risk score of patients encountered during the selected period.")
    
    # Example Trend Chart in Period Overview
    if 'encounter_date' in period_health_df_chw.columns and 'patient_id' in period_health_df_chw.columns:
        daily_visits_trend_period = get_trend_data(period_health_df_chw, value_col='patient_id', date_col='encounter_date', period='D', agg_func='nunique')
        if not daily_visits_trend_period.empty:
            st.plotly_chart(plot_annotated_line_chart(daily_visits_trend_data, "Daily Patients Visited in Period", y_axis_title="# Patients", height=app_config.COMPACT_PLOT_HEIGHT, date_format="%d %b"), use_container_width=True)
        else:
            st.caption("Not enough data for daily visits trend within the selected period.")
st.markdown("---")


# --- Tabs for Daily Alerts, Daily Task List, and Activity Trends (now uses date range) ---
tab_alerts, tab_tasks, tab_chw_activity_trends = st.tabs([
    f"🚨 Alerts ({selected_view_date_chw.strftime('%d %b')})",
    f"📋 Tasks ({selected_view_date_chw.strftime('%d %b')})",
    f"📈 Activity Trends ({selected_trend_start_chw.strftime('%d %b')} - {selected_trend_end_chw.strftime('%d %b')})"
])

with tab_alerts: # This tab still uses `patient_alerts_tasks_df` from the SINGLE selected_view_date_chw
    st.subheader(f"Critical Patient Alerts for {selected_view_date_chw.strftime('%B %d, %Y')} {zone_display_daily}")
    # ... (Alerts display logic remains the same, operating on patient_alerts_tasks_df)
    if not patient_alerts_tasks_df.empty:
        sort_col_alert = 'ai_followup_priority_score' if 'ai_followup_priority_score' in patient_alerts_tasks_df.columns else 'priority_score'
        alerts_to_display = patient_alerts_tasks_df.sort_values(by=sort_col_alert, ascending=False)
        temp_col_traffic_alert_tab = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in current_day_chw_df.columns and current_day_chw_df['vital_signs_temperature_celsius'].notna().any() else ('max_skin_temp_celsius' if 'max_skin_temp_celsius' in current_day_chw_df.columns else None)
        for _, alert_row in alerts_to_display.head(15).iterrows():
            priority_val = alert_row.get(sort_col_alert, 0)
            alert_status_light = "High" if priority_val >= 80 else ("Moderate" if priority_val >= 60 else "Low")
            details_parts = []
            if pd.notna(alert_row.get('ai_risk_score')): details_parts.append(f"AI Risk: {alert_row['ai_risk_score']:.0f}")
            if pd.notna(alert_row.get('ai_followup_priority_score')): details_parts.append(f"AI Prio: {alert_row['ai_followup_priority_score']:.0f}")
            if pd.notna(alert_row.get('min_spo2_pct')): details_parts.append(f"SpO2: {alert_row['min_spo2_pct']:.0f}%")
            actual_temp_col_in_row_alert = temp_col_traffic_alert_tab if temp_col_traffic_alert_tab and temp_col_traffic_alert_tab in alert_row and pd.notna(alert_row[temp_col_traffic_alert_tab]) else None
            if actual_temp_col_in_row_alert: details_parts.append(f"Temp: {alert_row[actual_temp_col_in_row_alert]:.1f}°C")
            if pd.notna(alert_row.get('fall_detected_today')) and alert_row['fall_detected_today'] > 0: details_parts.append(f"Falls: {int(alert_row['fall_detected_today'])}")
            msg = f"Patient {alert_row.get('patient_id','N/A')} ({alert_row.get('condition','N/A')})"
            detail_str = alert_row.get('alert_reason', 'Review Case') + (" | " + " / ".join(details_parts) if details_parts else "")
            render_traffic_light(message=msg, status=alert_status_light, details=detail_str)
    elif not current_day_chw_df.empty: st.success("✅ No critical patient alerts identified from today's encounters.")
    else: st.info("No CHW encounters recorded today, so no alerts to display.")


with tab_tasks: # This tab still uses `patient_alerts_tasks_df` from the SINGLE selected_view_date_chw
    st.subheader(f"Prioritized Task List for {selected_view_date_chw.strftime('%B %d, %Y')} {zone_display_daily}")
    # ... (Task list display logic remains the same, operating on patient_alerts_tasks_df) ...
    if not patient_alerts_tasks_df.empty:
        temp_col_for_task_table_tasks = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in current_day_chw_df.columns and current_day_chw_df['vital_signs_temperature_celsius'].notna().any() else ('max_skin_temp_celsius' if 'max_skin_temp_celsius' in current_day_chw_df.columns and current_day_chw_df['max_skin_temp_celsius'].notna().any() else None)
        cols_to_show_task = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'ai_followup_priority_score', 'alert_reason', 'referral_status', 'min_spo2_pct', (temp_col_for_task_table_tasks if temp_col_for_task_table_tasks else 'max_skin_temp_celsius'), 'fall_detected_today']
        task_df_for_display = patient_alerts_tasks_df[[col for col in cols_to_show_task if col in patient_alerts_tasks_df.columns]].copy()
        rename_col = temp_col_for_task_table_tasks if temp_col_for_task_table_tasks else ('max_skin_temp_celsius' if 'max_skin_temp_celsius' in task_df_for_display else None)
        if rename_col: task_df_for_display.rename(columns={rename_col: 'latest_temp_celsius'}, inplace=True, errors='ignore')
        sort_cols_tasks_tab = [col for col in ['ai_followup_priority_score', 'ai_risk_score'] if col in task_df_for_display.columns]
        task_df_display_final_sorted = task_df_for_display.sort_values(by=sort_cols_tasks_tab, ascending=[False]*len(sort_cols_tasks_tab)) if sort_cols_tasks_tab else task_df_for_display
        df_for_st_dataframe_tasks = task_df_display_final_sorted.copy()
        for col in df_for_st_dataframe_tasks.columns:
            if df_for_st_dataframe_tasks[col].dtype == 'object': df_for_st_dataframe_tasks[col] = df_for_st_dataframe_tasks[col].fillna('N/A').astype(str).replace(['nan','None','<NA>'],'N/A', regex=False)
        st.dataframe(df_for_st_dataframe_tasks, use_container_width=True, height=450, column_config={"patient_id": "Patient ID", "ai_risk_score": st.column_config.ProgressColumn("AI Risk",format="%d",min_value=0,max_value=100), "ai_followup_priority_score": st.column_config.ProgressColumn("AI Prio.",format="%d",min_value=0,max_value=100), "alert_reason": st.column_config.TextColumn("Reason",width="large"), "min_spo2_pct":st.column_config.NumberColumn("SpO2(%)",format="%d%%"), "latest_temp_celsius":st.column_config.NumberColumn("Temp(°C)",format="%.1f°C"), "fall_detected_today":st.column_config.NumberColumn("Falls",format="%d")}, hide_index=True )
        try: csv_chw_tasks = task_df_display_final_sorted.to_csv(index=False).encode('utf-8'); st.download_button(label="📥 Download Task List (CSV)", data=csv_chw_tasks, file_name=f"chw_tasks_{selected_view_date_chw.strftime('%Y%m%d')}.csv", mime="text/csv", key="chw_task_download_v8")
        except Exception as e_csv: logger.error(f"CHW Task CSV Download Error: {e_csv}"); st.warning("Could not prepare task list for download.")
    elif not current_day_chw_df.empty: st.info("No specific tasks from today's CHW encounters.")
    else: st.info("No CHW encounters recorded today, so no tasks to display.")

with tab_chw_activity_trends: # This tab now uses the selected_trend_date_range_chw
    st.subheader(f"My Activity Trends ({selected_trend_start_chw.strftime('%d %b %Y')} - {selected_trend_end_chw.strftime('%d %b %Y')})")
    if selected_trend_start_chw > selected_trend_end_chw:
        st.warning("Start date for trend period is after end date. Please adjust.")
    else:
        chw_trends_data_for_tab = period_health_df_chw.copy() # Uses data already filtered for trend range
        # Optionally filter by selected_chw_zone if trends should be zone-specific as well
        if selected_chw_zone != "All Zones" and 'zone_id' in chw_trends_data_for_tab.columns:
            chw_trends_data_for_tab = chw_trends_data_for_tab[chw_trends_data_for_tab['zone_id'] == selected_chw_zone]

        if not chw_trends_data_for_tab.empty:
            cols_chw_trend_tab_display = st.columns(2)
            with cols_chw_trend_tab_display[0]:
                visits_trend_data_range = get_trend_data(chw_trends_data_for_tab, value_col='patient_id', date_col='encounter_date', period='D', agg_func='nunique')
                if not visits_trend_data_range.empty: st.plotly_chart(plot_annotated_line_chart(visits_trend_data_range, f"Daily Patients Visited ({selected_chw_zone})", y_axis_title="# Patients", height=app_config.COMPACT_PLOT_HEIGHT-20, date_format="%a, %d %b"), use_container_width=True)
                else: st.caption(f"No patient visit data for trend in selected range/zone.")
            with cols_chw_trend_tab_display[1]:
                if 'ai_followup_priority_score' in chw_trends_data_for_tab.columns:
                    high_prio_df_trend_range = chw_trends_data_for_tab[chw_trends_data_for_tab['ai_followup_priority_score'] >= 80]
                    high_prio_trend_range = get_trend_data(high_prio_df_trend_range, value_col='patient_id', date_col='encounter_date', period='D', agg_func='nunique')
                    if not high_prio_trend_range.empty: st.plotly_chart(plot_annotated_line_chart(high_prio_trend_range, f"High Priority Follow-ups ({selected_chw_zone})", y_axis_title="# Follow-ups", height=app_config.COMPACT_PLOT_HEIGHT-20, date_format="%a, %d %b"), use_container_width=True)
                    else: st.caption(f"No high priority follow-up data for trend in selected range/zone.")
                else: st.caption("AI Follow-up Priority Score not available for trend.")
        else:
            st.info(f"Not enough data in the selected range ({selected_trend_start_chw.strftime('%d %b')} to {selected_trend_end_chw.strftime('%d %b %Y')}, Zone: {selected_chw_zone}) for activity trends.")
