# test/pages/1_chw_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
from datetime import date, timedelta # Import timedelta
import numpy as np

from config import app_config
from utils.core_data_processing import (
    load_health_records,
    get_chw_summary,
    get_patient_alerts_for_chw,
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
st.set_page_config(
    page_title="CHW Dashboard - Health Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_chw():
    css_path = app_config.STYLE_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info("CHW Dashboard: CSS loaded successfully.")
    else:
        logger.warning(f"CHW Dashboard: CSS file not found at {css_path}.")
load_css_chw()

# --- Data Loading and AI Enrichment ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading CHW data...")
def get_chw_dashboard_data_enriched():
    logger.info("CHW Dashboard: Loading and enriching health records...")
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    if health_df_raw.empty:
        logger.error("CHW Dashboard: Failed to load health records.")
        return pd.DataFrame()
    health_df_ai_enriched = apply_ai_models(health_df_raw)
    if health_df_ai_enriched.empty and not health_df_raw.empty:
        logger.warning("CHW Dashboard: AI enrichment failed, using raw data.")
        return health_df_raw
    elif health_df_ai_enriched.empty:
        return pd.DataFrame()
    logger.info(f"CHW Dashboard: Loaded and AI-enriched {len(health_df_ai_enriched)} records.")
    return health_df_ai_enriched

health_df_chw_main = get_chw_dashboard_data_enriched()

# --- Main Page Rendering ---
if health_df_chw_main.empty:
    st.error("🚨 **Critical Error:** Could not load or process health records for the CHW Dashboard.")
    logger.critical("CHW Dashboard: health_df_chw_main is empty.")
    st.stop()
else:
    st.title("🧑‍⚕️ Community Health Worker (CHW) Dashboard")
    st.markdown("**Daily Patient Prioritization, Field Insights, & Wellness Monitoring**")
    st.markdown("---")

    # --- Sidebar Filters ---
    if os.path.exists(app_config.APP_LOGO):
        st.sidebar.image(app_config.APP_LOGO, use_column_width='auto')
        st.sidebar.markdown("---")
    st.sidebar.header("🗓️ CHW Filters")

    # Determine available date range from the main DataFrame
    min_date_available_overall = health_df_chw_main['encounter_date'].min().date() if 'encounter_date' in health_df_chw_main and not health_df_chw_main['encounter_date'].dropna().empty else date.today() - pd.Timedelta(days=365) # Wider fallback
    max_date_available_overall = health_df_chw_main['encounter_date'].max().date() if 'encounter_date' in health_df_chw_main and not health_df_chw_main['encounter_date'].dropna().empty else date.today()
    if min_date_available_overall > max_date_available_overall: min_date_available_overall = max_date_available_overall

    # --- Filter 1: Single Date for Daily Snapshot ---
    st.sidebar.markdown("#### Daily Snapshot View")
    default_daily_view_date = max_date_available_overall # Default to latest data for daily view
    selected_view_date_chw = st.sidebar.date_input(
        "View Data For Date:", value=default_daily_view_date,
        min_value=min_date_available_overall, max_value=max_date_available_overall,
        key="chw_daily_view_date_selector_v8",
        help="Select a specific date for the daily summary, alerts, and task list."
    )

    # --- Filter 2: Date Range for Trends/Activity ---
    st.sidebar.markdown("---") # Separator
    st.sidebar.markdown("#### Activity Trends View")
    # Default trend range: last N days ending on the selected daily view date (or max overall if more logical)
    default_trend_end_date = selected_view_date_chw # Align trend end with daily view for context
    default_trend_start_date = default_trend_end_date - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
    if default_trend_start_date < min_date_available_overall:
        default_trend_start_date = min_date_available_overall
    
    selected_trend_date_range_chw = st.sidebar.date_input(
        "Select Date Range for Trends:",
        value=[default_trend_start_date, default_trend_end_date],
        min_value=min_date_available_overall,
        max_value=max_date_available_overall, # Trend end can't exceed max available data
        key="chw_trend_date_range_selector_v1",
        help="Select a date range for viewing activity trends."
    )
    
    # Ensure start is not after end for trend range
    if selected_trend_date_range_chw[0] > selected_trend_date_range_chw[1]:
        st.sidebar.warning("Trend start date cannot be after end date. Adjusting...")
        selected_trend_date_range_chw = (selected_trend_date_range_chw[1], selected_trend_date_range_chw[1])


    # Filter data for the *single selected_view_date_chw* for daily snapshot sections
    health_df_chw_main['encounter_date_obj'] = pd.to_datetime(health_df_chw_main['encounter_date']).dt.date
    current_day_chw_df = health_df_chw_main[health_df_chw_main['encounter_date_obj'] == selected_view_date_chw].copy()
    
    # Zone filter can apply to the daily snapshot data
    chw_zones = sorted(current_day_chw_df['zone_id'].unique().tolist()) if not current_day_chw_df.empty else ["N/A"]
    selected_chw_zone = "All Zones"
    if len(chw_zones) > 1 and chw_zones != ["N/A"]: # Only show if there are multiple zones for the day
        selected_chw_zone = st.sidebar.selectbox(
            "Filter Daily Snapshot by Zone:", options=["All Zones"] + chw_zones, index=0, key="chw_zone_filter_v2"
        )
        if selected_chw_zone != "All Zones":
            current_day_chw_df = current_day_chw_df[current_day_chw_df['zone_id'] == selected_chw_zone]
    
    # Initialize KPIs and alerts from the *daily filtered data*
    if current_day_chw_df.empty:
        zone_context_msg = f" in {selected_chw_zone}" if selected_chw_zone != "All Zones" else ""
        st.info(f"ℹ️ No CHW-related encounter data recorded for {selected_view_date_chw.strftime('%A, %B %d, %Y')}{zone_context_msg}.")
        chw_daily_kpis = get_chw_summary(pd.DataFrame(columns=health_df_chw_main.columns))
        patient_alerts_tasks_df = get_patient_alerts_for_chw(pd.DataFrame(columns=health_df_chw_main.columns))
    else:
        chw_daily_kpis = get_chw_summary(current_day_chw_df)
        patient_alerts_tasks_df = get_patient_alerts_for_chw(current_day_chw_df, risk_threshold_moderate=app_config.RISK_THRESHOLDS['chw_alert_moderate'], risk_threshold_high=app_config.RISK_THRESHOLDS['chw_alert_high'])
    
    zone_display_for_title = f"({selected_chw_zone})" if selected_chw_zone != "All Zones" else "(All Assigned Zones)"
    st.subheader(f"Daily Snapshot: {selected_view_date_chw.strftime('%A, %B %d, %Y')} {zone_display_for_title}")
    
    # KPIs based on selected_view_date_chw and selected_chw_zone
    kpi_cols_chw_overview = st.columns(4)
    # ... (KPI rendering unchanged - they already use chw_daily_kpis derived from current_day_chw_df)
    with kpi_cols_chw_overview[0]: render_kpi_card("Visits Today", str(chw_daily_kpis.get('visits_today', 0)), "🚶‍♀️", status="Good High" if chw_daily_kpis.get('visits_today', 0) >= 10 else "Low")
    with kpi_cols_chw_overview[1]:
        high_prio_tasks_count = 0
        if not current_day_chw_df.empty and 'ai_followup_priority_score' in current_day_chw_df.columns: high_prio_tasks_count = current_day_chw_df[current_day_chw_df['ai_followup_priority_score'] >= 80]['patient_id'].nunique()
        render_kpi_card("AI High-Prio Follow-ups Today", str(high_prio_tasks_count), "🎯", status="High" if high_prio_tasks_count > 2 else "Low", help_text="Patients needing follow-up based on high AI priority scores from today's encounters.")
    with kpi_cols_chw_overview[2]:
        avg_risk_visited = chw_daily_kpis.get('avg_patient_risk_visited_today', np.nan)
        render_kpi_card("Avg. Risk (Visited Today)", f"{avg_risk_visited:.0f}" if pd.notna(avg_risk_visited) else "N/A", "📈", status="High" if pd.notna(avg_risk_visited) and avg_risk_visited >= 70 else "Low")
    with kpi_cols_chw_overview[3]: render_kpi_card("Fever Alerts (Today)", str(chw_daily_kpis.get('patients_fever_visited_today', 0)), "🔥", status="High" if chw_daily_kpis.get('patients_fever_visited_today', 0) > 0 else "Low")
    st.markdown("---")


    # Epidemiology Snippet for CHW (for selected_view_date_chw)
    st.markdown(f"##### Local Epidemiology Watch - {selected_view_date_chw.strftime('%d %b %Y')} - {selected_chw_zone}")
    # ... (Epi snippet section using current_day_chw_df - logic unchanged from previous output) ...
    if not current_day_chw_df.empty:
        epi_cols_chw_local = st.columns(3)
        with epi_cols_chw_local[0]:
            symptomatic_conditions = ['TB', 'Pneumonia', 'Malaria', 'Dengue']
            new_symptomatic_df = current_day_chw_df[current_day_chw_df['condition'].isin(symptomatic_conditions) & (current_day_chw_df['patient_reported_symptoms'].str.contains('Fever|Cough|Chills|Headache|Aches', case=False, na=False))]
            new_symptomatic_count = new_symptomatic_df['patient_id'].nunique()
            render_kpi_card("New Symptomatic Cases (Key Cond.) Today", str(new_symptomatic_count), "🤒", status="High" if new_symptomatic_count > 2 else "Moderate")
        with epi_cols_chw_local[1]:
            key_condition_for_cluster = "Malaria"
            malaria_cases_today_zone = current_day_chw_df[current_day_chw_df['condition'] == key_condition_for_cluster]['patient_id'].nunique()
            render_kpi_card(f"New {key_condition_for_cluster} Cases Today", str(malaria_cases_today_zone), "🦟", status="High" if malaria_cases_today_zone >= 2 else "Low")
        with epi_cols_chw_local[2]:
            tb_contacts_val = chw_daily_kpis.get('tb_contacts_to_trace_today', 0)
            render_kpi_card("Pending TB Contact Traces Today", str(tb_contacts_val), "👥", status="High" if tb_contacts_val > 0 else "Low")
        
        high_risk_today_df_for_demo = current_day_chw_df[current_day_chw_df['ai_risk_score'] >= app_config.RISK_THRESHOLDS.get('high', 75)]
        if not high_risk_today_df_for_demo.empty and 'age' in high_risk_today_df_for_demo.columns:
            st.markdown("###### Demographics of High AI Risk Patients (Today)")
            age_bins = [0, 5, 18, 45, 65, np.inf]; age_labels = ['0-4', '5-17', '18-44', '45-64', '65+']
            high_risk_today_df_for_demo.loc[:, 'age_group'] = pd.cut(high_risk_today_df_for_demo['age'], bins=age_bins, labels=age_labels, right=False)
            age_group_counts = high_risk_today_df_for_demo['age_group'].value_counts().sort_index().reset_index()
            age_group_counts.columns = ['Age Group', 'Number of High-Risk Patients']
            if not age_group_counts.empty: st.plotly_chart(plot_bar_chart(age_group_counts, x_col='Age Group', y_col='Number of High-Risk Patients', title="High AI Risk Patients by Age Group (Today)", height=app_config.COMPACT_PLOT_HEIGHT-50), use_container_width=True)
            else: st.caption("No high AI risk patients with age data for breakdown today.")
        elif not current_day_chw_df.empty : st.caption("No high AI risk patients found today for demographic breakdown.")
    else:
        st.caption(f"No data for local epidemiology snapshot for {selected_chw_zone} on {selected_view_date_chw.strftime('%d %b %Y')}.")
    st.markdown("---")


    # Tabs: Alerts and Task list STILL use current_day_chw_df (for selected_view_date_chw)
    tab_alerts, tab_tasks, tab_chw_trends = st.tabs(["🚨 Critical Patient Alerts (Today)", "📋 Detailed Task List (Today)", f"📈 My Activity Trends"])
    
    with tab_alerts:
        # ... (Alerts display logic, same as before, uses patient_alerts_tasks_df which is based on current_day_chw_df) ...
        st.subheader(f"Critical Patient Alerts for {selected_view_date_chw.strftime('%B %d, %Y')}")
        if not patient_alerts_tasks_df.empty:
            # (Alerts display logic remains as previously provided)
            sort_col_alert = 'ai_followup_priority_score' if 'ai_followup_priority_score' in patient_alerts_tasks_df.columns else 'priority_score'
            alerts_to_display = patient_alerts_tasks_df.sort_values(by=sort_col_alert, ascending=False)
            temp_col_traffic_alert = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in current_day_chw_df.columns and current_day_chw_df['vital_signs_temperature_celsius'].notna().any() else ('max_skin_temp_celsius' if 'max_skin_temp_celsius' in current_day_chw_df.columns else None)
            for _, alert_row in alerts_to_display.head(15).iterrows():
                priority_val = alert_row.get(sort_col_alert, 0)
                alert_status_light = "High" if priority_val >= 80 else ("Moderate" if priority_val >= 60 else "Low")
                details_parts = []
                if pd.notna(alert_row.get('ai_risk_score')): details_parts.append(f"AI Risk: {alert_row['ai_risk_score']:.0f}")
                if pd.notna(alert_row.get('ai_followup_priority_score')): details_parts.append(f"AI Prio: {alert_row['ai_followup_priority_score']:.0f}")
                if pd.notna(alert_row.get('min_spo2_pct')): details_parts.append(f"SpO2: {alert_row['min_spo2_pct']:.0f}%")
                actual_temp_col_in_row_alert = temp_col_traffic_alert if temp_col_traffic_alert and temp_col_traffic_alert in alert_row and pd.notna(alert_row[temp_col_traffic_alert]) else None
                if actual_temp_col_in_row_alert: details_parts.append(f"Temp: {alert_row[actual_temp_col_in_row_alert]:.1f}°C")
                if pd.notna(alert_row.get('fall_detected_today')) and alert_row['fall_detected_today'] > 0: details_parts.append(f"Falls: {int(alert_row['fall_detected_today'])}")
                msg = f"Patient {alert_row.get('patient_id','N/A')} ({alert_row.get('condition','N/A')})"
                detail_str = alert_row.get('alert_reason', 'Review Case') + (" | " + " / ".join(details_parts) if details_parts else "")
                render_traffic_light(message=msg, status=alert_status_light, details=detail_str)
        elif not current_day_chw_df.empty: st.success("✅ No critical patient alerts identified from today's encounters.")
        else: st.info("No CHW encounters recorded today, so no alerts to display.")


    with tab_tasks:
        # ... (Task List logic, same as before, uses patient_alerts_tasks_df from current_day_chw_df) ...
        st.subheader(f"Prioritized Task List for {selected_view_date_chw.strftime('%B %d, %Y')}")
        if not patient_alerts_tasks_df.empty:
            # (Task list dataframe display logic remains as previously provided)
            temp_col_for_task_table_tasks = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in current_day_chw_df.columns and current_day_chw_df['vital_signs_temperature_celsius'].notna().any() else ('max_skin_temp_celsius' if 'max_skin_temp_celsius' in current_day_chw_df.columns and current_day_chw_df['max_skin_temp_celsius'].notna().any() else None)
            cols_to_show_task = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'ai_followup_priority_score', 'alert_reason', 'referral_status', 'min_spo2_pct', (temp_col_for_task_table_tasks if temp_col_for_task_table_tasks else 'max_skin_temp_celsius'), 'fall_detected_today']
            task_df_for_display = patient_alerts_tasks_df[[col for col in cols_to_show_task if col in patient_alerts_tasks_df.columns]].copy()
            rename_col = temp_col_for_task_table_tasks if temp_col_for_task_table_tasks else ('max_skin_temp_celsius' if 'max_skin_temp_celsius' in task_df_for_display else None)
            if rename_col: task_df_for_display.rename(columns={rename_col: 'latest_temp_celsius'}, inplace=True, errors='ignore')
            
            sort_cols_tasks_tab = [col for col in ['ai_followup_priority_score', 'ai_risk_score'] if col in task_df_for_display.columns]
            task_df_display_final_sorted = task_df_for_display.sort_values(by=sort_cols_tasks_tab, ascending=[False]*len(sort_cols_tasks_tab)) if sort_cols_tasks_tab else task_df_for_display
            df_for_st_dataframe_tasks = task_df_display_final_sorted.copy() # Make a copy for st.dataframe specific conversions
            for col in df_for_st_dataframe_tasks.columns: # Ensure serializable
                if df_for_st_dataframe_tasks[col].dtype == 'object': df_for_st_dataframe_tasks[col] = df_for_st_dataframe_tasks[col].fillna('N/A').astype(str).replace(['nan','None','<NA>'],'N/A', regex=False)
            st.dataframe(df_for_st_dataframe_tasks, use_container_width=True, height=450, column_config={"patient_id": "Patient ID", "ai_risk_score": st.column_config.ProgressColumn("AI Risk",format="%d",min_value=0,max_value=100), "ai_followup_priority_score": st.column_config.ProgressColumn("AI Prio.",format="%d",min_value=0,max_value=100), "alert_reason": st.column_config.TextColumn("Reason",width="large"), "min_spo2_pct":st.column_config.NumberColumn("SpO2(%)",format="%d%%"), "latest_temp_celsius":st.column_config.NumberColumn("Temp(°C)",format="%.1f°C"), "fall_detected_today":st.column_config.NumberColumn("Falls",format="%d")}, hide_index=True )
            try: csv_chw_tasks = task_df_display_final_sorted.to_csv(index=False).encode('utf-8'); st.download_button(label="📥 Download Task List (CSV)", data=csv_chw_tasks, file_name=f"chw_tasks_{selected_view_date_chw.strftime('%Y%m%d')}.csv", mime="text/csv", key="chw_task_download_v7")
            except Exception as e_csv: logger.error(f"CHW Task CSV Download Error: {e_csv}"); st.warning("Could not prepare task list for download.")
        elif not current_day_chw_df.empty: st.info("No specific tasks from today's CHW encounters.")
        else: st.info("No CHW encounters recorded today, so no tasks to display.")

    with tab_chw_trends:
        st.subheader(f"My Activity Trends (Selected Range: {selected_trend_date_range_chw[0].strftime('%d %b')} - {selected_trend_date_range_chw[1].strftime('%d %b %Y')})")
        
        # Use the main health_df for trends, then filter by the selected *trend date range*
        trends_base_df_chw = health_df_chw_main.copy()
        if not pd.api.types.is_datetime64_ns_dtype(trends_base_df_chw['encounter_date']):
            trends_base_df_chw['encounter_date'] = pd.to_datetime(trends_base_df_chw['encounter_date'], errors='coerce')
        trends_base_df_chw.dropna(subset=['encounter_date'], inplace=True)

        # Ensure date objects for comparison
        start_trend_filter = selected_trend_date_range_chw[0]
        end_trend_filter = selected_trend_date_range_chw[1]

        chw_trends_data_filtered_range = trends_base_df_chw[
            (trends_base_df_chw['encounter_date'].dt.date >= start_trend_filter) &
            (trends_base_df_chw['encounter_date'].dt.date <= end_trend_filter)
        ].copy()
        
        # Further filter for CHW-specific encounters if a general CHW flag is used in health_df_chw_main
        # Or, if the CHW system implies a specific user, this would filter by 'chw_id' or similar.
        # For this demo, we'll assume chw_trends_data_filtered_range already contains relevant CHW data for the period.
        # If 'selected_chw_zone' is NOT 'All Zones', we can apply it here too if trends should be zone-specific.
        if selected_chw_zone != "All Zones" and 'zone_id' in chw_trends_data_filtered_range.columns:
            chw_trends_data_filtered_range = chw_trends_data_filtered_range[chw_trends_data_filtered_range['zone_id'] == selected_chw_zone]


        if not chw_trends_data_filtered_range.empty:
            cols_chw_trend_tab_display = st.columns(2)
            with cols_chw_trend_tab_display[0]:
                visits_trend_data_range = get_trend_data(chw_trends_data_filtered_range, value_col='patient_id', date_col='encounter_date', period='D', agg_func='nunique')
                if not visits_trend_data_range.empty: st.plotly_chart(plot_annotated_line_chart(visits_trend_data_range, "Daily Patients Visited", y_axis_title="# Patients", height=app_config.COMPACT_PLOT_HEIGHT-20, date_format="%a, %d %b"), use_container_width=True)
                else: st.caption(f"No patient visit data for trend in selected range ({selected_chw_zone}).")
            with cols_chw_trend_tab_display[1]:
                if 'ai_followup_priority_score' in chw_trends_data_filtered_range.columns:
                    high_prio_df_trend_range = chw_trends_data_filtered_range[chw_trends_data_filtered_range['ai_followup_priority_score'] >= 80]
                    high_prio_trend_range = get_trend_data(high_prio_df_trend_range, value_col='patient_id', date_col='encounter_date', period='D', agg_func='nunique')
                    if not high_prio_trend_range.empty: st.plotly_chart(plot_annotated_line_chart(high_prio_trend_range, "High Priority Follow-ups", y_axis_title="# Follow-ups", height=app_config.COMPACT_PLOT_HEIGHT-20, date_format="%a, %d %b"), use_container_width=True)
                    else: st.caption(f"No high priority follow-up data for trend in selected range ({selected_chw_zone}).")
                else: st.caption("AI Follow-up Priority Score not available for trend.")
        else:
            st.info(f"Not enough data in the selected range ({start_trend_filter.strftime('%d %b %Y')} to {end_trend_filter.strftime('%d %b %Y')}, Zone: {selected_chw_zone}) for activity trends display.")
