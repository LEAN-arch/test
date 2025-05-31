# test/pages/1_chw_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
from datetime import date

# Assuming your project structure is such that 'config' and 'utils' are directly importable
# If 'health_hub' is the root, then from health_hub.config import app_config
# For this setup, using flat imports from a 'test' root perspective:
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    get_chw_summary,
    get_patient_alerts_for_chw,
    get_trend_data
)
from utils.ai_analytics_engine import apply_ai_models # Import the AI engine
from utils.ui_visualization_helpers import (
    render_kpi_card,
    render_traffic_light,
    plot_annotated_line_chart
)

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="CHW Dashboard - Health Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_chw(): # Renamed to avoid conflict if other pages have different load_css needs
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
    logger.info("CHW Dashboard: Attempting to load and enrich health records...")
    # Using the expanded CSV filename from Deliverable 1 (assuming app_config.HEALTH_RECORDS_CSV points to it)
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    if health_df_raw.empty:
        logger.error("CHW Dashboard: Failed to load health records or returned an empty DataFrame from core processing.")
        return pd.DataFrame() # Return empty DataFrame, will be handled by UI

    # Apply AI models to generate/update risk scores and priority scores
    # This call to apply_ai_models should come AFTER basic loading and cleaning in load_health_records
    health_df_ai_enriched = apply_ai_models(health_df_raw)
    if health_df_ai_enriched.empty and not health_df_raw.empty: # If AI enrichment failed but raw was fine
        logger.warning("CHW Dashboard: AI enrichment returned empty DF, using raw data instead.")
        health_df_ai_enriched = health_df_raw # Fallback
    elif health_df_ai_enriched.empty:
        logger.error("CHW Dashboard: Both raw load and AI enrichment resulted in empty DF.")
        return pd.DataFrame()
        
    logger.info(f"CHW Dashboard: Successfully loaded and AI-enriched {len(health_df_ai_enriched)} health records.")
    return health_df_ai_enriched

health_df_chw_main = get_chw_dashboard_data_enriched()

# --- Main Page Rendering ---
if health_df_chw_main.empty:
    st.error("🚨 **Critical Error:** Could not load or process necessary health records for the CHW Dashboard. Key insights and task lists cannot be generated. Please verify data sources, configurations, and AI model processing steps.")
    logger.critical("CHW Dashboard cannot render: health_df_chw_main is None or empty after loading and AI enrichment.")
    st.stop()
else:
    st.title("🧑‍⚕️ Community Health Worker (CHW) Dashboard")
    st.markdown("**Daily Patient Prioritization, Field Insights, & Wellness Monitoring**")
    st.markdown("---")

    # --- Sidebar Filters ---
    if os.path.exists(app_config.APP_LOGO):
        st.sidebar.image(app_config.APP_LOGO, use_column_width='auto')
        st.sidebar.markdown("---")
    else:
        logger.warning(f"Sidebar logo not found on CHW Dashboard at {app_config.APP_LOGO}")

    st.sidebar.header("🗓️ CHW Filters")

    # Date selection robust handling
    min_date_available = health_df_chw_main['encounter_date'].min().date() if 'encounter_date' in health_df_chw_main and not health_df_chw_main['encounter_date'].dropna().empty else date.today() - pd.Timedelta(days=90)
    max_date_available = health_df_chw_main['encounter_date'].max().date() if 'encounter_date' in health_df_chw_main and not health_df_chw_main['encounter_date'].dropna().empty else date.today()
    
    # Ensure min_date is not after max_date
    if min_date_available > max_date_available:
        min_date_available = max_date_available

    # Default to the latest date with data, or today if no data
    default_view_date = max_date_available

    selected_view_date_chw = st.sidebar.date_input(
        "View Data For Date:",
        value=default_view_date,
        min_value=min_date_available,
        max_value=max_date_available,
        key="chw_daily_view_date_selector_v5",
        help="Select the date for which you want to view daily summaries, tasks, and patient alerts."
    )

    # Filter data for the selected date
    # Ensure 'encounter_date_obj' is created correctly for filtering
    health_df_chw_main['encounter_date_obj'] = pd.to_datetime(health_df_chw_main['encounter_date']).dt.date
    current_day_chw_df = health_df_chw_main[health_df_chw_main['encounter_date_obj'] == selected_view_date_chw].copy()
    
    if current_day_chw_df.empty:
        st.info(f"ℹ️ No CHW-related encounter data recorded for {selected_view_date_chw.strftime('%A, %B %d, %Y')}.")
        # Allow page to load with empty sections, but log it
        logger.info(f"CHW Dashboard: No data found for selected date {selected_view_date_chw}.")
    else:
        logger.debug(f"CHW Dashboard: Data for selected date {selected_view_date_chw} has {len(current_day_chw_df)} CHW-related encounters.")


    # Get CHW summary KPIs for the filtered daily data
    chw_daily_kpis = get_chw_summary(current_day_chw_df)
    logger.debug(f"CHW Daily KPIs for {selected_view_date_chw}: {chw_daily_kpis}")

    st.subheader(f"Daily Snapshot: {selected_view_date_chw.strftime('%A, %B %d, %Y')}")
    
    kpi_cols_chw_overview = st.columns(4)
    with kpi_cols_chw_overview[0]:
        visits_val = chw_daily_kpis.get('visits_today', 0)
        render_kpi_card("Visits Today", str(visits_val), "🚶‍♀️",
                        status="Good High" if visits_val >= 10 else ("Moderate" if visits_val >= 5 else "Low"),
                        help_text="Total unique patients with CHW encounters on the selected date.")
    with kpi_cols_chw_overview[1]:
        # 'Key Disease Tasks' can be more dynamic based on AI followup score or critical referral needs
        high_priority_tasks = current_day_chw_df[current_day_chw_df['ai_followup_priority_score'] >= 80]['patient_id'].nunique() if 'ai_followup_priority_score' in current_day_chw_df else 0
        tb_contacts = chw_daily_kpis.get('tb_contacts_to_trace_today',0)
        key_tasks = high_priority_tasks + tb_contacts # Example, could add more task types
        render_kpi_card("High-Priority Tasks", str(key_tasks), "📋",
                        status="High" if key_tasks > 5 else ("Moderate" if key_tasks > 0 else "Low"),
                        help_text="Sum of patients with AI high-priority follow-up scores and pending TB contact tracing tasks.")
    with kpi_cols_chw_overview[2]:
        avg_risk_visited = chw_daily_kpis.get('avg_patient_risk_visited_today', 0.0)
        risk_display_text = f"{avg_risk_visited:.0f}" if pd.notna(avg_risk_visited) and avg_risk_visited > 0 else "N/A"
        risk_semantic = "High" if avg_risk_visited >= app_config.RISK_THRESHOLDS['high'] else \
                        "Moderate" if avg_risk_visited >= app_config.RISK_THRESHOLDS['moderate'] else "Low"
        if risk_display_text == "N/A": risk_semantic = "Neutral"
        render_kpi_card("Avg. Risk (Visited)", risk_display_text, "🎯", status=risk_semantic,
                        help_text="Average AI-calculated risk score of unique patients with CHW encounters today.")
    with kpi_cols_chw_overview[3]:
        followups_needed = chw_daily_kpis.get('high_risk_followups_today', 0) # Based on general high risk
        render_kpi_card("High AI Risk Follow-ups", str(followups_needed), "⚠️",
                        status="High" if followups_needed > 2 else ("Moderate" if followups_needed > 0 else "Low"),
                        help_text="Number of unique patients visited today flagged with high general AI risk.")

    st.markdown("##### Patient Wellness Indicators (For Visited Patients Today)")
    kpi_cols_chw_wellness = st.columns(4) # Added Falls Detected
    with kpi_cols_chw_wellness[0]:
        low_spo2 = chw_daily_kpis.get('patients_low_spo2_visited_today', 0)
        render_kpi_card("Low SpO2 Alerts", str(low_spo2), "💨",
                        status="High" if low_spo2 > 0 else "Low",
                        help_text=f"Patients with CHW encounters today and SpO2 < {app_config.SPO2_LOW_THRESHOLD_PCT}%.")
    with kpi_cols_chw_wellness[1]:
        fever = chw_daily_kpis.get('patients_fever_visited_today', 0)
        render_kpi_card("Fever Alerts", str(fever), "🔥", status="High" if fever > 0 else "Low",
                        help_text=f"Patients with CHW encounters today and temperature ≥ {app_config.SKIN_TEMP_FEVER_THRESHOLD_C}°C.")
    with kpi_cols_chw_wellness[2]:
        avg_steps_val = chw_daily_kpis.get('avg_patient_steps_visited_today', 0.0)
        steps_display_text = f"{avg_steps_val:,.0f}" if pd.notna(avg_steps_val) and avg_steps_val > 0 else "N/A"
        steps_status = "Good High" if avg_steps_val >= app_config.TARGET_DAILY_STEPS else \
                       "Moderate" if avg_steps_val >= app_config.TARGET_DAILY_STEPS * 0.6 else "Bad Low"
        if steps_display_text == "N/A": steps_status = "Neutral"
        render_kpi_card("Avg. Patient Steps", steps_display_text, "👣", status=steps_status,
                        help_text=f"Average daily steps of patients with CHW encounters. Target: {app_config.TARGET_DAILY_STEPS:,.0f} steps.")
    with kpi_cols_chw_wellness[3]:
        falls_val = chw_daily_kpis.get('patients_fall_detected_today', 0)
        render_kpi_card("Falls Detected", str(falls_val), "🤕", status="High" if falls_val > 0 else "Low",
                        help_text="Patients with CHW encounters today who had a fall detected.")
    st.markdown("---")

    tab_alerts, tab_tasks = st.tabs(["🚨 Critical Patient Alerts", "📋 Detailed Task List"])

    # Generate patient alerts/tasks using the AI-enriched daily data
    patient_alerts_tasks_df = get_patient_alerts_for_chw(
        current_day_chw_df, # This df now contains AI scores
        risk_threshold_moderate=app_config.RISK_THRESHOLDS['chw_alert_moderate'],
        risk_threshold_high=app_config.RISK_THRESHOLDS['chw_alert_high'] # Used as fallback if no AI prio
    )
    logger.debug(f"CHW Page: Patient alerts/tasks df for {selected_view_date_chw} has {len(patient_alerts_tasks_df)} rows. Columns: {patient_alerts_tasks_df.columns.tolist() if not patient_alerts_tasks_df.empty else 'N/A'}")


    with tab_alerts:
        st.subheader("Critical Patient Alerts for Today")
        if not patient_alerts_tasks_df.empty:
            # Sort by ai_followup_priority_score (if exists) or priority_score
            sort_col_alert = 'ai_followup_priority_score' if 'ai_followup_priority_score' in patient_alerts_tasks_df.columns else 'priority_score'
            alerts_to_display = patient_alerts_tasks_df.sort_values(by=sort_col_alert, ascending=False)

            for _, alert_row in alerts_to_display.head(15).iterrows():
                # Determine traffic light status based on AI Followup Prio or general Risk
                priority_val = alert_row.get(sort_col_alert, 0)
                alert_status_for_light = "High" if priority_val >= 80 else \
                                         "Moderate" if priority_val >= 60 else "Low" # Example thresholds for prio score

                alert_details_parts = []
                if pd.notna(alert_row.get('ai_risk_score')): alert_details_parts.append(f"AI Risk: {alert_row['ai_risk_score']:.0f}")
                if pd.notna(alert_row.get('ai_followup_priority_score')): alert_details_parts.append(f"AI Prio: {alert_row['ai_followup_priority_score']:.0f}")
                if pd.notna(alert_row.get('min_spo2_pct')): alert_details_parts.append(f"SpO2: {alert_row['min_spo2_pct']:.0f}%")
                
                temp_col_traffic = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in alert_row and pd.notna(alert_row['vital_signs_temperature_celsius']) else 'max_skin_temp_celsius'
                if pd.notna(alert_row.get(temp_col_traffic)): alert_details_parts.append(f"Temp: {alert_row[temp_col_traffic]:.1f}°C")
                if pd.notna(alert_row.get('fall_detected_today')) and alert_row['fall_detected_today'] > 0:
                    alert_details_parts.append(f"Falls: {int(alert_row['fall_detected_today'])}")

                alert_message = f"Patient {alert_row.get('patient_id','N/A')} ({alert_row.get('condition','N/A')})"
                alert_detail_string = alert_row.get('alert_reason', 'Review Case') + (" | " + " / ".join(alert_details_parts) if alert_details_parts else "")

                render_traffic_light(message=alert_message, status=alert_status_for_light, details=alert_detail_string)
        elif not current_day_chw_df.empty: # If there were CHW encounters but no alerts
            st.success("✅ No critical patient alerts identified for today based on current CHW encounters and criteria.")
        else: # No CHW encounters for the day
            st.info("No CHW encounters recorded for today, so no alerts to display.")

    with tab_tasks:
        st.subheader("Prioritized Task List for Today")
        if not patient_alerts_tasks_df.empty:
            # Show AI scores in the task list
            task_list_cols_to_show = ['patient_id', 'zone_id', 'condition',
                                      'ai_risk_score', 'ai_followup_priority_score',
                                      'alert_reason', 'referral_status',
                                      'min_spo2_pct', temp_col_traffic, # temp_col_traffic defined in alerts tab
                                      'fall_detected_today']
            task_df_display = patient_alerts_tasks_df[[col for col in task_list_cols_to_show if col in patient_alerts_tasks_df.columns]].copy()
            task_df_display.rename(columns={temp_col_traffic: 'latest_temp_celsius'}, inplace=True)


            # Sort by AI Followup Priority, then by AI Risk
            sort_cols_task = []
            if 'ai_followup_priority_score' in task_df_display.columns: sort_cols_task.append('ai_followup_priority_score')
            if 'ai_risk_score' in task_df_display.columns: sort_cols_task.append('ai_risk_score')
            
            if sort_cols_task:
                task_df_display_sorted = task_df_display.sort_values(by=sort_cols_task, ascending=[False, False])
            else:
                task_df_display_sorted = task_df_display # No AI scores to sort by specifically

            st.dataframe(
                task_df_display_sorted, use_container_width=True, height=450,
                column_config={
                    "patient_id": st.column_config.TextColumn("Patient ID"),
                    "zone_id": st.column_config.TextColumn("Zone"),
                    "ai_risk_score": st.column_config.ProgressColumn("AI Risk", help="Patient's AI-calculated risk score (0-100).", format="%d", min_value=0, max_value=100),
                    "ai_followup_priority_score": st.column_config.ProgressColumn("AI Followup Prio.", help="AI-calculated Follow-up Priority (0-100).", format="%d", min_value=0, max_value=100),
                    "alert_reason": st.column_config.TextColumn("Primary Task/Alert Reason", width="large"),
                    "min_spo2_pct": st.column_config.NumberColumn("Min SpO2 (%)", format="%d%%"),
                    "latest_temp_celsius": st.column_config.NumberColumn("Latest Temp (°C)", format="%.1f°C"),
                    "fall_detected_today": st.column_config.NumberColumn("Falls Today", format="%d"),
                },
                hide_index=True
            )
            try:
                csv_chw_tasks = task_df_display_sorted.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download Task List (CSV)", data=csv_chw_tasks, file_name=f"chw_tasks_{selected_view_date_chw.strftime('%Y%m%d')}.csv", mime="text/csv", key="chw_task_list_download_v5")
            except Exception as e_csv_chw:
                logger.error(f"CHW Dashboard: Error preparing task list CSV: {e_csv_chw}", exc_info=True)
                st.warning("Could not prepare task list for download.")
        elif not current_day_chw_df.empty:
            st.info("No specific tasks or follow-ups identified from today's CHW encounters.")
        else:
            st.info("No CHW encounters recorded for today, so no tasks to display.")

    st.markdown("---")
    st.subheader(f"Overall Patient Wellness & Activity Trends (Last {app_config.DEFAULT_DATE_RANGE_DAYS_TREND} Days ending {selected_view_date_chw.strftime('%B %d, %Y')})")

    trend_period_end_date_chw = pd.to_datetime(selected_view_date_chw).normalize() # Ensure it's a Timestamp
    trend_period_start_date_chw = trend_period_end_date_chw - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)

    # Use main health_df for trends (before daily filtering)
    # Ensure encounter_date is datetime for trend filtering
    health_df_for_trends_chw = health_df_chw_main.copy()
    if not pd.api.types.is_datetime64_ns_dtype(health_df_for_trends_chw['encounter_date']):
        health_df_for_trends_chw['encounter_date'] = pd.to_datetime(health_df_for_trends_chw['encounter_date'], errors='coerce')
    
    health_df_for_trends_chw.dropna(subset=['encounter_date'], inplace=True)

    chw_trend_df_source = health_df_for_trends_chw[
        (health_df_for_trends_chw['encounter_date'] >= trend_period_start_date_chw) &
        (health_df_for_trends_chw['encounter_date'] <= trend_period_end_date_chw)
    ].copy()

    logger.debug(f"CHW Page: Trend DF source has {len(chw_trend_df_source)} rows for period {trend_period_start_date_chw} to {trend_period_end_date_chw}.")

    if not chw_trend_df_source.empty:
        trend_cols_chw = st.columns(2)
        with trend_cols_chw[0]:
            # Trend of average AI risk score of patients encountered by CHWs
            # Filter for CHW encounters if specific flag exists (e.g. 'chw_visit' == 1 or specific 'encounter_type')
            chw_encounter_trend_df = chw_trend_df_source.copy() # By default, assume all data in scope is CHW-relevant
            if 'chw_visit' in chw_encounter_trend_df.columns:
                chw_encounter_trend_df = chw_encounter_trend_df[chw_encounter_trend_df['chw_visit'] == 1]
            
            if not chw_encounter_trend_df.empty and 'ai_risk_score' in chw_encounter_trend_df.columns:
                overall_risk_trend = get_trend_data(chw_encounter_trend_df, 'ai_risk_score', date_col='encounter_date', period='D', agg_func='mean')
                if not overall_risk_trend.empty:
                    st.plotly_chart(plot_annotated_line_chart(
                        overall_risk_trend, "Daily Avg. AI Risk Score (Visited Patients)",
                        y_axis_title="Avg. Risk Score", height=app_config.COMPACT_PLOT_HEIGHT,
                        target_line=app_config.TARGET_PATIENT_RISK_SCORE, target_label="Target Avg. Risk",
                        show_anomalies=True, date_format="%d %b"
                    ), use_container_width=True)
                else: st.caption("No AI risk score trend data available for CHW visited patients this period.")
            else: st.caption("Missing 'ai_risk_score' or no CHW encounters for risk trend.")

        with trend_cols_chw[1]:
            # Trend of daily CHW visits (unique patients seen by CHW)
            if not chw_encounter_trend_df.empty and 'patient_id' in chw_encounter_trend_df.columns: # using pre-filtered chw_encounter_trend_df
                visits_trend_chw_actual = get_trend_data(chw_encounter_trend_df, value_col='patient_id', date_col='encounter_date', period='D', agg_func='nunique')
                if not visits_trend_chw_actual.empty:
                    st.plotly_chart(plot_annotated_line_chart(
                        visits_trend_chw_actual, "Daily Unique Patients Visited by CHW",
                        y_axis_title="Number of Patients Visited", height=app_config.COMPACT_PLOT_HEIGHT,
                        show_anomalies=True, date_format="%d %b" # Example target daily visits: target_line=10
                    ), use_container_width=True)
                else: st.caption("No CHW visits trend data available for this period.")
            else: st.caption("No CHW encounter data for visits trend.")
    else:
        st.info(f"Not enough historical data (min {app_config.DEFAULT_DATE_RANGE_DAYS_TREND} days needed) ending on {selected_view_date_chw.strftime('%Y-%m-%d')} for overall CHW trends display.")
