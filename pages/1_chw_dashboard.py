# test/pages/1_chw_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
from datetime import date
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
    plot_bar_chart # For potential simple demographic bar
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

    if os.path.exists(app_config.APP_LOGO):
        st.sidebar.image(app_config.APP_LOGO, use_column_width='auto')
        st.sidebar.markdown("---")
    st.sidebar.header("🗓️ CHW Filters")

    # Date selection for CHW Daily View
    min_date_available = health_df_chw_main['encounter_date'].min().date() if 'encounter_date' in health_df_chw_main and not health_df_chw_main['encounter_date'].dropna().empty else date.today() - pd.Timedelta(days=90)
    max_date_available = health_df_chw_main['encounter_date'].max().date() if 'encounter_date' in health_df_chw_main and not health_df_chw_main['encounter_date'].dropna().empty else date.today()
    if min_date_available > max_date_available: min_date_available = max_date_available
    default_view_date = max_date_available

    selected_view_date_chw = st.sidebar.date_input(
        "View Data For Date:", value=default_view_date,
        min_value=min_date_available, max_value=max_date_available,
        key="chw_daily_view_date_selector_v7",
        help="Select the date for daily summaries, tasks, and alerts."
    )
    
    # Filter data for the selected date (already robustly done in previous iterations)
    health_df_chw_main['encounter_date_obj'] = pd.to_datetime(health_df_chw_main['encounter_date']).dt.date
    current_day_chw_df = health_df_chw_main[health_df_chw_main['encounter_date_obj'] == selected_view_date_chw].copy()

    # Zone filter specific to CHW (example, would need CHW-to-zone mapping in real system)
    chw_zones = sorted(current_day_chw_df['zone_id'].unique().tolist()) if not current_day_chw_df.empty else ["N/A"]
    selected_chw_zone = "All Zones" # Default
    if len(chw_zones) > 1 and chw_zones != ["N/A"]:
        selected_chw_zone = st.sidebar.selectbox(
            "Filter by Your Assigned Zone (for today's data):",
            options=["All Zones"] + chw_zones,
            index=0,
            key="chw_zone_filter_v1"
        )
        if selected_chw_zone != "All Zones":
            current_day_chw_df = current_day_chw_df[current_day_chw_df['zone_id'] == selected_chw_zone]
    
    # Derivations for the selected date and zone
    if current_day_chw_df.empty:
        zone_context = f" in {selected_chw_zone}" if selected_chw_zone != "All Zones" else ""
        st.info(f"ℹ️ No CHW-related encounter data recorded for {selected_view_date_chw.strftime('%A, %B %d, %Y')}{zone_context}.")
        chw_daily_kpis = get_chw_summary(pd.DataFrame(columns=health_df_chw_main.columns)) # Pass empty df with schema
        patient_alerts_tasks_df = get_patient_alerts_for_chw(pd.DataFrame(columns=health_df_chw_main.columns))
    else:
        chw_daily_kpis = get_chw_summary(current_day_chw_df)
        patient_alerts_tasks_df = get_patient_alerts_for_chw(current_day_chw_df, risk_threshold_moderate=app_config.RISK_THRESHOLDS['chw_alert_moderate'], risk_threshold_high=app_config.RISK_THRESHOLDS['chw_alert_high'])
    
    logger.debug(f"CHW Daily KPIs for {selected_view_date_chw} (Zone: {selected_chw_zone}): {chw_daily_kpis}")
    zone_display_for_title = f"({selected_chw_zone})" if selected_chw_zone != "All Zones" else "(All Assigned Zones)"
    st.subheader(f"Daily Snapshot: {selected_view_date_chw.strftime('%A, %B %d, %Y')} {zone_display_for_title}")
    
    # KPIs
    kpi_cols_chw_overview = st.columns(4)
    with kpi_cols_chw_overview[0]: render_kpi_card("Visits Today", str(chw_daily_kpis.get('visits_today', 0)), "🚶‍♀️", status="Good High" if chw_daily_kpis.get('visits_today', 0) >= 10 else "Low")
    with kpi_cols_chw_overview[1]:
        high_prio_tasks_count = 0
        if not current_day_chw_df.empty and 'ai_followup_priority_score' in current_day_chw_df.columns:
             high_prio_tasks_count = current_day_chw_df[current_day_chw_df['ai_followup_priority_score'] >= 80]['patient_id'].nunique()
        render_kpi_card("AI High-Prio Follow-ups", str(high_prio_tasks_count), "🎯", status="High" if high_prio_tasks_count > 2 else "Low", help_text="Patients needing follow-up based on high AI priority scores.")
    with kpi_cols_chw_overview[2]:
        avg_risk_visited = chw_daily_kpis.get('avg_patient_risk_visited_today', np.nan)
        render_kpi_card("Avg. Risk (Visited)", f"{avg_risk_visited:.0f}" if pd.notna(avg_risk_visited) else "N/A", "📈", status="High" if pd.notna(avg_risk_visited) and avg_risk_visited >= 70 else "Low")
    with kpi_cols_chw_overview[3]: render_kpi_card("Fever Alerts", str(chw_daily_kpis.get('patients_fever_visited_today', 0)), "🔥", status="High" if chw_daily_kpis.get('patients_fever_visited_today', 0) > 0 else "Low")
    st.markdown("---")

    # Epidemiology Snippet for CHW for the selected day & zone
    st.markdown(f"##### Epidemiology Watch - {selected_view_date_chw.strftime('%d %b %Y')} - {selected_chw_zone}")
    if not current_day_chw_df.empty:
        epi_cols_chw_local = st.columns(3)
        with epi_cols_chw_local[0]:
            # Symptom surveillance: new reports of specific syndromes
            # Example: Count patients seen today reporting 'Fever' AND ('Cough' OR 'Shortness of breath')
            resp_symptoms_df = current_day_chw_df[
                current_day_chw_df['patient_reported_symptoms'].str.contains('Fever', case=False, na=False) &
                (current_day_chw_df['patient_reported_symptoms'].str.contains('Cough|Shortness of breath', case=False, na=False))
            ]
            new_resp_syndrome_count = resp_symptoms_df['patient_id'].nunique()
            render_kpi_card("New Respiratory Syndromes", str(new_resp_syndrome_count), "🌬️", status="High" if new_resp_syndrome_count > 1 else "Low", help_text="Patients seen today with Fever + Cough/Shortness of Breath.")

        with epi_cols_chw_local[1]:
            # Simple cluster indication: More than X cases of a specific condition in the zone today
            key_condition_for_cluster = "Malaria" # Example
            malaria_cases_today_zone = current_day_chw_df[current_day_chw_df['condition'] == key_condition_for_cluster]['patient_id'].nunique()
            render_kpi_card(f"New {key_condition_for_cluster} Cases", str(malaria_cases_today_zone), "🦟", status="High" if malaria_cases_today_zone >= 2 else "Low", help_text=f"New {key_condition_for_cluster} cases identified in this zone today.")

        with epi_cols_chw_local[2]:
            # Contacts for tracing KPI from chw_daily_kpis
            tb_contacts_val = chw_daily_kpis.get('tb_contacts_to_trace_today', 0)
            render_kpi_card("TB Contacts to Trace", str(tb_contacts_val), "👥", status="High" if tb_contacts_val > 0 else "Low", help_text="Number of TB patient contacts identified today needing tracing/follow-up.")
        
        # Basic Demography for High-Risk Patients Today (Example)
        high_risk_today_df = current_day_chw_df[current_day_chw_df['ai_risk_score'] >= app_config.RISK_THRESHOLDS.get('high', 75)]
        if not high_risk_today_df.empty and 'age' in high_risk_today_df.columns:
            st.markdown("###### Demographics of High AI Risk Patients Seen Today")
            # Create age groups for patients with high risk score from current_day_chw_df
            age_bins = [0, 5, 18, 45, 65, np.inf]
            age_labels = ['0-4', '5-17', '18-44', '45-64', '65+']
            high_risk_today_df.loc[:, 'age_group'] = pd.cut(high_risk_today_df['age'], bins=age_bins, labels=age_labels, right=False)
            age_group_counts = high_risk_today_df['age_group'].value_counts().sort_index().reset_index()
            age_group_counts.columns = ['Age Group', 'Number of High-Risk Patients']
            if not age_group_counts.empty:
                st.plotly_chart(plot_bar_chart(age_group_counts, x_col='Age Group', y_col='Number of High-Risk Patients', title="High AI Risk Patients by Age Group (Today)", height=app_config.COMPACT_PLOT_HEIGHT-50), use_container_width=True)
            else: st.caption("No high AI risk patients with age data for demographic breakdown today.")
        elif not current_day_chw_df.empty : st.caption("No high AI risk patients found today for demographic breakdown.")


    else:
        st.caption(f"No data to display local epidemiology snapshot for {selected_chw_zone} on {selected_view_date_chw.strftime('%d %b %Y')}.")
    st.markdown("---")


    tab_alerts, tab_tasks, tab_chw_trends = st.tabs(["🚨 Critical Patient Alerts", "📋 Detailed Task List", "📈 My Activity Trends (Last 7 Days)"])
    
    with tab_alerts:
        # ... (Alerts display logic from before, using patient_alerts_tasks_df) ...
        st.subheader(f"Critical Patient Alerts for {selected_view_date_chw.strftime('%B %d, %Y')}")
        if not patient_alerts_tasks_df.empty:
            sort_col_alert = 'ai_followup_priority_score' if 'ai_followup_priority_score' in patient_alerts_tasks_df.columns else 'priority_score'
            alerts_to_display = patient_alerts_tasks_df.sort_values(by=sort_col_alert, ascending=False)
            # Define temp_col_traffic consistently for the selected day's data
            temp_col_traffic_alert_tab = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in current_day_chw_df.columns and current_day_chw_df['vital_signs_temperature_celsius'].notna().any() else ('max_skin_temp_celsius' if 'max_skin_temp_celsius' in current_day_chw_df.columns and current_day_chw_df['max_skin_temp_celsius'].notna().any() else None)
            
            for _, alert_row in alerts_to_display.head(15).iterrows():
                priority_val = alert_row.get(sort_col_alert, 0)
                alert_status_light = "High" if priority_val >= 80 else ("Moderate" if priority_val >= 60 else "Low")
                details_parts = []
                if pd.notna(alert_row.get('ai_risk_score')): details_parts.append(f"AI Risk: {alert_row['ai_risk_score']:.0f}")
                if pd.notna(alert_row.get('ai_followup_priority_score')): details_parts.append(f"AI Prio: {alert_row['ai_followup_priority_score']:.0f}")
                if pd.notna(alert_row.get('min_spo2_pct')): details_parts.append(f"SpO2: {alert_row['min_spo2_pct']:.0f}%")
                
                temp_val_display = alert_row.get(temp_col_traffic_alert_tab) if temp_col_traffic_alert_tab else None # Get temp from consistent column
                if pd.notna(temp_val_display): details_parts.append(f"Temp: {temp_val_display:.1f}°C")
                if pd.notna(alert_row.get('fall_detected_today')) and alert_row['fall_detected_today'] > 0: details_parts.append(f"Falls: {int(alert_row['fall_detected_today'])}")
                
                msg = f"Patient {alert_row.get('patient_id','N/A')} ({alert_row.get('condition','N/A')})"
                detail_str = alert_row.get('alert_reason', 'Review Case') + (" | " + " / ".join(details_parts) if details_parts else "")
                render_traffic_light(message=msg, status=alert_status_light, details=detail_str)
        elif not current_day_chw_df.empty: st.success("✅ No critical patient alerts identified from today's encounters.")
        else: st.info("No CHW encounters recorded today, so no alerts to display.")

    with tab_tasks:
        # ... (Task list display logic from before, using patient_alerts_tasks_df) ...
        st.subheader(f"Prioritized Task List for {selected_view_date_chw.strftime('%B %d, %Y')}")
        if not patient_alerts_tasks_df.empty:
            temp_col_task_list = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in current_day_chw_df.columns and current_day_chw_df['vital_signs_temperature_celsius'].notna().any() else ('max_skin_temp_celsius' if 'max_skin_temp_celsius' in current_day_chw_df.columns and current_day_chw_df['max_skin_temp_celsius'].notna().any() else None)
            cols_to_show_task = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'ai_followup_priority_score', 'alert_reason', 'referral_status', 'min_spo2_pct', (temp_col_task_list if temp_col_task_list else 'max_skin_temp_celsius'), 'fall_detected_today'] # ensure a temp column name is always there
            task_df_for_display = patient_alerts_tasks_df[[col for col in cols_to_show_task if col in patient_alerts_tasks_df.columns]].copy()
            rename_col = temp_col_task_list if temp_col_task_list else ('max_skin_temp_celsius' if 'max_skin_temp_celsius' in task_df_for_display else None)
            if rename_col: task_df_for_display.rename(columns={rename_col: 'latest_temp_celsius'}, inplace=True, errors='ignore')
            
            sort_cols_tasks_tab = [col for col in ['ai_followup_priority_score', 'ai_risk_score'] if col in task_df_for_display.columns]
            task_df_display_final_sorted = task_df_for_display.sort_values(by=sort_cols_tasks_tab, ascending=[False]*len(sort_cols_tasks_tab)) if sort_cols_tasks_tab else task_df_for_display
            
            # Ensure DataFrame passed to st.dataframe is serializable
            df_for_st_dataframe_tasks = task_df_display_final_sorted.copy()
            for col in df_for_st_dataframe_tasks.columns:
                if df_for_st_dataframe_tasks[col].dtype == 'object':
                    df_for_st_dataframe_tasks[col] = df_for_st_dataframe_tasks[col].fillna('N/A').astype(str).replace(['nan','None','<NA>'],'N/A', regex=False)

            st.dataframe(df_for_st_dataframe_tasks, use_container_width=True, height=450, column_config={"patient_id": "Patient ID", "ai_risk_score": st.column_config.ProgressColumn("AI Risk",format="%d",min_value=0,max_value=100), "ai_followup_priority_score": st.column_config.ProgressColumn("AI Prio.",format="%d",min_value=0,max_value=100), "alert_reason": st.column_config.TextColumn("Reason",width="large"), "min_spo2_pct":st.column_config.NumberColumn("SpO2(%)",format="%d%%"), "latest_temp_celsius":st.column_config.NumberColumn("Temp(°C)",format="%.1f°C"), "fall_detected_today":st.column_config.NumberColumn("Falls",format="%d")}, hide_index=True )
            try: csv_chw_tasks = task_df_display_final_sorted.to_csv(index=False).encode('utf-8'); st.download_button(label="📥 Download Task List (CSV)", data=csv_chw_tasks, file_name=f"chw_tasks_{selected_view_date_chw.strftime('%Y%m%d')}.csv", mime="text/csv", key="chw_task_download_v7")
            except Exception as e_csv: logger.error(f"CHW Task CSV Download Error: {e_csv}"); st.warning("Could not prepare task list for download.")
        elif not current_day_chw_df.empty: st.info("No specific tasks from today's CHW encounters.")
        else: st.info("No CHW encounters recorded today, so no tasks to display.")

    with tab_chw_trends:
        st.subheader(f"My Recent Activity Trends (Last 7 Days ending {selected_view_date_chw.strftime('%B %d, %Y')})")
        trend_end_chw = pd.to_datetime(selected_view_date_chw).normalize()
        trend_start_chw = trend_end_chw - pd.Timedelta(days=6)
        # Use the main CHW df for trends, then filter by date.
        # Ensure 'encounter_date' is datetime64[ns]
        trends_base_df = health_df_chw_main.copy()
        if not pd.api.types.is_datetime64_ns_dtype(trends_base_df['encounter_date']):
            trends_base_df['encounter_date'] = pd.to_datetime(trends_base_df['encounter_date'], errors='coerce')
        trends_base_df.dropna(subset=['encounter_date'], inplace=True) # Important after coercion

        chw_trends_data = trends_base_df[(trends_base_df['encounter_date'] >= trend_start_chw) & (trends_base_df['encounter_date'] <= trend_end_chw)].copy()
        
        # Placeholder for CHW-specific filter if CHW login was implemented
        # chw_id_for_filter = "CHW001" # example
        # chw_trends_data = chw_trends_data[chw_trends_data['chw_id_column'] == chw_id_for_filter]

        if not chw_trends_data.empty:
            cols_chw_trend_tab_display = st.columns(2)
            with cols_chw_trend_tab_display[0]:
                visits_trend_data = get_trend_data(chw_trends_data, value_col='patient_id', date_col='encounter_date', period='D', agg_func='nunique')
                if not visits_trend_data.empty: st.plotly_chart(plot_annotated_line_chart(visits_trend_data, "Daily Patients Visited", y_axis_title="# Patients", height=app_config.COMPACT_PLOT_HEIGHT-20, date_format="%a, %d %b"), use_container_width=True)
                else: st.caption("No visit data for 7-day trend.")
            with cols_chw_trend_tab_display[1]:
                if 'ai_followup_priority_score' in chw_trends_data.columns:
                    high_prio_df_trend = chw_trends_data[chw_trends_data['ai_followup_priority_score'] >= 80]
                    high_prio_trend = get_trend_data(high_prio_df_trend, value_col='patient_id', date_col='encounter_date', period='D', agg_func='nunique')
                    if not high_prio_trend.empty: st.plotly_chart(plot_annotated_line_chart(high_prio_trend, "High Priority Follow-ups", y_axis_title="# Follow-ups", height=app_config.COMPACT_PLOT_HEIGHT-20, date_format="%a, %d %b"), use_container_width=True)
                    else: st.caption("No high priority follow-up data for 7-day trend.")
                else: st.caption("AI Follow-up Priority Score not available for trend.")
        else:
            st.info(f"Not enough data in the last 7 days ending {selected_view_date_chw.strftime('%Y-%m-%d')} for activity trends display.")
