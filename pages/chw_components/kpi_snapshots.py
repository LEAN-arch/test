# test/pages/chw_components/kpi_snapshots.py
import streamlit as st
import pandas as pd
import numpy as np
from config import app_config
from utils.ui_visualization_helpers import render_kpi_card

def render_chw_daily_kpis(chw_daily_kpis, current_day_chw_df):
    # Subheader for "Daily Snapshot KPIs" is managed by the main 1_chw_dashboard.py page
    
    kpi_cols_chw_overview = st.columns(4)
    with kpi_cols_chw_overview[0]:
        visits_val = chw_daily_kpis.get('visits_today', 0)
        render_kpi_card("Visits Today", str(visits_val), "🚶‍♀️", 
                        status="Good High" if visits_val >= 10 else ("Moderate" if visits_val >= 5 else "Low"), 
                        help_text="Total unique patients with CHW encounters on the selected date.")
    with kpi_cols_chw_overview[1]:
        high_priority_tasks_count = 0
        # Ensure column exists and has data before trying to access it for nunique()
        if not current_day_chw_df.empty and \
           'ai_followup_priority_score' in current_day_chw_df.columns and \
           current_day_chw_df['ai_followup_priority_score'].notna().any() :
             high_priority_tasks_count = current_day_chw_df[current_day_chw_df['ai_followup_priority_score'] >= 80]['patient_id'].nunique()
        render_kpi_card("AI High-Prio Follow-ups", str(high_priority_tasks_count), "🎯", 
                        status="High" if high_priority_tasks_count > 2 else ("Moderate" if high_priority_tasks_count > 0 else "Low"), 
                        help_text="Patients needing follow-up based on high AI priority scores from today's encounters.")
    with kpi_cols_chw_overview[2]:
        avg_risk_visited = chw_daily_kpis.get('avg_patient_risk_visited_today', np.nan) # Default to NaN
        risk_display_text = f"{avg_risk_visited:.0f}" if pd.notna(avg_risk_visited) else "N/A"
        risk_semantic = "Neutral" # Default status
        if pd.notna(avg_risk_visited):
            if avg_risk_visited >= app_config.RISK_THRESHOLDS['high']: risk_semantic = "High"
            elif avg_risk_visited >= app_config.RISK_THRESHOLDS['moderate']: risk_semantic = "Moderate"
            else: risk_semantic = "Low"
        render_kpi_card("Avg. Risk (Visited Today)", risk_display_text, "📈", 
                        status=risk_semantic, 
                        help_text="Average AI risk score of unique patients with CHW encounters today.")
    with kpi_cols_chw_overview[3]:
        fever_val = chw_daily_kpis.get('patients_fever_visited_today', 0)
        render_kpi_card("Fever Alerts (Today)", str(fever_val), "🔥", 
                        status="High" if fever_val > 0 else "Low", 
                        help_text=f"Patients with CHW encounters today and temperature ≥ {app_config.SKIN_TEMP_FEVER_THRESHOLD_C}°C.")

    st.markdown("##### Patient Wellness Indicators (Visited Patients Today)")
    kpi_cols_chw_wellness = st.columns(3)
    with kpi_cols_chw_wellness[0]:
        low_spo2 = chw_daily_kpis.get('patients_low_spo2_visited_today', 0)
        render_kpi_card("Low SpO2 Alerts", str(low_spo2), "💨", 
                        status="High" if low_spo2 > 0 else "Low", 
                        help_text=f"Patients with SpO2 < {app_config.SPO2_LOW_THRESHOLD_PCT}%.")
    with kpi_cols_chw_wellness[1]:
        avg_steps_val = chw_daily_kpis.get('avg_patient_steps_visited_today', np.nan)
        steps_display_text = f"{avg_steps_val:,.0f}" if pd.notna(avg_steps_val) else "N/A"
        steps_status = "Neutral" # Default status
        if pd.notna(avg_steps_val):
            if avg_steps_val >= app_config.TARGET_DAILY_STEPS : steps_status = "Good High"
            elif avg_steps_val >= app_config.TARGET_DAILY_STEPS * 0.6 : steps_status = "Moderate"
            else: steps_status = "Bad Low"
        render_kpi_card("Avg. Patient Steps", steps_display_text, "👣", 
                        status=steps_status, 
                        help_text=f"Avg. daily steps of patients with CHW encounters. Target: {app_config.TARGET_DAILY_STEPS:,.0f} steps.")
    with kpi_cols_chw_wellness[2]:
        falls_val = chw_daily_kpis.get('patients_fall_detected_today', 0)
        render_kpi_card("Falls Detected", str(falls_val), "🤕", 
                        status="High" if falls_val > 0 else "Low", 
                        help_text="Patients with a fall detected today among visited.")
    # The "---" markdown separator is handled by the main page script before calling the next component.
