# test/pages/chw_components/alerts_display.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.ui_visualization_helpers import render_traffic_light
# from config import app_config # Not strictly needed if thresholds passed or implicit in alert_df

def render_chw_alerts_tab(patient_alerts_tasks_df, current_day_chw_df, selected_view_date_chw, selected_chw_zone_daily):
    # Main subheader (e.g., "Critical Patient Alerts for ...") is managed by the main 1_chw_dashboard.py page.
    # This component just renders the list of traffic lights.

    if patient_alerts_tasks_df.empty:
        # Determine message based on whether there was any CHW activity that day for that zone
        zone_context_msg_alerts = f" in {selected_chw_zone_daily}" if selected_chw_zone_daily != "All Zones" else " across all assigned zones"
        if not current_day_chw_df.empty: # Data existed for the day/zone, but no alerts generated from it
            st.success(f"✅ No critical patient alerts identified from encounters on {selected_view_date_chw.strftime('%d %b %Y')}{zone_context_msg_alerts}.")
        # If current_day_chw_df is also empty, the main page (1_chw_dashboard.py) already shows a more general "no data" message
        # so we don't need to repeat "No CHW encounters..." here. Just returning is fine.
        return

    # Sort alerts for display (highest priority first)
    # Determine sort column: prefer ai_followup_priority_score, then priority_score (composite), then ai_risk_score
    sort_col_alert = None
    if 'ai_followup_priority_score' in patient_alerts_tasks_df.columns and patient_alerts_tasks_df['ai_followup_priority_score'].notna().any():
        sort_col_alert = 'ai_followup_priority_score'
    elif 'priority_score' in patient_alerts_tasks_df.columns and patient_alerts_tasks_df['priority_score'].notna().any():
        sort_col_alert = 'priority_score'
    elif 'ai_risk_score' in patient_alerts_tasks_df.columns and patient_alerts_tasks_df['ai_risk_score'].notna().any():
        sort_col_alert = 'ai_risk_score' # Fallback sort
    
    if sort_col_alert:
        alerts_to_display = patient_alerts_tasks_df.sort_values(by=sort_col_alert, ascending=False)
    else: 
        alerts_to_display = patient_alerts_tasks_df # Display as is if no suitable sort column with data
        st.caption("Note: Alerts not sorted by AI priority as priority scores are unavailable.")
        
    # Determine the temperature column to use from the daily data context for consistency in display
    # This ensures we look for the temperature value under the same column name for each alert_row.
    temp_col_to_use_in_alert_details = None
    # Use current_day_chw_df for checking column preference, as patient_alerts_tasks_df is a derivative.
    base_df_for_temp_check = current_day_chw_df if not current_day_chw_df.empty else patient_alerts_tasks_df 

    if 'vital_signs_temperature_celsius' in base_df_for_temp_check.columns and base_df_for_temp_check['vital_signs_temperature_celsius'].notna().any():
        temp_col_to_use_in_alert_details = 'vital_signs_temperature_celsius'
    elif 'max_skin_temp_celsius' in base_df_for_temp_check.columns and base_df_for_temp_check['max_skin_temp_celsius'].notna().any():
        temp_col_to_use_in_alert_details = 'max_skin_temp_celsius'

    for _, alert_row in alerts_to_display.head(15).iterrows():
        priority_display_val = alert_row.get(sort_col_alert, 0) if sort_col_alert else alert_row.get('ai_risk_score', 0) # Get the value used for sorting/status
        
        alert_status_light = "High" if priority_display_val >= 80 else \
                             ("Moderate" if priority_display_val >= 60 else "Low")
        
        details_parts = []
        if pd.notna(alert_row.get('ai_risk_score')): 
            details_parts.append(f"AI Risk: {alert_row['ai_risk_score']:.0f}")
        if pd.notna(alert_row.get('ai_followup_priority_score')): 
            details_parts.append(f"AI Prio: {alert_row['ai_followup_priority_score']:.0f}")
        if pd.notna(alert_row.get('min_spo2_pct')): 
            details_parts.append(f"SpO2: {alert_row['min_spo2_pct']:.0f}%")
        
        temp_val_for_display = alert_row.get(temp_col_to_use_in_alert_details) if temp_col_to_use_in_alert_details and temp_col_to_use_in_alert_details in alert_row else None
        if pd.notna(temp_val_for_display): 
            details_parts.append(f"Temp: {temp_val_for_display:.1f}°C")
            
        if pd.notna(alert_row.get('fall_detected_today')) and alert_row['fall_detected_today'] > 0:
            details_parts.append(f"Falls: {int(alert_row['fall_detected_today'])}")
        
        patient_id_str = str(alert_row.get('patient_id','N/A'))
        condition_str = str(alert_row.get('condition','N/A'))
        display_condition_str = f" ({condition_str})" if condition_str and condition_str.lower() not in ['unknown', 'n/a'] else ""
            
        msg = f"Patient {patient_id_str}{display_condition_str}"
        reason_str = str(alert_row.get('alert_reason', 'Review Case'))
        detail_str = reason_str + (" | " + " / ".join(details_parts) if details_parts else "")
        
        render_traffic_light(message=msg, status=alert_status_light, details=detail_str)
