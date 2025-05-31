# test/pages/chw_components/alerts_display.py
import streamlit as st
import pandas as pd
from utils.ui_visualization_helpers import render_traffic_light
from config import app_config # For thresholds

def render_chw_alerts_tab(patient_alerts_tasks_df, current_day_chw_df, selected_view_date_chw, selected_chw_zone_daily):
    zone_context_msg = f"in {selected_chw_zone_daily}" if selected_chw_zone_daily != "All Zones" else "across all assigned zones"
    st.subheader(f"Critical Patient Alerts for {selected_view_date_chw.strftime('%B %d, %Y')} {zone_context_msg}")
    
    if not patient_alerts_tasks_df.empty:
        sort_col_alert = 'ai_followup_priority_score' if 'ai_followup_priority_score' in patient_alerts_tasks_df.columns else 'priority_score'
        alerts_to_display = patient_alerts_tasks_df.sort_values(by=sort_col_alert, ascending=False)
        
        # Determine temp column consistently for the day's data context
        temp_col_base_df = current_day_chw_df if not current_day_chw_df.empty else patient_alerts_tasks_df # Fallback if daily is empty
        temp_col_traffic_alert = None
        if 'vital_signs_temperature_celsius' in temp_col_base_df.columns and temp_col_base_df['vital_signs_temperature_celsius'].notna().any():
            temp_col_traffic_alert = 'vital_signs_temperature_celsius'
        elif 'max_skin_temp_celsius' in temp_col_base_df.columns and temp_col_base_df['max_skin_temp_celsius'].notna().any():
            temp_col_traffic_alert = 'max_skin_temp_celsius'

        for _, alert_row in alerts_to_display.head(15).iterrows():
            priority_val = alert_row.get(sort_col_alert, 0)
            alert_status_light = "High" if priority_val >= 80 else ("Moderate" if priority_val >= 60 else "Low")
            details_parts = []
            if pd.notna(alert_row.get('ai_risk_score')): details_parts.append(f"AI Risk: {alert_row['ai_risk_score']:.0f}")
            if pd.notna(alert_row.get('ai_followup_priority_score')): details_parts.append(f"AI Prio: {alert_row['ai_followup_priority_score']:.0f}")
            if pd.notna(alert_row.get('min_spo2_pct')): details_parts.append(f"SpO2: {alert_row['min_spo2_pct']:.0f}%")
            
            temp_val_display = alert_row.get(temp_col_traffic_alert) if temp_col_traffic_alert and temp_col_traffic_alert in alert_row else None
            if pd.notna(temp_val_display): details_parts.append(f"Temp: {temp_val_display:.1f}°C")
            if pd.notna(alert_row.get('fall_detected_today')) and alert_row['fall_detected_today'] > 0: details_parts.append(f"Falls: {int(alert_row['fall_detected_today'])}")
            
            msg = f"Patient {alert_row.get('patient_id','N/A')} ({alert_row.get('condition','N/A')})"
            detail_str = alert_row.get('alert_reason', 'Review Case') + (" | " + " / ".join(details_parts) if details_parts else "")
            render_traffic_light(message=msg, status=alert_status_light, details=detail_str)
    elif not current_day_chw_df.empty: # If there were encounters today but no alerts from them
        st.success("✅ No critical patient alerts identified from today's encounters.")
    else: # No encounters for the day (current_day_chw_df is empty)
        st.info(f"No CHW encounters recorded on {selected_view_date_chw.strftime('%d %b %Y')} {zone_context_msg}, so no alerts to display.")
