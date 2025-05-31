# test/pages/chw_components/alerts_display.py
import streamlit as st
import pandas as pd
import numpy as np # For np.nan
from utils.ui_visualization_helpers import render_traffic_light
from config import app_config # Not directly used here but good for context

def render_chw_alerts**File 9: `test/pages/chw_components/alerts_display.py` (Final Version)**
```python
# test/pages/chw_components/alerts_display.py
import streamlit as st
import pandas as pd
import numpy as np # Added for np.nan potentially in fallbacks
from utils.ui_visualization_helpers import render_traffic_light
from config import app_config # For thresholds if needed directly

def render_chw_alerts_tab(patient_alerts_tasks_df, current_day_chw_df,_tab(patient_alerts_tasks_df, current_day_chw_df, selected_view_date_chw, selected_chw_zone_daily):
    # Main subheader is managed by the main dashboard page (1_chw_dashboard.py)
    
    if patient_alerts_tasks_df.empty:
        zone_context_msg_alerts = f" in {selected_chw_zone_daily}" if selected_chw_zone_daily != "All Zones" else " across all assigned zones"
        if not current_day_chw_df.empty: # If there were encounters today but no alerts generated
            st.success(f"✅ No critical patient alerts identified from encounters on {selected_view_date_chw.strftime('%d %b %Y')}{zone_context_msg_alerts}.")
        # else: ( selected_view_date_chw, selected_chw_zone_daily):
    # Main subheader for this tab is handled in 1_chw_dashboard.py before calling this.
    
    # Use the context of the daily snapshot zone filter for the subheader message
    zone_context_msg = f"in {selected_chw_zone_daily}" if selected_chw_zone_daily != "All Zones" else "across all assigned zones"
    
    if not patient_alerts_tasks_df.empty:
        sort_col_alert = 'ai_followup_priority_score' if 'ai_followup_priority_score' in patient_alerts_tasks_df.columns else 'priority_score'
        alerts_to_display = patient_alerts_tasks_df.sort_values(by=sort_col_alert, ascending=False)
        
        # Determine the appropriate temperature column to use based on data availability in the *daily context*
        # This temp_col can then be reliably accessed in the alert_row if it exists
        temp_col_traffic_alert = None
        if not current_day_chw_df.empty : # Ensure current_day_chw_df is not empty before checking columns
            if 'vital_signs_temperature_celsius' in current_day_chw_df.columns and current_day_chw_df['vital_signs_temperature_celsius'].notna().any():
                temp_col_traffic_alert = 'vital_signs_temperature_celsius'
            elif 'max_skin_temp_celsius' in current_day_chw_df.columns and current_day_chw_df['max_skin_temp_celsius'].notna().any():
                temp_col_traffic_alert = 'max_skin_temp_celsius'

if current_day_chw_df is also empty, main page already shows general "no data")
        #    st.info(f"No CHW encounters recorded on {selected_view_date_chw.strftime('%d %b %Y')}{zone_context_msg_alerts}, so no alerts to display.")
        return

    # Sort alerts for display (highest priority first)
    sort_col_alert = 'ai_followup_priority_score' if 'ai_followup_priority_score' in patient_alerts_tasks_df.columns and patient_alerts_tasks_df['ai_followup_priority_score'].notna().any() else \
                     ('priority_score' if 'priority_score' in patient_alerts_tasks_df.columns and patient_alerts_tasks_df['priority_score'].notna().any() else None)
    
    if sort_col_alert:
        alerts_to_display = patient_alerts_tasks_df.sort_values(by=sort_col_alert, ascending=False)
    else: # Fallback if no priority scores are available for some reason
        alerts_to_display = patient_alerts_tasks_df
        
    # Determine temperature column to use, based on availability in the daily data context
    # (since alerts_df is a subset and might not have all original columns if it was heavily processed)
    temp_col_to_use_in_alerts = None
    base_df_for_temp_col = current_day_chw_df if not current_day_chw_df.empty else patient_alerts_tasks_df
    if 'vital_signs_temperature_celsius' in base_df_for_temp_col.columns and base_df_for_temp_col['vital_signs_temperature_celsius'].notna().any():
        temp_col_to_use_in_alerts = 'vital_signs_temperature_celsius'
    elif 'max_skin_temp_celsius' in base_df_for_temp_col.columns and base_df_for_temp_col['max_skin_temp_celsius'].notna().any():
        temp_col_to_use_in_alerts = 'max_skin_temp_celsius'

    for _, alert_row in alerts_to_display.head(15).iterrows():
        priority_val = alert_row.get(sort_col_alert, 0) if sort_col_alert else alert_row.get('ai_risk_score', 0) # Fallback to ai_risk_score if no prio
        alert_status_light = "High" if priority_val >= 80 else ("Moderate" if priority_val >= 60 else "Low")
        
        details_parts = []
        if pd.notna(alert_row.get('ai_risk_score')): details_parts.append(f"AI Risk: {alert_row['ai_risk_score']:.0f}")
        if pd.notna(alert_row.get('ai_followup_priority_score')): details_parts.append(f"AI Prio: {alert_row['ai_followup_priority_score']:.0f}")
        if pd.notna(alert_row.get('min_spo2_pct')): details_parts.append(f"SpO2: {alert_row['min_spo2_pct']:.0f}%")
        
        temp_val_display = alert_row.get(temp_col_to_use_in_alerts) if temp_col_to_use_in_alerts and temp_col_to_use_in_alerts in alert_row else None
        if pd.notna(temp_val_display): details_parts.append(f"Temp: {temp_val_display:.1f}°C")
            
        if pd.notna(alert_row.get('fall_detected_today')) and alert_row['fall_detected_today'] > 0:
            details_parts.append(f"Falls: {int(alert_row['fall_detected_today'])}")
        
        patient_info_str = str(alert_row.get('patient_id','N/A'))
        condition_str = str(alert_row.get('condition','N/A'))
        if condition_str != "Unknown" and condition_str : patient_info_str += f" ({condition_str})"
            
        msg = f"Patient {patient_info_str}"
        detail_str = str(alert_row.get('alert_reason', 'Review Case')) + (" | " + " / ".join(details_parts) if details_parts else "")
        render_traffic_light(message=msg, status=alert_status_light, details=detail_str)
