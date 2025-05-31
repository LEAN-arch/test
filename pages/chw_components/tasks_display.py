# test/pages/chw_components/tasks_display.py
import streamlit as st
import pandas as pd
import logging # Added for potential logging within component
import numpy as np # For np.nan

# from config import app_config # Not strictly needed here for task display logic
logger = logging.getLogger(__name__)

def render_chw_tasks_tab(patient_alerts_tasks_df, current_day_chw_df, selected_view_date_chw, selected_chw_zone_daily):
    # Main subheader is managed by the main 1_chw_dashboard.py page
    
    zone_context_msg_tasks = f"in {selected_chw_zone_daily}" if selected_chw_zone_daily != "All Zones" else "across all assigned zones"

    if patient_alerts_tasks_df.empty:
        if not current_day_chw_df.empty: # Encounters happened, but no tasks derived
            st.info(f"No specific tasks or follow-ups identified from today's CHW encounters {zone_context_msg_tasks}.")
        # If current_day_chw_df is also empty, main page handles general "no data" message.
        return

    # Determine the temperature column name that was likely used in alert generation/daily summary
    # This makes sure we pick up the right temperature data for the task list display
    temp_col_for_task_table = None
    base_df_for_temp_check_tasks = current_day_chw_df if not current_day_chw_df.empty else patient_alerts_tasks_df # Fallback
    
    if 'vital_signs_temperature_celsius' in base_df_for_temp_check_tasks.columns and \
       base_df_for_temp_check_tasks['vital_signs_temperature_celsius'].notna().any():
        temp_col_for_task_table = 'vital_signs_temperature_celsius'
    elif 'max_skin_temp_celsius' in base_df_for_temp_check_tasks.columns and \
         base_df_for_temp_check_tasks['max_skin_temp_celsius'].notna().any():
        temp_col_for_task_table = 'max_skin_temp_celsius'
    
    cols_to_show_task = ['patient_id', 'zone_id', 'condition', 'age', # Added age for context
                         'ai_risk_score', 'ai_followup_priority_score',
                         'alert_reason', 'referral_status', 'min_spo2_pct', 'fall_detected_today']
    
    # Add the determined temperature column to the list if it's valid
    # Fallback to 'max_skin_temp_celsius' if none was determined but it might exist in patient_alerts_tasks_df
    final_temp_col_in_list = temp_col_for_task_table if temp_col_for_task_table else 'max_skin_temp_celsius'
    if final_temp_col_in_list not in cols_to_show_task: # Add if not already there
        cols_to_show_task.append(final_temp_col_in_list)

    # Select only existing columns to avoid KeyErrors
    task_df_for_display = patient_alerts_tasks_df[[col for col in cols_to_show_task if col in patient_alerts_tasks_df.columns]].copy()
    
    # Rename the temperature column to a generic 'latest_temp_celsius' for display
    if final_temp_col_in_list in task_df_for_display.columns:
        task_df_for_display.rename(columns={final_temp_col_in_list: 'latest_temp_celsius'}, inplace=True, errors='ignore')
    
    sort_cols_tasks_tab = [col for col in ['ai_followup_priority_score', 'ai_risk_score'] if col in task_df_for_display.columns and task_df_for_display[col].notna().any()]
    task_df_display_final_sorted = task_df_for_display.sort_values(by=sort_cols_tasks_tab, ascending=[False]*len(sort_cols_tasks_tab)) if sort_cols_tasks_tab else task_df_for_display
    
    # Prepare DataFrame for st.dataframe serialization
    df_for_st_dataframe_tasks = task_df_display_final_sorted.copy()
    for col in df_for_st_dataframe_tasks.columns:
        # Handle date columns explicitly if any are directly displayed (encounter_date not in default view here)
        # Convert object columns to string, handling NaNs appropriately
        if df_for_st_dataframe_tasks[col].dtype == 'object':
            df_for_st_dataframe_tasks[col] = df_for_st_dataframe_tasks[col].fillna('N/A').astype(str).replace(['nan','None','<NA>','NaT'],'N/A', regex=False)
        # Ensure numeric columns intended for Progress/Number are float/int, with NaNs handled as PyArrow expects
        elif col in ['ai_risk_score', 'ai_followup_priority_score', 'min_spo2_pct', 'latest_temp_celsius', 'fall_detected_today', 'age']:
             df_for_st_dataframe_tasks[col] = pd.to_numeric(df_for_st_dataframe_tasks[col], errors='coerce') # Let NaNs pass for now


    st.dataframe(
        df_for_st_dataframe_tasks, use_container_width=True, height=450,
        column_config={
            "patient_id": "Patient ID", 
            "zone_id": st.column_config.TextColumn("Zone", width="small"),
            "age": st.column_config.NumberColumn("Age", format="%d yrs", width="small"),
            "ai_risk_score": st.column_config.ProgressColumn("AI Risk",format="%d",min_value=0,max_value=100, width="medium"),
            "ai_followup_priority_score": st.column_config.ProgressColumn("AI Prio.",format="%d",min_value=0,max_value=100, width="medium"),
            "alert_reason": st.column_config.TextColumn("Reason",width="large"),
            "min_spo2_pct":st.column_config.NumberColumn("SpO2(%)",format="%d%%", width="small"),
            "latest_temp_celsius":st.column_config.NumberColumn("Temp(°C)",format="%.1f°C", width="small"),
            "fall_detected_today":st.column_config.NumberColumn("Falls",format="%d", width="small")
        }, hide_index=True
    )
    try:
        csv_chw_tasks = task_df_display_final_sorted.to_csv(index=False).encode('utf-8') # Use the df before potential further display conversions
        st.download_button(label="📥 Download Task List (CSV)", data=csv_chw_tasks, file_name=f"chw_tasks_{selected_view_date_chw.strftime('%Y%m%d')}.csv", mime="text/csv", key="chw_task_download_v9")
    except Exception as e_csv:
        logger.error(f"CHW Task CSV Download Error: {e_csv}", exc_info=True)
        st.warning("Could not prepare task list for download.")
