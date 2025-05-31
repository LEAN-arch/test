# test/pages/chw_components/tasks_display.py
import streamlit as st
import pandas as pd
import logging # Added for potential logging within component

logger = logging.getLogger(__name__) # Added logger

def render_chw_tasks_tab(patient_alerts_tasks_df, current_day_chw_df, selected_view_date_chw, selected_chw_zone_daily):
    zone_context_msg = f"in {selected_chw_zone_daily}" if selected_chw_zone_daily != "All Zones" else "across all assigned zones"
    st.subheader(f"Prioritized Task List for {selected_view_date_chw.strftime('%B %d, %Y')} {zone_context_msg}")
    
    if not patient_alerts_tasks_df.empty:
        temp_col_task_list = None
        if 'vital_signs_temperature_celsius' in current_day_chw_df.columns and current_day_chw_df['vital_signs_temperature_celsius'].notna().any():
            temp_col_task_list = 'vital_signs_temperature_celsius'
        elif 'max_skin_temp_celsius' in current_day_chw_df.columns and current_day_chw_df['max_skin_temp_celsius'].notna().any():
            temp_col_task_list = 'max_skin_temp_celsius'
        
        cols_to_show_task = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'ai_followup_priority_score', 'alert_reason', 'referral_status', 'min_spo2_pct', 'fall_detected_today']
        if temp_col_task_list: cols_to_show_task.append(temp_col_task_list) # Add if determined
        else: cols_to_show_task.append('max_skin_temp_celsius') # Default fallback if neither primary temp col had data
            
        task_df_for_display = patient_alerts_tasks_df[[col for col in cols_to_show_task if col in patient_alerts_tasks_df.columns]].copy()
        
        rename_col = temp_col_task_list if temp_col_task_list and temp_col_task_list in task_df_for_display.columns else \
                     ('max_skin_temp_celsius' if 'max_skin_temp_celsius' in task_df_for_display.columns else None)
        if rename_col : task_df_for_display.rename(columns={rename_col: 'latest_temp_celsius'}, inplace=True, errors='ignore')
        
        sort_cols_tasks_tab = [col for col in ['ai_followup_priority_score', 'ai_risk_score'] if col in task_df_for_display.columns]
        task_df_display_final_sorted = task_df_for_display.sort_values(by=sort_cols_tasks_tab, ascending=[False]*len(sort_cols_tasks_tab)) if sort_cols_tasks_tab else task_df_for_display
        
        df_for_st_dataframe_tasks = task_df_display_final_sorted.copy()
        for col in df_for_st_dataframe_tasks.columns:
            if df_for_st_dataframe_tasks[col].dtype == 'object':
                df_for_st_dataframe_tasks[col] = df_for_st_dataframe_tasks[col].fillna('N/A').astype(str).replace(['nan','None','<NA>'],'N/A', regex=False)

        st.dataframe(
            df_for_st_dataframe_tasks, use_container_width=True, height=450,
            column_config={
                "patient_id": "Patient ID", "zone_id": "Zone",
                "ai_risk_score": st.column_config.ProgressColumn("AI Risk",format="%d",min_value=0,max_value=100),
                "ai_followup_priority_score": st.column_config.ProgressColumn("AI Prio.",format="%d",min_value=0,max_value=100),
                "alert_reason": st.column_config.TextColumn("Reason",width="large"),
                "min_spo2_pct":st.column_config.NumberColumn("SpO2(%)",format="%d%%"),
                "latest_temp_celsius":st.column_config.NumberColumn("Temp(°C)",format="%.1f°C"), # 'latest_temp_celsius' used here
                "fall_detected_today":st.column_config.NumberColumn("Falls",format="%d")
            }, hide_index=True
        )
        try:
            csv_chw_tasks = task_df_display_final_sorted.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Task List (CSV)", data=csv_chw_tasks, file_name=f"chw_tasks_{selected_view_date_chw.strftime('%Y%m%d')}.csv", mime="text/csv", key="chw_task_download_v9") # Incremented key
        except Exception as e_csv:
            logger.error(f"CHW Task CSV Download Error: {e_csv}")
            st.warning("Could not prepare task list for download.")
    elif not current_day_chw_df.empty:
        st.info("No specific tasks from today's CHW encounters.")
    else:
        st.info(f"No CHW encounters recorded on {selected_view_date_chw.strftime('%d %b %Y')} {zone_context_msg}, so no tasks to display.")
