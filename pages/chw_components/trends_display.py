# test/pages/chw_components/trends_display.py
import streamlit as st
import pandas as pd
import numpy as np # For np.nan
from config import app_config
from utils.core_data_processing import get_trend_data
from utils.ui_visualization_helpers import plot_annotated_line_chart

def render_chw_activity_trends_tab(health_df_chw_main, selected_trend_start_chw, selected_trend_end_chw, selected_chw_zone_daily):
    # Main subheader is managed by the main dashboard page (1_chw_dashboard.py)
    
    if selected_trend_start_chw > selected_trend_end_chw:
        st.warning("Trend period error: Start date is after end date. Please adjust date range filter in the sidebar.")
        return 

    # Ensure health_df_chw_main is not empty and has the necessary date column
    if health_df_chw_main.empty or 'encounter_date' not in health_df_chw_main.columns:
        st.info(f"Not enough data to display activity trends for the selected period/zone.")
        return

    trends_base_df_chw = health_df_chw_main.copy()
    
    # Ensure 'encounter_date' is datetime64[ns] for filtering and get_trend_data
    if not pd.api.types.is_datetime64_ns_dtype(trends_base_df_chw['encounter_date']):
        trends_base_df_chw['encounter_date'] = pd.to_datetime(trends_base_df_chw['encounter_date'], errors='coerce')
    trends_base_df_chw.dropna(subset=['encounter_date'], inplace=True) # Remove rows where date conversion failed

    if trends_base_df_chw.empty: # After potential NaT drop
        st.info(f"No valid date data found for trends in selected period/zone.")
        return

    # Filter by selected trend date range
    # Use .dt.date for comparison with datetime.date objects from selector
    chw_trends_data_filtered_range = trends_base_df_chw[
        (trends_base_df_chw['encounter_date'].dt.date >= selected_trend_start_chw) &
        (trends_base_df_chw['encounter_date'].dt.date <= selected_trend_end_chw)
    ].copy()
    
    zone_trend_context = "across all assigned zones" # Default
    if selected_chw_zone_daily != "All Zones" and 'zone_id' in chw_trends_data_filtered_range.columns:
        chw_trends_data_filtered_range = chw_trends_data_filtered_range[chw_trends_data_filtered_range['zone_id'] == selected_chw_zone_daily]
        zone_trend_context = f"for {selected_chw_zone_daily}"


    if not chw_trends_data_filtered_range.empty:
        cols_chw_trend_tab_display = st.columns(2)
        with cols_chw_trend_tab_display[0]:
            # Use the 'encounter_date' column (which is datetime64[ns]) for get_trend_data
            visits_trend_data_range = get_trend_data(chw_trends_data_filtered_range, value_col='patient_id', date_col='encounter_date', period='D', agg_func='nunique')
            if not visits_trend_data_range.empty:
                st.plotly_chart(plot_annotated_line_chart(
                    visits_trend_data_range, 
                    f"Daily Patients Visited {zone_trend_context}", 
                    y_axis_title="# Patients", 
                    height=app_config.COMPACT_PLOT_HEIGHT-20, 
                    date_format="%a, %d %b",
                    y_is_count=True # Values are counts of patients
                ), use_container_width=True)
            else:
                st.caption(f"No patient visit data for trend in selected range {zone_trend_context}.")
        
        with cols_chw_trend_tab_display[1]:
            if 'ai_followup_priority_score' in chw_trends_data_filtered_range.columns and \
               chw_trends_data_filtered_range['ai_followup_priority_score'].notna().any():
                
                high_prio_df_trend_range = chw_trends_data_filtered_range[chw_trends_data_filtered_range['ai_followup_priority_score'] >= 80]
                high_prio_trend_range_data = get_trend_data(high_prio_df_trend_range, value_col='patient_id', date_col='encounter_date', period='D', agg_func='nunique')
                
                if not high_prio_trend_range_data.empty:
                    st.plotly_chart(plot_annotated_line_chart(
                        high_prio_trend_range_data, 
                        f"High Priority Follow-ups {zone_trend_context}", 
                        y_axis_title="# Follow-ups", 
                        height=app_config.COMPACT_PLOT_HEIGHT-20, 
                        date_format="%a, %d %b",
                        y_is_count=True # Values are counts of follow-ups
                    ), use_container_width=True)
                else:
                    st.caption(f"No high priority follow-up data for trend in selected range {zone_trend_context}.")
            else:
                st.caption(f"AI Follow-up Priority Score not available for trend {zone_trend_context}.")
    else:
        st.info(f"Not enough data in the selected range ({selected_trend_start_chw.strftime('%d %b %Y')} to {selected_trend_end_chw.strftime('%d %b %Y')}, Zone: {selected_chw_zone_daily}) for activity trends display.")
