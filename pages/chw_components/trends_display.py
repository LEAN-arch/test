# test/pages/chw_components/trends_display.py
import streamlit as st
import pandas as pd
from config import app_config
from utils.core_data_processing import get_trend_data
from utils.ui_visualization_helpers import plot_annotated_line_chart

def render_chw_activity_trends_tab(health_df_chw_main, selected_trend_start_chw, selected_trend_end_chw, selected_chw_zone):
    st.subheader(f"My Activity Trends ({selected_trend_start_chw.strftime('%d %b %Y')} - {selected_trend_end_chw.strftime('%d %b %Y')})")
    
    if selected_trend_start_chw > selected_trend_end_chw:
        st.warning("Start date for trend period is after end date. Please adjust date range filter in the sidebar.")
        return # Exit if date range is invalid

    trends_base_df_chw = health_df_chw_main.copy()
    # Ensure 'encounter_date' is datetime for filtering. It should be already by this point from main page.
    if not pd.api.types.is_datetime64_ns_dtype(trends_base_df_chw['encounter_date']):
        trends_base_df_chw['encounter_date'] = pd.to_datetime(trends_base_df_chw['encounter_date'], errors='coerce')
    trends_base_df_chw.dropna(subset=['encounter_date'], inplace=True) # Crucial for date comparison

    # Convert Series of datetime.date objects to Pandas Timestamps for proper comparison if needed
    # The 'encounter_date' column itself in trends_base_df_chw should be datetime64[ns] for Grouper
    chw_trends_data_filtered_range = trends_base_df_chw[
        (trends_base_df_chw['encounter_date'].dt.date >= selected_trend_start_chw) &
        (trends_base_df_chw['encounter_date'].dt.date <= selected_trend_end_chw)
    ].copy()
    
    # Optional: Filter by zone selected for daily snapshot, if trends should also reflect that zone
    if selected_chw_zone != "All Zones" and 'zone_id' in chw_trends_data_filtered_range.columns:
        chw_trends_data_filtered_range = chw_trends_data_filtered_range[chw_trends_data_filtered_range['zone_id'] == selected_chw_zone]
        zone_trend_context = f"for {selected_chw_zone}"
    else:
        zone_trend_context = "across all assigned zones"


    if not chw_trends_data_filtered_range.empty:
        cols_chw_trend_tab_display = st.columns(2)
        with cols_chw_trend_tab_display[0]:
            visits_trend_data_range = get_trend_data(chw_trends_data_filtered_range, value_col='patient_id', date_col='encounter_date', period='D', agg_func='nunique')
            if not visits_trend_data_range.empty:
                st.plotly_chart(plot_annotated_line_chart(visits_trend_data_range, f"Daily Patients Visited {zone_trend_context}", y_axis_title="# Patients", height=app_config.COMPACT_PLOT_HEIGHT-20, date_format="%a, %d %b"), use_container_width=True)
            else:
                st.caption(f"No patient visit data for trend in selected range/zone.")
        with cols_chw_trend_tab_display[1]:
            if 'ai_followup_priority_score' in chw_trends_data_filtered_range.columns:
                high_prio_df_trend_range = chw_trends_data_filtered_range[chw_trends_data_filtered_range['ai_followup_priority_score'] >= 80]
                high_prio_trend_range_data = get_trend_data(high_prio_df_trend_range, value_col='patient_id', date_col='encounter_date', period='D', agg_func='nunique')
                if not high_prio_trend_range_data.empty:
                    st.plotly_chart(plot_annotated_line_chart(high_prio_trend_range_data, f"High Priority Follow-ups {zone_trend_context}", y_axis_title="# Follow-ups", height=app_config.COMPACT_PLOT_HEIGHT-20, date_format="%a, %d %b"), use_container_width=True)
                else:
                    st.caption(f"No high priority follow-up data for trend in selected range/zone.")
            else:
                st.caption("AI Follow-up Priority Score not available for trend.")
    else:
        st.info(f"Not enough data in the selected range ({selected_trend_start_chw.strftime('%d %b %Y')} to {selected_trend_end_chw.strftime('%d %b %Y')}, Zone: {selected_chw_zone}) for activity trends display.")
