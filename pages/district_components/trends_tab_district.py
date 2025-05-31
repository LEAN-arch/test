# test/pages/district_components/trends_tab_district.py
import streamlit as st
import pandas as pd
import numpy as np
from config import app_config
from utils.core_data_processing import get_trend_data
from utils.ui_visualization_helpers import plot_annotated_line_chart, plot_bar_chart

def render_district_trends_tab(filtered_health_for_trends_dist, filtered_iot_for_trends_dist, selected_start_date_dist_trends, selected_end_date_dist_trends):
    st.header("📈 District-Wide Health & Environmental Trends")
    
    if filtered_health_for_trends_dist.empty and (filtered_iot_for_trends_dist is None or filtered_iot_for_trends_dist.empty):
        st.info(f"No health or environmental data available for the selected trend period: {selected_start_date_dist_trends.strftime('%d %b %Y')} to {selected_end_date_dist_trends.strftime('%d %b %Y')}.")
        return

    st.markdown(f"Displaying trends from **{selected_start_date_dist_trends.strftime('%d %b %Y')}** to **{selected_end_date_dist_trends.strftime('%d %b %Y')}**.")
    
    # --- Disease Incidence Trends ---
    st.subheader("Key Disease Incidence Trends (New Cases Identified per Week, District-Wide)")
    # For "new cases", we generally count the first time a patient is recorded with a condition in the period.
    
    key_conditions_for_inc_trends = app_config.KEY_CONDITIONS_FOR_TRENDS[:4] # Example: TB, Malaria, HIV, Pneumonia
    
    if not filtered_health_for_trends_dist.empty and 'condition' in filtered_health_for_trends_dist.columns and 'patient_id' in filtered_health_for_trends_dist.columns and 'encounter_date' in filtered_health_for_trends_dist.columns:
        # Prepare data for incidence calculation (first occurrence of condition per patient in period)
        inc_df_source = filtered_health_for_trends_dist.sort_values('encounter_date')
        inc_df_source['is_first_occurrence_in_period'] = ~inc_df_source.duplicated(subset=['patient_id', 'condition'], keep='first')
        new_cases_in_period_df = inc_df_source[inc_df_source['is_first_occurrence_in_period']]

        cols_disease_trends_dist = st.columns(min(len(key_conditions_for_inc_trends), 2)) # Max 2 charts per row
        col_idx = 0
        for condition_name in key_conditions_for_inc_trends:
            condition_trend_df = new_cases_in_period_df[new_cases_in_period_df['condition'].str.contains(condition_name, case=False, na=False)]
            if not condition_trend_df.empty:
                # Weekly new cases for this condition
                weekly_new_cases = get_trend_data(condition_trend_df, 'patient_id', date_col='encounter_date', period='W-Mon', agg_func='count') # Count of new unique patients per week
                if not weekly_new_cases.empty:
                    with cols_disease_trends_dist[col_idx % 2]:
                        st.plotly_chart(plot_annotated_line_chart(weekly_new_cases, f"Weekly New {condition_name} Cases", y_axis_title=f"New {condition_name} Cases", height=app_config.COMPACT_PLOT_HEIGHT, date_format="%U, %Y (Wk)"), use_container_width=True)
                    col_idx += 1
                elif col_idx % 2 == 0 and len(key_conditions_for_inc_trends) > col_idx +1 : # if this col empty but more conditions
                    with cols_disease_trends_dist[col_idx % 2]: st.caption(f"No new {condition_name} case trend data.") # keep alignment
                    col_idx +=1 # ensure next plot uses next col
                elif col_idx % 2 == 0 : # if this col empty and it's the last one for the row
                     with cols_disease_trends_dist[col_idx % 2]: st.caption(f"No new {condition_name} case trend data.")


        if col_idx == 0: # If no disease trends were plotted
            st.caption("No specific disease incidence trend data available for the selected key conditions in this period.")
    else:
        st.caption("Health data missing required columns ('condition', 'patient_id', 'encounter_date') for disease incidence trends.")

    # --- Overall AI Risk Score Trend ---
    st.markdown("---")
    st.subheader("Population Health Metric Trends")
    cols_wellness_env_dist_trends = st.columns(2)
    with cols_wellness_env_dist_trends[0]:
        if not filtered_health_for_trends_dist.empty and 'ai_risk_score' in filtered_health_for_trends_dist.columns:
            # Daily average of AI risk scores for all encounters in the period
            overall_risk_trend_dist = get_trend_data(filtered_health_for_trends_dist, 'ai_risk_score', date_col='encounter_date', period='D', agg_func='mean')
            if not overall_risk_trend_dist.empty:
                st.plotly_chart(plot_annotated_line_chart(overall_risk_trend_dist, "Daily Avg. Patient AI Risk Score (District-Wide)", y_axis_title="Avg. AI Risk Score", target_line=app_config.TARGET_PATIENT_RISK_SCORE, height=app_config.COMPACT_PLOT_HEIGHT, date_format="%d %b"), use_container_width=True)
            else: st.caption("No AI risk score trend data available for this period.")
        else: st.caption("AI Risk Score data missing for trend analysis.")

    # --- Avg Daily Steps Trend (Example Wellness) ---
    with cols_wellness_env_dist_trends[1]:
        if not filtered_health_for_trends_dist.empty and 'avg_daily_steps' in filtered_health_for_trends_dist.columns:
            steps_trends_dist_page = get_trend_data(filtered_health_for_trends_dist, 'avg_daily_steps', date_col='encounter_date', period='W-Mon', agg_func='mean') # Weekly average
            if not steps_trends_dist_page.empty:
                st.plotly_chart(plot_annotated_line_chart(steps_trends_dist_page, "Weekly Avg. Patient Daily Steps (District)", y_axis_title="Average Steps", target_line=app_config.TARGET_DAILY_STEPS, target_label=f"Target {app_config.TARGET_DAILY_STEPS} Steps", height=app_config.COMPACT_PLOT_HEIGHT, date_format="%U, %Y (Wk)"), use_container_width=True)
            else: st.caption("No patient steps trend data for this period.")
        else: st.caption("Average daily steps data missing for trends.")
    
    # --- Clinic Environmental Trend (Example: CO2) ---
    st.markdown("---")
    st.subheader("Clinic Environmental Trends (District Average)")
    if filtered_iot_for_trends_dist is not None and not filtered_iot_for_trends_dist.empty and 'avg_co2_ppm' in filtered_iot_for_trends_dist.columns:
        # Daily average of CO2 levels across all monitored clinics in the district
        co2_trends_dist_iot_page = get_trend_data(filtered_iot_for_trends_dist, 'avg_co2_ppm', date_col='timestamp', period='D', agg_func='mean')
        if not co2_trends_dist_iot_page.empty:
            st.plotly_chart(plot_annotated_line_chart(co2_trends_dist_iot_page, "Daily Avg. CO2 (All Monitored Clinics - District)", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, target_label=f"Alert >{app_config.CO2_LEVEL_ALERT_PPM}ppm", height=app_config.COMPACT_PLOT_HEIGHT, date_format="%d %b"), use_container_width=True)
        else: st.caption("No clinic CO2 trend data from IoT for this period.")
    else: st.caption("Clinic CO2 data missing or no IoT data for this period for environmental trends.")
