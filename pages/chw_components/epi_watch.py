# test/pages/chw_components/epi_watch.py
import streamlit as st
import pandas as pd
import numpy as np
from config import app_config
from utils.ui_visualization_helpers import render_kpi_card, plot_bar_chart

def render_chw_epi_watch(current_day_chw_df, chw_daily_kpis, selected_chw_zone, selected_view_date_chw):
    st.markdown(f"##### Local Epidemiology Watch - {selected_view_date_chw.strftime('%d %b %Y')} - {selected_chw_zone}")
    if not current_day_chw_df.empty:
        epi_cols_chw_local = st.columns(3)
        with epi_cols_chw_local[0]:
            symptomatic_conditions = ['TB', 'Pneumonia', 'Malaria', 'Dengue']
            symptom_keywords = 'Fever|Cough|Chills|Headache|Aches|Diarrhea' # Added Diarrhea for broader syndromic
            
            # Ensure patient_reported_symptoms is string and handle NaNs
            prs_series = current_day_chw_df['patient_reported_symptoms'].astype(str).fillna('')
            new_symptomatic_df = current_day_chw_df[
                current_day_chw_df['condition'].isin(symptomatic_conditions) &
                (prs_series.str.contains(symptom_keywords, case=False, na=False))
            ]
            new_symptomatic_count = new_symptomatic_df['patient_id'].nunique()
            render_kpi_card("New Symptomatic Cases (Key Cond.) Today", str(new_symptomatic_count), "🤒", status="High" if new_symptomatic_count > 1 else "Low", help_text=f"Unique patients for key conditions reporting new symptoms like {symptom_keywords.replace('|',', ')}.")

        with epi_cols_chw_local[1]:
            key_condition_for_cluster = "Malaria" 
            malaria_cases_today_zone = current_day_chw_df[current_day_chw_df['condition'] == key_condition_for_cluster]['patient_id'].nunique()
            render_kpi_card(f"New {key_condition_for_cluster} Cases Today", str(malaria_cases_today_zone), "🦟", status="High" if malaria_cases_today_zone >= 2 else "Low", help_text=f"New {key_condition_for_cluster} cases identified in this zone today.")

        with epi_cols_chw_local[2]:
            tb_contacts_val = chw_daily_kpis.get('tb_contacts_to_trace_today', 0)
            render_kpi_card("Pending TB Contact Traces Today", str(tb_contacts_val), "👥", status="High" if tb_contacts_val > 0 else "Low", help_text="Number of new TB cases from today for whom contact tracing is now pending.")
        
        high_risk_today_df_for_demo = current_day_chw_df[current_day_chw_df.get('ai_risk_score', pd.Series(dtype=float)) >= app_config.RISK_THRESHOLDS.get('high', 75)]
        if not high_risk_today_df_for_demo.empty and 'age' in high_risk_today_df_for_demo.columns:
            st.markdown("###### Demographics of High AI Risk Patients (Today)")
            age_bins = [0, 5, 18, 45, 65, np.inf]; age_labels = ['0-4', '5-17', '18-44', '45-64', '65+']
            high_risk_today_df_for_demo = high_risk_today_df_for_demo.copy() # Explicit copy
            high_risk_today_df_for_demo.loc[:, 'age_group'] = pd.cut(high_risk_today_df_for_demo['age'], bins=age_bins, labels=age_labels, right=False)
            age_group_counts = high_risk_today_df_for_demo['age_group'].value_counts().sort_index().reset_index()
            age_group_counts.columns = ['Age Group', 'Number of High-Risk Patients']
            if not age_group_counts.empty:
                st.plotly_chart(plot_bar_chart(age_group_counts, x_col='Age Group', y_col='Number of High-Risk Patients', title="High AI Risk Patients by Age Group (Today)", height=app_config.COMPACT_PLOT_HEIGHT-50), use_container_width=True)
            else: st.caption("No high AI risk patients with age data for demographic breakdown today.")
        elif not current_day_chw_df.empty : st.caption("No high AI risk patients found today for demographic breakdown.")
    else:
        st.caption(f"No data to display local epidemiology snapshot for {selected_chw_zone} on {selected_view_date_chw.strftime('%d %b %Y')}.")
    st.markdown("---")
