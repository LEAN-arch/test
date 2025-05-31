# test/pages/chw_components/epi_watch.py
import streamlit as st
import pandas as pd
import numpy as np
from config import app_config
from utils.ui_visualization_helpers import render_kpi_card, plot_bar_chart

def render_chw_epi_watch(current_day_chw_df, chw_daily_kpis, selected_chw_zone, selected_view_date_chw):
    # Main subheader "Epidemiology Watch..." is in the main 1_chw_dashboard.py before calling this.
    # This function renders the content *within* that section.

    if current_day_chw_df.empty:
        # Message already handled by main page if current_day_chw_df is empty globally.
        # This specific component can just return if its input is empty.
        st.caption(f"No daily encounter data available to display local epidemiology for {selected_chw_zone} on {selected_view_date_chw.strftime('%d %b %Y')}.")
        return

    epi_cols_chw_local = st.columns(3)
    with epi_cols_chw_local[0]:
        symptomatic_conditions = ['TB', 'Pneumonia', 'Malaria', 'Dengue'] # Key conditions for symptom surveillance
        symptom_keywords = 'Fever|Cough|Chills|Headache|Aches|Diarrhea' # Keywords for acute symptoms
        
        new_symptomatic_count = 0
        if 'patient_reported_symptoms' in current_day_chw_df.columns and 'condition' in current_day_chw_df.columns:
            prs_series = current_day_chw_df['patient_reported_symptoms'].astype(str).fillna('')
            new_symptomatic_df = current_day_chw_df[
                current_day_chw_df['condition'].isin(symptomatic_conditions) &
                (prs_series.str.contains(symptom_keywords, case=False, na=False))
            ]
            new_symptomatic_count = new_symptomatic_df['patient_id'].nunique()
        render_kpi_card("New Symptomatic Cases (Key Cond.)", str(new_symptomatic_count), "🤒", 
                        status="High" if new_symptomatic_count > 1 else ("Moderate" if new_symptomatic_count > 0 else "Low"), 
                        help_text=f"Unique patients with key conditions reporting symptoms like '{symptom_keywords.replace('|',', ')}' today.")

    with epi_cols_chw_local[1]:
        key_condition_for_cluster = "Malaria" 
        malaria_cases_today_zone = 0
        if 'condition' in current_day_chw_df.columns:
            malaria_cases_today_zone = current_day_chw_df[current_day_chw_df['condition'] == key_condition_for_cluster]['patient_id'].nunique()
        render_kpi_card(f"New {key_condition_for_cluster} Cases Today", str(malaria_cases_today_zone), "🦟", 
                        status="High" if malaria_cases_today_zone >= 2 else ("Moderate" if malaria_cases_today_zone > 0 else "Low"), 
                        help_text=f"New {key_condition_for_cluster} cases identified in this zone today.")

    with epi_cols_chw_local[2]:
        tb_contacts_val = chw_daily_kpis.get('tb_contacts_to_trace_today', 0)
        render_kpi_card("Pending TB Contact Traces", str(tb_contacts_val), "👥", 
                        status="High" if tb_contacts_val > 0 else "Low", 
                        help_text="New TB patient contacts from today needing tracing/follow-up.")
    
    # Demographics of High-Risk Patients Today
    st.markdown("###### Demographics of High AI Risk Patients (Today)")
    if 'ai_risk_score' in current_day_chw_df.columns:
        high_risk_today_df_for_demo = current_day_chw_df[current_day_chw_df['ai_risk_score'] >= app_config.RISK_THRESHOLDS.get('high', 75)]
        if not high_risk_today_df_for_demo.empty and 'age' in high_risk_today_df_for_demo.columns:
            age_bins = [0, 5, 18, 45, 65, np.inf]; age_labels = ['0-4', '5-17', '18-44', '45-64', '65+']
            # Use .loc to avoid SettingWithCopyWarning when creating 'age_group'
            high_risk_today_df_for_demo = high_risk_today_df_for_demo.copy() # Make a copy before adding column
            high_risk_today_df_for_demo.loc[:, 'age_group'] = pd.cut(high_risk_today_df_for_demo['age'], bins=age_bins, labels=age_labels, right=False)
            
            age_group_counts = high_risk_today_df_for_demo['age_group'].value_counts().sort_index().reset_index()
            age_group_counts.columns = ['Age Group', 'Number of High-Risk Patients']
            if not age_group_counts.empty:
                st.plotly_chart(plot_bar_chart(
                    age_group_counts, 
                    x_col='Age Group', 
                    y_col='Number of High-Risk Patients', 
                    title="High AI Risk Patients by Age Group (Today)", 
                    height=app_config.COMPACT_PLOT_HEIGHT-50,
                    y_is_count=True, # Explicitly pass this for correct scaling
                    text_format='d'   # And text format
                    ), use_container_width=True)
            else: 
                st.caption("No high AI risk patients with age data for demographic breakdown today.")
        elif not current_day_chw_df.empty : # if there's daily data, but no high risk patients
             st.caption("No high AI risk patients found today for demographic breakdown.")
        # If current_day_chw_df was empty, the main message is already shown
    else:
        st.caption("AI Risk Score data not available for demographic breakdown.")
    # The "---" separator is now also handled by the main page script after this component call.
