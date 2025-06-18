import streamlit as st
import pandas as pd
import numpy as np
from config import app_config
from utils.core_data_processing import get_trend_data
from utils.ui_visualization_helpers import plot_bar_chart, plot_annotated_line_chart, plot_donut_chart

def render_clinic_epi_module(filtered_health_df_clinic, date_range_display_str):
    if filtered_health_df_clinic.empty:
        st.info(f"No health data available for epidemiology analysis for the period {date_range_display_str}.")
        return

    st.header(f"Clinic-Level Epidemiology Analysis {date_range_display_str}")

    epi_cols_clinic = st.columns(2)
    with epi_cols_clinic[0]:
        st.subheader("Symptom Trends (Weekly Top 5)")
        if 'patient_reported_symptoms' in filtered_health_df_clinic.columns and \
           filtered_health_df_clinic['patient_reported_symptoms'].notna().any() and \
           'encounter_date' in filtered_health_df_clinic.columns:
            
            symptoms_df = filtered_health_df_clinic[['encounter_date', 'patient_reported_symptoms']].copy()
            # FIX: Avoid inplace=True to prevent SettingWithCopyWarning
            symptoms_df = symptoms_df.dropna(subset=['patient_reported_symptoms', 'encounter_date'])
            symptoms_df = symptoms_df[~symptoms_df['patient_reported_symptoms'].str.lower().isin(["unknown", "n/a", "none"])] # Exclude generic unknowns

            if not symptoms_df.empty:
                # Explode semi-colon separated symptoms and clean them
                symptoms_exploded = symptoms_df.assign(symptom=symptoms_df['patient_reported_symptoms'].str.split(';')) \
                                               .explode('symptom')
                symptoms_exploded['symptom'] = symptoms_exploded['symptom'].str.strip().str.title()
                # FIX: Avoid inplace=True
                symptoms_exploded = symptoms_exploded.dropna(subset=['symptom'])
                symptoms_exploded = symptoms_exploded[symptoms_exploded['symptom'] != ''] # Remove empty strings after split
                
                if not symptoms_exploded.empty:
                    top_n_symptoms = symptoms_exploded['symptom'].value_counts().nlargest(5).index.tolist()
                    symptoms_to_plot = symptoms_exploded[symptoms_exploded['symptom'].isin(top_n_symptoms)].copy() # Use .copy() to be explicit
                    
                    if not symptoms_to_plot.empty:
                        # Ensure 'encounter_date' is datetime for get_trend_data
                        if not pd.api.types.is_datetime64_ns_dtype(symptoms_to_plot['encounter_date']):
                             # FIX: Use .loc for assignment to avoid SettingWithCopyWarning
                             symptoms_to_plot['encounter_date'] = pd.to_datetime(symptoms_to_plot['encounter_date'], errors='coerce')
                             symptoms_to_plot = symptoms_to_plot.dropna(subset=['encounter_date'])

                        if not symptoms_to_plot.empty:
                            symptom_trends_data = symptoms_to_plot.groupby([pd.Grouper(key='encounter_date', freq='W-Mon'), 'symptom']).size().reset_index(name='count')
                            # FIX: Avoid inplace=True
                            symptom_trends_data = symptom_trends_data.rename(columns={'encounter_date': 'week_start_date'})
                            fig_symptoms = plot_bar_chart(symptom_trends_data, x_col='week_start_date', y_col='count', color_col='symptom', title="Weekly Symptom Frequency (Top 5)", barmode='group', height=app_config.DEFAULT_PLOT_HEIGHT, y_axis_title="Number of Mentions", x_axis_title="Week Start", y_is_count=True, text_format='d')
                            st.plotly_chart(fig_symptoms, use_container_width=True)
                        else: st.caption("No valid date data for symptom trends.")
                    else: st.caption("Not enough distinct symptom data to plot trends (Top 5).")
                else: st.caption("No parsable symptom data reported.")
            else: st.caption("No patient reported symptoms data (excluding generic unknowns).")
        else: st.caption("Patient reported symptoms or encounter date data not available for symptom trends.")

    with epi_cols_clinic[1]:
        st.subheader("Test Positivity Rate Trends")
        mal_rdt_key_config = "RDT-Malaria" 
        mal_rdt_display = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_rdt_key_config, {}).get("display_name", "Malaria RDT")
        
        if 'test_type' in filtered_health_df_clinic.columns and 'test_result' in filtered_health_df_clinic.columns:
            malaria_df_trend = filtered_health_df_clinic[
                (filtered_health_df_clinic['test_type'] == mal_rdt_key_config) &
                (~filtered_health_df_clinic['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown', 'N/A', 'Indeterminate']))
            ].copy()

            if not malaria_df_trend.empty and 'encounter_date' in malaria_df_trend.columns:
                # FIX: Use .loc for assignment on the copy
                malaria_df_trend['is_positive'] = (malaria_df_trend['test_result'] == 'Positive')
                if not pd.api.types.is_datetime64_ns_dtype(malaria_df_trend['encounter_date']):
                    malaria_df_trend['encounter_date'] = pd.to_datetime(malaria_df_trend['encounter_date'], errors='coerce')
                # FIX: Avoid inplace=True
                malaria_df_trend = malaria_df_trend.dropna(subset=['encounter_date'])
                
                if not malaria_df_trend.empty:
                    weekly_malaria_pos_rate = get_trend_data(malaria_df_trend, value_col='is_positive', date_col='encounter_date', period='W-Mon', agg_func='mean') * 100
                    if not weekly_malaria_pos_rate.empty:
                        st.plotly_chart(plot_annotated_line_chart(weekly_malaria_pos_rate, f"Weekly {mal_rdt_display} Positivity Rate", y_axis_title="Positivity Rate (%)", target_line=app_config.TARGET_MALARIA_POSITIVITY_RATE, height=app_config.COMPACT_PLOT_HEIGHT, y_is_count=False), use_container_width=True)
                    else: st.caption(f"No aggregated positivity trend data for {mal_rdt_display}.")
                else: st.caption(f"No valid date data for {mal_rdt_display} positivity trend after cleaning.")
            else: st.caption(f"No valid test data for {mal_rdt_display} in this period.")
        else: st.caption(f"Test type/result data missing for {mal_rdt_display} trends.")

    st.markdown("---")
    st.subheader("Demographic Breakdown for Selected Condition")
    if 'condition' in filtered_health_df_clinic.columns:
        conditions_for_demog_clinic = ["All Conditions"] + sorted(filtered_health_df_clinic['condition'].dropna().unique().tolist())
        selected_condition_demog_clinic = st.selectbox("Select Condition for Demographic Analysis:", options=conditions_for_demog_clinic, index=0, key="clinic_demog_cond_select_epi_v1")
        
        condition_df_demog_source_clinic = filtered_health_df_clinic
        if selected_condition_demog_clinic != "All Conditions":
            condition_df_demog_source_clinic = filtered_health_df_clinic[filtered_health_df_clinic['condition'] == selected_condition_demog_clinic]
        
        new_cases_demog_df_clinic = pd.DataFrame()
        if 'encounter_date' in condition_df_demog_source_clinic.columns and \
           'patient_id' in condition_df_demog_source_clinic.columns and \
           'condition' in condition_df_demog_source_clinic.columns and \
           not condition_df_demog_source_clinic.empty:
            
            cond_df_demog_copy = condition_df_demog_source_clinic.copy()
            # FIX: Avoid inplace=True
            cond_df_demog_copy = cond_df_demog_copy.sort_values('encounter_date')
            new_cases_demog_df_clinic = cond_df_demog_copy.drop_duplicates(subset=['patient_id', 'condition'], keep='first')

        if not new_cases_demog_df_clinic.empty:
            demog_breakdown_cols_epi_clinic = st.columns(2)
            with demog_breakdown_cols_epi_clinic[0]:
                if 'age' in new_cases_demog_df_clinic.columns and new_cases_demog_df_clinic['age'].notna().any():
                    age_bins_clinic = [0, 5, 18, 35, 50, 65, np.inf]; age_labels_clinic = ['0-4', '5-17', '18-34', '35-49', '50-64', '65+']
                    new_cases_demog_df_clinic_age = new_cases_demog_df_clinic.copy()
                    # FIX: Use .loc for assignment on the copy
                    new_cases_demog_df_clinic_age['age_group_clinic_epi_display'] = pd.cut(new_cases_demog_df_clinic_age['age'], bins=age_bins_clinic, labels=age_labels_clinic, right=False)
                    age_dist_demog_clinic = new_cases_demog_df_clinic_age['age_group_clinic_epi_display'].value_counts().sort_index().reset_index()
                    age_dist_demog_clinic.columns = ['Age Group', 'New Cases']
                    if not age_dist_demog_clinic.empty: st.plotly_chart(plot_bar_chart(age_dist_demog_clinic, 'Age Group', 'New Cases', f"{selected_condition_demog_clinic} - Cases by Age Group", height=300, y_is_count=True, text_format='d'), use_container_width=True)
                    else: st.caption("No data for age distribution.")
                else: st.caption("Age data not available for selected condition.")
            with demog_breakdown_cols_epi_clinic[1]:
                if 'gender' in new_cases_demog_df_clinic.columns and new_cases_demog_df_clinic['gender'].notna().any():
                    gender_dist_demog_clinic = new_cases_demog_df_clinic['gender'].value_counts().reset_index()
                    gender_dist_demog_clinic.columns = ['Gender', 'New Cases']
                    if not gender_dist_demog_clinic.empty: st.plotly_chart(plot_donut_chart(gender_dist_demog_clinic, 'Gender', 'New Cases', f"{selected_condition_demog_clinic} - Cases by Gender", height=300, values_are_counts=True), use_container_width=True)
                    else: st.caption("No data for gender distribution.")
                else: st.caption("Gender data not available for selected condition.")
        elif selected_condition_demog_clinic != "All Conditions": st.caption(f"No '{selected_condition_demog_clinic}' cases found for demographic breakdown.")
        else: st.caption(f"No cases found for demographic breakdown with current filters.")
    else: st.caption("Condition data unavailable for demographic breakdown.")

    st.markdown("---")
    st.subheader("Referral Funnel Analysis (Simplified)")
    if 'referral_status' in filtered_health_df_clinic.columns and filtered_health_df_clinic['referral_status'].notna().any() and \
       'encounter_id' in filtered_health_df_clinic.columns:
        referral_df_funnel_epi = filtered_health_df_clinic[filtered_health_df_clinic['referral_status'].str.lower().isin(['pending', 'completed', 'initiated', 'service provided', 'attended', 'missed appointment', 'declined'])].copy()
        if not referral_df_funnel_epi.empty:
            total_referrals_made_epi = referral_df_funnel_epi['encounter_id'].nunique()
            completed_referrals_epi = 0
            if 'referral_outcome' in referral_df_funnel_epi.columns:
                 completed_referrals_epi = referral_df_funnel_epi[referral_df_funnel_epi['referral_outcome'].isin(['Completed', 'Service Provided', 'Attended'])]['encounter_id'].nunique()
            pending_referrals_epi = referral_df_funnel_epi[referral_df_funnel_epi['referral_status'] == 'Pending']['encounter_id'].nunique()
            
            funnel_data_epi = pd.DataFrame([
                {'Stage': 'Referrals Initiated', 'Count': total_referrals_made_epi},
                {'Stage': 'Referrals Completed (Outcome)', 'Count': completed_referrals_epi},
                {'Stage': 'Referrals Still Pending', 'Count': pending_referrals_epi},
            ])
            
            funnel_data_epi_plot = funnel_data_epi[funnel_data_epi['Count'] > 0]
            if funnel_data_epi_plot.empty and total_referrals_made_epi > 0 :
                 funnel_data_epi_plot = funnel_data_epi.head(1)

            if not funnel_data_epi_plot.empty :
                 st.plotly_chart(plot_bar_chart(funnel_data_epi_plot, 'Stage', 'Count', "Referral Funnel Stages", height=350, y_axis_title="Number of Referrals", orientation='v', y_is_count=True, text_format='d'), use_container_width=True)
            else: st.caption("No data to display for referral funnel with current filters.")
        else: st.caption("No actionable referral records found for funnel analysis.")
    else: st.caption("Referral status or encounter ID data not available for funnel analysis.")
    st.markdown("---")
