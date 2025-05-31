# test/pages/clinic_components/epi_module.py
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

    st.header(f"Klinik Seviyesinde Epidemiyoloji Analizi {date_range_display_str}")

    epi_cols_clinic = st.columns(2)
    with epi_cols_clinic[0]:
        st.subheader("Semptom Trendleri (Haftalık)")
        if 'patient_reported_symptoms' in filtered_health_df_clinic.columns and filtered_health_df_clinic['patient_reported_symptoms'].notna().any():
            symptoms_df = filtered_health_df_clinic[['encounter_date', 'patient_reported_symptoms']].copy()
            symptoms_df.dropna(subset=['patient_reported_symptoms'], inplace=True)
            symptoms_df = symptoms_df[symptoms_df['patient_reported_symptoms'].str.lower() != "unknown"] # Exclude 'Unknown'
            if not symptoms_df.empty:
                symptoms_exploded = symptoms_df.assign(symptom=symptoms_df['patient_reported_symptoms'].str.split(';')).explode('symptom')
                symptoms_exploded['symptom'] = symptoms_exploded['symptom'].str.strip().str.title()
                symptoms_exploded.dropna(subset=['symptom'], inplace=True)
                symptoms_exploded = symptoms_exploded[symptoms_exploded['symptom'] != '']


                top_n_symptoms = symptoms_exploded['symptom'].value_counts().nlargest(5).index.tolist()
                symptoms_to_plot = symptoms_exploded[symptoms_exploded['symptom'].isin(top_n_symptoms)]
                
                if not symptoms_to_plot.empty:
                    symptom_trends = symptoms_to_plot.groupby([pd.Grouper(key='encounter_date', freq='W-Mon'), 'symptom']).size().reset_index(name='count')
                    symptom_trends.rename(columns={'encounter_date': 'week_start_date'}, inplace=True)
                    fig_symptoms = plot_bar_chart(symptom_trends, x_col='week_start_date', y_col='count', color_col='symptom', title="Haftalık Semptom Görülme Sıklığı (Top 5)", barmode='stack', height=app_config.DEFAULT_PLOT_HEIGHT, y_axis_title="Karşılaşma Sayısı", x_axis_title="Hafta")
                    st.plotly_chart(fig_symptoms, use_container_width=True)
                else: st.caption("Semptom trendlerini gösterecek yeterli veri yok (Top 5).")
            else: st.caption("Hasta tarafından bildirilen semptom verisi (Unknown hariç) yok.")
        else: st.caption("Patient reported symptoms data not available.")

    with epi_cols_clinic[1]:
        st.subheader("Test Pozitiflik Oranı Trendleri")
        # Example: Malaria RDT Positivity Trend
        # test_type should ideally hold the original key from app_config.KEY_TEST_TYPES_FOR_ANALYSIS
        mal_rdt_key_config = "RDT-Malaria" # This should be the key used in your data's 'test_type' column
        mal_rdt_display = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_rdt_key_config, {}).get("display_name", "Malaria RDT")
        
        malaria_df_trend = filtered_health_df_clinic[
            (filtered_health_df_clinic['test_type'] == mal_rdt_key_config) &
            (~filtered_health_df_clinic['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown', 'N/A', 'Indeterminate']))
        ].copy()

        if not malaria_df_trend.empty:
            malaria_df_trend.loc[:, 'is_positive'] = malaria_df_trend['test_result'] == 'Positive'
            # Ensure 'encounter_date' is datetime before passing to get_trend_data
            if not pd.api.types.is_datetime64_ns_dtype(malaria_df_trend['encounter_date']):
                malaria_df_trend.loc[:,'encounter_date'] = pd.to_datetime(malaria_df_trend['encounter_date'], errors='coerce')
            malaria_df_trend.dropna(subset=['encounter_date'], inplace=True) # Drop if date conversion failed
            
            weekly_malaria_pos_rate = get_trend_data(malaria_df_trend, value_col='is_positive', date_col='encounter_date', period='W-Mon', agg_func='mean') * 100
            if not weekly_malaria_pos_rate.empty:
                st.plotly_chart(plot_annotated_line_chart(weekly_malaria_pos_rate, f"{mal_rdt_display} Pozitiflik Oranı (%)", y_axis_title="Pozitiflik Oranı (%)", target_line=app_config.TARGET_MALARIA_POSITIVITY_RATE, height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
            else: st.caption(f"{mal_rdt_display} için pozitiflik trend verisi yok.")
        else: st.caption(f"{mal_rdt_display} için test verisi yok.")

    st.markdown("---")
    st.subheader("Seçili Durum için Demografik Dağılım")
    conditions_for_demog = ["All Conditions"] + sorted(filtered_health_df_clinic['condition'].dropna().unique().tolist())
    selected_condition_demog = st.selectbox("Demografik Analiz için Durum Seçin:", options=conditions_for_demog, index=0, key="clinic_demog_cond_select_v2")
    
    condition_df_demog_source = filtered_health_df_clinic
    if selected_condition_demog != "All Conditions":
        condition_df_demog_source = filtered_health_df_clinic[filtered_health_df_clinic['condition'] == selected_condition_demog]
    
    # Consider only first diagnosis of this condition per patient in period for "new cases"
    condition_df_demog = condition_df_demog_source.copy()
    if 'encounter_date' in condition_df_demog.columns and 'patient_id' in condition_df_demog.columns and 'condition' in condition_df_demog.columns :
        condition_df_demog.sort_values('encounter_date', inplace=True)
        new_cases_demog_df = condition_df_demog.drop_duplicates(subset=['patient_id', 'condition'], keep='first')
    else: new_cases_demog_df = pd.DataFrame() # empty if essential columns are missing


    if not new_cases_demog_df.empty:
        demog_breakdown_cols_epi = st.columns(2)
        with demog_breakdown_cols_epi[0]:
            if 'age' in new_cases_demog_df.columns and new_cases_demog_df['age'].notna().any():
                age_bins = [0, 5, 18, 35, 50, 65, np.inf]; age_labels = ['0-4', '5-17', '18-34', '35-49', '50-64', '65+']
                new_cases_demog_df = new_cases_demog_df.copy() # Avoid SettingWithCopy
                new_cases_demog_df.loc[:, 'age_group_clinic_epi'] = pd.cut(new_cases_demog_df['age'], bins=age_bins, labels=age_labels, right=False)
                age_dist_demog = new_cases_demog_df['age_group_clinic_epi'].value_counts().sort_index().reset_index()
                age_dist_demog.columns = ['Age Group', 'New Cases']
                if not age_dist_demog.empty: st.plotly_chart(plot_bar_chart(age_dist_demog, 'Age Group', 'New Cases', f"{selected_condition_demog} - Yaş Gruplarına Göre Vaka Sayısı", height=300), use_container_width=True)
                else: st.caption("Yaş dağılımı için veri yok.")
            else: st.caption("Yaş verisi mevcut değil.")
        with demog_breakdown_cols_epi[1]:
            if 'gender' in new_cases_demog_df.columns and new_cases_demog_df['gender'].notna().any():
                gender_dist_demog = new_cases_demog_df['gender'].value_counts().reset_index()
                gender_dist_demog.columns = ['Gender', 'New Cases']
                if not gender_dist_demog.empty: st.plotly_chart(plot_donut_chart(gender_dist_demog, 'Gender', 'New Cases', f"{selected_condition_demog} - Cinsiyete Göre Vaka Sayısı", height=300), use_container_width=True)
                else: st.caption("Cinsiyet dağılımı için veri yok.")
            else: st.caption("Cinsiyet verisi mevcut değil.")
    elif selected_condition_demog == "All Conditions" and filtered_health_df_clinic.empty:
         st.caption(f"Seçili '{selected_condition_demog}' durumu için bu dönemde vaka bulunamadı.")
    elif selected_condition_demog != "All Conditions":
         st.caption(f"Seçili '{selected_condition_demog}' durumu için bu dönemde vaka bulunamadı.")


    # Referral Funnel - simplified example for demo
    st.markdown("---")
    st.subheader("Referral Funnel Analizi (Basit)")
    if 'referral_status' in filtered_health_df_clinic.columns and filtered_health_df_clinic['referral_status'].notna().any():
        referral_df_funnel = filtered_health_df_clinic[filtered_health_df_clinic['referral_status'] != 'N/A'].copy()
        if not referral_df_funnel.empty:
            total_referrals_made = referral_df_funnel['encounter_id'].nunique()
            completed_referrals = referral_df_funnel[referral_df_funnel['referral_outcome'].isin(['Completed', 'Service Provided', 'Attended'])]['encounter_id'].nunique()
            pending_referrals = referral_df_funnel[referral_df_funnel['referral_status'] == 'Pending']['encounter_id'].nunique()
            
            funnel_data = pd.DataFrame([
                {'stage': 'Referrals Made', 'count': total_referrals_made},
                {'stage': 'Referrals Completed', 'count': completed_referrals},
                {'stage': 'Referrals Still Pending', 'count': pending_referrals},
            ])
            if not funnel_data.empty and funnel_data['count'].sum() > 0:
                 st.plotly_chart(plot_bar_chart(funnel_data, 'stage', 'count', "Referral Durumları", height=350, y_axis_title="Sevk Sayısı"), use_container_width=True)
            else: st.caption("Sevk hunisi için veri yok.")
        else: st.caption("Analiz edilecek sevk kaydı bulunamadı.")
    else: st.caption("Referral status data not available.")
    st.markdown("---")
