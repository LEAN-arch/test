# test/pages/4_population_analytics_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from datetime import date, timedelta
import plotly.express as px # For scatter plot or more complex direct plots

from config import app_config
from utils.core_data_processing import (
    load_health_records,
    load_zone_data,
    get_trend_data
)
from utils.ai_analytics_engine import apply_ai_models
from utils.ui_visualization_helpers import (
    plot_bar_chart,
    plot_donut_chart,
    plot_annotated_line_chart,
    # plot_heatmap # Can add if relevant data for heatmap becomes available
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="Population Analytics - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_pop_analytics():
    css_path = app_config.STYLE_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path) as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else: logger.warning(f"Population Analytics CSS file not found: {css_path}.")
load_css_pop_analytics()

# --- Data Loading ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading population analytics data...")
def get_population_analytics_data():
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    health_df = apply_ai_models(health_df_raw) if not health_df_raw.empty else pd.DataFrame()
    zone_gdf = load_zone_data()
    zone_attributes_df = pd.DataFrame(zone_gdf.drop(columns=[zone_gdf.geometry.name], errors='ignore')) if zone_gdf is not None and not zone_gdf.empty else pd.DataFrame(columns=['zone_id', 'name', 'population', 'socio_economic_index'])
    return health_df, zone_attributes_df

health_df_pop, zone_attr_df_pop = get_population_analytics_data()

if health_df_pop.empty:
    st.error("🚨 Data Error: Could not load health records. Population analytics cannot be displayed."); st.stop()

st.title("📊 Population Health & Systems Analytics")
st.markdown("Explore demographic distributions, epidemiological patterns, clinical trends, and health system factors across the population.")
st.markdown("---")

# --- Sidebar Filters ---
if os.path.exists(app_config.APP_LOGO): st.sidebar.image(app_config.APP_LOGO, use_column_width='auto'); st.sidebar.markdown("---")
st.sidebar.header("🔎 Analytics Filters")

min_date_pop = health_df_pop['encounter_date'].min().date() if 'encounter_date' in health_df_pop and not health_df_pop['encounter_date'].dropna().empty else date.today() - timedelta(days=365)
max_date_pop = health_df_pop['encounter_date'].max().date() if 'encounter_date' in health_df_pop and not health_df_pop['encounter_date'].dropna().empty else date.today()
if min_date_pop > max_date_pop: min_date_pop = max_date_pop
default_pop_start_date = min_date_pop; default_pop_end_date = max_date_pop
selected_start_date_pop, selected_end_date_pop = st.sidebar.date_input("Select Date Range:", value=[default_pop_start_date, default_pop_end_date], min_value=min_date_pop, max_value=max_date_pop, key="pop_analytics_date_range_v2")
if selected_start_date_pop > selected_end_date_pop: st.sidebar.error("Start date must be before end date."); selected_start_date_pop = selected_end_date_pop

health_df_pop['encounter_date_obj'] = pd.to_datetime(health_df_pop['encounter_date']).dt.date
filtered_health_df = health_df_pop[
    (health_df_pop['encounter_date_obj'].notna()) &
    (health_df_pop['encounter_date_obj'] >= selected_start_date_pop) &
    (health_df_pop['encounter_date_obj'] <= selected_end_date_pop)
].copy()

if filtered_health_df.empty:
    st.warning(f"No health encounter data for selected period: {selected_start_date_pop.strftime('%d %b %Y')} - {selected_end_date_pop.strftime('%d %b %Y')}."); st.stop()

conditions_list = ["All Conditions"] + sorted(filtered_health_df['condition'].dropna().unique().tolist())
selected_condition_filter_pop = st.sidebar.selectbox("Filter by Condition (Optional):", options=conditions_list, index=0, key="pop_condition_filter_v2")
analytics_df = filtered_health_df.copy() # Start with date-filtered data
if selected_condition_filter_pop != "All Conditions":
    analytics_df = analytics_df[analytics_df['condition'] == selected_condition_filter_pop]
    if analytics_df.empty: st.warning(f"No data for '{selected_condition_filter_pop}' in selected period. Showing overall or adjust filters.")
    # If analytics_df becomes empty after condition filter, plots might show "no data" captions.

# Zone filter
zones_list = ["All Zones"] + sorted(analytics_df['zone_id'].dropna().unique().tolist())
selected_zone_filter_pop = st.sidebar.selectbox("Filter by Zone (Optional):", options=zones_list, index=0, key="pop_zone_filter_v1")
if selected_zone_filter_pop != "All Zones":
    analytics_df = analytics_df[analytics_df['zone_id'] == selected_zone_filter_pop]
    if analytics_df.empty: st.warning(f"No data for '{selected_zone_filter_pop}' with current filters.")


# --- Tabbed Interface ---
tab_epi_overview, tab_demographics_sdoh, tab_clinical_dx, tab_systems_equity = st.tabs([
    "📈 Epidemiological Overview", 
    "🧑‍⚕️ Demographics & SDOH", 
    "🧬 Clinical & Diagnostics", 
    "🌍 Systems & Equity"
])

with tab_epi_overview:
    st.header(f"Epidemiological Overview for '{selected_condition_filter_pop}' in '{selected_zone_filter_pop}'")
    if analytics_df.empty:
        st.info("No data available for epidemiological overview with current filters.")
    else:
        epi_overview_cols = st.columns(2)
        with epi_overview_cols[0]:
            st.subheader("Condition Case Counts")
            # Unique patients for each condition in the filtered data
            condition_counts = analytics_df.groupby('condition')['patient_id'].nunique().nlargest(10).reset_index(name='unique_patients')
            if not condition_counts.empty:
                st.plotly_chart(plot_bar_chart(condition_counts, 'condition', 'unique_patients', "Top 10 Conditions by Unique Patient Count", height=400, orientation='h'), use_container_width=True)
            else: st.caption("No condition data to display counts.")
        
        with epi_overview_cols[1]:
            st.subheader("AI Risk Score Distribution")
            if 'ai_risk_score' in analytics_df.columns and analytics_df['ai_risk_score'].notna().any():
                # Simple histogram of AI risk scores
                fig_risk_dist = px.histogram(analytics_df.dropna(subset=['ai_risk_score']), x="ai_risk_score", nbins=20, title="Distribution of Patient AI Risk Scores")
                fig_risk_dist.update_layout(bargap=0.1, height=400)
                st.plotly_chart(fig_risk_dist, use_container_width=True)
            else: st.caption("AI Risk Score data not available for distribution.")

        st.markdown("---")
        st.subheader(f"Incidence Trend for Top Conditions (Weekly New Cases)")
        # Top 3 conditions by volume for trend display
        if not analytics_df.empty and 'condition' in analytics_df.columns:
            top_conditions_for_trend = analytics_df['condition'].value_counts().nlargest(3).index.tolist()
            if top_conditions_for_trend:
                inc_trend_cols = st.columns(len(top_conditions_for_trend))
                df_for_inc_trend = analytics_df.copy()
                # Calculate first occurrence in period for "new case" definition
                df_for_inc_trend.sort_values('encounter_date', inplace=True)
                df_for_inc_trend['is_first_in_period'] = ~df_for_inc_trend.duplicated(subset=['patient_id', 'condition'], keep='first')
                new_cases_df_trend = df_for_inc_trend[df_for_inc_trend['is_first_in_period']]

                for i, cond_name in enumerate(top_conditions_for_trend):
                    cond_trend_data = new_cases_df_trend[new_cases_df_trend['condition'] == cond_name]
                    if not cond_trend_data.empty:
                        weekly_new_cases_trend = get_trend_data(cond_trend_data, 'patient_id', date_col='encounter_date', period='W-Mon', agg_func='count')
                        if not weekly_new_cases_trend.empty:
                            with inc_trend_cols[i]:
                                st.plotly_chart(plot_annotated_line_chart(weekly_new_cases_trend, f"Weekly New {cond_name} Cases", y_axis_title="New Cases", height=300, date_format="%U, %Y (Wk)"), use_container_width=True)
                        else: with inc_trend_cols[i]: st.caption(f"No trend data for {cond_name}.")
                    else: with inc_trend_cols[i]: st.caption(f"No data for {cond_name} trends.")
            else: st.caption("No top conditions found for incidence trend.")
        else: st.caption("Condition data missing for incidence trends.")

with tab_demographics_sdoh:
    # (Demographics & SDOH section remains largely the same as before, ensure it uses `analytics_df`)
    st.header("Demographics & Social Determinants of Health (SDOH)")
    if analytics_df.empty: st.info("No data available for Demographics/SDOH analysis with current filters.")
    else:
        demo_cols_sdoh = st.columns(2)
        with demo_cols_sdoh[0]: # Age Distribution
            st.subheader("Age Distribution of Patients")
            if 'age' in analytics_df.columns and analytics_df['age'].notna().any():
                age_bins = [0, 5, 12, 18, 35, 50, 65, np.inf]; age_labels = ['0-4 yrs', '5-11 yrs', '12-17 yrs', '18-34 yrs', '35-49 yrs', '50-64 yrs', '65+ yrs']
                analytics_df_copy_age = analytics_df.copy() # Avoid SettingWithCopyWarning
                analytics_df_copy_age.loc[:, 'age_group_pop_sdoh'] = pd.cut(analytics_df_copy_age['age'], bins=age_bins, labels=age_labels, right=False)
                age_dist_df_sdoh = analytics_df_copy_age['age_group_pop_sdoh'].value_counts().sort_index().reset_index(); age_dist_df_sdoh.columns = ['Age Group', 'Patient Encounters']
                st.plotly_chart(plot_bar_chart(age_dist_df_sdoh, 'Age Group', 'Patient Encounters', "Encounters by Age Group", height=350), use_container_width=True)
            else: st.caption("Age data not available.")
        with demo_cols_sdoh[1]: # Gender Distribution
            st.subheader("Gender Distribution of Patients")
            if 'gender' in analytics_df.columns:
                gender_dist_df_sdoh = analytics_df['gender'].value_counts().reset_index(); gender_dist_df_sdoh.columns = ['Gender', 'Patient Encounters']
                st.plotly_chart(plot_donut_chart(gender_dist_df_sdoh, 'Gender', 'Patient Encounters', "Encounters by Gender", height=350), use_container_width=True)
            else: st.caption("Gender data not available.")
        
        st.markdown("---")
        st.subheader("Geographic & Socio-Economic Context (Zone Level Averages)")
        if not zone_attr_df_pop.empty and 'zone_id' in analytics_df.columns and analytics_df['zone_id'].notna().any():
            # Calculate unique patients and avg risk *from already filtered analytics_df*
            patients_per_zone_filtered = analytics_df.groupby('zone_id')['patient_id'].nunique().reset_index(name='patients_in_filtered_period')
            avg_risk_per_zone_filtered = analytics_df.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='avg_risk_in_filtered_period')
            
            zone_analysis_df_sdoh = zone_attr_df_pop.merge(patients_per_zone_filtered, on='zone_id', how='left')
            zone_analysis_df_sdoh = zone_analysis_df_sdoh.merge(avg_risk_per_zone_filtered, on='zone_id', how='left')
            zone_analysis_df_sdoh['patients_in_filtered_period'].fillna(0, inplace=True)
            
            if not zone_analysis_df_sdoh.empty:
                sdoh_context_cols = st.columns(2)
                with sdoh_context_cols[0]:
                    if 'socio_economic_index' in zone_analysis_df_sdoh.columns:
                        st.plotly_chart(plot_bar_chart(zone_analysis_df_sdoh.sort_values('socio_economic_index'), 'name', 'socio_economic_index', 'Socio-Economic Index by Zone', height=350, y_axis_title="SES Index"), use_container_width=True)
                with sdoh_context_cols[1]:
                    if 'avg_travel_time_clinic_min' in zone_analysis_df_sdoh.columns:
                        st.plotly_chart(plot_bar_chart(zone_analysis_df_sdoh.sort_values('avg_travel_time_clinic_min'), 'name', 'avg_travel_time_clinic_min', 'Avg. Travel Time to Clinic by Zone', height=350, y_axis_title="Minutes"), use_container_width=True)
            else: st.info("No zone-level data to display after merging with filtered patient encounters.")
        else: st.info("Zone attribute data or zone_id in health records unavailable.")


with tab_clinical_dx:
    st.header("Clinical & Diagnostic Data Patterns")
    if analytics_df.empty: st.info("No data for clinical/diagnostic analysis with current filters.")
    else:
        # ... (Top Presenting Symptoms, Test Result Distribution (Top 5 Tests) as before) ...
        # ... (Comorbidity Patterns, Medication Adherence as before, using analytics_df) ...
        clin_dx_cols = st.columns(2)
        with clin_dx_cols[0]:
            st.subheader("Top Presenting Symptoms (from encounters)")
            if 'patient_reported_symptoms' in analytics_df.columns and analytics_df['patient_reported_symptoms'].notna().any():
                symptoms_exploded_dx = analytics_df['patient_reported_symptoms'].str.split(';').explode().str.strip().replace('',np.nan).dropna()
                symptom_counts_dx = symptoms_exploded_dx.value_counts().nlargest(10).reset_index(); symptom_counts_dx.columns = ['Symptom', 'Frequency']
                if not symptom_counts_dx.empty: st.plotly_chart(plot_bar_chart(symptom_counts_dx, 'Symptom', 'Frequency', "Top 10 Reported Symptoms", height=400, orientation='h'), use_container_width=True)
                else: st.caption("No distinct symptom data.")
            else: st.caption("Patient reported symptoms data unavailable.")
        with clin_dx_cols[1]:
            st.subheader("Test Result Distribution (Top 5 Tests by Volume)")
            if 'test_type' in analytics_df.columns and 'test_result' in analytics_df.columns:
                top_5_tests_dx = analytics_df['test_type'].value_counts().nlargest(5).index.tolist()
                top_tests_df_dx = analytics_df[analytics_df['test_type'].isin(top_5_tests_dx)]
                test_results_dist_dx = top_tests_df_dx[~top_tests_df_dx['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown', 'N/A', 'Indeterminate'])]
                if not test_results_dist_dx.empty:
                    test_result_summary_dx = test_results_dist_dx.groupby(['test_type', 'test_result'])['patient_id'].nunique().reset_index(name='count')
                    st.plotly_chart(plot_bar_chart(test_result_summary_dx, 'test_type', 'count', "Conclusive Test Results Distribution", color_col='test_result', barmode='group', height=400), use_container_width=True)
                else: st.caption("No conclusive test result data for top tests.")
            else: st.caption("Test type or result data unavailable.")
        
        # Additional: Overall Test Positivity Rate Trend for a key test type
        st.markdown("---"); st.subheader("Overall Test Positivity Rate Trend (e.g., Malaria RDT)")
        # Ensure test_type uses keys, not display names, for reliable matching if KEY_TEST_TYPES_FOR_ANALYSIS is used.
        mal_rdt_key_trend_pop = "RDT-Malaria" # Key used in data
        mal_rdt_disp_pop = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_rdt_key_trend_pop,{}).get("display_name", mal_rdt_key_trend_pop)
        mal_df_pop = analytics_df[(analytics_df['test_type'] == mal_rdt_key_trend_pop) & (~analytics_df['test_result'].isin(['Pending','Rejected Sample','Unknown']))].copy()
        if not mal_df_pop.empty:
            mal_df_pop.loc[:,'is_positive'] = mal_df_pop['test_result'] == 'Positive'
            if not pd.api.types.is_datetime64_ns_dtype(mal_df_pop['encounter_date']): mal_df_pop.loc[:,'encounter_date'] = pd.to_datetime(mal_df_pop['encounter_date'], errors='coerce')
            mal_df_pop.dropna(subset=['encounter_date'], inplace=True)
            weekly_pos_rate_pop = get_trend_data(mal_df_pop, 'is_positive', date_col='encounter_date', period='W-Mon', agg_func='mean') * 100
            if not weekly_pos_rate_pop.empty: st.plotly_chart(plot_annotated_line_chart(weekly_pos_rate_pop, f"Weekly {mal_rdt_disp_pop} Positivity Rate", y_axis_title="Positivity (%)", height=350, target_line=app_config.TARGET_MALARIA_POSITIVITY_RATE), use_container_width=True)
            else: st.caption(f"No data for {mal_rdt_disp_pop} positivity trend.")
        else: st.caption(f"No {mal_rdt_disp_pop} test data for positivity trend.")


with tab_systems_equity:
    st.header("Health Systems Context & Equity Insights")
    if analytics_df.empty: st.info("No data available for systems/equity analysis with current filters.")
    else:
        # (Patient Encounters by Clinic ID, Referral Pathway Performance (Status Dist), Avg Test TAT Trend as before, using analytics_df)
        # ...
        sys_equity_cols = st.columns(2)
        with sys_equity_cols[0]:
            st.subheader("Encounters by Clinic ID (Top 10)")
            if 'clinic_id' in analytics_df.columns:
                facility_load_sys = analytics_df['clinic_id'].value_counts().nlargest(10).reset_index(); facility_load_sys.columns = ['Clinic ID', 'Number of Encounters']
                st.plotly_chart(plot_bar_chart(facility_load_sys, 'Clinic ID', 'Number of Encounters', "Top 10 Clinics by Encounter Volume", height=400, orientation='h'), use_container_width=True)
        with sys_equity_cols[1]:
            st.subheader("Referral Status Distribution")
            if 'referral_status' in analytics_df.columns and analytics_df['referral_status'].notna().any():
                ref_data_sys = analytics_df[analytics_df['referral_status'].str.lower() != "n/a"].copy()
                if not ref_data_sys.empty:
                    ref_status_counts_sys = ref_data_sys['referral_status'].value_counts().reset_index(); ref_status_counts_sys.columns = ['Referral Status', 'Count']
                    st.plotly_chart(plot_donut_chart(ref_status_counts_sys, 'Referral Status', 'Count', "Distribution of Referral Statuses", height=400), use_container_width=True)
                else: st.caption("No active referral data.")
        
        st.markdown("---")
        st.subheader("AI Risk Score Distribution by Zone Socio-Economic Index")
        if 'zone_id' in analytics_df.columns and 'ai_risk_score' in analytics_df.columns and \
           not zone_attr_df_pop.empty and 'socio_economic_index' in zone_attr_df_pop.columns and 'name' in zone_attr_df_pop.columns: # Ensure 'name' for hover
            avg_risk_zone_equity = analytics_df.groupby('zone_id')['ai_risk_score'].mean().reset_index()
            equity_df_plot = zone_attr_df_pop.merge(avg_risk_zone_equity, on='zone_id', how='inner') # inner merge to only get zones with both info
            equity_df_plot.dropna(subset=['ai_risk_score', 'socio_economic_index'], inplace=True)
            if not equity_df_plot.empty:
                fig_equity_risk_ses = px.scatter(equity_df_plot, x='socio_economic_index', y='ai_risk_score', 
                                       text='name',  # Show zone name on points if not too cluttered
                                       size='population' if 'population' in equity_df_plot.columns else None, 
                                       color='ai_risk_score',
                                       title="Avg. Patient AI Risk vs. Zone Socio-Economic Index",
                                       labels={'socio_economic_index': "Socio-Economic Index (Zone)", 'ai_risk_score': "Avg. Patient AI Risk Score in Zone (Period)"},
                                       height=450, color_continuous_scale="Reds", hover_name='name')
                fig_equity_risk_ses.update_traces(textposition='top center')
                st.plotly_chart(fig_equity_risk_ses, use_container_width=True)
            else: st.caption("Not enough data to plot AI Risk vs Zone SES.")
        else: st.caption("Required data (AI risk, zone_id, zone attributes with SES and name) not fully available.")


# --- Footer ---
st.markdown("---")
st.caption(app_config.APP_FOOTER)
