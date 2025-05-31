# test/pages/4_population_analytics_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from datetime import date, timedelta
import plotly.express as px

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
    plot_annotated_line_chart
)

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Population Analytics - Health Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_pop_analytics():
    css_path = app_config.STYLE_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info("Population Analytics Dashboard: CSS loaded successfully.")
    else:
        logger.warning(f"Population Analytics Dashboard: CSS file not found at {css_path}.")
load_css_pop_analytics()

# --- Data Loading ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading population analytics data...")
def get_population_analytics_data():
    logger.info("Population Analytics: Loading health records and zone data...")
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    health_df = apply_ai_models(health_df_raw) if not health_df_raw.empty else pd.DataFrame()
    
    zone_gdf = load_zone_data()
    if zone_gdf is not None and not zone_gdf.empty and hasattr(zone_gdf, 'geometry') and zone_gdf.geometry.name in zone_gdf.columns:
        zone_attributes_df = pd.DataFrame(zone_gdf.drop(columns=[zone_gdf.geometry.name], errors='ignore'))
        logger.info(f"Loaded {len(zone_attributes_df)} zone attributes for SDOH context.")
    else:
        logger.warning("Zone attributes data not available or GDF invalid. SDOH visualizations by zone will be limited.")
        zone_attributes_df = pd.DataFrame(columns=['zone_id', 'name', 'population', 'socio_economic_index'])

    if health_df.empty:
        logger.error("Population Analytics: Failed to load or process health records.")
    return health_df, zone_attributes_df

health_df_pop, zone_attr_df_pop = get_population_analytics_data()

if health_df_pop.empty:
    st.error("🚨 **Data Error:** Could not load health records. Population analytics cannot be displayed.")
    logger.critical("Population Analytics: health_df_pop is empty.")
    st.stop()

st.title("📊 Population Health & Systems Analytics")
st.markdown("Explore demographic distributions, epidemiological patterns, clinical trends, and health system factors across the population.")
st.markdown("---")

# --- Sidebar Filters ---
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, use_column_width='auto'); st.sidebar.markdown("---")
st.sidebar.header("🔎 Analytics Filters")

min_date_pop = date.today() - timedelta(days=365)
max_date_pop = date.today()
if 'encounter_date' in health_df_pop.columns and not health_df_pop['encounter_date'].dropna().empty:
    min_date_pop = health_df_pop['encounter_date'].dropna().min().date()
    max_date_pop = health_df_pop['encounter_date'].dropna().max().date()
if min_date_pop > max_date_pop: min_date_pop = max_date_pop

default_pop_start_date = min_date_pop
default_pop_end_date = max_date_pop

selected_start_date_pop, selected_end_date_pop = st.sidebar.date_input(
    "Select Date Range for Analysis:", value=[default_pop_start_date, default_pop_end_date],
    min_value=min_date_pop, max_value=max_date_pop, key="pop_analytics_date_range_v3"
)
if selected_start_date_pop > selected_end_date_pop:
    st.sidebar.error("Error: Start date must be before end date."); selected_start_date_pop = selected_end_date_pop

analytics_df_base = pd.DataFrame(columns=health_df_pop.columns) # Initialize with schema
if 'encounter_date' in health_df_pop.columns:
    health_df_pop['encounter_date_obj'] = pd.to_datetime(health_df_pop['encounter_date'], errors='coerce').dt.date
    analytics_df_base = health_df_pop[
        (health_df_pop['encounter_date_obj'].notna()) &
        (health_df_pop['encounter_date_obj'] >= selected_start_date_pop) &
        (health_df_pop['encounter_date_obj'] <= selected_end_date_pop)
    ].copy()

if analytics_df_base.empty:
    st.warning(f"No health encounter data found for period: {selected_start_date_pop.strftime('%d %b %Y')} - {selected_end_date_pop.strftime('%d %b %Y')}. Adjust filters."); st.stop()

conditions_list_pop = ["All Conditions"] + sorted(analytics_df_base['condition'].dropna().unique().tolist())
selected_condition_filter_pop = st.sidebar.selectbox("Filter by Condition (Optional):", options=conditions_list_pop, index=0, key="pop_condition_filter_v3")
analytics_df = analytics_df_base.copy()
if selected_condition_filter_pop != "All Conditions":
    analytics_df = analytics_df[analytics_df['condition'] == selected_condition_filter_pop]

zones_list_pop = ["All Zones"] + sorted(analytics_df_base['zone_id'].dropna().unique().tolist())
selected_zone_filter_pop = st.sidebar.selectbox("Filter by Zone (Optional):", options=zones_list_pop, index=0, key="pop_zone_filter_v2")
if selected_zone_filter_pop != "All Zones":
    analytics_df = analytics_df[analytics_df['zone_id'] == selected_zone_filter_pop]

analytics_df_display = analytics_df.copy() # What's actually used by tabs
if analytics_df.empty :
    st.warning(f"No data for '{selected_condition_filter_pop}' in zone '{selected_zone_filter_pop}' for the selected period. Displaying broader data for the period if available, or adjust filters.")
    analytics_df_display = analytics_df_base.copy() # Fallback to only date-filtered
    if selected_zone_filter_pop != "All Zones": # Respect zone if only condition filter resulted in empty
         analytics_df_display = analytics_df_display[analytics_df_display['zone_id'] == selected_zone_filter_pop]
    if analytics_df_display.empty: # If even that fallback is empty, use full base (unlikely if first check passed)
         analytics_df_display = health_df_pop.copy() 

# --- Tabbed Interface ---
tab_epi_overview, tab_demographics_sdoh, tab_clinical_dx, tab_systems_equity = st.tabs([
    "📈 Epidemiological Overview", "🧑‍⚕️ Demographics & SDOH", "🧬 Clinical & Diagnostics", "🌍 Systems & Equity"
])

with tab_epi_overview:
    st.header(f"Epidemiological Overview for '{selected_condition_filter_pop}' in '{selected_zone_filter_pop}'")
    if analytics_df_display.empty:
        st.info("No data available for epidemiological overview with current filters.")
    else:
        epi_overview_cols = st.columns(2)
        with epi_overview_cols[0]:
            st.subheader("Condition Case Counts")
            if 'condition' in analytics_df_display.columns and analytics_df_display['condition'].notna().any():
                condition_counts_epi = analytics_df_display.groupby('condition')['patient_id'].nunique().nlargest(10).reset_index(name='unique_patients')
                if not condition_counts_epi.empty:
                    condition_counts_epi['condition'] = condition_counts_epi['condition'].astype(str)
                    st.plotly_chart(plot_bar_chart(condition_counts_epi, x_col='condition', y_col='unique_patients', title="Top 10 Conditions by Unique Patient Count", height=400, orientation='h', y_is_count=True, text_format='d'), use_container_width=True)
                else: st.caption("No condition data to display counts after aggregation.")
            else: st.caption("Condition column missing or empty.")
        
        with epi_overview_cols[1]:
            st.subheader("AI Risk Score Distribution")
            if 'ai_risk_score' in analytics_df_display.columns and analytics_df_display['ai_risk_score'].notna().any():
                fig_risk_dist_epi = px.histogram(analytics_df_display.dropna(subset=['ai_risk_score']), x="ai_risk_score", nbins=20, title="Distribution of Patient AI Risk Scores")
                fig_risk_dist_epi.update_layout(bargap=0.1, height=400); st.plotly_chart(fig_risk_dist_epi, use_container_width=True)
            else: st.caption("AI Risk Score data not available for distribution.")

        st.markdown("---")
        st.subheader(f"Incidence Trend for Top Conditions (Weekly New Cases)")
        if not analytics_df_display.empty and 'condition' in analytics_df_display.columns and 'patient_id' in analytics_df_display.columns and 'encounter_date' in analytics_df_display.columns:
            top_n_for_trend_plot_epi = 3 
            df_for_top_cond_trend_epi = analytics_df_display.copy()

            if selected_condition_filter_pop != "All Conditions":
                top_conditions_for_trend_epi = [selected_condition_filter_pop]
                if selected_condition_filter_pop not in df_for_top_cond_trend_epi['condition'].unique(): top_conditions_for_trend_epi = []
            else: top_conditions_for_trend_epi = df_for_top_cond_trend_epi['condition'].value_counts().nlargest(top_n_for_trend_plot_epi).index.tolist()

            if top_conditions_for_trend_epi:
                num_charts_to_plot_epi = len(top_conditions_for_trend_epi)
                inc_trend_cols_epi = st.columns(num_charts_to_plot_epi if num_charts_to_plot_epi > 0 else 1)
                
                df_for_inc_trend_calc_epi = df_for_top_cond_trend_epi.copy()
                if not pd.api.types.is_datetime64_ns_dtype(df_for_inc_trend_calc_epi['encounter_date']): df_for_inc_trend_calc_epi['encounter_date'] = pd.to_datetime(df_for_inc_trend_calc_epi['encounter_date'], errors='coerce')
                df_for_inc_trend_calc_epi.dropna(subset=['encounter_date'], inplace=True)
                if not df_for_inc_trend_calc_epi.empty : # Check after dropna
                    df_for_inc_trend_calc_epi.sort_values('encounter_date', inplace=True)
                    df_for_inc_trend_calc_epi['is_first_in_period'] = ~df_for_inc_trend_calc_epi.duplicated(subset=['patient_id', 'condition'], keep='first')
                    new_cases_df_trend_epi = df_for_inc_trend_calc_epi[df_for_inc_trend_calc_epi['is_first_in_period']]

                    for i, cond_name_epi in enumerate(top_conditions_for_trend_epi):
                        current_col_epi = inc_trend_cols_epi[i % num_charts_to_plot_epi] # Safe indexing
                        condition_trend_data_epi = new_cases_df_trend_epi[new_cases_df_trend_epi['condition'] == cond_name_epi]
                        with current_col_epi:
                            if not condition_trend_data_epi.empty:
                                weekly_new_cases_trend_epi = get_trend_data(condition_trend_data_epi, 'patient_id', date_col='encounter_date', period='W-Mon', agg_func='count')
                                if not weekly_new_cases_trend_epi.empty: st.plotly_chart(plot_annotated_line_chart(weekly_new_cases_trend_epi, f"Weekly New {cond_name_epi} Cases", y_axis_title="New Cases", height=300, date_format="%U, %Y (Wk)", y_is_count=True), use_container_width=True)
                                else: st.caption(f"No trend data to plot for {cond_name_epi}.")
                            else: st.caption(f"No new cases data for {cond_name_epi} in this period.")
                else: st.caption("Not enough valid date data for incidence calculation.") # After date cleaning for trend
            else: 
                if selected_condition_filter_pop != "All Conditions": st.caption(f"No data for '{selected_condition_filter_pop}' for incidence trend.")
                else: st.caption("No top conditions for incidence trends with current filters.")
        else: st.caption("Required data missing for incidence trends.")

with tab_demographics_sdoh:
    st.header("Demographics & Social Determinants of Health (SDOH)")
    if analytics_df_display.empty: st.info("No data for Demographics/SDOH analysis with current filters.")
    else:
        demo_cols_sdoh_page = st.columns(2)
        with demo_cols_sdoh_page[0]:
            st.subheader("Age Distribution of Patients")
            if 'age' in analytics_df_display.columns and analytics_df_display['age'].notna().any():
                age_bins_sdoh = [0, 5, 12, 18, 35, 50, 65, np.inf]; age_labels_sdoh = ['0-4 yrs', '5-11 yrs', '12-17 yrs', '18-34 yrs', '35-49 yrs', '50-64 yrs', '65+ yrs']
                analytics_df_copy_age_sdoh = analytics_df_display.copy()
                analytics_df_copy_age_sdoh.loc[:, 'age_group_pop_sdoh'] = pd.cut(analytics_df_copy_age_sdoh['age'], bins=age_bins_sdoh, labels=age_labels_sdoh, right=False)
                age_dist_df_sdoh = analytics_df_copy_age_sdoh['age_group_pop_sdoh'].value_counts().sort_index().reset_index(); age_dist_df_sdoh.columns = ['Age Group', 'Patient Encounters']
                if not age_dist_df_sdoh.empty : st.plotly_chart(plot_bar_chart(age_dist_df_sdoh, 'Age Group', 'Patient Encounters', "Encounters by Age Group", height=350, y_is_count=True, text_format='d'), use_container_width=True)
                else: st.caption("No age data to plot.")
            else: st.caption("Age data not available.")
        with demo_cols_sdoh_page[1]:
            st.subheader("Gender Distribution of Patients")
            if 'gender' in analytics_df_display.columns and analytics_df_display['gender'].notna().any():
                gender_dist_df_sdoh = analytics_df_display['gender'].value_counts().reset_index(); gender_dist_df_sdoh.columns = ['Gender', 'Patient Encounters']
                if not gender_dist_df_sdoh.empty: st.plotly_chart(plot_donut_chart(gender_dist_df_sdoh, 'Gender', 'Patient Encounters', "Encounters by Gender", height=350, values_are_counts=True), use_container_width=True)
                else: st.caption("No gender data to plot.")
            else: st.caption("Gender data not available.")
        
        st.markdown("---")
        st.subheader("Geographic & Socio-Economic Context (Zone Level Averages)")
        if not zone_attr_df_pop.empty and 'zone_id' in analytics_df_display.columns and analytics_df_display['zone_id'].notna().any():
            patients_per_zone_filtered_sdoh = analytics_df_display.groupby('zone_id')['patient_id'].nunique().reset_index(name='patients_in_filtered_period')
            avg_risk_per_zone_filtered_sdoh = analytics_df_display.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='avg_risk_in_filtered_period')
            
            zone_analysis_df_sdoh = zone_attr_df_pop.copy() # Start with all zone attributes
            # Ensure zone_id dtypes match for merge
            zone_analysis_df_sdoh['zone_id'] = zone_analysis_df_sdoh['zone_id'].astype(str)
            patients_per_zone_filtered_sdoh['zone_id'] = patients_per_zone_filtered_sdoh['zone_id'].astype(str)
            avg_risk_per_zone_filtered_sdoh['zone_id'] = avg_risk_per_zone_filtered_sdoh['zone_id'].astype(str)

            zone_analysis_df_sdoh = zone_analysis_df_sdoh.merge(patients_per_zone_filtered_sdoh, on='zone_id', how='left')
            zone_analysis_df_sdoh = zone_analysis_df_sdoh.merge(avg_risk_per_zone_filtered_sdoh, on='zone_id', how='left')
            zone_analysis_df_sdoh['patients_in_filtered_period'].fillna(0, inplace=True)
            # avg_risk_in_filtered_period can remain NaN if no patients from that zone in filtered data
            
            if not zone_analysis_df_sdoh.empty:
                sdoh_context_cols_display = st.columns(2)
                with sdoh_context_cols_display[0]:
                    if 'socio_economic_index' in zone_analysis_df_sdoh.columns and zone_analysis_df_sdoh['socio_economic_index'].notna().any():
                         st.plotly_chart(plot_bar_chart(zone_analysis_df_sdoh.sort_values('socio_economic_index'), 'name', 'socio_economic_index', 'Socio-Economic Index by Zone', height=350, y_axis_title="SES Index", text_format=".2f"), use_container_width=True)
                with sdoh_context_cols_display[1]:
                    if 'avg_travel_time_clinic_min' in zone_analysis_df_sdoh.columns and zone_analysis_df_sdoh['avg_travel_time_clinic_min'].notna().any():
                        st.plotly_chart(plot_bar_chart(zone_analysis_df_sdoh.sort_values('avg_travel_time_clinic_min'), 'name', 'avg_travel_time_clinic_min', 'Avg. Travel Time to Clinic by Zone', height=350, y_axis_title="Minutes", y_is_count=False, text_format=".0f"), use_container_width=True)
            else: st.info("No zone-level data to display after merging.")
        else: st.info("Zone attribute data or zone_id in health records unavailable for SDOH context.")

with tab_clinical_dx:
    st.header("Clinical & Diagnostic Data Patterns")
    if analytics_df_display.empty: st.info("No data for clinical/diagnostic analysis with current filters.")
    else:
        clin_dx_cols_page = st.columns(2)
        with clin_dx_cols_page[0]:
            st.subheader("Top Presenting Symptoms (from encounters)")
            if 'patient_reported_symptoms' in analytics_df_display.columns and analytics_df_display['patient_reported_symptoms'].notna().any():
                symptoms_exploded_dx_page = analytics_df_display['patient_reported_symptoms'].str.split(';').explode().str.strip().replace(['', 'Unknown', 'N/A'],np.nan).dropna()
                if not symptoms_exploded_dx_page.empty:
                    symptom_counts_dx_page = symptoms_exploded_dx_page.value_counts().nlargest(10).reset_index(); symptom_counts_dx_page.columns = ['Symptom', 'Frequency']
                    st.plotly_chart(plot_bar_chart(symptom_counts_dx_page, 'Symptom', 'Frequency', "Top 10 Reported Symptoms", height=400, orientation='h', y_is_count=True, text_format='d'), use_container_width=True)
                else: st.caption("No distinct symptoms reported.")
            else: st.caption("Patient reported symptoms data unavailable.")
        with clin_dx_cols_page[1]:
            st.subheader("Test Result Distribution (Top 5 Tests by Volume)")
            if 'test_type' in analytics_df_display.columns and 'test_result' in analytics_df_display.columns:
                top_5_tests_dx_page = analytics_df_display['test_type'].value_counts().nlargest(5).index.tolist()
                top_tests_df_dx_page = analytics_df_display[analytics_df_display['test_type'].isin(top_5_tests_dx_page)]
                test_results_dist_dx_page = top_tests_df_dx_page[~top_tests_df_dx_page['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown', 'N/A', 'Indeterminate'])]
                if not test_results_dist_dx_page.empty:
                    test_result_summary_dx_page = test_results_dist_dx_page.groupby(['test_type', 'test_result'])['patient_id'].nunique().reset_index(name='count')
                    st.plotly_chart(plot_bar_chart(test_result_summary_dx_page, 'test_type', 'count', "Conclusive Test Results Distribution", color_col='test_result', barmode='group', height=400, y_is_count=True, text_format='d'), use_container_width=True)
                else: st.caption("No conclusive test result data for top tests.")
            else: st.caption("Test type or result data unavailable.")
        
        st.markdown("---"); st.subheader("Overall Test Positivity Rate Trend (e.g., Malaria RDT)")
        mal_rdt_key_pop_trend = "RDT-Malaria"
        mal_rdt_disp_pop_trend = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_rdt_key_pop_trend,{}).get("display_name", mal_rdt_key_pop_trend)
        mal_df_pop_trend_source = analytics_df_display[(analytics_df_display['test_type'] == mal_rdt_key_pop_trend) & (~analytics_df_display['test_result'].isin(['Pending','Rejected Sample','Unknown']))].copy()
        if not mal_df_pop_trend_source.empty:
            mal_df_pop_trend_source.loc[:,'is_positive'] = mal_df_pop_trend_source['test_result'] == 'Positive'
            if not pd.api.types.is_datetime64_ns_dtype(mal_df_pop_trend_source['encounter_date']): mal_df_pop_trend_source.loc[:,'encounter_date'] = pd.to_datetime(mal_df_pop_trend_source['encounter_date'], errors='coerce')
            mal_df_pop_trend_source.dropna(subset=['encounter_date'], inplace=True)
            if not mal_df_pop_trend_source.empty :
                weekly_pos_rate_pop_trend = get_trend_data(mal_df_pop_trend_source, 'is_positive', date_col='encounter_date', period='W-Mon', agg_func='mean') * 100
                if not weekly_pos_rate_pop_trend.empty: st.plotly_chart(plot_annotated_line_chart(weekly_pos_rate_pop_trend, f"Weekly {mal_rdt_disp_pop_trend} Positivity Rate", y_axis_title="Positivity (%)", height=350, target_line=app_config.TARGET_MALARIA_POSITIVITY_RATE, y_is_count=False), use_container_width=True) # Pos rate is not a count
                else: st.caption(f"No data for {mal_rdt_disp_pop_trend} positivity trend.")
            else: st.caption(f"No valid date data for {mal_rdt_disp_pop_trend} positivity trend.")
        else: st.caption(f"No {mal_rdt_disp_pop_trend} test data for positivity trend.")

with tab_systems_equity:
    st.header("Health Systems Context & Equity Insights")
    if analytics_df_display.empty: st.info("No data for systems/equity analysis with current filters.")
    else:
        sys_equity_cols_page = st.columns(2)
        with sys_equity_cols_page[0]:
            st.subheader("Encounters by Clinic ID (Top 10)")
            if 'clinic_id' in analytics_df_display.columns and analytics_df_display['clinic_id'].notna().any():
                facility_load_sys_page = analytics_df_display['clinic_id'].value_counts().nlargest(10).reset_index(); facility_load_sys_page.columns = ['Clinic ID', 'Number of Encounters']
                st.plotly_chart(plot_bar_chart(facility_load_sys_page, 'Clinic ID', 'Number of Encounters', "Top 10 Clinics by Encounter Volume", height=400, orientation='h', y_is_count=True, text_format='d'), use_container_width=True)
        with sys_equity_cols_page[1]:
            st.subheader("Referral Status Distribution")
            if 'referral_status' in analytics_df_display.columns and analytics_df_display['referral_status'].notna().any():
                ref_data_sys_page = analytics_df_display[analytics_df_display['referral_status'].str.lower().isin(['pending', 'completed', 'initiated', 'missed appointment', 'declined'])].copy() # Filter for specific known statuses
                if not ref_data_sys_page.empty:
                    ref_status_counts_sys_page = ref_data_sys_page['referral_status'].value_counts().reset_index(); ref_status_counts_sys_page.columns = ['Referral Status', 'Count']
                    st.plotly_chart(plot_donut_chart(ref_status_counts_sys_page, 'Referral Status', 'Count', "Distribution of Referral Statuses", height=400, values_are_counts=True), use_container_width=True)
                else: st.caption("No active referral data for status distribution after filtering for known statuses.")
            else: st.caption("Referral status data unavailable.")
        
        st.markdown("---")
        st.subheader("AI Risk Score Distribution by Zone Socio-Economic Index")
        if 'zone_id' in analytics_df_display.columns and 'ai_risk_score' in analytics_df_display.columns and \
           not zone_attr_df_pop.empty and 'socio_economic_index' in zone_attr_df_pop.columns and \
           'name' in zone_attr_df_pop.columns:
            
            avg_risk_zone_equity_page = analytics_df_display.groupby('zone_id')['ai_risk_score'].mean().reset_index()
            zone_attr_df_pop_copy_equity = zone_attr_df_pop.copy()
            zone_attr_df_pop_copy_equity['zone_id'] = zone_attr_df_pop_copy_equity['zone_id'].astype(str)
            avg_risk_zone_equity_page['zone_id'] = avg_risk_zone_equity_page['zone_id'].astype(str)
            
            equity_df_plot_page = zone_attr_df_pop_copy_equity.merge(avg_risk_zone_equity_page, on='zone_id', how='inner')
            equity_df_plot_page.dropna(subset=['ai_risk_score', 'socio_economic_index'], inplace=True)

            if not equity_df_plot_page.empty:
                fig_equity_risk_ses_page = px.scatter(equity_df_plot_page, x='socio_economic_index', y='ai_risk_score', 
                                       text='name', size='population' if 'population' in equity_df_plot_page.columns else None, 
                                       color='ai_risk_score',
                                       title="Avg. Patient AI Risk vs. Zone Socio-Economic Index",
                                       labels={'socio_economic_index': "Socio-Economic Index (Zone)", 'ai_risk_score': "Avg. Patient AI Risk Score in Zone (Period)"},
                                       height=450, color_continuous_scale="Reds", hover_name='name')
                fig_equity_risk_ses_page.update_traces(textposition='top center')
                st.plotly_chart(fig_equity_risk_ses_page, use_container_width=True)
            else: st.caption("Not enough data to plot AI Risk vs Zone SES.")
        else: st.caption("Required data for AI Risk vs. Zone SES analysis not fully available.")

# --- Footer ---
st.markdown("---"); st.caption(app_config.APP_FOOTER)
