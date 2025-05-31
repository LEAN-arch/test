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
    page_title="Population Dashboard - Health Hub", 
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_pop_dashboard(): 
    css_path = app_config.STYLE_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info("Population Dashboard: CSS loaded successfully.")
    else:
        logger.warning(f"Population Dashboard: CSS file not found at {css_path}.")
load_css_pop_dashboard()

# --- Data Loading ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading population data...")
def get_population_dashboard_data(): 
    logger.info("Population Dashboard: Loading health records and zone data...")
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
        logger.error("Population Dashboard: Failed to load or process health records.")
    return health_df, zone_attributes_df

health_df_pop, zone_attr_df_pop = get_population_dashboard_data() 

if health_df_pop.empty:
    st.error("🚨 **Data Error:** Could not load health records. Population Dashboard cannot be displayed.")
    logger.critical("Population Dashboard: health_df_pop is empty.")
    st.stop()

st.title("📊 Population Dashboard") 
st.markdown("""
    Explore demographic distributions, epidemiological patterns, clinical trends, 
    and health system factors across the population. Use the filters to narrow your analysis.
""")
st.markdown("---")

# --- Sidebar Filters ---
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, width=100) 
    st.sidebar.markdown("---")
else:
    logger.warning(f"Sidebar logo not found on Population Dashboard at {app_config.APP_LOGO}")

st.sidebar.header("🔎 Population Filters")
min_date_pop = health_df_pop['encounter_date'].min().date() if 'encounter_date' in health_df_pop and not health_df_pop['encounter_date'].dropna().empty else date.today() - timedelta(days=365)
max_date_pop = health_df_pop['encounter_date'].max().date() if 'encounter_date' in health_df_pop and not health_df_pop['encounter_date'].dropna().empty else date.today()
if min_date_pop > max_date_pop: min_date_pop = max_date_pop

default_pop_start_date = min_date_pop; default_pop_end_date = max_date_pop
selected_start_date_pop, selected_end_date_pop = st.sidebar.date_input(
    "Select Date Range for Analysis:", value=[default_pop_start_date, default_pop_end_date],
    min_value=min_date_pop, max_value=max_date_pop, key="pop_dashboard_date_range_v1"
)
if selected_start_date_pop > selected_end_date_pop: st.sidebar.error("Start date must be before end date."); selected_start_date_pop = selected_end_date_pop

# Prepare encounter_date_obj for filtering
if 'encounter_date' in health_df_pop.columns:
    health_df_pop['encounter_date_obj'] = pd.to_datetime(health_df_pop['encounter_date'], errors='coerce').dt.date
else: # Should not happen if load_health_records works
    health_df_pop['encounter_date_obj'] = pd.NaT 

analytics_df_base = health_df_pop[
    (health_df_pop['encounter_date_obj'].notna()) &
    (health_df_pop['encounter_date_obj'] >= selected_start_date_pop) &
    (health_df_pop['encounter_date_obj'] <= selected_end_date_pop)
].copy()

if analytics_df_base.empty: st.warning(f"No health encounter data found for period: {selected_start_date_pop.strftime('%d %b %Y')} - {selected_end_date_pop.strftime('%d %b %Y')}. Adjust filters."); st.stop()

conditions_list_pop_dash = ["All Conditions"] + sorted(analytics_df_base['condition'].dropna().unique().tolist())
selected_condition_filter_pop_dash = st.sidebar.selectbox("Filter by Condition (Optional):", options=conditions_list_pop_dash, index=0, key="pop_dashboard_condition_filter_v1") # Using _pop_dash for var names
analytics_df = analytics_df_base.copy()
if selected_condition_filter_pop_dash != "All Conditions": analytics_df = analytics_df[analytics_df['condition'] == selected_condition_filter_pop_dash]

zones_list_pop_dash = ["All Zones"] + sorted(analytics_df_base['zone_id'].dropna().unique().tolist())
selected_zone_filter_pop_dash = st.sidebar.selectbox("Filter by Zone (Optional):", options=zones_list_pop_dash, index=0, key="pop_dashboard_zone_filter_v1")
if selected_zone_filter_pop_dash != "All Zones": analytics_df = analytics_df[analytics_df['zone_id'] == selected_zone_filter_pop_dash]

analytics_df_display = analytics_df.copy() # This is the final DataFrame used by tabs
if analytics_df.empty : # If filters resulted in empty, try a broader scope for display
    st.warning(f"No data for '{selected_condition_filter_pop_dash}' in zone '{selected_zone_filter_pop_dash}' for the selected period. Displaying data for all conditions in selected zone/period or broader if needed.")
    analytics_df_display = analytics_df_base.copy() 
    if selected_zone_filter_pop_dash != "All Zones": analytics_df_display = analytics_df_display[analytics_df_display['zone_id'] == selected_zone_filter_pop_dash]
    # If still empty, analytics_df_display will be empty but from a known base, tabs will show "no data"

# --- Tabbed Interface ---
tab_epi_overview, tab_demographics_sdoh, tab_clinical_dx, tab_systems_equity = st.tabs([
    "📈 Epidemiological Overview", "🧑‍⚕️ Demographics & SDOH", "🧬 Clinical & Diagnostics", "🌍 Systems & Equity"
])

with tab_epi_overview:
    st.header(f"Epidemiological Overview for '{selected_condition_filter_pop_dash}' in '{selected_zone_filter_pop_dash}'")
    if analytics_df_display.empty:
        st.info("No data available for epidemiological overview with current filters.")
    else:
        epi_overview_cols = st.columns(2)
        with epi_overview_cols[0]:
            st.subheader("Condition Case Counts")
            if 'condition' in analytics_df_display.columns and analytics_df_display['condition'].notna().any():
                condition_counts_epi_tab = analytics_df_display.groupby('condition')['patient_id'].nunique().nlargest(10).reset_index(name='unique_patients')
                if not condition_counts_epi_tab.empty:
                    condition_counts_epi_tab['condition'] = condition_counts_epi_tab['condition'].astype(str)
                    st.plotly_chart(plot_bar_chart(condition_counts_epi_tab, x_col='condition', y_col='unique_patients', title="Top 10 Conditions by Unique Patient Count", height=400, orientation='h', y_is_count=True, text_format='d'), use_container_width=True)
                else: st.caption("No condition data to display counts after aggregation.")
            else: st.caption("Condition column missing or empty.")
        
        with epi_overview_cols[1]:
            st.subheader("AI Risk Score Distribution")
            if 'ai_risk_score' in analytics_df_display.columns and analytics_df_display['ai_risk_score'].notna().any():
                fig_risk_dist_epi_tab = px.histogram(analytics_df_display.dropna(subset=['ai_risk_score']), x="ai_risk_score", nbins=20, title="Distribution of Patient AI Risk Scores")
                fig_risk_dist_epi_tab.update_layout(bargap=0.1, height=400); st.plotly_chart(fig_risk_dist_epi_tab, use_container_width=True)
            else: st.caption("AI Risk Score data not available for distribution.")

        st.markdown("---")
        st.subheader(f"Incidence Trend for Top Conditions (Weekly New Cases)")
        if not analytics_df_display.empty and 'condition' in analytics_df_display.columns and 'patient_id' in analytics_df_display.columns and 'encounter_date' in analytics_df_display.columns:
            top_n_for_trend_plot_epi_tab = 3 
            df_for_top_cond_trend_epi_tab = analytics_df_display.copy()

            if selected_condition_filter_pop_dash != "All Conditions":
                top_conditions_for_trend_epi_tab = [selected_condition_filter_pop_dash]
                if selected_condition_filter_pop_dash not in df_for_top_cond_trend_epi_tab['condition'].unique(): top_conditions_for_trend_epi_tab = []
            else: top_conditions_for_trend_epi_tab = df_for_top_cond_trend_epi_tab['condition'].value_counts().nlargest(top_n_for_trend_plot_epi_tab).index.tolist()

            if top_conditions_for_trend_epi_tab:
                num_charts_to_plot_epi_tab = len(top_conditions_for_trend_epi_tab)
                inc_trend_cols_epi_tab = st.columns(num_charts_to_plot_epi_tab if num_charts_to_plot_epi_tab > 0 else 1)
                
                df_for_inc_trend_calc_epi_tab = df_for_top_cond_trend_epi_tab.copy()
                # Ensure 'encounter_date' is datetime64[ns] for Grouper
                if not pd.api.types.is_datetime64_ns_dtype(df_for_inc_trend_calc_epi_tab['encounter_date']): df_for_inc_trend_calc_epi_tab.loc[:,'encounter_date'] = pd.to_datetime(df_for_inc_trend_calc_epi_tab['encounter_date'], errors='coerce')
                df_for_inc_trend_calc_epi_tab.dropna(subset=['encounter_date'], inplace=True)
                
                if not df_for_inc_trend_calc_epi_tab.empty:
                    df_for_inc_trend_calc_epi_tab.sort_values('encounter_date', inplace=True)
                    df_for_inc_trend_calc_epi_tab['is_first_in_period'] = ~df_for_inc_trend_calc_epi_tab.duplicated(subset=['patient_id', 'condition'], keep='first')
                    new_cases_df_trend_epi_tab = df_for_inc_trend_calc_epi_tab[df_for_inc_trend_calc_epi_tab['is_first_in_period']]

                    for i, cond_name_epi_tab in enumerate(top_conditions_for_trend_epi_tab):
                        current_col_epi_tab = inc_trend_cols_epi_tab[i % num_charts_to_plot_epi_tab]
                        condition_trend_data_epi_tab = new_cases_df_trend_epi_tab[new_cases_df_trend_epi_tab['condition'] == cond_name_epi_tab]
                        with current_col_epi_tab:
                            if not condition_trend_data_epi_tab.empty:
                                weekly_new_cases_trend_val = get_trend_data(condition_trend_data_epi_tab, 'patient_id', date_col='encounter_date', period='W-Mon', agg_func='count')
                                if not weekly_new_cases_trend_val.empty: st.plotly_chart(plot_annotated_line_chart(weekly_new_cases_trend_val, f"Weekly New {cond_name_epi_tab} Cases", y_axis_title="New Cases", height=300, date_format="%U, %Y (Wk)", y_is_count=True), use_container_width=True)
                                else: st.caption(f"No trend data to plot for {cond_name_epi_tab}.")
                            else: st.caption(f"No new cases data for {cond_name_epi_tab} in this period.")
                else: st.caption("Not enough valid date data for incidence calculation.")
            else: 
                if selected_condition_filter_pop_dash != "All Conditions": st.caption(f"No data for '{selected_condition_filter_pop_dash}' for incidence trend.")
                else: st.caption("No top conditions for incidence trends with current filters.")
        else: st.caption("Required data missing for incidence trends.")

with tab_demographics_sdoh:
    st.header("Demographics & Social Determinants of Health (SDOH)")
    if analytics_df_display.empty: st.info("No data for Demographics/SDOH analysis with current filters.")
    else:
        demo_cols_sdoh_page_tab = st.columns(2)
        with demo_cols_sdoh_page_tab[0]:
            st.subheader("Age Distribution of Patients")
            if 'age' in analytics_df_display.columns and analytics_df_display['age'].notna().any():
                age_bins_sdoh_tab = [0, 5, 12, 18, 35, 50, 65, np.inf]; age_labels_sdoh_tab = ['0-4 yrs', '5-11 yrs', '12-17 yrs', '18-34 yrs', '35-49 yrs', '50-64 yrs', '65+ yrs']
                analytics_df_age_sdoh_tab = analytics_df_display.copy()
                analytics_df_age_sdoh_tab.loc[:, 'age_group_pop_sdoh_tab'] = pd.cut(analytics_df_age_sdoh_tab['age'], bins=age_bins_sdoh_tab, labels=age_labels_sdoh_tab, right=False)
                age_dist_df_sdoh_tab = analytics_df_age_sdoh_tab['age_group_pop_sdoh_tab'].value_counts().sort_index().reset_index(); age_dist_df_sdoh_tab.columns = ['Age Group', 'Patient Encounters']
                if not age_dist_df_sdoh_tab.empty : st.plotly_chart(plot_bar_chart(age_dist_df_sdoh_tab, 'Age Group', 'Patient Encounters', "Encounters by Age Group", height=350, y_is_count=True, text_format='d'), use_container_width=True)
                else: st.caption("No age data to plot.")
            else: st.caption("Age data not available.")
        with demo_cols_sdoh_page_tab[1]:
            st.subheader("Gender Distribution of Patients")
            if 'gender' in analytics_df_display.columns and analytics_df_display['gender'].notna().any():
                gender_dist_df_sdoh_tab = analytics_df_display['gender'].value_counts().reset_index(); gender_dist_df_sdoh_tab.columns = ['Gender', 'Patient Encounters']
                if not gender_dist_df_sdoh_tab.empty: st.plotly_chart(plot_donut_chart(gender_dist_df_sdoh_tab, 'Gender', 'Patient Encounters', "Encounters by Gender", height=350, values_are_counts=True), use_container_width=True)
                else: st.caption("No gender data to plot.")
            else: st.caption("Gender data not available.")
        
        st.markdown("---")
        st.subheader("Geographic & Socio-Economic Context (Zone Level Averages)")
        if not zone_attr_df_pop.empty and 'zone_id' in analytics_df_display.columns and analytics_df_display['zone_id'].notna().any():
            patients_per_zone_filtered_sdoh_tab = analytics_df_display.groupby('zone_id')['patient_id'].nunique().reset_index(name='patients_in_filtered_period')
            avg_risk_per_zone_filtered_sdoh_tab = analytics_df_display.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='avg_risk_in_filtered_period')
            
            zone_analysis_df_sdoh_tab = zone_attr_df_pop.copy()
            zone_analysis_df_sdoh_tab['zone_id'] = zone_analysis_df_sdoh_tab['zone_id'].astype(str)
            patients_per_zone_filtered_sdoh_tab['zone_id'] = patients_per_zone_filtered_sdoh_tab['zone_id'].astype(str)
            avg_risk_per_zone_filtered_sdoh_tab['zone_id'] = avg_risk_per_zone_filtered_sdoh_tab['zone_id'].astype(str)

            zone_analysis_df_sdoh_tab = zone_analysis_df_sdoh_tab.merge(patients_per_zone_filtered_sdoh_tab, on='zone_id', how='left')
            zone_analysis_df_sdoh_tab = zone_analysis_df_sdoh_tab.merge(avg_risk_per_zone_filtered_sdoh_tab, on='zone_id', how='left')
            zone_analysis_df_sdoh_tab['patients_in_filtered_period'].fillna(0, inplace=True)
            
            if not zone_analysis_df_sdoh_tab.empty:
                sdoh_context_cols_display_tab = st.columns(2)
                with sdoh_context_cols_display_tab[0]:
                    if 'socio_economic_index' in zone_analysis_df_sdoh_tab.columns and zone_analysis_df_sdoh_tab['socio_economic_index'].notna().any():
                         st.plotly_chart(plot_bar_chart(zone_analysis_df_sdoh_tab.sort_values('socio_economic_index'), 'name', 'socio_economic_index', 'Socio-Economic Index by Zone', height=350, y_axis_title="SES Index", text_format=".2f"), use_container_width=True)
                with sdoh_context_cols_display_tab[1]:
                    if 'avg_travel_time_clinic_min' in zone_analysis_df_sdoh_tab.columns and zone_analysis_df_sdoh_tab['avg_travel_time_clinic_min'].notna().any():
                        st.plotly_chart(plot_bar_chart(zone_analysis_df_sdoh_tab.sort_values('avg_travel_time_clinic_min'), 'name', 'avg_travel_time_clinic_min', 'Avg. Travel Time to Clinic by Zone', height=350, y_axis_title="Minutes", y_is_count=False, text_format=".0f"), use_container_width=True)
            else: st.info("No zone-level data to display after merging.")
        else: st.info("Zone attribute data or zone_id in health records unavailable for SDOH context.")

with tab_clinical_dx:
    st.header("Clinical & Diagnostic Data Patterns")
    if analytics_df_display.empty: st.info("No data for clinical/diagnostic analysis with current filters.")
    else:
        clin_dx_cols_page_tab = st.columns(2)
        with clin_dx_cols_page_tab[0]:
            st.subheader("Top Presenting Symptoms (from encounters)")
            if 'patient_reported_symptoms' in analytics_df_display.columns and analytics_df_display['patient_reported_symptoms'].notna().any():
                symptoms_exploded_dx_page_tab = analytics_df_display['patient_reported_symptoms'].str.split(';').explode().str.strip().replace(['', 'Unknown', 'N/A'],np.nan).dropna()
                if not symptoms_exploded_dx_page_tab.empty:
                    symptom_counts_dx_page_tab = symptoms_exploded_dx_page_tab.value_counts().nlargest(10).reset_index(); symptom_counts_dx_page_tab.columns = ['Symptom', 'Frequency']
                    st.plotly_chart(plot_bar_chart(symptom_counts_dx_page_tab, 'Symptom', 'Frequency', "Top 10 Reported Symptoms", height=400, orientation='h', y_is_count=True, text_format='d'), use_container_width=True)
                else: st.caption("No distinct symptoms reported.")
            else: st.caption("Patient reported symptoms data unavailable.")
        with clin_dx_cols_page_tab[1]:
            st.subheader("Test Result Distribution (Top 5 Tests by Volume)")
            if 'test_type' in analytics_df_display.columns and 'test_result' in analytics_df_display.columns:
                top_5_tests_dx_page_tab = analytics_df_display['test_type'].value_counts().nlargest(5).index.tolist()
                top_tests_df_dx_page_tab = analytics_df_display[analytics_df_display['test_type'].isin(top_5_tests_dx_page_tab)]
                test_results_dist_dx_page_tab = top_tests_df_dx_page_tab[~top_tests_df_dx_page_tab['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown', 'N/A', 'Indeterminate'])]
                if not test_results_dist_dx_page_tab.empty:
                    test_result_summary_dx_page_tab = test_results_dist_dx_page_tab.groupby(['test_type', 'test_result'])['patient_id'].nunique().reset_index(name='count')
                    st.plotly_chart(plot_bar_chart(test_result_summary_dx_page_tab, 'test_type', 'count', "Conclusive Test Results Distribution", color_col='test_result', barmode='group', height=400, y_is_count=True, text_format='d'), use_container_width=True)
                else: st.caption("No conclusive test result data for top tests.")
            else: st.caption("Test type or result data unavailable.")
        
        st.markdown("---"); st.subheader("Overall Test Positivity Rate Trend (e.g., Malaria RDT)")
        mal_rdt_key_pop_tab = "RDT-Malaria"; mal_rdt_disp_pop_tab = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_rdt_key_pop_tab,{}).get("display_name", mal_rdt_key_pop_tab)
        mal_df_pop_trend_tab_source = analytics_df_display[(analytics_df_display['test_type'] == mal_rdt_key_pop_tab) & (~analytics_df_display['test_result'].isin(['Pending','Rejected Sample','Unknown']))].copy()
        if not mal_df_pop_trend_tab_source.empty:
            mal_df_pop_trend_tab_source.loc[:,'is_positive'] = mal_df_pop_trend_tab_source['test_result'] == 'Positive'
            if not pd.api.types.is_datetime64_ns_dtype(mal_df_pop_trend_tab_source['encounter_date']): mal_df_pop_trend_tab_source.loc[:,'encounter_date'] = pd.to_datetime(mal_df_pop_trend_tab_source['encounter_date'], errors='coerce')
            mal_df_pop_trend_tab_source.dropna(subset=['encounter_date'], inplace=True)
            if not mal_df_pop_trend_tab_source.empty :
                weekly_pos_rate_pop_trend_val = get_trend_data(mal_df_pop_trend_tab_source, 'is_positive', date_col='encounter_date', period='W-Mon', agg_func='mean') * 100
                if not weekly_pos_rate_pop_trend_val.empty: st.plotly_chart(plot_annotated_line_chart(weekly_pos_rate_pop_trend_val, f"Weekly {mal_rdt_disp_pop_tab} Positivity Rate", y_axis_title="Positivity (%)", height=350, target_line=app_config.TARGET_MALARIA_POSITIVITY_RATE, y_is_count=False), use_container_width=True)
                else: st.caption(f"No data for {mal_rdt_disp_pop_tab} positivity trend.")
            else: st.caption(f"No valid date data for {mal_rdt_disp_pop_tab} positivity trend.")
        else: st.caption(f"No {mal_rdt_disp_pop_tab} test data for positivity trend.")

with tab_systems_equity:
    st.header("Health Systems Context & Equity Insights")
    if analytics_df_display.empty: st.info("No data for systems/equity analysis with current filters.")
    else:
        sys_equity_cols_page_tab = st.columns(2)
        with sys_equity_cols_page_tab[0]:
            st.subheader("Encounters by Clinic ID (Top 10)")
            if 'clinic_id' in analytics_df_display.columns and analytics_df_display['clinic_id'].notna().any():
                facility_load_sys_page_tab = analytics_df_display['clinic_id'].value_counts().nlargest(10).reset_index(); facility_load_sys_page_tab.columns = ['Clinic ID', 'Number of Encounters']
                if not facility_load_sys_page_tab.empty : st.plotly_chart(plot_bar_chart(facility_load_sys_page_tab, 'Clinic ID', 'Number of Encounters', "Top 10 Clinics by Encounter Volume", height=400, orientation='h', y_is_count=True, text_format='d'), use_container_width=True)
                else: st.caption("No encounter data by clinic ID.")
            else: st.caption("Clinic ID data not available.")
        with sys_equity_cols_page_tab[1]:
            st.subheader("Referral Status Distribution")
            if 'referral_status' in analytics_df_display.columns and analytics_df_display['referral_status'].notna().any():
                ref_data_sys_page_tab = analytics_df_display[analytics_df_display['referral_status'].str.lower().isin(['pending', 'completed', 'initiated', 'service provided', 'attended', 'missed appointment', 'declined'])].copy()
                if not ref_data_sys_page_tab.empty:
                    ref_status_counts_sys_page_tab = ref_data_sys_page_tab['referral_status'].value_counts().reset_index(); ref_status_counts_sys_page_tab.columns = ['Referral Status', 'Count']
                    if not ref_status_counts_sys_page_tab.empty : st.plotly_chart(plot_donut_chart(ref_status_counts_sys_page_tab, 'Referral Status', 'Count', "Distribution of Referral Statuses", height=400, values_are_counts=True), use_container_width=True)
                    else: st.caption("No referral status counts to plot.")
                else: st.caption("No actionable referral data for status distribution after filtering for known statuses.")
            else: st.caption("Referral status data unavailable.")
        
        st.markdown("---")
        st.subheader("AI Risk Score Distribution by Zone Socio-Economic Index")
        if 'zone_id' in analytics_df_display.columns and 'ai_risk_score' in analytics_df_display.columns and \
           not zone_attr_df_pop.empty and 'socio_economic_index' in zone_attr_df_pop.columns and \
           'name' in zone_attr_df_pop.columns: 
            
            avg_risk_zone_equity_page_tab = analytics_df_display.groupby('zone_id')['ai_risk_score'].mean().reset_index()
            zone_attr_df_pop_copy_equity_tab = zone_attr_df_pop.copy() # Use the full zone attributes
            zone_attr_df_pop_copy_equity_tab['zone_id'] = zone_attr_df_pop_copy_equity_tab['zone_id'].astype(str)
            avg_risk_zone_equity_page_tab['zone_id'] = avg_risk_zone_equity_page_tab['zone_id'].astype(str)
            
            equity_df_plot_page_tab = zone_attr_df_pop_copy_equity_tab.merge(avg_risk_zone_equity_page_tab, on='zone_id', how='left') # Left merge to keep all zones
            # We are plotting avg_risk vs SES for zones that *have patients in the filtered analytics_df_display*
            # So, we filter after merge or use inner merge if we only care about zones with data.
            # For now, let's use inner to only plot zones with risk data from the filtered set.
            equity_df_plot_page_tab = zone_attr_df_pop_copy_equity_tab.merge(avg_risk_zone_equity_page_tab, on='zone_id', how='inner')

            equity_df_plot_page_tab.dropna(subset=['ai_risk_score', 'socio_economic_index'], inplace=True)

            if not equity_df_plot_page_tab.empty:
                fig_equity_risk_ses_page_tab = px.scatter(equity_df_plot_page_tab, x='socio_economic_index', y='ai_risk_score', 
                                       text='name', size='population' if 'population' in equity_df_plot_page_tab.columns else None, 
                                       color='ai_risk_score',
                                       title="Avg. Patient AI Risk vs. Zone Socio-Economic Index",
                                       labels={'socio_economic_index': "Socio-Economic Index (Zone)", 'ai_risk_score': "Avg. Patient AI Risk Score in Zone (Period)"},
                                       height=450, color_continuous_scale="Reds", hover_name='name')
                fig_equity_risk_ses_page_tab.update_traces(textposition='top center')
                st.plotly_chart(fig_equity_risk_ses_page_tab, use_container_width=True)
            else: st.caption("Not enough data to plot AI Risk vs Zone SES after merging/filtering.")
        else: st.caption("Required data for AI Risk vs. Zone SES analysis not fully available (needs patient AI risk & zone_id, and zone attributes with SES, name, population).")

# --- Footer ---
st.markdown("---"); st.caption(app_config.APP_FOOTER)
