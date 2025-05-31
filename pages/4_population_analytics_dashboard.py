# test/pages/4_population_analytics_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from datetime import date, timedelta

from config import app_config
from utils.core_data_processing import (
    load_health_records,
    load_zone_data, # To get zone_attributes for SDOH context
    get_trend_data
)
from utils.ai_analytics_engine import apply_ai_models # For AI risk score context
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
    # Apply AI models to have ai_risk_score for demographic stratification by risk
    health_df = apply_ai_models(health_df_raw) if not health_df_raw.empty else pd.DataFrame()
    
    zone_attributes_df = None
    # Attempt to load zone_attributes from load_zone_data, which returns a GeoDataFrame
    # We only need the non-geometry part for attributes.
    zone_gdf = load_zone_data()
    if zone_gdf is not None and not zone_gdf.empty:
        zone_attributes_df = pd.DataFrame(zone_gdf.drop(columns=[zone_gdf.geometry.name], errors='ignore'))
        logger.info(f"Loaded {len(zone_attributes_df)} zone attributes for SDOH context.")
    else:
        logger.warning("Zone attributes data not available. SDOH visualizations by zone will be limited.")
        zone_attributes_df = pd.DataFrame(columns=['zone_id', 'name', 'population', 'socio_economic_index']) # Schema for fallback

    if health_df.empty:
        logger.error("Population Analytics: Failed to load or process health records.")
    return health_df, zone_attributes_df

health_df_pop, zone_attr_df_pop = get_population_analytics_data()

# --- Main Page Rendering ---
if health_df_pop.empty:
    st.error("🚨 **Data Error:** Could not load health records. Population analytics cannot be displayed.")
    logger.critical("Population Analytics: health_df_pop is empty.")
    st.stop()

st.title("📊 Population Health & Systems Analytics")
st.markdown("""
    Explore demographic distributions, social determinants of health (SDOH) patterns, clinical trends, 
    and health system contextual factors across the population. Use the filters to narrow your analysis.
""")
st.markdown("---")

# --- Sidebar Filters ---
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, use_column_width='auto')
    st.sidebar.markdown("---")
st.sidebar.header("🔎 Analytics Filters")

# Date Range Filter (applies to most visualizations on this page)
min_date_pop = health_df_pop['encounter_date'].min().date() if 'encounter_date' in health_df_pop and not health_df_pop['encounter_date'].dropna().empty else date.today() - timedelta(days=365)
max_date_pop = health_df_pop['encounter_date'].max().date() if 'encounter_date' in health_df_pop and not health_df_pop['encounter_date'].dropna().empty else date.today()
if min_date_pop > max_date_pop: min_date_pop = max_date_pop

default_pop_start_date = min_date_pop # Default to full available range initially
default_pop_end_date = max_date_pop

selected_start_date_pop, selected_end_date_pop = st.sidebar.date_input(
    "Select Date Range for Analysis:",
    value=[default_pop_start_date, default_pop_end_date],
    min_value=min_date_pop, max_value=max_date_pop,
    key="pop_analytics_date_range_v1"
)
if selected_start_date_pop > selected_end_date_pop:
    st.sidebar.error("Error: Start date must be before end date.")
    selected_start_date_pop = selected_end_date_pop # Correct to single day if error

# Filter main health dataframe by selected date range
health_df_pop['encounter_date_obj'] = pd.to_datetime(health_df_pop['encounter_date']).dt.date # Ensure obj for comparison
filtered_health_df = health_df_pop[
    (health_df_pop['encounter_date_obj'] >= selected_start_date_pop) &
    (health_df_pop['encounter_date_obj'] <= selected_end_date_pop)
].copy()

if filtered_health_df.empty:
    st.warning(f"No health encounter data found for the selected period: {selected_start_date_pop.strftime('%d %b %Y')} to {selected_end_date_pop.strftime('%d %b %Y')}. Please adjust the date range.")
    st.stop()

# Optional Condition Filter
conditions_list = ["All Conditions"] + sorted(filtered_health_df['condition'].unique().tolist())
selected_condition_filter = st.sidebar.selectbox(
    "Filter by Condition (Optional):", options=conditions_list, index=0, key="pop_condition_filter_v1"
)
if selected_condition_filter != "All Conditions":
    filtered_health_df = filtered_health_df[filtered_health_df['condition'] == selected_condition_filter]
    if filtered_health_df.empty:
        st.warning(f"No data found for '{selected_condition_filter}' in the selected period. Displaying overall data or select 'All Conditions'.")
        # Fallback to all conditions if filter results in empty, or let it be empty.
        # For simplicity, we'll allow it to show "No data for X" messages in plots if truly empty.

# --- Tabbed Interface for Different Analytics Categories ---
tab_demographics, tab_clinical, tab_systems, tab_equity = st.tabs([
    "🧑‍⚕️ Demographics & SDOH", 
    "🧬 Clinical & Diagnostics", 
    "🌍 Health Systems Context", 
    "❤️ Equity Insights"
])

with tab_demographics:
    st.header("Demographics & Social Determinants of Health (SDOH)")
    if filtered_health_df.empty:
        st.info("No data available for demographic analysis with the current filters.")
    else:
        demo_cols = st.columns(2)
        with demo_cols[0]:
            st.subheader("Age Distribution")
            if 'age' in filtered_health_df.columns and filtered_health_df['age'].notna().any():
                age_bins = [0, 5, 12, 18, 35, 50, 65, np.inf]
                age_labels = ['0-4 yrs', '5-11 yrs', '12-17 yrs', '18-34 yrs', '35-49 yrs', '50-64 yrs', '65+ yrs']
                filtered_health_df.loc[:, 'age_group_pop'] = pd.cut(filtered_health_df['age'], bins=age_bins, labels=age_labels, right=False)
                age_dist_df = filtered_health_df['age_group_pop'].value_counts().sort_index().reset_index()
                age_dist_df.columns = ['Age Group', 'Number of Patients']
                st.plotly_chart(plot_bar_chart(age_dist_df, 'Age Group', 'Number of Patients', "Patient Encounters by Age Group", height=350), use_container_width=True)
            else:
                st.caption("Age data not available for distribution.")

        with demo_cols[1]:
            st.subheader("Gender Distribution")
            if 'gender' in filtered_health_df.columns:
                gender_dist_df = filtered_health_df['gender'].value_counts().reset_index()
                gender_dist_df.columns = ['Gender', 'Number of Patients']
                st.plotly_chart(plot_donut_chart(gender_dist_df, 'Gender', 'Number of Patients', "Patient Encounters by Gender", height=350), use_container_width=True)
            else:
                st.caption("Gender data not available for distribution.")
        
        st.markdown("---")
        st.subheader("Geographic & Socio-Economic Context (Zone Level)")
        # Merge filtered health data (unique patients per zone) with zone attributes
        if not zone_attr_df_pop.empty and 'zone_id' in filtered_health_df.columns:
            patient_counts_by_zone = filtered_health_df.groupby('zone_id')['patient_id'].nunique().reset_index(name='patients_in_period')
            zone_analysis_df = zone_attr_df_pop.merge(patient_counts_by_zone, on='zone_id', how='left')
            zone_analysis_df['patients_in_period'].fillna(0, inplace=True)
            
            if not zone_analysis_df.empty:
                sdoh_cols = st.columns(2)
                with sdoh_cols[0]:
                    if 'socio_economic_index' in zone_analysis_df.columns and zone_analysis_df['socio_economic_index'].notna().any():
                         # Avg SES of zones with patients in period
                        avg_ses_display = zone_analysis_df[zone_analysis_df['patients_in_period'] > 0]['socio_economic_index'].mean()
                        st.metric("Avg. Socio-Economic Index (Patient Zones)", f"{avg_ses_display:.2f}" if pd.notna(avg_ses_display) else "N/A", help="Average SES index of zones where patients had encounters in the period.")
                        st.plotly_chart(plot_bar_chart(zone_analysis_df.sort_values('socio_economic_index'), 'name', 'socio_economic_index', 'Socio-Economic Index by Zone', height=350, y_axis_title="SES Index"), use_container_width=True)
                    else: st.caption("Socio-Economic Index data by zone not available.")
                with sdoh_cols[1]:
                    if 'avg_travel_time_clinic_min' in zone_analysis_df.columns and zone_analysis_df['avg_travel_time_clinic_min'].notna().any():
                        avg_travel_display = zone_analysis_df[zone_analysis_df['patients_in_period'] > 0]['avg_travel_time_clinic_min'].mean()
                        st.metric("Avg. Travel Time to Clinic (Patient Zones)", f"{avg_travel_display:.0f} min" if pd.notna(avg_travel_display) else "N/A", help="Average travel time for zones where patients had encounters.")
                        st.plotly_chart(plot_bar_chart(zone_analysis_df.sort_values('avg_travel_time_clinic_min'), 'name', 'avg_travel_time_clinic_min', 'Avg. Travel Time to Clinic by Zone', height=350, y_axis_title="Minutes"), use_container_width=True)
                    else: st.caption("Avg. Travel Time data by zone not available.")
            else: st.info("No zone-level data to display after merging with patient encounters.")
        else:
            st.info("Zone attribute data not available or no zone information in health records for this period.")
        # Placeholders for other SDOH if data was available (education, occupation, income, language, disability)
        # st.markdown("###### Education Level (Placeholder - Requires Data)")
        # st.markdown("###### Occupation Risks (Placeholder - Requires Data)")

with tab_clinical:
    st.header("Clinical & Diagnostic Data Patterns")
    if filtered_health_df.empty:
        st.info("No data available for clinical/diagnostic analysis with the current filters.")
    else:
        clin_cols = st.columns(2)
        with clin_cols[0]:
            st.subheader("Top Presenting Symptoms")
            if 'patient_reported_symptoms' in filtered_health_df.columns and filtered_health_df['patient_reported_symptoms'].notna().any():
                # Symptoms are often semi-colon separated lists. We need to split and count.
                symptoms_exploded = filtered_health_df['patient_reported_symptoms'].str.split(';').explode()
                symptoms_cleaned = symptoms_exploded.str.strip().replace('', np.nan).dropna()
                symptom_counts = symptoms_cleaned.value_counts().nlargest(10).reset_index()
                symptom_counts.columns = ['Symptom', 'Frequency']
                if not symptom_counts.empty:
                    st.plotly_chart(plot_bar_chart(symptom_counts, 'Symptom', 'Frequency', "Top 10 Reported Symptoms (Encounters)", height=400, orientation='h'), use_container_width=True)
                else: st.caption("No distinct symptom data reported.")
            else: st.caption("Patient reported symptoms data not available.")

        with clin_cols[1]:
            st.subheader("Test Result Distribution (Top 5 Tests)")
            if 'test_type' in filtered_health_df.columns and 'test_result' in filtered_health_df.columns:
                top_5_tests = filtered_health_df['test_type'].value_counts().nlargest(5).index.tolist()
                top_tests_df = filtered_health_df[filtered_health_df['test_type'].isin(top_5_tests)]
                # Filter out non-conclusive results for this visualization
                test_results_dist = top_tests_df[~top_tests_df['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown', 'N/A', 'Indeterminate'])]
                if not test_results_dist.empty:
                    test_result_summary = test_results_dist.groupby(['test_type', 'test_result'])['patient_id'].nunique().reset_index(name='count')
                    st.plotly_chart(plot_bar_chart(test_result_summary, 'test_type', 'count', "Conclusive Test Results (Top 5 Tests by Volume)", color_col='test_result', barmode='group', height=400), use_container_width=True)
                else: st.caption("No conclusive test result data for top tests.")
            else: st.caption("Test type or result data not available.")

        st.markdown("---")
        st.subheader("Comorbidity Patterns (Example: Hypertension with Diabetes)")
        if 'condition' in filtered_health_df.columns and 'key_chronic_conditions_summary' in filtered_health_df.columns:
            # Patients with a specific primary condition from 'condition' column
            primary_condition = "Diabetes" # Example
            if selected_condition_filter != "All Conditions": primary_condition = selected_condition_filter # Use selected if not All

            df_primary_cond = filtered_health_df[filtered_health_df['condition'].str.contains(primary_condition, case=False, na=False)]
            
            if not df_primary_cond.empty:
                # Check key_chronic_conditions_summary for co-occurrence
                comorbidity_check = "Hypertension" # Example
                df_primary_cond_comorbid = df_primary_cond[df_primary_cond['key_chronic_conditions_summary'].str.contains(comorbidity_check, case=False, na=False)]
                
                total_primary = df_primary_cond['patient_id'].nunique()
                total_comorbid = df_primary_cond_comorbid['patient_id'].nunique()
                
                st.metric(f"Patients with {primary_condition}", total_primary)
                st.metric(f"Patients with {primary_condition} AND {comorbidity_check}", total_comorbid,
                          delta=f"{(total_comorbid/total_primary*100 if total_primary > 0 else 0):.1f}% of {primary_condition} cases",
                          delta_color="off" if total_comorbid == 0 else ("inverse" if total_comorbid/total_primary > 0.5 else "normal") ) # Example delta logic
            else:
                st.caption(f"No patients with primary condition '{primary_condition}' found in this period to analyze comorbidity.")
        else: st.caption("Data for condition or chronic conditions summary not available.")
        
        # Vaccination status trend (Placeholder - needs 'vaccination_status' and 'vaccine_name' columns)
        # Medication adherence distribution (using 'medication_adherence_self_report')
        if 'medication_adherence_self_report' in filtered_health_df.columns:
            st.markdown("---"); st.subheader("Medication Adherence (Self-Reported)")
            adherence_counts = filtered_health_df['medication_adherence_self_report'].value_counts().reset_index()
            adherence_counts.columns = ['Adherence Level', 'Patient Count']
            # Filter out "Unknown" if it dominates and isn't informative for distribution
            adherence_counts_filtered = adherence_counts[adherence_counts['Adherence Level'] != "Unknown"]
            if not adherence_counts_filtered.empty:
                st.plotly_chart(plot_donut_chart(adherence_counts_filtered, 'Adherence Level', 'Patient Count', "Self-Reported Medication Adherence", height=350), use_container_width=True)
            else: st.caption("No medication adherence data (excluding 'Unknown') reported.")


with tab_systems:
    st.header("Health Systems & Contextual Data Insights")
    if filtered_health_df.empty:
        st.info("No data available for health systems analysis with the current filters.")
    else:
        sys_cols = st.columns(2)
        with sys_cols[0]:
            st.subheader("Patient Encounters by Facility Type (Clinic ID)")
            if 'clinic_id' in filtered_health_df.columns:
                # Simple count by clinic_id for now. "Facility type" would need mapping clinic_id to type.
                facility_load = filtered_health_df['clinic_id'].value_counts().nlargest(10).reset_index()
                facility_load.columns = ['Clinic ID', 'Number of Encounters']
                st.plotly_chart(plot_bar_chart(facility_load, 'Clinic ID', 'Number of Encounters', "Top 10 Clinics by Encounter Volume", height=400, orientation='h'), use_container_width=True)
            else: st.caption("Clinic ID data not available.")

        with sys_cols[1]:
            st.subheader("Referral Pathway Performance (Example: Status Distribution)")
            if 'referral_status' in filtered_health_df.columns:
                referral_data = filtered_health_df[filtered_health_df['referral_status'] != "N/A"] # Exclude N/A for clarity
                if not referral_data.empty:
                    referral_status_counts = referral_data['referral_status'].value_counts().reset_index()
                    referral_status_counts.columns = ['Referral Status', 'Count']
                    st.plotly_chart(plot_donut_chart(referral_status_counts, 'Referral Status', 'Count', "Distribution of Referral Statuses", height=400), use_container_width=True)
                else: st.caption("No referral data (excluding 'N/A') for status distribution.")
            else: st.caption("Referral status data not available.")

        st.markdown("---")
        # Time to Care / Delays (e.g., Avg Test Turnaround Trend)
        st.subheader("Average Test Turnaround Time (TAT) Trend")
        if 'test_turnaround_days' in filtered_health_df.columns and 'encounter_date' in filtered_health_df.columns:
            # Use only conclusive tests for TAT trend
            conclusive_tests_for_tat = filtered_health_df[
                ~filtered_health_df['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown', 'N/A', 'Indeterminate']) &
                filtered_health_df['test_turnaround_days'].notna()
            ]
            if not conclusive_tests_for_tat.empty:
                avg_tat_trend = get_trend_data(conclusive_tests_for_tat, 'test_turnaround_days', date_col='encounter_date', period='W', agg_func='mean') # Weekly average
                if not avg_tat_trend.empty:
                    st.plotly_chart(plot_annotated_line_chart(avg_tat_trend, "Weekly Avg. Test TAT (All Conclusive Tests)", y_axis_title="Avg. TAT (Days)", target_line=app_config.TARGET_TEST_TURNAROUND_DAYS, height=400, date_format="%U, %Y (Week)"), use_container_width=True)
                else: st.caption("Not enough data for TAT trend.")
            else: st.caption("No conclusive tests with turnaround data for TAT trend.")
        else: st.caption("Test turnaround or encounter date data not available for TAT trend.")
        
        # Placeholders for other system data if available:
        # st.markdown("###### Diagnostic Test Availability Trends (Placeholder)")
        # st.markdown("###### Workforce Availability Indicators (Placeholder)")


with tab_equity:
    st.header("Equity & Ethics-Oriented Insights")
    if filtered_health_df.empty:
        st.info("No data available for equity analysis with the current filters.")
    else:
        st.info("Equity-oriented visualizations would typically compare outcomes or access across different SDOH groups (e.g., risk scores by socio-economic quintiles from zone data, access times by ethnicity if available and appropriate). This requires careful handling of sensitive data and clear ethical guidelines. Current visualizations are limited by available fields.")

        st.subheader("AI Risk Score Distribution by Zone (Socio-Economic Context)")
        if 'zone_id' in filtered_health_df.columns and 'ai_risk_score' in filtered_health_df.columns and not zone_attr_df_pop.empty and 'socio_economic_index' in zone_attr_df_pop.columns:
            # Calculate average AI risk per zone from health data for the period
            avg_risk_per_zone_period = filtered_health_df.groupby('zone_id')['ai_risk_score'].mean().reset_index()
            
            # Merge with zone attributes that include socio_economic_index
            equity_zone_df = zone_attr_df_pop.merge(avg_risk_per_zone_period, on='zone_id', how='left')
            equity_zone_df.dropna(subset=['ai_risk_score', 'socio_economic_index'], inplace=True)

            if not equity_zone_df.empty:
                # Simple scatter plot (example: would be better as box plots per SES quintile)
                fig_equity = px.scatter(equity_zone_df, x='socio_economic_index', y='ai_risk_score', 
                                       hover_name='name', size='population', color='ai_risk_score',
                                       title="Avg. Patient AI Risk vs. Zone Socio-Economic Index",
                                       labels={'socio_economic_index': "Socio-Economic Index (Zone)", 'ai_risk_score': "Avg. Patient AI Risk Score (Zone)"},
                                       height=450, color_continuous_scale="Reds")
                st.plotly_chart(fig_equity, use_container_width=True)
                st.caption("Each dot represents a zone. Size indicates population. Color intensity reflects AI risk. Ideally, further analysis would stratify by SES quintiles.")
            else:
                st.caption("Not enough data to plot AI risk vs. socio-economic index by zone for the selected period.")
        else:
            st.caption("Required data for AI Risk vs. Zone SES analysis not available (needs ai_risk_score, zone_id in health data, and socio_economic_index in zone attributes).")
            
        # Placeholder for other equity metrics
        # st.markdown("###### Access to Digital Tools (Placeholder - Requires 'digital_access_level' data)")
        # st.markdown("###### Insurance Coverage Impact (Placeholder - Requires 'insurance_status' data)")

# --- Footer ---
st.markdown("---")
st.caption(app_config.APP_FOOTER)
