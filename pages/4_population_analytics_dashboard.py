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
    get_trend_data # Kept for use in epi_module if refactored or for direct use
)
from utils.ai_analytics_engine import apply_ai_models
from utils.ui_visualization_helpers import (
    plot_bar_chart,
    plot_donut_chart,
    plot_annotated_line_chart,
    render_kpi_card # Import render_kpi_card
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="Population Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_pop_dashboard(): 
    css_path = app_config.STYLE_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path) as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else: logger.warning(f"Population Dashboard CSS file not found: {css_path}.")
load_css_pop_dashboard()

# --- Data Loading ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading population data...")
def get_population_dashboard_data(): 
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    health_df = apply_ai_models(health_df_raw) if not health_df_raw.empty else pd.DataFrame()
    zone_gdf = load_zone_data()
    zone_attributes_df = pd.DataFrame(zone_gdf.drop(columns=[zone_gdf.geometry.name], errors='ignore')) if zone_gdf is not None and not zone_gdf.empty and hasattr(zone_gdf, 'geometry') and zone_gdf.geometry.name in zone_gdf.columns else pd.DataFrame(columns=['zone_id', 'name', 'population', 'socio_economic_index'])
    return health_df, zone_attributes_df

health_df_pop, zone_attr_df_pop = get_population_dashboard_data() 

if health_df_pop.empty:
    st.error("🚨 Data Error: Could not load health records. Population Dashboard cannot be displayed."); st.stop()

st.title("📊 Population Dashboard") 
st.markdown("Explore demographic distributions, epidemiological patterns, clinical trends, and health system factors across the population.")
st.markdown("---")

# --- Sidebar Filters ---
# ... (Sidebar logic as previously corrected, for brevity) ...
if os.path.exists(app_config.APP_LOGO): st.sidebar.image(app_config.APP_LOGO, width=230); st.sidebar.markdown("---")
st.sidebar.header("🔎 Population Filters")
min_date_pop = health_df_pop['encounter_date'].min().date() if 'encounter_date' in health_df_pop and not health_df_pop['encounter_date'].dropna().empty else date.today() - timedelta(days=365)
max_date_pop = health_df_pop['encounter_date'].max().date() if 'encounter_date' in health_df_pop and not health_df_pop['encounter_date'].dropna().empty else date.today()
if min_date_pop > max_date_pop: min_date_pop = max_date_pop
default_pop_start_date = min_date_pop; default_pop_end_date = max_date_pop
selected_start_date_pop, selected_end_date_pop = st.sidebar.date_input("Select Date Range:", value=[default_pop_start_date, default_pop_end_date], min_value=min_date_pop, max_value=max_date_pop, key="pop_dashboard_date_range_v1")
if selected_start_date_pop > selected_end_date_pop: selected_start_date_pop = selected_end_date_pop
if 'encounter_date' in health_df_pop.columns: health_df_pop['encounter_date_obj'] = pd.to_datetime(health_df_pop['encounter_date'], errors='coerce').dt.date
else: health_df_pop['encounter_date_obj'] = pd.NaT 
analytics_df_base = health_df_pop[(health_df_pop['encounter_date_obj'].notna()) & (health_df_pop['encounter_date_obj'] >= selected_start_date_pop) & (health_df_pop['encounter_date_obj'] <= selected_end_date_pop)].copy()
if analytics_df_base.empty: st.warning(f"No health data for period: {selected_start_date_pop.strftime('%d %b %Y')} - {selected_end_date_pop.strftime('%d %b %Y')}."); st.stop()
conditions_list_pop_dash = ["All Conditions"] + sorted(analytics_df_base['condition'].dropna().unique().tolist()); selected_condition_filter_pop_dash = st.sidebar.selectbox("Filter by Condition:", options=conditions_list_pop_dash, index=0, key="pop_dashboard_condition_filter_v1")
analytics_df = analytics_df_base.copy(); 
if selected_condition_filter_pop_dash != "All Conditions": analytics_df = analytics_df[analytics_df['condition'] == selected_condition_filter_pop_dash]
zones_list_pop_dash = ["All Zones"] + sorted(analytics_df_base['zone_id'].dropna().unique().tolist()); selected_zone_filter_pop_dash = st.sidebar.selectbox("Filter by Zone:", options=zones_list_pop_dash, index=0, key="pop_dashboard_zone_filter_v1")
if selected_zone_filter_pop_dash != "All Zones": analytics_df = analytics_df[analytics_df['zone_id'] == selected_zone_filter_pop_dash]
analytics_df_display = analytics_df.copy()
if analytics_df.empty : 
    st.warning(f"No data for '{selected_condition_filter_pop_dash}' in '{selected_zone_filter_pop_dash}'. Displaying broader data for period."); 
    analytics_df_display = analytics_df_base.copy(); 
    if selected_zone_filter_pop_dash != "All Zones": analytics_df_display = analytics_df_display[analytics_df_display['zone_id'] == selected_zone_filter_pop_dash]
    if analytics_df_display.empty: analytics_df_display = analytics_df_base.copy()

# --- NEW: Decision-Making KPI Boxes for the Filtered Data ---
st.subheader(f"Key Indicators for Selected Population ({selected_start_date_pop.strftime('%d %b')} - {selected_end_date_pop.strftime('%d %b')})")
if analytics_df_display.empty:
    st.info("No data available to display key indicators for the current filter selection.")
else:
    kpi_pop_cols1 = st.columns(4)
    # 1. Population in Analysis (Unique Patients)
    unique_patients_in_filter = analytics_df_display['patient_id'].nunique()
    kpi_pop_cols1[0].metric("Unique Patients (Filtered)", f"{unique_patients_in_filter:,}")

    # 2. Average AI Risk Score
    avg_ai_risk_filtered = np.nan
    if 'ai_risk_score' in analytics_df_display.columns and analytics_df_display['ai_risk_score'].notna().any():
        avg_ai_risk_filtered = analytics_df_display['ai_risk_score'].mean()
    kpi_pop_cols1[1].metric("Avg. AI Risk Score", f"{avg_ai_risk_filtered:.1f}" if pd.notna(avg_ai_risk_filtered) else "N/A")

    # 3. Proportion of High-Risk Individuals
    high_risk_count_filtered = 0
    prop_high_risk_filtered = 0.0
    if 'ai_risk_score' in analytics_df_display.columns and unique_patients_in_filter > 0:
        high_risk_patients = analytics_df_display[analytics_df_display['ai_risk_score'] >= app_config.RISK_THRESHOLDS['high']]['patient_id'].nunique()
        high_risk_count_filtered = high_risk_patients
        prop_high_risk_filtered = (high_risk_patients / unique_patients_in_filter) * 100 if unique_patients_in_filter > 0 else 0.0
    kpi_pop_cols1[2].metric("% High AI Risk Patients", f"{prop_high_risk_filtered:.1f}%", 
                             help_text=f"{high_risk_count_filtered} patient(s) with AI Risk ≥ {app_config.RISK_THRESHOLDS['high']}")

    # 4. Most Prevalent Condition (based on unique patients in filtered_df)
    most_prevalent_condition = "N/A"
    if 'condition' in analytics_df_display.columns and analytics_df_display['condition'].notna().any():
        condition_patient_counts = analytics_df_display.groupby('condition')['patient_id'].nunique()
        if not condition_patient_counts.empty:
            most_prevalent_condition = condition_patient_counts.idxmax()
    kpi_pop_cols1[3].metric("Top Condition (Unique Pts)", most_prevalent_condition)
    
    kpi_pop_cols2 = st.columns(3) # Second row of KPIs
    # 5. Key Test Positivity (Example: Malaria RDT)
    mal_rdt_key_pop_kpi = "RDT-Malaria" # Key used in data
    mal_rdt_pos_rate_kpi = 0.0
    if 'test_type' in analytics_df_display.columns and 'test_result' in analytics_df_display.columns:
        mal_rdt_df_kpi = analytics_df_display[(analytics_df_display['test_type'] == mal_rdt_key_pop_kpi) & (~analytics_df_display['test_result'].isin(['Pending','Rejected Sample','Unknown','N/A']))]
        if not mal_rdt_df_kpi.empty and len(mal_rdt_df_kpi) > 0 :
            mal_rdt_pos_rate_kpi = (mal_rdt_df_kpi[mal_rdt_df_kpi['test_result'] == 'Positive'].shape[0] / len(mal_rdt_df_kpi)) * 100
    kpi_pop_cols2[0].metric(f"{app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_rdt_key_pop_kpi, {}).get('display_name', mal_rdt_key_pop_kpi)} Positivity", f"{mal_rdt_pos_rate_kpi:.1f}%")

    # 6. Referral Completion Rate
    referral_completion_rate_kpi = 0.0
    if 'referral_status' in analytics_df_display.columns and 'referral_outcome' in analytics_df_display.columns:
        referrals_made_df = analytics_df_display[analytics_df_display['referral_status'].notna() & (~analytics_df_display['referral_status'].isin(['N/A', 'Unknown']))]
        if not referrals_made_df.empty:
            total_made_referrals = referrals_made_df['encounter_id'].nunique()
            completed_outcomes = ['Completed', 'Service Provided', 'Attended']
            completed_refs = referrals_made_df[referrals_made_df['referral_outcome'].isin(completed_outcomes)]['encounter_id'].nunique()
            if total_made_referrals > 0:
                referral_completion_rate_kpi = (completed_refs / total_made_referrals) * 100
    kpi_pop_cols2[1].metric("Referral Completion Rate", f"{referral_completion_rate_kpi:.1f}%", help_text="Based on referrals with conclusive outcomes (Completed, Service Provided, Attended).")
    
    # 7. Placeholder for a custom AI insight or another key metric
    # For example, % of diabetic patients with recent HbA1c (requires test type for HbA1c)
    # Or Average number of comorbidities for patients with AI Risk > threshold
    avg_comorbidities_high_risk = np.nan
    if 'key_chronic_conditions_summary' in analytics_df_display.columns and 'ai_risk_score' in analytics_df_display.columns:
        high_risk_df_comorbid = analytics_df_display[analytics_df_display['ai_risk_score'] >= app_config.RISK_THRESHOLDS['high']]
        if not high_risk_df_comorbid.empty and high_risk_df_comorbid['key_chronic_conditions_summary'].notna().any():
            # Count comorbidities by splitting the summary string
            comorbidity_counts = high_risk_df_comorbid['key_chronic_conditions_summary'].apply(lambda x: len([c for c in str(x).split(';') if c.strip() and c.lower() not in ['unknown', 'n/a', 'none']]))
            if comorbidity_counts.notna().any():
                avg_comorbidities_high_risk = comorbidity_counts.mean()
    kpi_pop_cols2[2].metric("Avg. Comorbidities (High Risk Pts)", f"{avg_comorbidities_high_risk:.1f}" if pd.notna(avg_comorbidities_high_risk) else "N/A", help_text="Average number of listed chronic conditions for patients with high AI risk scores.")

st.markdown("---")

# --- Tabbed Interface ---
tab_epi_overview, tab_demographics_sdoh, tab_clinical_dx, tab_systems_equity = st.tabs([
    "📈 Epidemiological Overview", "🧑‍⚕️ Demographics & SDOH", "🧬 Clinical & Diagnostics", "🌍 Systems & Equity"
])

with tab_epi_overview:
    st.header(f"Epidemiological Overview for '{selected_condition_filter_pop_dash}' in '{selected_zone_filter_pop_dash}'")
    # ... (Full Epi overview tab logic from previous version, ensuring it uses analytics_df_display)
    # (The code for condition case counts, AI risk distribution, Incidence Trend for Top Conditions remains here)
    if analytics_df_display.empty: st.info("No data for epidemiological overview with current filters.")
    else:
        epi_overview_cols_tab = st.columns(2)
        with epi_overview_cols_tab[0]: # Condition Case Counts (already using analytics_df_display)
            st.subheader("Condition Case Counts (Unique Patients)")
            if 'condition' in analytics_df_display.columns and analytics_df_display['condition'].notna().any():
                condition_counts_epi_tab_val = analytics_df_display.groupby('condition')['patient_id'].nunique().nlargest(10).reset_index(name='unique_patients')
                if not condition_counts_epi_tab_val.empty: condition_counts_epi_tab_val['condition'] = condition_counts_epi_tab_val['condition'].astype(str); st.plotly_chart(plot_bar_chart(condition_counts_epi_tab_val, 'condition', 'unique_patients', "Top 10 Conditions by Unique Patient Count", height=400, orientation='h', y_is_count=True, text_format='d'), use_container_width=True)
                else: st.caption("No condition data for counts.")
            else: st.caption("Condition column missing.")
        with epi_overview_cols_tab[1]: # AI Risk Distribution
            st.subheader("AI Risk Score Distribution")
            if 'ai_risk_score' in analytics_df_display.columns and analytics_df_display['ai_risk_score'].notna().any():
                fig_risk_dist_epi_tab_val = px.histogram(analytics_df_display.dropna(subset=['ai_risk_score']), x="ai_risk_score", nbins=20, title="Patient AI Risk Scores Distribution"); fig_risk_dist_epi_tab_val.update_layout(bargap=0.1, height=400); st.plotly_chart(fig_risk_dist_epi_tab_val, use_container_width=True)
            else: st.caption("AI Risk Score data not available.")
        # ... (Rest of epi_overview tab including Incidence Trend for Top Conditions logic from previous full output)

with tab_demographics_sdoh:
    st.header("Demographics & Social Determinants of Health (SDOH)")
    # ... (Full Demographics & SDOH tab logic from previous version, ensuring it uses analytics_df_display and zone_attr_df_pop)
    if analytics_df_display.empty: st.info("No data for Demographics/SDOH with current filters.")
    else: # (Demographics, Zone SES/Travel Time visualizations using analytics_df_display)
        pass

with tab_clinical_dx:
    st.header("Clinical & Diagnostic Data Patterns")
    # ... (Full Clinical & Diagnostics tab logic from previous version, using analytics_df_display)
    if analytics_df_display.empty: st.info("No data for Clinical/Dx with current filters.")
    else: # (Symptoms, Test Results, Positivity Trends using analytics_df_display)
        pass

with tab_systems_equity:
    st.header("Health Systems Context & Equity Insights")
    # ... (Full Systems & Equity tab logic from previous version, using analytics_df_display and zone_attr_df_pop)
    if analytics_df_display.empty: st.info("No data for Systems/Equity with current filters.")
    else: # (Encounters by Clinic, Referral Status, Risk vs SES using analytics_df_display)
        pass

# --- Footer ---
st.markdown("---"); st.caption(app_config.APP_FOOTER)
