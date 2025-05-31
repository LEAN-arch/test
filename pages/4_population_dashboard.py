# test/pages/4_population_analytics_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys 
import logging
from datetime import date, timedelta
import plotly.express as px
import html 

# --- Explicitly add project root to sys.path for robust imports ---
PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_PATH)

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
    if zone_gdf is not None and not zone_gdf.empty and hasattr(zone_gdf, 'geometry') and hasattr(zone_gdf.geometry, 'name') and zone_gdf.geometry.name in zone_gdf.columns:
        zone_attributes_df = pd.DataFrame(zone_gdf.drop(columns=[zone_gdf.geometry.name], errors='ignore'))
    else:
        zone_attributes_df = pd.DataFrame(columns=['zone_id', 'name', 'population', 'socio_economic_index'])
    return health_df, zone_attributes_df

health_df_pop, zone_attr_df_pop = get_population_dashboard_data() 

if health_df_pop.empty:
    st.error("🚨 **Data Error:** Could not load health records. Population Dashboard cannot be displayed."); st.stop()

st.title("📊 Population Dashboard") 
st.markdown("Explore demographic distributions, epidemiological patterns, clinical trends, and health system factors across the population.")
st.markdown("---")

# --- Sidebar Filters ---
if os.path.exists(app_config.APP_LOGO): st.sidebar.image(app_config.APP_LOGO, width=100); st.sidebar.markdown("---")
else: logger.warning(f"Sidebar logo not found on Population Dashboard at {app_config.APP_LOGO}")
st.sidebar.header("🔎 Population Filters")

min_date_pop = date.today() - timedelta(days=365*2); max_date_pop = date.today()
if 'encounter_date' in health_df_pop.columns and health_df_pop['encounter_date'].notna().any():
    min_date_pop_series = health_df_pop['encounter_date'].dropna()
    if not min_date_pop_series.empty:
        min_date_pop = min_date_pop_series.min().date()
        max_date_pop = min_date_pop_series.max().date()
if min_date_pop > max_date_pop: min_date_pop = max_date_pop
default_pop_start_date = min_date_pop; default_pop_end_date = max_date_pop
selected_start_date_pop, selected_end_date_pop = st.sidebar.date_input("Select Date Range:", value=[default_pop_start_date, default_pop_end_date], min_value=min_date_pop, max_value=max_date_pop, key="pop_dashboard_date_range_v3")
if selected_start_date_pop > selected_end_date_pop: selected_start_date_pop = selected_end_date_pop

if 'encounter_date' in health_df_pop.columns:
    health_df_pop.loc[:, 'encounter_date_obj'] = pd.to_datetime(health_df_pop['encounter_date'], errors='coerce').dt.date
else: health_df_pop['encounter_date_obj'] = pd.NaT 
analytics_df_base = health_df_pop[ (health_df_pop['encounter_date_obj'].notna()) & (health_df_pop['encounter_date_obj'] >= selected_start_date_pop) & (health_df_pop['encounter_date_obj'] <= selected_end_date_pop) ].copy()

if analytics_df_base.empty: st.warning(f"No health data for period: {selected_start_date_pop.strftime('%d %b %Y')} - {selected_end_date_pop.strftime('%d %b %Y')}. Adjust filters."); st.stop()

conditions_list_pop_dash = ["All Conditions"] + sorted(analytics_df_base['condition'].dropna().unique().tolist()); selected_condition_filter_pop_dash = st.sidebar.selectbox("Filter by Condition (Optional):", options=conditions_list_pop_dash, index=0, key="pop_dashboard_condition_filter_v3")
analytics_df_after_cond = analytics_df_base.copy(); 
if selected_condition_filter_pop_dash != "All Conditions": analytics_df_after_cond = analytics_df_after_cond[analytics_df_after_cond['condition'] == selected_condition_filter_pop_dash]
zones_list_pop_dash = ["All Zones"] + sorted(analytics_df_base['zone_id'].dropna().unique().tolist()); selected_zone_filter_pop_dash = st.sidebar.selectbox("Filter by Zone (Optional):", options=zones_list_pop_dash, index=0, key="pop_dashboard_zone_filter_v3")
analytics_df_after_zone = analytics_df_after_cond.copy()
if selected_zone_filter_pop_dash != "All Zones": analytics_df_after_zone = analytics_df_after_zone[analytics_df_after_zone['zone_id'] == selected_zone_filter_pop_dash]

analytics_df_display = analytics_df_after_zone.copy()
if analytics_df_display.empty and (selected_condition_filter_pop_dash != "All Conditions" or selected_zone_filter_pop_dash != "All Zones"):
    warning_message = f"No data for '{selected_condition_filter_pop_dash}' in zone '{selected_zone_filter_pop_dash}' for the selected period. "
    analytics_df_display = analytics_df_base.copy()
    if selected_zone_filter_pop_dash != "All Zones" and 'zone_id' in analytics_df_display.columns:
        analytics_df_display = analytics_df_display[analytics_df_display['zone_id'] == selected_zone_filter_pop_dash]
        if analytics_df_display.empty: # If zone filter also made it empty with "All Conditions"
            analytics_df_display = analytics_df_base.copy() # Reset to only date-filtered
            warning_message += f"Showing data for all conditions in selected period for zone '{selected_zone_filter_pop_dash}' if available, or all zones."
        else: warning_message += f"Showing data for '{selected_zone_filter_pop_dash}' and all conditions."

    elif selected_condition_filter_pop_dash != "All Conditions" and 'condition' in analytics_df_base.columns: # Zone was "All Zones" but condition made it empty
        temp_df_cond_fallback = analytics_df_base[analytics_df_base['condition'] == selected_condition_filter_pop_dash]
        if not temp_df_cond_fallback.empty: analytics_df_display = temp_df_cond_fallback
        else: analytics_df_display = analytics_df_base.copy() # Fallback to just date if condition also not found
        warning_message += f"Showing data for '{selected_condition_filter_pop_dash}' across all zones if available, or overall data for period."
    else: # Both were "All" or some other combo, but analytics_df_display is still empty
        analytics_df_display = analytics_df_base.copy()
        warning_message += "Displaying overall data for the selected period."
    st.warning(warning_message)
    if analytics_df_display.empty: # Ultimate fallback if analytics_df_base was also somehow empty after initial check
        st.error("No data available for display after all fallbacks. Please check data source and date filters."); st.stop()

# --- Decision-Making KPI Boxes ---
st.subheader(f"Key Indicators ({selected_start_date_pop.strftime('%d %b')} - {selected_end_date_pop.strftime('%d %b')}, Cond: {selected_condition_filter_pop_dash}, Zone: {selected_zone_filter_pop_dash})")
if analytics_df_display.empty:
    st.info("No data to display key indicators for the current filter selection.")
else:
    kpi_pop_cols1 = st.columns(4)
    unique_patients_in_filter = analytics_df_display['patient_id'].nunique() if 'patient_id' in analytics_df_display.columns else 0
    kpi_pop_cols1[0].metric("Unique Patients (Filtered)", f"{unique_patients_in_filter:,}")

    avg_ai_risk_filtered = np.nan
    if 'ai_risk_score' in analytics_df_display.columns and analytics_df_display['ai_risk_score'].notna().any(): avg_ai_risk_filtered = analytics_df_display['ai_risk_score'].mean()
    kpi_pop_cols1[1].metric("Avg. AI Risk Score", f"{avg_ai_risk_filtered:.1f}" if pd.notna(avg_ai_risk_filtered) else "N/A")

    high_risk_count_filtered = 0; prop_high_risk_filtered = 0.0
    if 'ai_risk_score' in analytics_df_display.columns and analytics_df_display['ai_risk_score'].notna().any() and 'patient_id' in analytics_df_display.columns and unique_patients_in_filter > 0:
        high_risk_patients_df = analytics_df_display[pd.to_numeric(analytics_df_display['ai_risk_score'], errors='coerce') >= app_config.RISK_THRESHOLDS['high']]
        if not high_risk_patients_df.empty: high_risk_count_filtered = high_risk_patients_df['patient_id'].nunique()
        prop_high_risk_filtered = (high_risk_count_filtered / unique_patients_in_filter) * 100
    value_prop_high_risk = f"{prop_high_risk_filtered:.1f}%" if unique_patients_in_filter > 0 and pd.notna(prop_high_risk_filtered) else ("0.0%" if unique_patients_in_filter > 0 else "N/A")
    help_text_prop_high_risk = f"{int(high_risk_count_filtered)} unique patient(s) with AI Risk Score ≥ {app_config.RISK_THRESHOLDS['high']}"
    kpi_pop_cols1[2].metric(label="% High AI Risk Patients", value=value_prop_high_risk, help=help_text_prop_high_risk)

    with kpi_pop_cols1[3]:
        most_prevalent_condition_val = "N/A"; most_prevalent_condition_count = 0
        if 'condition' in analytics_df_display.columns and analytics_df_display['condition'].notna().any():
            # Count occurrences of condition for prevalence in encounters, not unique patients
            condition_encounter_counts = analytics_df_display['condition'].value_counts()
            if not condition_encounter_counts.empty: most_prevalent_condition_val = condition_encounter_counts.idxmax(); most_prevalent_condition_count = condition_encounter_counts.max()
        st.markdown(f"""<div class="custom-markdown-kpi-box"><div class="custom-kpi-label-top-condition">Top Condition (Encounters)</div><div class="custom-kpi-value-large">{html.escape(str(most_prevalent_condition_val))}</div><div class="custom-kpi-subtext-small">{html.escape(f"{most_prevalent_condition_count} encounters") if most_prevalent_condition_val != "N/A" else ""}</div></div>""", unsafe_allow_html=True)
    
    kpi_pop_cols2 = st.columns(3)
    mal_rdt_key_kpi = "RDT-Malaria"; mal_rdt_pos_rate_kpi_val = 0.0
    if 'test_type' in analytics_df_display.columns and 'test_result' in analytics_df_display.columns:
        mal_rdt_df_kpi_calc = analytics_df_display[(analytics_df_display['test_type'] == mal_rdt_key_kpi) & (~analytics_df_display['test_result'].isin(['Pending','Rejected Sample','Unknown','N/A']))]
        if not mal_rdt_df_kpi_calc.empty and len(mal_rdt_df_kpi_calc) > 0 : mal_rdt_pos_rate_kpi_val = (mal_rdt_df_kpi_calc[mal_rdt_df_kpi_calc['test_result'] == 'Positive'].shape[0] / len(mal_rdt_df_kpi_calc)) * 100
    kpi_pop_cols2[0].metric(f"{app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_rdt_key_kpi, {}).get('display_name', mal_rdt_key_kpi)} Positivity", f"{mal_rdt_pos_rate_kpi_val:.1f}%")

    referral_completion_rate_kpi_val = 0.0
    if 'referral_status' in analytics_df_display.columns and 'referral_outcome' in analytics_df_display.columns and 'encounter_id' in analytics_df_display.columns:
        referrals_made_df_kpi = analytics_df_display[analytics_df_display['referral_status'].notna() & (~analytics_df_display['referral_status'].isin(['N/A', 'Unknown']))]
        if not referrals_made_df_kpi.empty:
            total_made_referrals_kpi = referrals_made_df_kpi['encounter_id'].nunique()
            completed_outcomes_kpi = ['Completed', 'Service Provided', 'Attended', 'Attended Consult', 'Attended Followup'] # Expanded
            completed_refs_kpi = referrals_made_df_kpi[referrals_made_df_kpi['referral_outcome'].isin(completed_outcomes_kpi)]['encounter_id'].nunique()
            if total_made_referrals_kpi > 0: referral_completion_rate_kpi_val = (completed_refs_kpi / total_made_referrals_kpi) * 100
    kpi_pop_cols2[1].metric("Referral Completion Rate", f"{referral_completion_rate_kpi_val:.1f}%", help="Based on referrals with conclusive positive outcomes.")
    
    avg_comorbidities_high_risk_val = np.nan
    if 'key_chronic_conditions_summary' in analytics_df_display.columns and 'ai_risk_score' in analytics_df_display.columns and analytics_df_display['ai_risk_score'].notna().any():
        high_risk_df_comorbid_kpi = analytics_df_display[pd.to_numeric(analytics_df_display['ai_risk_score'], errors='coerce') >= app_config.RISK_THRESHOLDS['high']]
        if not high_risk_df_comorbid_kpi.empty and high_risk_df_comorbid_kpi['key_chronic_conditions_summary'].notna().any():
            comorbidity_counts_kpi = high_risk_df_comorbid_kpi['key_chronic_conditions_summary'].apply(lambda x: len([c for c in str(x).split(';') if c.strip() and c.lower() not in ['unknown', 'n/a', 'none']]))
            if comorbidity_counts_kpi.notna().any(): avg_comorbidities_high_risk_val = comorbidity_counts_kpi.mean()
    kpi_pop_cols2[2].metric("Avg. Comorbidities (High Risk Pts)", f"{avg_comorbidities_high_risk_val:.1f}" if pd.notna(avg_comorbidities_high_risk_val) else "N/A")
st.markdown("---")

# --- Tabbed Interface ---
tab_epi_overview, tab_demographics_sdoh, tab_clinical_dx, tab_systems_equity = st.tabs([
    "📈 Epidemiological Overview", "🧑‍⚕️ Demographics & SDOH", "🧬 Clinical & Diagnostics", "🌍 Systems & Equity"
])

with tab_epi_overview:
    st.header(f"Epidemiological Overview for '{selected_condition_filter_pop_dash}' in '{selected_zone_filter_pop_dash}'")
    if analytics_df_display.empty: st.info("No data for epidemiological overview with current filters.")
    else:
        epi_overview_cols_tab = st.columns(2)
        with epi_overview_cols_tab[0]:
            st.subheader("Condition Case Counts (Unique Patients)")
            if 'condition' in analytics_df_display.columns and analytics_df_display['condition'].notna().any():
                condition_counts_data = analytics_df_display.groupby('condition')['patient_id'].nunique().nlargest(10).reset_index(name='unique_patients')
                if not condition_counts_data.empty: 
                    condition_counts_data.loc[:, 'condition']=condition_counts_data['condition'].astype(str)
                    st.plotly_chart(plot_bar_chart(condition_counts_data,'condition','unique_patients',"Top Conditions by Unique Patients",height=400,orientation='h',y_is_count=True,text_format='d'),use_container_width=True)
                else: st.caption("No condition data for counts.")
            else: st.caption("Condition column missing.")
        with epi_overview_cols_tab[1]:
            st.subheader("AI Risk Score Distribution")
            if 'ai_risk_score' in analytics_df_display.columns and analytics_df_display['ai_risk_score'].notna().any(): 
                fig_risk=px.histogram(analytics_df_display.dropna(subset=['ai_risk_score']),x="ai_risk_score",nbins=20,title="Patient AI Risk Scores")
                fig_risk.update_layout(bargap=0.1,height=400); st.plotly_chart(fig_risk,use_container_width=True)
            else: st.caption("AI Risk Score data unavailable.")
        
        st.markdown("---"); st.subheader(f"Incidence Trend for Top Conditions (Weekly New Cases)")
        if not analytics_df_display.empty and all(col in analytics_df_display.columns for col in ['condition', 'patient_id', 'encounter_date']):
            top_n_for_trend_plot_epi_tab = 3 
            source_df_for_trends_epi_tab = analytics_df_display.copy()
            if selected_condition_filter_pop_dash != "All Conditions":
                top_conditions_for_trend_epi_val = [selected_condition_filter_pop_dash] if selected_condition_filter_pop_dash in source_df_for_trends_epi_tab['condition'].unique() else []
            else: top_conditions_for_trend_epi_val = source_df_for_trends_epi_tab['condition'].value_counts().nlargest(top_n_for_trend_plot_epi_tab).index.tolist()
            if top_conditions_for_trend_epi_val:
                num_charts_epi_val = len(top_conditions_for_trend_epi_val)
                inc_trend_cols_plot_val = st.columns(num_charts_epi_val if num_charts_epi_val > 0 else 1)
                df_for_inc_calc_plot_val = source_df_for_trends_epi_tab.copy()
                if not pd.api.types.is_datetime64_ns_dtype(df_for_inc_calc_plot_val['encounter_date']): df_for_inc_calc_plot_val.loc[:,'encounter_date'] = pd.to_datetime(df_for_inc_calc_plot_val['encounter_date'], errors='coerce')
                df_for_inc_calc_plot_val.dropna(subset=['encounter_date'], inplace=True)
                if not df_for_inc_calc_plot_val.empty :
                    df_for_inc_calc_plot_val.sort_values('encounter_date', inplace=True); df_for_inc_calc_plot_val.loc[:,'is_first_in_period'] = ~df_for_inc_calc_plot_val.duplicated(subset=['patient_id', 'condition'], keep='first'); new_cases_df_plot_val = df_for_inc_calc_plot_val[df_for_inc_calc_plot_val['is_first_in_period']]
                    for i, cond_name_val_item in enumerate(top_conditions_for_trend_epi_val):
                        current_col_for_plot_item = inc_trend_cols_plot_val[i % num_charts_epi_val if num_charts_epi_val > 0 else 0]
                        condition_trend_data_item = new_cases_df_plot_val[new_cases_df_plot_val['condition'] == cond_name_val_item]
                        with current_col_for_plot_item:
                            if not condition_trend_data_item.empty:
                                weekly_new_cases_item_val = get_trend_data(condition_trend_data_item, 'patient_id', date_col='encounter_date', period='W-Mon', agg_func='count')
                                if not weekly_new_cases_item_val.empty: st.plotly_chart(plot_annotated_line_chart(weekly_new_cases_item_val, f"Weekly New {cond_name_val_item} Cases", y_axis_title="New Cases", height=300, date_format="%U, %Y (Wk)", y_is_count=True), use_container_width=True)
                                else: st.caption(f"No trend data for {cond_name_val_item}.")
                            else: st.caption(f"No new cases for {cond_name_val_item}.")
                else: st.caption("Not enough valid date data for incidence.")
            else: st.caption(f"No data for '{selected_condition_filter_pop_dash if selected_condition_filter_pop_dash != 'All Conditions' else 'top conditions'}' for incidence trend.")
        else: st.caption("Required data missing for incidence trends.")

with tab_demographics_sdoh:
    st.header("Demographics & Social Determinants of Health (SDOH)")
    if analytics_df_display.empty: st.info("No data for Demographics/SDOH with current filters.")
    else:
        demo_cols_sdoh_page_tab_val = st.columns(2)
        with demo_cols_sdoh_page_tab_val[0]: # Age Distribution
            st.subheader("Age Distribution of Patients")
            if 'age' in analytics_df_display.columns and analytics_df_display['age'].notna().any():
                age_bins_val_sdoh = [0,5,12,18,35,50,65,np.inf]; age_labels_val_sdoh = ['0-4','5-11','12-17','18-34','35-49','50-64','65+']
                age_df_val_sdoh = analytics_df_display.copy(); age_df_val_sdoh.loc[:, 'age_group_pop_sdoh_val'] = pd.cut(age_df_val_sdoh['age'], bins=age_bins_val_sdoh, labels=age_labels_val_sdoh, right=False)
                age_dist_val_sdoh = age_df_val_sdoh['age_group_pop_sdoh_val'].value_counts().sort_index().reset_index(); age_dist_val_sdoh.columns=['Age Group','Patient Encounters']
                if not age_dist_val_sdoh.empty : st.plotly_chart(plot_bar_chart(age_dist_val_sdoh, 'Age Group', 'Patient Encounters', "Encounters by Age Group", height=350, y_is_count=True, text_format='d'), use_container_width=True)
                else: st.caption("No age data to plot.")
        with demo_cols_sdoh_page_tab_val[1]: # Gender Distribution
            st.subheader("Gender Distribution of Patients")
            if 'gender' in analytics_df_display.columns and analytics_df_display['gender'].notna().any():
                gender_dist_val_sdoh = analytics_df_display['gender'].value_counts().reset_index(); gender_dist_val_sdoh.columns = ['Gender','Patient Encounters']
                if not gender_dist_val_sdoh.empty: st.plotly_chart(plot_donut_chart(gender_dist_val_sdoh, 'Gender', 'Patient Encounters', "Encounters by Gender", height=350, values_are_counts=True), use_container_width=True)
                else: st.caption("No gender data to plot.")
        st.markdown("---"); st.subheader("Geographic & Socio-Economic Context (Zone Level Averages)")
        if not zone_attr_df_pop.empty and 'zone_id' in analytics_df_display.columns and analytics_df_display['zone_id'].notna().any():
            patients_zone_sdoh_tab = analytics_df_display.groupby('zone_id')['patient_id'].nunique().reset_index(name='patients_filtered')
            risk_zone_sdoh_tab = analytics_df_display.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='avg_risk_filtered')
            zone_analysis_sdoh_tab_val = zone_attr_df_pop.copy(); zone_analysis_sdoh_tab_val.loc[:,'zone_id']=zone_analysis_sdoh_tab_val['zone_id'].astype(str)
            for df_m in [patients_zone_sdoh_tab, risk_zone_sdoh_tab]: df_m.loc[:,'zone_id'] = df_m['zone_id'].astype(str); zone_analysis_sdoh_tab_val=zone_analysis_sdoh_tab_val.merge(df_m,on='zone_id',how='left')
            zone_analysis_sdoh_tab_val['patients_filtered'].fillna(0,inplace=True)
            if not zone_analysis_sdoh_tab_val.empty:
                sdoh_cols_viz_tab = st.columns(2)
                with sdoh_cols_viz_tab[0]: 
                    if 'socio_economic_index' in zone_analysis_sdoh_tab_val and zone_analysis_sdoh_tab_val['socio_economic_index'].notna().any(): st.plotly_chart(plot_bar_chart(zone_analysis_sdoh_tab_val.sort_values('socio_economic_index'),'name','socio_economic_index','Zone Socio-Economic Index',height=350,y_axis_title="SES Index",text_format=".2f"),use_container_width=True)
                with sdoh_cols_viz_tab[1]:
                    if 'avg_travel_time_clinic_min' in zone_analysis_sdoh_tab_val and zone_analysis_sdoh_tab_val['avg_travel_time_clinic_min'].notna().any(): st.plotly_chart(plot_bar_chart(zone_analysis_sdoh_tab_val.sort_values('avg_travel_time_clinic_min'),'name','avg_travel_time_clinic_min','Zone Avg Travel Time to Clinic',height=350,y_axis_title="Minutes",y_is_count=False,text_format=".0f"),use_container_width=True)
            else: st.info("No zone data after merge for SDOH context.")
        else: st.info("Zone attributes or health record zone_id unavailable for SDOH context.")

with tab_clinical_dx:
    st.header("Clinical & Diagnostic Data Patterns")
    if analytics_df_display.empty: st.info("No data for Clinical/Dx with current filters.")
    else:
        clin_dx_cols_page_tab_val = st.columns(2)
        with clin_dx_cols_page_tab_val[0]:
            st.subheader("Top Presenting Symptoms (from encounters)")
            if 'patient_reported_symptoms' in analytics_df_display.columns and analytics_df_display['patient_reported_symptoms'].notna().any():
                symptoms_dx_val = analytics_df_display['patient_reported_symptoms'].str.split(';').explode().str.strip().replace(['','Unknown','N/A'],np.nan).dropna()
                if not symptoms_dx_val.empty:
                    s_counts_dx = symptoms_dx_val.value_counts().nlargest(10).reset_index(); s_counts_dx.columns = ['Symptom','Frequency']
                    st.plotly_chart(plot_bar_chart(s_counts_dx,'Symptom','Frequency',"Top 10 Reported Symptoms",height=400,orientation='h',y_is_count=True,text_format='d'),use_container_width=True)
                else: st.caption("No distinct symptoms reported.")
        with clin_dx_cols_page_tab_val[1]:
            st.subheader("Test Result Distribution (Top 5 Tests by Volume)")
            if 'test_type' in analytics_df_display.columns and 'test_result' in analytics_df_display.columns:
                top_5_t = analytics_df_display['test_type'].value_counts().nlargest(5).index.tolist()
                top_t_df = analytics_df_display[analytics_df_display['test_type'].isin(top_5_t)]
                t_res_dist = top_t_df[~top_t_df['test_result'].isin(['Pending','Rejected Sample','Unknown','N/A','Indeterminate'])]
                if not t_res_dist.empty:
                    t_res_sum = t_res_dist.groupby(['test_type','test_result'])['patient_id'].nunique().reset_index(name='count')
                    st.plotly_chart(plot_bar_chart(t_res_sum,'test_type','count',"Conclusive Test Results Distribution",color_col='test_result',barmode='group',height=400,y_is_count=True,text_format='d'),use_container_width=True)
                else: st.caption("No conclusive test result data for top tests.")
        st.markdown("---"); st.subheader("Overall Test Positivity Rate Trend (e.g., Malaria RDT)")
        mal_key_trend = "RDT-Malaria"; mal_disp_trend = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_key_trend,{}).get("display_name",mal_key_trend)
        mal_df_trend_source = analytics_df_display[(analytics_df_display.get('test_type')==mal_key_trend)&(~analytics_df_display.get('test_result',pd.Series(dtype=str)).isin(['Pending','Rejected Sample','Unknown']))].copy()
        if not mal_df_trend_source.empty and 'encounter_date' in mal_df_trend_source.columns:
            mal_df_trend_source.loc[:,'is_positive'] = mal_df_trend_source['test_result']=='Positive'
            if not pd.api.types.is_datetime64_ns_dtype(mal_df_trend_source['encounter_date']): mal_df_trend_source.loc[:,'encounter_date']=pd.to_datetime(mal_df_trend_source['encounter_date'],errors='coerce')
            mal_df_trend_source.dropna(subset=['encounter_date'],inplace=True)
            if not mal_df_trend_source.empty:
                weekly_pos_trend = get_trend_data(mal_df_trend_source,'is_positive','encounter_date','W-Mon','mean')*100
                if not weekly_pos_trend.empty: st.plotly_chart(plot_annotated_line_chart(weekly_pos_trend,f"Weekly {mal_disp_trend} Positivity Rate",y_axis_title="Positivity (%)",height=350,target_line=app_config.TARGET_MALARIA_POSITIVITY_RATE,y_is_count=False),use_container_width=True)
                else: st.caption(f"No data for {mal_disp_trend} positivity trend.")
            else: st.caption(f"No valid date data for {mal_disp_trend} positivity trend.")
        else: st.caption(f"No {mal_disp_trend} test data for positivity trend.")


with tab_systems_equity:
    st.header("Health Systems Context & Equity Insights")
    if analytics_df_display.empty: st.info("No data for systems/equity analysis with current filters.")
    else:
        sys_eq_cols = st.columns(2)
        with sys_eq_cols[0]:
            st.subheader("Encounters by Clinic ID (Top 10)")
            if 'clinic_id' in analytics_df_display.columns and analytics_df_display['clinic_id'].notna().any():
                fac_load_sys = analytics_df_display['clinic_id'].value_counts().nlargest(10).reset_index(); fac_load_sys.columns=['Clinic ID','Encounters']
                if not fac_load_sys.empty: st.plotly_chart(plot_bar_chart(fac_load_sys,'Clinic ID','Encounters',"Top 10 Clinics by Encounter Volume",height=400,orientation='h',y_is_count=True,text_format='d'),use_container_width=True)
        with sys_eq_cols[1]:
            st.subheader("Referral Status Distribution")
            if 'referral_status' in analytics_df_display.columns and analytics_df_display['referral_status'].notna().any():
                ref_data_sys = analytics_df_display[analytics_df_display['referral_status'].str.lower().isin(['pending','completed','initiated','service provided','attended','missed appointment','declined'])].copy()
                if not ref_data_sys.empty:
                    ref_counts_sys=ref_data_sys['referral_status'].value_counts().reset_index();ref_counts_sys.columns=['Referral Status','Count']
                    if not ref_counts_sys.empty: st.plotly_chart(plot_donut_chart(ref_counts_sys,'Referral Status','Count',"Referral Statuses",height=400,values_are_counts=True),use_container_width=True)
                else: st.caption("No actionable referral data after filtering.")
        st.markdown("---"); st.subheader("AI Risk Score Distribution by Zone Socio-Economic Index")
        if all(c in analytics_df_display for c in ['zone_id','ai_risk_score']) and not zone_attr_df_pop.empty and all(c in zone_attr_df_pop for c in ['socio_economic_index','name']):
            avg_risk_zone_eq = analytics_df_display.groupby('zone_id')['ai_risk_score'].mean().reset_index()
            zone_attr_copy_eq = zone_attr_df_pop.copy(); zone_attr_copy_eq['zone_id']=zone_attr_copy_eq['zone_id'].astype(str); avg_risk_zone_eq['zone_id']=avg_risk_zone_eq['zone_id'].astype(str)
            equity_plot_df_final = zone_attr_copy_eq.merge(avg_risk_zone_eq, on='zone_id', how='inner')
            equity_plot_df_final.dropna(subset=['ai_risk_score','socio_economic_index'],inplace=True)
            if not equity_plot_df_final.empty:
                fig_eq_risk_ses = px.scatter(equity_plot_df_final,x='socio_economic_index',y='ai_risk_score',text='name',size='population' if 'population' in equity_plot_df_final else None,color='ai_risk_score',title="Avg. Patient AI Risk vs. Zone Socio-Economic Index",labels={'socio_economic_index':"SES Index (Zone)",'ai_risk_score':"Avg. Patient AI Risk (Zone)"},height=450,color_continuous_scale="Reds",hover_name='name')
                fig_eq_risk_ses.update_traces(textposition='top center'); st.plotly_chart(fig_eq_risk_ses,use_container_width=True)
            else: st.caption("Not enough data to plot AI Risk vs Zone SES.")
        else: st.caption("Required data for AI Risk vs. Zone SES analysis not fully available.")

st.markdown("---"); st.caption(app_config.APP_FOOTER)
