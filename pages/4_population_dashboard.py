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
if os.path.exists(app_config.APP_LOGO): st.sidebar.image(app_config.APP_LOGO, width=230); st.sidebar.markdown("---")
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
selected_start_date_pop, selected_end_date_pop = st.sidebar.date_input("Select Date Range:", value=[default_pop_start_date, default_pop_end_date], min_value=min_date_pop, max_value=max_date_pop, key="pop_dashboard_date_range_v3_final_corrected")
if selected_start_date_pop > selected_end_date_pop: selected_start_date_pop = selected_end_date_pop

if 'encounter_date' in health_df_pop.columns:
    health_df_pop.loc[:, 'encounter_date_obj'] = pd.to_datetime(health_df_pop['encounter_date'], errors='coerce').dt.date
else: health_df_pop['encounter_date_obj'] = pd.NaT 
analytics_df_base = health_df_pop[ (health_df_pop['encounter_date_obj'].notna()) & (health_df_pop['encounter_date_obj'] >= selected_start_date_pop) & (health_df_pop['encounter_date_obj'] <= selected_end_date_pop) ].copy()

if analytics_df_base.empty: st.warning(f"No health data for period: {selected_start_date_pop.strftime('%d %b %Y')} - {selected_end_date_pop.strftime('%d %b %Y')}. Adjust filters."); st.stop()

conditions_list_pop_final = ["All Conditions"] + sorted(analytics_df_base['condition'].dropna().unique().tolist()); selected_condition_filter_pop_dash = st.sidebar.selectbox("Filter by Condition (Optional):", options=conditions_list_pop_final, index=0, key="pop_dashboard_condition_filter_v3_final")
analytics_df_after_cond_filter = analytics_df_base.copy(); 
if selected_condition_filter_pop_dash != "All Conditions": analytics_df_after_cond_filter = analytics_df_after_cond_filter[analytics_df_after_cond_filter['condition'] == selected_condition_filter_pop_dash]
zones_list_pop_final = ["All Zones"] + sorted(analytics_df_base['zone_id'].dropna().unique().tolist()); selected_zone_filter_pop_dash = st.sidebar.selectbox("Filter by Zone (Optional):", options=zones_list_pop_final, index=0, key="pop_dashboard_zone_filter_v3_final")
analytics_df_after_zone_filter = analytics_df_after_cond_filter.copy()
if selected_zone_filter_pop_dash != "All Zones": analytics_df_after_zone_filter = analytics_df_after_zone_filter[analytics_df_after_zone_filter['zone_id'] == selected_zone_filter_pop_dash]

analytics_df_display = analytics_df_after_zone_filter.copy()
if analytics_df_display.empty and (selected_condition_filter_pop_dash != "All Conditions" or selected_zone_filter_pop_dash != "All Zones"):
    warning_msg_txt = f"No data for '{selected_condition_filter_pop_dash}' in zone '{selected_zone_filter_pop_dash}'. "
    analytics_df_display = analytics_df_base.copy() 
    if selected_zone_filter_pop_dash != "All Zones" and 'zone_id' in analytics_df_display.columns: 
        analytics_df_display = analytics_df_display[analytics_df_display['zone_id'] == selected_zone_filter_pop_dash]
        if analytics_df_display.empty: warning_msg_txt += f"Showing all conditions for zone '{selected_zone_filter_pop_dash}'."
        else: warning_msg_txt += f"Showing all conditions for '{selected_zone_filter_pop_dash}'."
    elif selected_condition_filter_pop_dash != "All Conditions" and 'condition' in analytics_df_base.columns:
        temp_df_fallback = analytics_df_base[analytics_df_base['condition'] == selected_condition_filter_pop_dash]
        if not temp_df_fallback.empty: analytics_df_display = temp_df_fallback; warning_msg_txt += f"Showing data for '{selected_condition_filter_pop_dash}' across all zones."
        else: warning_msg_txt += "Displaying overall data for period."
    else: warning_msg_txt += "Displaying overall data for period."
    st.warning(warning_msg_txt)
    if analytics_df_display.empty: st.error("No data after all fallbacks."); st.stop()

# --- Decision-Making KPI Boxes ---
st.subheader(f"Key Indicators ({selected_start_date_pop.strftime('%d %b')} - {selected_end_date_pop.strftime('%d %b')}, Cond: {selected_condition_filter_pop_dash}, Zone: {selected_zone_filter_pop_dash})")
if analytics_df_display.empty: st.info("No data to display key indicators for current filter selection.")
else:
    kpi_pop_cols1 = st.columns(4)
    unique_patients_val = analytics_df_display['patient_id'].nunique() if 'patient_id' in analytics_df_display.columns else 0
    kpi_pop_cols1[0].metric("Unique Patients (Filtered)", f"{unique_patients_val:,}")
    avg_ai_risk_val = np.nan
    if 'ai_risk_score' in analytics_df_display.columns and analytics_df_display['ai_risk_score'].notna().any(): avg_ai_risk_val = analytics_df_display['ai_risk_score'].mean()
    kpi_pop_cols1[1].metric("Avg. AI Risk Score", f"{avg_ai_risk_val:.1f}" if pd.notna(avg_ai_risk_val) else "N/A")
    high_risk_count_val = 0; prop_high_risk_val = 0.0
    if 'ai_risk_score' in analytics_df_display.columns and analytics_df_display['ai_risk_score'].notna().any() and 'patient_id' in analytics_df_display.columns and unique_patients_val > 0:
        high_risk_df = analytics_df_display[pd.to_numeric(analytics_df_display['ai_risk_score'], errors='coerce') >= app_config.RISK_THRESHOLDS['high']]
        if not high_risk_df.empty: high_risk_count_val = high_risk_df['patient_id'].nunique()
        prop_high_risk_val = (high_risk_count_val / unique_patients_val) * 100
    val_prop_risk_display = f"{prop_high_risk_val:.1f}%" if unique_patients_val > 0 and pd.notna(prop_high_risk_val) else ("0.0%" if unique_patients_val > 0 else "N/A")
    help_text_prop_risk = f"{int(high_risk_count_val)} unique patient(s) with AI Risk Score ≥ {app_config.RISK_THRESHOLDS['high']}"
    kpi_pop_cols1[2].metric(label="% High AI Risk Patients", value=val_prop_risk_display, help=help_text_prop_risk)
    with kpi_pop_cols1[3]:
        top_cond_name = "N/A"; top_cond_enc_count = 0
        if 'condition' in analytics_df_display.columns and analytics_df_display['condition'].notna().any():
            cond_enc_counts = analytics_df_display['condition'].value_counts()
            if not cond_enc_counts.empty: top_cond_name = cond_enc_counts.idxmax(); top_cond_enc_count = cond_enc_counts.max()
        st.markdown(f"""<div class="custom-markdown-kpi-box highlight-red-edge"><div class="custom-kpi-label-top-condition">Top Condition (Encounters)</div><div class="custom-kpi-value-large">{html.escape(str(top_cond_name))}</div><div class="custom-kpi-subtext-small">{html.escape(f"{top_cond_enc_count} encounters") if top_cond_name != "N/A" else ""}</div></div>""", unsafe_allow_html=True)
    kpi_pop_cols2 = st.columns(3)
    mal_rdt_key_kpi_pop = "RDT-Malaria"; mal_rdt_pos_rate_val_pop = 0.0
    if 'test_type' in analytics_df_display.columns and 'test_result' in analytics_df_display.columns:
        mal_rdt_df_kpi_pop = analytics_df_display[(analytics_df_display['test_type'] == mal_rdt_key_kpi_pop) & (~analytics_df_display['test_result'].isin(['Pending','Rejected Sample','Unknown','N/A']))]
        if not mal_rdt_df_kpi_pop.empty and len(mal_rdt_df_kpi_pop) > 0 : mal_rdt_pos_rate_val_pop = (mal_rdt_df_kpi_pop[mal_rdt_df_kpi_pop['test_result'] == 'Positive'].shape[0] / len(mal_rdt_df_kpi_pop)) * 100
    kpi_pop_cols2[0].metric(f"{app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_rdt_key_kpi_pop, {}).get('display_name', mal_rdt_key_kpi_pop)} Positivity", f"{mal_rdt_pos_rate_val_pop:.1f}%")
    ref_compl_rate_val_pop = 0.0
    if all(c in analytics_df_display for c in ['referral_status', 'referral_outcome', 'encounter_id']):
        refs_made_df_pop = analytics_df_display[analytics_df_display['referral_status'].notna() & (~analytics_df_display['referral_status'].isin(['N/A', 'Unknown']))]
        if not refs_made_df_pop.empty:
            total_made_refs_pop = refs_made_df_pop['encounter_id'].nunique()
            compl_outcomes_pop = ['Completed', 'Service Provided', 'Attended', 'Attended Consult', 'Attended Followup']
            compl_refs_pop = refs_made_df_pop[refs_made_df_pop['referral_outcome'].isin(compl_outcomes_pop)]['encounter_id'].nunique()
            if total_made_refs_pop > 0: ref_compl_rate_val_pop = (compl_refs_pop / total_made_refs_pop) * 100
    kpi_pop_cols2[1].metric("Referral Completion Rate", f"{ref_compl_rate_val_pop:.1f}%", help="Based on conclusive positive outcomes.")
    avg_comorb_hr_val_pop = np.nan
    if all(c in analytics_df_display for c in ['key_chronic_conditions_summary', 'ai_risk_score']) and analytics_df_display['ai_risk_score'].notna().any():
        hr_df_comorb_pop = analytics_df_display[pd.to_numeric(analytics_df_display['ai_risk_score'], errors='coerce') >= app_config.RISK_THRESHOLDS['high']]
        if not hr_df_comorb_pop.empty and hr_df_comorb_pop['key_chronic_conditions_summary'].notna().any():
            comorb_counts_pop = hr_df_comorb_pop['key_chronic_conditions_summary'].apply(lambda x: len([c for c in str(x).split(';') if c.strip() and c.lower() not in ['unknown', 'n/a', 'none']]))
            if comorb_counts_pop.notna().any(): avg_comorb_hr_val_pop = comorb_counts_pop.mean()
    kpi_pop_cols2[2].metric("Avg. Comorbidities (High Risk Pts)", f"{avg_comorb_hr_val_pop:.1f}" if pd.notna(avg_comorb_hr_val_pop) else "N/A")
st.markdown("---")

# --- Tabbed Interface ---
tab_epi_overview, tab_demographics_sdoh, tab_clinical_dx, tab_systems_equity = st.tabs([
    "📈 Epidemiological Overview", "🧑‍⚕️ Demographics & SDOH", "🧬 Clinical & Diagnostics", "🌍 Systems & Equity"
])

with tab_epi_overview:
    st.header(f"Epidemiological Overview for '{selected_condition_filter_pop_dash}' in '{selected_zone_filter_pop_dash}'")
    if analytics_df_display.empty: st.info("No data for epidemiological overview with current filters.")
    else:
        epi_overview_cols_tab_content = st.columns(2)
        with epi_overview_cols_tab_content[0]:
            st.subheader("Condition Case Counts (Unique Patients)")
            if 'condition' in analytics_df_display.columns and analytics_df_display['condition'].notna().any() and 'patient_id' in analytics_df_display.columns:
                condition_counts_epi_content = analytics_df_display.groupby('condition')['patient_id'].nunique().nlargest(10).reset_index(name='unique_patients')
                if not condition_counts_epi_content.empty: 
                    condition_counts_epi_content.loc[:, 'condition']=condition_counts_epi_content['condition'].astype(str)
                    st.plotly_chart(plot_bar_chart(condition_counts_epi_content,'condition','unique_patients',"Top Conditions by Unique Patients",height=400,orientation='h',y_is_count=True,text_format='d'),use_container_width=True)
                    st.caption("""**Significance:** Highlights frequent conditions by unique patient volume, guiding resource allocation.""")
                else: st.caption("No condition data for counts.")
            else: st.caption("Condition/Patient ID column missing.")
        with epi_overview_cols_tab_content[1]:
            st.subheader("AI Risk Score Distribution")
            if 'ai_risk_score' in analytics_df_display.columns and analytics_df_display['ai_risk_score'].notna().any():
                risk_scores_plot_content = analytics_df_display['ai_risk_score'].dropna()
                if not risk_scores_plot_content.empty: 
                    fig_risk_content=px.histogram(risk_scores_plot_content, x="ai_risk_score",nbins=20,title="Patient AI Risk Scores")
                    fig_risk_content.update_layout(bargap=0.1,height=400); st.plotly_chart(fig_risk_content,use_container_width=True)
                    st.caption(f"""**Significance:** Shows AI risk distribution (0-100). Lower scores skew = healthier; higher scores skew (> {app_config.RISK_THRESHOLDS['moderate']}-{app_config.RISK_THRESHOLDS['high']}) = vulnerable population. Informs risk stratification.""")
                else: st.caption("No valid AI Risk Scores to plot.")
            else: st.caption("AI Risk Score data unavailable.")
        
        st.markdown("---"); st.subheader(f"Incidence Trend for Top Conditions (Weekly New Cases)")
        if not analytics_df_display.empty and all(col in analytics_df_display.columns for col in ['condition', 'patient_id', 'encounter_date']):
            top_n_epi_trend = 3 
            source_df_trends_val = analytics_df_display.copy()
            if selected_condition_filter_pop_dash != "All Conditions": top_conds_for_trend_val = [selected_condition_filter_pop_dash] if selected_condition_filter_pop_dash in source_df_trends_val['condition'].unique() else []
            else: top_conds_for_trend_val = source_df_trends_val['condition'].value_counts().nlargest(top_n_epi_trend).index.tolist()
            if top_conds_for_trend_val:
                num_charts_val = len(top_conds_for_trend_val); inc_trend_cols_val = st.columns(num_charts_val if num_charts_val > 0 else 1)
                df_inc_calc_val = source_df_trends_val.copy()
                if not pd.api.types.is_datetime64_ns_dtype(df_inc_calc_val['encounter_date']): df_inc_calc_val.loc[:,'encounter_date'] = pd.to_datetime(df_inc_calc_val['encounter_date'], errors='coerce')
                df_inc_calc_val.dropna(subset=['encounter_date'], inplace=True)
                if not df_inc_calc_val.empty :
                    df_inc_calc_val.sort_values('encounter_date', inplace=True); df_inc_calc_val.loc[:,'is_first_in_period'] = ~df_inc_calc_val.duplicated(subset=['patient_id', 'condition'], keep='first'); new_cases_trend_plot_val = df_inc_calc_val[df_inc_calc_val['is_first_in_period']]
                    for i, cond_name_item_val in enumerate(top_conds_for_trend_val):
                        curr_col_item_val = inc_trend_cols_val[i % num_charts_val if num_charts_val > 0 else 0]
                        cond_trend_data_item_val = new_cases_trend_plot_val[new_cases_trend_plot_val['condition'] == cond_name_item_val]
                        with curr_col_item_val:
                            if not cond_trend_data_item_val.empty:
                                weekly_new_val_plot = get_trend_data(cond_trend_data_item_val, 'patient_id', date_col='encounter_date', period='W-Mon', agg_func='count')
                                if not weekly_new_val_plot.empty: st.plotly_chart(plot_annotated_line_chart(weekly_new_val_plot, f"Weekly New {cond_name_item_val} Cases", y_axis_title="New Cases", height=300, date_format="%U, %Y (Wk)", y_is_count=True), use_container_width=True)
                                else: st.caption(f"No trend data for {cond_name_item_val}.")
                            else: st.caption(f"No new cases for {cond_name_item_val}.")
                else: st.caption("Not enough valid date data for incidence.")
            else: st.caption(f"No data for '{selected_condition_filter_pop_dash if selected_condition_filter_pop_dash != 'All Conditions' else 'top conditions'}' for incidence trend.")
        else: st.caption("Required data missing for incidence trends.")

with tab_demographics_sdoh:
    st.header("Demographics & Social Determinants of Health (SDOH)")
    if analytics_df_display.empty: st.info("No data for Demographics/SDOH with current filters.")
    else:
        demo_cols_sdoh_val = st.columns(2)
        with demo_cols_sdoh_val[0]:
            st.subheader("Age Distribution of Patients")
            if 'age' in analytics_df_display.columns and analytics_df_display['age'].notna().any():
                age_bins_final = [0,5,12,18,35,50,65,np.inf]; age_labels_final = ['0-4','5-17','12-17','18-34','35-49','50-64','65+']
                age_df_final = analytics_df_display.copy(); age_df_final.loc[:, 'age_group_final'] = pd.cut(age_df_final['age'], bins=age_bins_final, labels=age_labels_final, right=False)
                age_dist_final = age_df_final['age_group_final'].value_counts().sort_index().reset_index(); age_dist_final.columns=['Age Group','Patient Encounters']
                if not age_dist_final.empty : st.plotly_chart(plot_bar_chart(age_dist_final, 'Age Group', 'Patient Encounters', "Encounters by Age Group", height=350, y_is_count=True, text_format='d'), use_container_width=True)
                else: st.caption("No age data to plot.")
        with demo_cols_sdoh_val[1]:
            st.subheader("Gender Distribution of Patients")
            if 'gender' in analytics_df_display.columns and analytics_df_display['gender'].notna().any():
                gender_dist_final = analytics_df_display['gender'].value_counts().reset_index(); gender_dist_final.columns = ['Gender','Patient Encounters']
                if not gender_dist_final.empty: st.plotly_chart(plot_donut_chart(gender_dist_final, 'Gender', 'Patient Encounters', "Encounters by Gender", height=350, values_are_counts=True), use_container_width=True)
                else: st.caption("No gender data to plot.")
        st.markdown("---"); st.subheader("Geographic & Socio-Economic Context (Zone Level Averages)")
        if not zone_attr_df_pop.empty and 'zone_id' in analytics_df_display.columns and analytics_df_display['zone_id'].notna().any():
            patients_zone_sdoh_final = analytics_df_display.groupby('zone_id')['patient_id'].nunique().reset_index(name='patients_filtered')
            risk_zone_sdoh_final = analytics_df_display.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='avg_risk_filtered')
            zone_analysis_sdoh_final = zone_attr_df_pop.copy(); zone_analysis_sdoh_final.loc[:,'zone_id']=zone_analysis_sdoh_final['zone_id'].astype(str)
            for df_merge in [patients_zone_sdoh_final, risk_zone_sdoh_final]: df_merge.loc[:,'zone_id'] = df_merge['zone_id'].astype(str); zone_analysis_sdoh_final=zone_analysis_sdoh_final.merge(df_merge,on='zone_id',how='left')
            zone_analysis_sdoh_final['patients_filtered'].fillna(0,inplace=True)
            if not zone_analysis_sdoh_final.empty:
                sdoh_cols_viz_final = st.columns(2)
                with sdoh_cols_viz_final[0]: 
                    if 'socio_economic_index' in zone_analysis_sdoh_final and zone_analysis_sdoh_final['socio_economic_index'].notna().any(): st.plotly_chart(plot_bar_chart(zone_analysis_sdoh_final.sort_values('socio_economic_index'),'name','socio_economic_index','Zone Socio-Economic Index',height=350,y_axis_title="SES Index",text_format=".2f"),use_container_width=True)
                with sdoh_cols_viz_final[1]:
                    if 'avg_travel_time_clinic_min' in zone_analysis_sdoh_final and zone_analysis_sdoh_final['avg_travel_time_clinic_min'].notna().any(): st.plotly_chart(plot_bar_chart(zone_analysis_sdoh_final.sort_values('avg_travel_time_clinic_min'),'name','avg_travel_time_clinic_min','Zone Avg Travel Time to Clinic',height=350,y_axis_title="Minutes",y_is_count=False,text_format=".0f"),use_container_width=True)
            else: st.info("No zone data after merge for SDOH context.")
        else: st.info("Zone attribute data or health record zone_id unavailable for SDOH context.")

with tab_clinical_dx:
    st.header("Clinical & Diagnostic Data Patterns")
    if analytics_df_display.empty: st.info("No data for Clinical/Dx with current filters.")
    else:
        clin_dx_cols_final = st.columns(2)
        with clin_dx_cols_final[0]:
            st.subheader("Top Presenting Symptoms (from encounters)")
            if 'patient_reported_symptoms' in analytics_df_display.columns and analytics_df_display['patient_reported_symptoms'].notna().any():
                symptoms_dx_final = analytics_df_display['patient_reported_symptoms'].str.split(';').explode().str.strip().replace(['','Unknown','N/A'],np.nan).dropna()
                if not symptoms_dx_final.empty:
                    s_counts_dx_final = symptoms_dx_final.value_counts().nlargest(10).reset_index(); s_counts_dx_final.columns = ['Symptom','Frequency']
                    st.plotly_chart(plot_bar_chart(s_counts_dx_final,'Symptom','Frequency',"Top 10 Reported Symptoms",height=400,orientation='h',y_is_count=True,text_format='d'),use_container_width=True)
                else: st.caption("No distinct symptoms reported.")
        with clin_dx_cols_final[1]:
            st.subheader("Test Result Distribution (Top 5 Tests by Volume)")
            if 'test_type' in analytics_df_display.columns and 'test_result' in analytics_df_display.columns:
                top_5_t_final = analytics_df_display['test_type'].value_counts().nlargest(5).index.tolist()
                top_t_df_final = analytics_df_display[analytics_df_display['test_type'].isin(top_5_t_final)]
                t_res_dist_final = top_t_df_final[~top_t_df_final['test_result'].isin(['Pending','Rejected Sample','Unknown','N/A','Indeterminate'])]
                if not t_res_dist_final.empty:
                    t_res_sum_final = t_res_dist_final.groupby(['test_type','test_result'])['patient_id'].nunique().reset_index(name='count')
                    st.plotly_chart(plot_bar_chart(t_res_sum_final,'test_type','count',"Conclusive Test Results Distribution",color_col='test_result',barmode='group',height=400,y_is_count=True,text_format='d'),use_container_width=True)
                else: st.caption("No conclusive test result data for top tests.")
        st.markdown("---"); st.subheader("Overall Test Positivity Rate Trend (e.g., Malaria RDT)")
        mal_key_final = "RDT-Malaria"; mal_disp_final = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_key_final,{}).get("display_name",mal_key_final)
        mal_df_final = analytics_df_display[(analytics_df_display.get('test_type')==mal_key_final)&(~analytics_df_display.get('test_result',pd.Series(dtype=str)).isin(['Pending','Rejected Sample','Unknown']))].copy()
        if not mal_df_final.empty and 'encounter_date' in mal_df_final.columns:
            mal_df_final.loc[:,'is_positive'] = mal_df_final['test_result']=='Positive'
            if not pd.api.types.is_datetime64_ns_dtype(mal_df_final['encounter_date']): mal_df_final.loc[:,'encounter_date']=pd.to_datetime(mal_df_final['encounter_date'],errors='coerce')
            mal_df_final.dropna(subset=['encounter_date'],inplace=True)
            if not mal_df_final.empty:
                weekly_pos_final = get_trend_data(mal_df_final,'is_positive','encounter_date','W-Mon','mean')*100
                if not weekly_pos_final.empty: st.plotly_chart(plot_annotated_line_chart(weekly_pos_final,f"Weekly {mal_disp_final} Positivity Rate",y_axis_title="Positivity (%)",height=350,target_line=app_config.TARGET_MALARIA_POSITIVITY_RATE,y_is_count=False),use_container_width=True)
                else: st.caption(f"No data for {mal_disp_final} positivity trend.")
        else: st.caption(f"No {mal_disp_final} test data for positivity trend.")

with tab_systems_equity:
    st.header("Health Systems Context & Equity Insights")
    if analytics_df_display.empty: st.info("No data for systems/equity analysis with current filters.")
    else:
        sys_eq_cols_final = st.columns(2)
        with sys_eq_cols_final[0]:
            st.subheader("Encounters by Clinic ID (Top 10)")
            if 'clinic_id' in analytics_df_display.columns and analytics_df_display['clinic_id'].notna().any():
                fac_load_final = analytics_df_display['clinic_id'].value_counts().nlargest(10).reset_index(); fac_load_final.columns=['Clinic ID','Encounters']
                if not fac_load_final.empty: st.plotly_chart(plot_bar_chart(fac_load_final,'Clinic ID','Encounters',"Top 10 Clinics by Encounter Volume",height=400,orientation='h',y_is_count=True,text_format='d'),use_container_width=True)
        with sys_eq_cols_final[1]:
            st.subheader("Referral Status Distribution")
            if 'referral_status' in analytics_df_display.columns and analytics_df_display['referral_status'].notna().any():
                ref_data_final = analytics_df_display[analytics_df_display['referral_status'].str.lower().isin(['pending','completed','initiated','service provided','attended','missed appointment','declined'])].copy()
                if not ref_data_final.empty:
                    ref_counts_final=ref_data_final['referral_status'].value_counts().reset_index();ref_counts_final.columns=['Referral Status','Count']
                    if not ref_counts_final.empty: st.plotly_chart(plot_donut_chart(ref_counts_final,'Referral Status','Count',"Referral Statuses",height=400,values_are_counts=True),use_container_width=True)
                else: st.caption("No actionable referral data after filtering.")
        st.markdown("---"); st.subheader("AI Risk Score Distribution by Zone Socio-Economic Index")
        if all(c in analytics_df_display for c in ['zone_id','ai_risk_score']) and not zone_attr_df_pop.empty and all(c in zone_attr_df_pop for c in ['socio_economic_index','name']):
            avg_risk_zone_eq_final = analytics_df_display.groupby('zone_id')['ai_risk_score'].mean().reset_index()
            zone_attr_copy_eq_final = zone_attr_df_pop.copy(); zone_attr_copy_eq_final['zone_id']=zone_attr_copy_eq_final['zone_id'].astype(str); avg_risk_zone_eq_final['zone_id']=avg_risk_zone_eq_final['zone_id'].astype(str)
            equity_plot_df_final = zone_attr_copy_eq_final.merge(avg_risk_zone_eq_final, on='zone_id', how='inner')
            equity_plot_df_final.dropna(subset=['ai_risk_score','socio_economic_index'],inplace=True)
            if not equity_plot_df_final.empty:
                fig_eq_risk_ses_final = px.scatter(equity_plot_df_final,x='socio_economic_index',y='ai_risk_score',text='name',size='population' if 'population' in equity_plot_df_final else None,color='ai_risk_score',title="Avg. Patient AI Risk vs. Zone Socio-Economic Index",labels={'socio_economic_index':"SES Index (Zone)",'ai_risk_score':"Avg. Patient AI Risk (Zone)"},height=450,color_continuous_scale="Reds",hover_name='name')
                fig_eq_risk_ses_final.update_traces(textposition='top center'); st.plotly_chart(fig_eq_risk_ses_final,use_container_width=True)
            else: st.caption("Not enough data to plot AI Risk vs Zone SES.")
        else: st.caption("Required data for AI Risk vs. Zone SES analysis not fully available.")

st.markdown("---"); st.caption(app_config.APP_FOOTER)
