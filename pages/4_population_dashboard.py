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
    load_zone_data, # Used to get zone_attributes
    get_trend_data
)
from utils.ai_analytics_engine import apply_ai_models
from utils.ui_visualization_helpers import (
    plot_bar_chart,
    plot_donut_chart,
    plot_annotated_line_chart
    # render_kpi_card not used here, st.metric or custom markdown for KPIs
)

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Population Dashboard - Health Hub", 
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__) # Get logger after basic imports

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
    # Apply AI models only if raw data loading was successful
    health_df = apply_ai_models(health_df_raw) if not health_df_raw.empty else pd.DataFrame(columns=health_df_raw.columns if not health_df_raw.empty else [])
    
    zone_gdf = load_zone_data() # This returns a GeoDataFrame
    zone_attributes_df = pd.DataFrame(columns=['zone_id', 'name', 'population', 'socio_economic_index']) # Default schema
    if zone_gdf is not None and not zone_gdf.empty:
        geom_col = zone_gdf.geometry.name if hasattr(zone_gdf, 'geometry') and hasattr(zone_gdf.geometry, 'name') else 'geometry'
        if geom_col in zone_gdf.columns:
            zone_attributes_df = pd.DataFrame(zone_gdf.drop(columns=[geom_col], errors='ignore'))
        else: # Fallback if no distinct geometry column identified
            zone_attributes_df = pd.DataFrame(zone_gdf)
        logger.info(f"Loaded {len(zone_attributes_df)} zone attributes for SDOH context.")
    else:
        logger.warning("Zone attributes data (from GeoDataFrame) not available or GDF invalid.")
    
    if health_df.empty: logger.error("Population Dashboard: Health data is empty after loading/AI processing.")
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

min_date_pop_overall = date.today() - timedelta(days=365*2); max_date_pop_overall = date.today()
if 'encounter_date' in health_df_pop.columns and health_df_pop['encounter_date'].notna().any():
    min_date_pop_series_val = health_df_pop['encounter_date'].dropna()
    if not min_date_pop_series_val.empty:
        min_date_pop_overall = min_date_pop_series_val.min().date()
        max_date_pop_overall = min_date_pop_series_val.max().date()
if min_date_pop_overall > max_date_pop_overall: min_date_pop_overall = max_date_pop_overall
default_pop_start_date_val = min_date_pop_overall; default_pop_end_date_val = max_date_pop_overall
selected_start_date_pop, selected_end_date_pop = st.sidebar.date_input("Select Date Range:", value=[default_pop_start_date_val, default_pop_end_date_val], min_value=min_date_pop_overall, max_value=max_date_pop_overall, key="pop_dashboard_date_range_final") # Consistent key
if selected_start_date_pop > selected_end_date_pop: selected_start_date_pop = selected_end_date_pop

# Prepare encounter_date_obj for filtering and ensure it exists
if 'encounter_date' in health_df_pop.columns:
    health_df_pop.loc[:, 'encounter_date_obj'] = pd.to_datetime(health_df_pop['encounter_date'], errors='coerce').dt.date
else: health_df_pop['encounter_date_obj'] = pd.NaT 

analytics_df_base = health_df_pop[ (health_df_pop['encounter_date_obj'].notna()) & (health_df_pop['encounter_date_obj'] >= selected_start_date_pop) & (health_df_pop['encounter_date_obj'] <= selected_end_date_pop) ].copy()

if analytics_df_base.empty: st.warning(f"No health data for period: {selected_start_date_pop.strftime('%d %b %Y')} - {selected_end_date_pop.strftime('%d %b %Y')}. Adjust filters."); st.stop()

# Optional Condition Filter
conditions_list_pop_val = ["All Conditions"] + sorted(analytics_df_base['condition'].dropna().unique().tolist())
selected_condition_filter_pop = st.sidebar.selectbox("Filter by Condition (Optional):", options=conditions_list_pop_val, index=0, key="pop_condition_filter_final")
analytics_df_after_condition = analytics_df_base.copy()
if selected_condition_filter_pop != "All Conditions": analytics_df_after_condition = analytics_df_after_condition[analytics_df_after_condition['condition'] == selected_condition_filter_pop]

# Optional Zone Filter
zones_list_pop_val = ["All Zones"] + sorted(analytics_df_base['zone_id'].dropna().unique().tolist())
selected_zone_filter_pop = st.sidebar.selectbox("Filter by Zone (Optional):", options=zones_list_pop_val, index=0, key="pop_zone_filter_final")
analytics_df_after_zone = analytics_df_after_condition.copy()
if selected_zone_filter_pop != "All Zones": analytics_df_after_zone = analytics_df_after_zone[analytics_df_after_zone['zone_id'] == selected_zone_filter_pop]

# analytics_df_display is what the tabs will use
analytics_df_display = analytics_df_after_zone.copy()
if analytics_df_display.empty and (selected_condition_filter_pop != "All Conditions" or selected_zone_filter_pop != "All Zones"):
    warning_msg = f"No data found for '{selected_condition_filter_pop}' in zone '{selected_zone_filter_pop}'. "
    # Try to provide a sensible fallback
    analytics_df_display = analytics_df_base.copy() # Start with just date-filtered
    if selected_zone_filter_pop != "All Zones": # Apply zone if it was specified
        analytics_df_display = analytics_df_display[analytics_df_display['zone_id'] == selected_zone_filter_pop]
        if analytics_df_display.empty: # If zone + all conditions is empty
            warning_msg += f" Also no data found for just zone '{selected_zone_filter_pop}'. Displaying overall period data."
            analytics_df_display = analytics_df_base.copy() # Revert to only date-filtered
        else: warning_msg += f" Showing data for zone '{selected_zone_filter_pop}' across all conditions."
    elif selected_condition_filter_pop != "All Conditions": # Zone was "All Zones", but condition filter emptied it
        analytics_df_display = analytics_df_base[analytics_df_base['condition'] == selected_condition_filter_pop]
        if analytics_df_display.empty: warning_msg += f" Also no data found for just condition '{selected_condition_filter_pop}'. Displaying overall period data."
        else: warning_msg += f" Showing data for '{selected_condition_filter_pop}' across all zones."
    else: # Should not be reached if analytics_df_base wasn't empty, but for safety
        warning_msg += " Displaying overall data for selected period."
    st.warning(warning_msg)
    if analytics_df_display.empty : st.error("No data for display after all fallbacks. Adjust filters."); st.stop()

# --- Decision-Making KPI Boxes ---
st.subheader(f"Key Indicators ({selected_start_date_pop.strftime('%d %b')} - {selected_end_date_pop.strftime('%d %b')}, Cond: {selected_condition_filter_pop}, Zone: {selected_zone_filter_pop})")
if analytics_df_display.empty: st.info("No data to display key indicators for current filter selection.")
else:
    kpi_cols1 = st.columns(4)
    unique_patients = analytics_df_display.get('patient_id', pd.Series(dtype=str)).nunique()
    kpi_cols1[0].metric("Unique Patients (Filtered)", f"{unique_patients:,}")

    avg_risk = np.nan
    if 'ai_risk_score' in analytics_df_display.columns and analytics_df_display['ai_risk_score'].notna().any(): avg_risk = analytics_df_display['ai_risk_score'].mean()
    kpi_cols1[1].metric("Avg. AI Risk Score", f"{avg_risk:.1f}" if pd.notna(avg_risk) else "N/A")

    high_risk_count = 0; prop_high_risk = 0.0
    if 'ai_risk_score' in analytics_df_display.columns and analytics_df_display['ai_risk_score'].notna().any() and unique_patients > 0:
        high_risk_df_kpi = analytics_df_display[pd.to_numeric(analytics_df_display['ai_risk_score'], errors='coerce') >= app_config.RISK_THRESHOLDS['high']]
        if not high_risk_df_kpi.empty: high_risk_count = high_risk_df_kpi['patient_id'].nunique()
        prop_high_risk = (high_risk_count / unique_patients) * 100
    value_prop_risk = f"{prop_high_risk:.1f}%" if unique_patients > 0 and pd.notna(prop_high_risk) else ("0.0%" if unique_patients > 0 else "N/A")
    kpi_cols1[2].metric(label="% High AI Risk Patients", value=value_prop_risk, help=f"{int(high_risk_count)} unique patient(s) with AI Risk Score ≥ {app_config.RISK_THRESHOLDS['high']}")

    with kpi_cols1[3]:
        top_cond_name, top_cond_count = "N/A", 0
        if 'condition' in analytics_df_display.columns and analytics_df_display['condition'].notna().any():
            cond_counts_kpi = analytics_df_display['condition'].value_counts()
            if not cond_counts_kpi.empty: top_cond_name = cond_counts_kpi.idxmax(); top_cond_count = cond_counts_kpi.max()
        st.markdown(f"""<div class="custom-markdown-kpi-box highlight-red-edge"><div class="custom-kpi-label-top-condition">{html.escape("Top Condition (Encounters)")}</div><div class="custom-kpi-value-large">{html.escape(str(top_cond_name))}</div><div class="custom-kpi-subtext-small">{html.escape(f"{top_cond_count} encounters") if top_cond_name != "N/A" else ""}</div></div>""", unsafe_allow_html=True)
    
    kpi_pop_cols2 = st.columns(3)
    # ... (Malaria Positivity KPI) ...

    # 6. Referral Completion Rate
    referral_completion_rate_kpi_val = 0.0 # Initialized
    if 'referral_status' in analytics_df_display.columns and \
       'referral_outcome' in analytics_df_display.columns and \
       'encounter_id' in analytics_df_display.columns: # Ensure all needed columns are present
        
        referrals_made_df_kpi = analytics_df_display[
            analytics_df_display['referral_status'].notna() & 
            (~analytics_df_display['referral_status'].isin(['N/A', 'Unknown']))
        ]
        if not referrals_made_df_kpi.empty:
            total_made_referrals_kpi = referrals_made_df_kpi['encounter_id'].nunique()
            completed_outcomes_kpi = ['Completed', 'Service Provided', 'Attended', 'Attended Consult', 'Attended Followup']
            
            # Ensure 'referral_outcome' column is accessed safely
            if 'referral_outcome' in referrals_made_df_kpi.columns:
                completed_refs_kpi = referrals_made_df_kpi[
                    referrals_made_df_kpi['referral_outcome'].isin(completed_outcomes_kpi)
                ]['encounter_id'].nunique()
                if total_made_referrals_kpi > 0:
                    referral_completion_rate_kpi_val = (completed_refs_kpi / total_made_referrals_kpi) * 100
            # If 'referral_outcome' is somehow not in referrals_made_df_kpi (though it should be if in analytics_df_display)
            # referral_completion_rate_kpi_val remains 0.0
            
    # The problematic line uses ref_comp_rate
    # kpi_pop_cols2[1].metric("Referral Completion Rate", f"{ref_comp_rate:.1f}%", help="Based on conclusive positive outcomes.")
    # It should use referral_completion_rate_kpi_val
    kpi_pop_cols2[1].metric("Referral Completion Rate", f"{referral_completion_rate_kpi_val:.1f}%", help="Based on conclusive positive outcomes.")


# --- Tabbed Interface ---
tab_epi_overview, tab_demographics_sdoh, tab_clinical_dx, tab_systems_equity = st.tabs([
    "📈 Epidemiological Overview", "🧑‍⚕️ Demographics & SDOH", "🧬 Clinical & Diagnostics", "🌍 Systems & Equity"
])

with tab_epi_overview:
    st.header(f"Epidemiological Overview for '{selected_condition_filter_pop}' in '{selected_zone_filter_pop}'") # Use _pop vars for titles
    if analytics_df_display.empty: st.info("No data for epidemiological overview with current filters.")
    else:
        epi_overview_cols_content = st.columns(2)
        with epi_overview_cols_content[0]:
            st.subheader("Condition Case Counts (Unique Patients)")
            if 'condition' in analytics_df_display.columns and analytics_df_display['condition'].notna().any() and 'patient_id' in analytics_df_display.columns:
                cond_counts = analytics_df_display.groupby('condition')['patient_id'].nunique().nlargest(10).reset_index(name='unique_patients')
                if not cond_counts.empty: cond_counts.loc[:, 'condition']=cond_counts['condition'].astype(str); st.plotly_chart(plot_bar_chart(cond_counts,'condition','unique_patients',"Top Conditions by Unique Patients",height=400,orientation='h',y_is_count=True,text_format='d'),use_container_width=True); st.caption("""**Significance:** Highlights frequent conditions by unique patient volume, guiding resource allocation.""")
                else: st.caption("No condition data for counts.")
            else: st.caption("Condition/Patient ID column missing.")
        with epi_overview_cols_content[1]:
            st.subheader("AI Risk Score Distribution")
            if 'ai_risk_score' in analytics_df_display.columns and analytics_df_display['ai_risk_score'].notna().any():
                risk_scores_plot = analytics_df_display['ai_risk_score'].dropna()
                if not risk_scores_plot.empty: fig_risk_content_plot=px.histogram(risk_scores_plot, x="ai_risk_score",nbins=20,title="Patient AI Risk Scores"); fig_risk_content_plot.update_layout(bargap=0.1,height=400); st.plotly_chart(fig_risk_content_plot,use_container_width=True); st.caption(f"""**Significance:** Distribution of AI risk scores (0-100). Skew to low = healthier; skew to high (> {app_config.RISK_THRESHOLDS['moderate']}-{app_config.RISK_THRESHOLDS['high']}) = vulnerable. Informs risk stratification.""")
                else: st.caption("No valid AI Risk Scores to plot.")
            else: st.caption("AI Risk Score data unavailable.")
        
        st.markdown("---"); st.subheader(f"Incidence Trend for Top Conditions (Weekly New Cases)")
        if not analytics_df_display.empty and all(col in analytics_df_display.columns for col in ['condition','patient_id','encounter_date']):
            top_n = 3; source_trends = analytics_df_display.copy()
            top_conds = [selected_condition_filter_pop] if selected_condition_filter_pop!="All Conditions" and selected_condition_filter_pop in source_trends['condition'].unique() else source_trends['condition'].value_counts().nlargest(top_n).index.tolist()
            if top_conds:
                num_c = len(top_conds); inc_cols = st.columns(num_c if num_c > 0 else 1)
                df_inc_calc = source_trends.copy()
                if not pd.api.types.is_datetime64_ns_dtype(df_inc_calc['encounter_date']): df_inc_calc.loc[:,'encounter_date'] = pd.to_datetime(df_inc_calc['encounter_date'], errors='coerce')
                df_inc_calc.dropna(subset=['encounter_date'], inplace=True)
                if not df_inc_calc.empty :
                    df_inc_calc.sort_values('encounter_date', inplace=True); df_inc_calc.loc[:,'is_first'] = ~df_inc_calc.duplicated(subset=['patient_id', 'condition'], keep='first'); new_cases_df = df_inc_calc[df_inc_calc['is_first']]
                    for i, c_name in enumerate(top_conds):
                        curr_c = inc_cols[i % num_c if num_c > 0 else 0]
                        cond_trend_df = new_cases_df[new_cases_df['condition'] == c_name]
                        with curr_c:
                            if not cond_trend_df.empty:
                                weekly_new = get_trend_data(cond_trend_df,'patient_id','encounter_date','W-Mon','count')
                                if not weekly_new.empty: st.plotly_chart(plot_annotated_line_chart(weekly_new,f"Weekly New {c_name} Cases",y_axis_title="New Cases",height=300,date_format="%U, %Y (Wk)",y_is_count=True),use_container_width=True)
                                else: st.caption(f"No trend for {c_name}.")
                            else: st.caption(f"No new cases for {c_name}.")
                else: st.caption("No valid date data for incidence.")
            else: st.caption(f"No data for '{selected_condition_filter_pop if selected_condition_filter_pop != 'All Conditions' else 'top conditions'}' for trend.")
        else: st.caption("Required data missing for incidence trends.")

with tab_demographics_sdoh:
    st.header("Demographics & Social Determinants of Health (SDOH)")
    if analytics_df_display.empty: st.info("No data for Demographics/SDOH with current filters.")
    else:
        demo_sdoh_cols = st.columns(2)
        with demo_sdoh_cols[0]:
            st.subheader("Age Distribution")
            if 'age' in analytics_df_display.columns and analytics_df_display['age'].notna().any():
                age_bins_final_sdoh = [0,5,12,18,35,50,65,np.inf]; age_labels_final_sdoh = ['0-4','5-17','12-17','18-34','35-49','50-64','65+']
                age_df_final_sdoh = analytics_df_display.copy(); age_df_final_sdoh.loc[:, 'age_group_final'] = pd.cut(age_df_final_sdoh['age'], bins=age_bins_final_sdoh, labels=age_labels_final_sdoh, right=False)
                age_dist_final_sdoh = age_df_final_sdoh['age_group_final'].value_counts().sort_index().reset_index(); age_dist_final_sdoh.columns=['Age Group','Encounters']
                if not age_dist_final_sdoh.empty : st.plotly_chart(plot_bar_chart(age_dist_final_sdoh, 'Age Group', 'Encounters', "Encounters by Age Group", height=350, y_is_count=True, text_format='d'), use_container_width=True)
                else: st.caption("No age data.")
        with demo_sdoh_cols[1]:
            st.subheader("Gender Distribution")
            if 'gender' in analytics_df_display.columns and analytics_df_display['gender'].notna().any():
                gender_dist_final_sdoh = analytics_df_display['gender'].value_counts().reset_index(); gender_dist_final_sdoh.columns = ['Gender','Encounters']
                if not gender_dist_final_sdoh.empty: st.plotly_chart(plot_donut_chart(gender_dist_final_sdoh, 'Gender', 'Encounters', "Encounters by Gender", height=350, values_are_counts=True), use_container_width=True)
                else: st.caption("No gender data.")
        st.markdown("---"); st.subheader("Geographic & Socio-Economic Context (Zone Averages)")
        if not zone_attr_df_pop.empty and 'zone_id' in analytics_df_display.columns and analytics_df_display['zone_id'].notna().any():
            patients_zone_final_sdoh = analytics_df_display.groupby('zone_id')['patient_id'].nunique().reset_index(name='patients_filtered')
            risk_zone_final_sdoh = analytics_df_display.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='avg_risk_filtered')
            zone_analysis_final_sdoh = zone_attr_df_pop.copy(); zone_analysis_final_sdoh.loc[:,'zone_id']=zone_analysis_final_sdoh['zone_id'].astype(str)
            for df_mrg in [patients_zone_final_sdoh, risk_zone_final_sdoh]: df_mrg.loc[:,'zone_id'] = df_mrg['zone_id'].astype(str); zone_analysis_final_sdoh=zone_analysis_final_sdoh.merge(df_mrg,on='zone_id',how='left')
            zone_analysis_final_sdoh['patients_filtered'].fillna(0,inplace=True)
            if not zone_analysis_final_sdoh.empty:
                sdoh_cols_viz_final_val = st.columns(2)
                with sdoh_cols_viz_final_val[0]: 
                    if 'socio_economic_index' in zone_analysis_final_sdoh and zone_analysis_final_sdoh['socio_economic_index'].notna().any(): st.plotly_chart(plot_bar_chart(zone_analysis_final_sdoh.sort_values('socio_economic_index'),'name','socio_economic_index','Zone Socio-Economic Index',height=350,y_axis_title="SES Index",text_format=".2f"),use_container_width=True)
                with sdoh_cols_viz_final_val[1]:
                    if 'avg_travel_time_clinic_min' in zone_analysis_final_sdoh and zone_analysis_final_sdoh['avg_travel_time_clinic_min'].notna().any(): st.plotly_chart(plot_bar_chart(zone_analysis_final_sdoh.sort_values('avg_travel_time_clinic_min'),'name','avg_travel_time_clinic_min','Zone Avg Travel Time to Clinic',height=350,y_axis_title="Minutes",y_is_count=False,text_format=".0f"),use_container_width=True)
            else: st.info("No zone data after merge for SDOH.")
        else: st.info("Zone attributes or health record zone_id unavailable for SDOH context.")

with tab_clinical_dx:
    st.header("Clinical & Diagnostic Data Patterns")
    if analytics_df_display.empty: st.info("No data for Clinical/Dx with current filters.")
    else:
        clin_dx_cols_final_val = st.columns(2)
        with clin_dx_cols_final_val[0]:
            st.subheader("Top Presenting Symptoms")
            if 'patient_reported_symptoms' in analytics_df_display.columns and analytics_df_display['patient_reported_symptoms'].notna().any():
                symptoms_final_val = analytics_df_display['patient_reported_symptoms'].str.split(';').explode().str.strip().replace(['','Unknown','N/A'],np.nan).dropna()
                if not symptoms_final_val.empty:
                    s_counts_final = symptoms_final_val.value_counts().nlargest(10).reset_index(); s_counts_final.columns = ['Symptom','Frequency']
                    st.plotly_chart(plot_bar_chart(s_counts_final,'Symptom','Frequency',"Top 10 Reported Symptoms",height=400,orientation='h',y_is_count=True,text_format='d'),use_container_width=True)
                else: st.caption("No distinct symptoms.")
        with clin_dx_cols_final_val[1]:
            st.subheader("Test Result Distribution (Top 5 Tests)")
            if 'test_type' in analytics_df_display.columns and 'test_result' in analytics_df_display.columns:
                top_5_t_val = analytics_df_display['test_type'].value_counts().nlargest(5).index.tolist()
                top_t_df_val = analytics_df_display[analytics_df_display['test_type'].isin(top_5_t_val)]
                t_res_dist_val = top_t_df_val[~top_t_df_val['test_result'].isin(['Pending','Rejected Sample','Unknown','N/A','Indeterminate'])]
                if not t_res_dist_val.empty:
                    t_res_sum_val = t_res_dist_val.groupby(['test_type','test_result'])['patient_id'].nunique().reset_index(name='count')
                    st.plotly_chart(plot_bar_chart(t_res_sum_val,'test_type','count',"Conclusive Test Results",color_col='test_result',barmode='group',height=400,y_is_count=True,text_format='d'),use_container_width=True)
                else: st.caption("No conclusive test result data.")
        st.markdown("---"); st.subheader("Overall Test Positivity Rate Trend (e.g., Malaria RDT)")
        mal_key = "RDT-Malaria"; mal_disp = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_key,{}).get("display_name",mal_key)
        mal_df_source = analytics_df_display[(analytics_df_display.get('test_type')==mal_key)&(~analytics_df_display.get('test_result',pd.Series(dtype=str)).isin(['Pending','Rejected Sample','Unknown']))].copy()
        if not mal_df_source.empty and 'encounter_date' in mal_df_source.columns:
            mal_df_source.loc[:,'is_positive'] = mal_df_source['test_result']=='Positive'
            if not pd.api.types.is_datetime64_ns_dtype(mal_df_source['encounter_date']): mal_df_source.loc[:,'encounter_date']=pd.to_datetime(mal_df_source['encounter_date'],errors='coerce')
            mal_df_source.dropna(subset=['encounter_date'],inplace=True)
            if not mal_df_source.empty:
                weekly_pos_rate = get_trend_data(mal_df_source,'is_positive','encounter_date','W-Mon','mean')*100
                if not weekly_pos_rate.empty: st.plotly_chart(plot_annotated_line_chart(weekly_pos_rate,f"Weekly {mal_disp} Positivity Rate",y_axis_title="Positivity (%)",height=350,target_line=app_config.TARGET_MALARIA_POSITIVITY_RATE,y_is_count=False),use_container_width=True)
                else: st.caption(f"No data for {mal_disp} trend.")
        else: st.caption(f"No {mal_disp} data for trend.")


with tab_systems_equity:
    st.header("Health Systems Context & Equity Insights")
    if analytics_df_display.empty: st.info("No data for systems/equity analysis with current filters.")
    else:
        sys_eq_cols_final_val = st.columns(2)
        with sys_eq_cols_final_val[0]:
            st.subheader("Encounters by Clinic ID (Top 10)")
            if 'clinic_id' in analytics_df_display.columns and analytics_df_display['clinic_id'].notna().any():
                fac_load_val = analytics_df_display['clinic_id'].value_counts().nlargest(10).reset_index(); fac_load_val.columns=['Clinic ID','Encounters']
                if not fac_load_val.empty: st.plotly_chart(plot_bar_chart(fac_load_val,'Clinic ID','Encounters',"Top 10 Clinics by Encounter Volume",height=400,orientation='h',y_is_count=True,text_format='d'),use_container_width=True)
        with sys_eq_cols_final_val[1]:
            st.subheader("Referral Status Distribution")
            if 'referral_status' in analytics_df_display.columns and analytics_df_display['referral_status'].notna().any():
                ref_data_val = analytics_df_display[analytics_df_display['referral_status'].str.lower().isin(['pending','completed','initiated','service provided','attended','missed appointment','declined'])].copy()
                if not ref_data_val.empty:
                    ref_counts_val=ref_data_val['referral_status'].value_counts().reset_index();ref_counts_val.columns=['Referral Status','Count']
                    if not ref_counts_val.empty: st.plotly_chart(plot_donut_chart(ref_counts_val,'Referral Status','Count',"Referral Statuses",height=400,values_are_counts=True),use_container_width=True)
                else: st.caption("No actionable referral data after filtering.")
        st.markdown("---"); st.subheader("AI Risk Score Distribution by Zone Socio-Economic Index")
        if all(c in analytics_df_display for c in ['zone_id','ai_risk_score']) and not zone_attr_df_pop.empty and all(c in zone_attr_df_pop for c in ['socio_economic_index','name']):
            avg_risk_zone_eq_final_val = analytics_df_display.groupby('zone_id')['ai_risk_score'].mean().reset_index()
            zone_attr_copy_eq_final_val = zone_attr_df_pop.copy(); zone_attr_copy_eq_final_val['zone_id']=zone_attr_copy_eq_final_val['zone_id'].astype(str); avg_risk_zone_eq_final_val['zone_id']=avg_risk_zone_eq_final_val['zone_id'].astype(str)
            equity_plot_df_val = zone_attr_copy_eq_final_val.merge(avg_risk_zone_eq_final_val, on='zone_id', how='inner')
            equity_plot_df_val.dropna(subset=['ai_risk_score','socio_economic_index'],inplace=True)
            if not equity_plot_df_val.empty:
                fig_eq_risk_ses_val = px.scatter(equity_plot_df_val,x='socio_economic_index',y='ai_risk_score',text='name',size='population' if 'population' in equity_plot_df_val else None,color='ai_risk_score',title="Avg. Patient AI Risk vs. Zone Socio-Economic Index",labels={'socio_economic_index':"SES Index (Zone)",'ai_risk_score':"Avg. Patient AI Risk (Zone)"},height=450,color_continuous_scale="Reds",hover_name='name')
                fig_eq_risk_ses_val.update_traces(textposition='top center'); st.plotly_chart(fig_eq_risk_ses_val,use_container_width=True)
            else: st.caption("Not enough data to plot AI Risk vs Zone SES.")
        else: st.caption("Required data for AI Risk vs. Zone SES analysis not fully available.")

st.markdown("---"); st.caption(app_config.APP_FOOTER)
