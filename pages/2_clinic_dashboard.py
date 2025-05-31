# test/pages/2_clinic_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
from datetime import date
import numpy as np

# Assuming flat import structure for this 'test' setup
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    load_iot_clinic_environment_data,
    get_clinic_summary,
    get_clinic_environmental_summary,
    get_trend_data,
    get_supply_forecast_data,
    get_patient_alerts_for_clinic
)
from utils.ai_analytics_engine import (
    apply_ai_models,
    SupplyForecastingModel
)
from utils.ui_visualization_helpers import (
    render_kpi_card,
    plot_donut_chart,
    plot_annotated_line_chart,
    plot_bar_chart
)

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Clinic Dashboard - Health Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_clinic():
    css_path = app_config.STYLE_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info("Clinic Dashboard: CSS loaded successfully.")
    else:
        logger.warning(f"Clinic Dashboard: CSS file not found at {css_path}.")
load_css_clinic()

# --- Data Loading and AI Enrichment ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading clinic operational data...")
def get_clinic_dashboard_data_enriched():
    logger.info("Clinic Dashboard: Attempting to load and enrich health records and IoT data...")
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    iot_df = load_iot_clinic_environment_data(file_path=app_config.IOT_CLINIC_ENVIRONMENT_CSV)

    if health_df_raw.empty:
        logger.error("Clinic Dashboard: Failed to load health records from core processing.")
        return pd.DataFrame(), iot_df if iot_df is not None else pd.DataFrame()

    health_df_ai_enriched = apply_ai_models(health_df_raw)
    if health_df_ai_enriched.empty and not health_df_raw.empty:
        logger.warning("Clinic Dashboard: AI enrichment failed, using raw health data.")
        health_df_ai_enriched = health_df_raw
    elif health_df_ai_enriched.empty:
        logger.error("Clinic Dashboard: Raw load and AI enrichment resulted in empty health DF.")
        return pd.DataFrame(), iot_df if iot_df is not None else pd.DataFrame()

    logger.info(f"Clinic Dashboard: Loaded {len(health_df_ai_enriched)} AI-enriched health records and {len(iot_df) if iot_df is not None else 0} IoT records.")
    return health_df_ai_enriched, iot_df if iot_df is not None else pd.DataFrame()

health_df_clinic_main, iot_df_clinic_main = get_clinic_dashboard_data_enriched()

# --- Main Page Rendering ---
health_data_available = health_df_clinic_main is not None and not health_df_clinic_main.empty
iot_data_available = iot_df_clinic_main is not None and not iot_df_clinic_main.empty

if not health_data_available:
    st.error("🚨 **Critical Error:** Health records data is unavailable or failed to process. Most Clinic Dashboard features will not function. Please check data sources, configurations, and AI processing steps.")
    logger.critical("Clinic Dashboard cannot render meaningfully: health_df_clinic_main is None or empty.")

st.title("🏥 Clinic Operations & Environmental Dashboard")
st.markdown("**Monitoring Service Efficiency, Quality of Care, Resource Management, and Facility Environment**")
st.markdown("---")

# --- Sidebar Filters & Date Range Setup ---
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, use_column_width='auto')
    st.sidebar.markdown("---")
else:
    logger.warning(f"Sidebar logo not found on Clinic Dashboard at {app_config.APP_LOGO}")

st.sidebar.header("🗓️ Clinic Filters")

all_potential_timestamps_clinic = []
if health_data_available and 'encounter_date' in health_df_clinic_main.columns:
    encounter_dates_ts = pd.to_datetime(health_df_clinic_main['encounter_date'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(encounter_dates_ts):
        encounter_dates_ts = encounter_dates_ts.dt.tz_localize(None)
    all_potential_timestamps_clinic.extend(encounter_dates_ts.dropna())

if iot_data_available and 'timestamp' in iot_df_clinic_main.columns:
    iot_timestamps_ts = pd.to_datetime(iot_df_clinic_main['timestamp'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(iot_timestamps_ts):
        iot_timestamps_ts = iot_timestamps_ts.dt.tz_localize(None)
    all_potential_timestamps_clinic.extend(iot_timestamps_ts.dropna())

all_valid_timestamps_clinic = [ts for ts in all_potential_timestamps_clinic if isinstance(ts, pd.Timestamp)]

if all_valid_timestamps_clinic:
    min_ts_data_clinic = min(all_valid_timestamps_clinic)
    max_ts_data_clinic = max(all_valid_timestamps_clinic)
    min_date_data_clinic = min_ts_data_clinic.date()
    max_date_data_clinic = max_ts_data_clinic.date()
    logger.info(f"Clinic date range from data: {min_date_data_clinic} to {max_date_data_clinic}")
else:
    logger.warning("No valid dates found in health or IoT data for Clinic dashboard. Using default fallback date range.")
    min_date_data_clinic = date.today() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND * 3)
    max_date_data_clinic = date.today()

if min_date_data_clinic > max_date_data_clinic:
    min_date_data_clinic = max_date_data_clinic

default_end_val_clinic = max_date_data_clinic
default_start_val_clinic = max_date_data_clinic - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if default_start_val_clinic < min_date_data_clinic:
    default_start_val_clinic = min_date_data_clinic

selected_start_date_cl, selected_end_date_cl = st.sidebar.date_input(
    "Select Date Range for Analysis:",
    value=[default_start_val_clinic, default_end_val_clinic],
    min_value=min_date_data_clinic,
    max_value=max_date_data_clinic,
    key="clinic_dashboard_date_range_selector_v11",
    help="This date range applies to most charts and Key Performance Indicators (KPIs)."
)

# --- Filter dataframes based on selected date range (Corrected for TypeError) ---
filtered_health_df_clinic = pd.DataFrame()
filtered_iot_df_clinic = pd.DataFrame()

if health_data_available:
    health_df_clinic_main['encounter_date'] = pd.to_datetime(health_df_clinic_main['encounter_date'], errors='coerce')
    health_df_clinic_main['encounter_date_obj'] = health_df_clinic_main['encounter_date'].dt.date
    valid_dates_for_filtering_health = health_df_clinic_main['encounter_date_obj'].notna()
    df_health_to_filter = health_df_clinic_main[valid_dates_for_filtering_health].copy()
    if not df_health_to_filter.empty:
        filtered_health_df_clinic = df_health_to_filter[
            (df_health_to_filter['encounter_date_obj'] >= selected_start_date_cl) &
            (df_health_to_filter['encounter_date_obj'] <= selected_end_date_cl)
        ].copy()
    else:
        filtered_health_df_clinic = pd.DataFrame(columns=health_df_clinic_main.columns) # Ensure schema if empty
    logger.info(f"Clinic Health Data: Filtered from {selected_start_date_cl} to {selected_end_date_cl}, resulting in {len(filtered_health_df_clinic)} encounters.")
    if filtered_health_df_clinic.empty and health_df_clinic_main[health_df_clinic_main['encounter_date'].notna()].shape[0] > 0:
        st.info(f"ℹ️ No health encounter data available for the selected period: {selected_start_date_cl.strftime('%d %b %Y')} to {selected_end_date_cl.strftime('%d %b %Y')}.")

if iot_data_available:
    iot_df_clinic_main['timestamp'] = pd.to_datetime(iot_df_clinic_main['timestamp'], errors='coerce')
    iot_df_clinic_main['timestamp_date_obj'] = iot_df_clinic_main['timestamp'].dt.date
    valid_dates_for_filtering_iot = iot_df_clinic_main['timestamp_date_obj'].notna()
    df_iot_to_filter = iot_df_clinic_main[valid_dates_for_filtering_iot].copy()
    if not df_iot_to_filter.empty:
        filtered_iot_df_clinic = df_iot_to_filter[
            (df_iot_to_filter['timestamp_date_obj'] >= selected_start_date_cl) &
            (df_iot_to_filter['timestamp_date_obj'] <= selected_end_date_cl)
        ].copy()
    else:
        filtered_iot_df_clinic = pd.DataFrame(columns=iot_df_clinic_main.columns) # Ensure schema if empty
    logger.info(f"Clinic IoT Data: Filtered from {selected_start_date_cl} to {selected_end_date_cl}, resulting in {len(filtered_iot_df_clinic)} records.")
    if filtered_iot_df_clinic.empty and iot_df_clinic_main[iot_df_clinic_main['timestamp'].notna()].shape[0] > 0:
         st.info(f"ℹ️ No IoT environmental data available for the selected period: {selected_start_date_cl.strftime('%d %b %Y')} to {selected_end_date_cl.strftime('%d %b %Y')}.")
elif not iot_data_available:
    st.info("ℹ️ IoT environmental data is currently unavailable for this system. Environment monitoring section will be skipped.")

# --- Display KPIs ---
date_range_display_str = f"({selected_start_date_cl.strftime('%d %b %Y')} - {selected_end_date_cl.strftime('%d %b %Y')})"
clinic_service_kpis = get_clinic_summary(filtered_health_df_clinic if not filtered_health_df_clinic.empty else pd.DataFrame(columns=health_df_clinic_main.columns if health_data_available else []))
if not filtered_health_df_clinic.empty:
    logger.debug(f"Clinic Service KPIs calculated for period: {clinic_service_kpis}")
else:
    logger.info("Filtered health data is empty, clinic service KPIs will show defaults or N/A.")

st.subheader(f"Overall Clinic Performance Summary {date_range_display_str}")
kpi_cols_main_clinic = st.columns(4)
with kpi_cols_main_clinic[0]:
    overall_tat_val = clinic_service_kpis.get('overall_avg_test_turnaround', 0.0)
    render_kpi_card("Overall Avg. TAT", f"{overall_tat_val:.1f}d", "⏱️", status="High" if overall_tat_val > (app_config.TARGET_TEST_TURNAROUND_DAYS + 1) else ("Moderate" if overall_tat_val > app_config.TARGET_TEST_TURNAROUND_DAYS else "Low"), help_text=f"Average TAT for all conclusive tests. Target: ≤{app_config.TARGET_TEST_TURNAROUND_DAYS} days.")
with kpi_cols_main_clinic[1]:
    perc_met_tat_val = clinic_service_kpis.get('overall_perc_met_tat', 0.0)
    render_kpi_card("% Critical Tests TAT Met", f"{perc_met_tat_val:.1f}%", "🎯",status="Good High" if perc_met_tat_val >= app_config.TARGET_OVERALL_TESTS_MEETING_TAT_PCT else "Bad Low", help_text=f"Critical tests meeting TAT. Target: ≥{app_config.TARGET_OVERALL_TESTS_MEETING_TAT_PCT}%")
with kpi_cols_main_clinic[2]:
    pending_crit_val = clinic_service_kpis.get('total_pending_critical_tests',0); render_kpi_card("Pending Critical Tests", str(pending_crit_val),"⏳", status="High" if pending_crit_val > 10 else "Moderate", help_text="Critical tests awaiting results.")
with kpi_cols_main_clinic[3]:
    rejection_rate_val = clinic_service_kpis.get('sample_rejection_rate',0.0); render_kpi_card("Sample Rejection Rate", f"{rejection_rate_val:.1f}%", "🚫",status="High" if rejection_rate_val > app_config.TARGET_SAMPLE_REJECTION_RATE_PCT else "Low",help_text=f"Overall sample rejection rate. Target: <{app_config.TARGET_SAMPLE_REJECTION_RATE_PCT}%")

st.markdown("##### Disease-Specific Test Positivity Rates (Selected Period)")
test_details_for_kpis = clinic_service_kpis.get("test_summary_details", {})
kpi_cols_disease_pos = st.columns(4)

tb_gx_key = next((k for k, v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if "genexpert" in v.get("display_name", "").lower()), "Sputum-GeneXpert")
tb_gx_display_name = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(tb_gx_key, {}).get("display_name", "TB GeneXpert")
with kpi_cols_disease_pos[0]:
    tb_pos_rate = test_details_for_kpis.get(tb_gx_display_name, {}).get("positive_rate",0.0)
    render_kpi_card(f"{tb_gx_display_name} Pos.", f"{tb_pos_rate:.1f}%", "🫁", status="High" if tb_pos_rate > 10 else "Moderate")
mal_rdt_key = next((k for k, v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if "rdt-malaria" in k.lower()), "RDT-Malaria")
mal_rdt_display_name = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_rdt_key, {}).get("display_name", "Malaria RDT")
with kpi_cols_disease_pos[1]:
    mal_pos_rate = test_details_for_kpis.get(mal_rdt_display_name, {}).get("positive_rate",0.0)
    render_kpi_card(f"{mal_rdt_display_name} Pos.", f"{mal_pos_rate:.1f}%", "🦟", status="High" if mal_pos_rate > app_config.TARGET_MALARIA_POSITIVITY_RATE else "Low")
hiv_rapid_key = next((k for k, v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if "hiv-rapid" in k.lower()), "HIV-Rapid")
hiv_rapid_display_name = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(hiv_rapid_key, {}).get("display_name", "HIV Rapid Test")
with kpi_cols_disease_pos[2]:
    hiv_pos_rate = test_details_for_kpis.get(hiv_rapid_display_name, {}).get("positive_rate",0.0)
    render_kpi_card(f"{hiv_rapid_display_name} Pos.", f"{hiv_pos_rate:.1f}%", "🩸", status="Moderate" if hiv_pos_rate > 2 else "Low")
with kpi_cols_disease_pos[3]:
    drug_stockouts_val = clinic_service_kpis.get('key_drug_stockouts',0); render_kpi_card("Key Drug Stockouts", str(drug_stockouts_val), "💊", status="High" if drug_stockouts_val > 0 else "Low", help_text=f"Key drugs with <{app_config.CRITICAL_SUPPLY_DAYS} days supply.")

if not filtered_iot_df_clinic.empty:
    st.subheader(f"Clinic Environment Snapshot {date_range_display_str}")
    clinic_env_kpis = get_clinic_environmental_summary(filtered_iot_df_clinic); logger.debug(f"Clinic Env KPIs: {clinic_env_kpis}")
    kpi_cols_env_clinic = st.columns(4)
    with kpi_cols_env_clinic[0]: avg_co2 = clinic_env_kpis.get('avg_co2_overall',0); co2_alert = clinic_env_kpis.get('rooms_co2_alert_latest',0); render_kpi_card("Avg. CO2",f"{avg_co2:.0f} ppm","💨",status="High" if co2_alert > 0 else "Low", help_text=f"Period Avg. {co2_alert} room(s) > {app_config.CO2_LEVEL_ALERT_PPM}ppm now.")
    with kpi_cols_env_clinic[1]: avg_pm25 = clinic_env_kpis.get('avg_pm25_overall',0); pm25_alert = clinic_env_kpis.get('rooms_pm25_alert_latest',0); render_kpi_card("Avg. PM2.5",f"{avg_pm25:.1f} µg/m³","🌫️",status="High" if pm25_alert > 0 else "Low",help_text=f"Period Avg. {pm25_alert} room(s) > {app_config.PM25_ALERT_UGM3}µg/m³ now.")
    with kpi_cols_env_clinic[2]: avg_occ = clinic_env_kpis.get('avg_occupancy_overall',0); occ_alert = clinic_env_kpis.get('high_occupancy_alert_latest',False); render_kpi_card("Avg. Occupancy",f"{avg_occ:.1f} ppl","👨‍👩‍👧‍👦",status="High" if occ_alert else "Low",help_text=f"Avg Waiting Room Occupancy. Target < {app_config.TARGET_WAITING_ROOM_OCCUPANCY}. Alert if any high.")
    with kpi_cols_env_clinic[3]: noise_alert = clinic_env_kpis.get('rooms_noise_alert_latest',0); render_kpi_card("Noise Alerts",str(noise_alert),"🔊",status="High" if noise_alert > 0 else "Low",help_text=f"Rooms > {app_config.NOISE_LEVEL_ALERT_DB}dB now.")
elif iot_data_available:
    st.info("No IoT data for selected period for Environmental KPIs.")
st.markdown("---")

tab_titles_clinic = ["🔬 Detailed Testing Insights", "💊 Supply Chain Management", "🧍 Patient Focus & Alerts", "🌿 Clinic Environment Details"]
tab_tests, tab_supplies, tab_patients, tab_environment = st.tabs(tab_titles_clinic)

with tab_tests:
    st.subheader("🔬 In-depth Laboratory Testing Performance & Trends")
    if filtered_health_df_clinic.empty:
        st.info("No health data available for the selected period to display detailed testing insights.")
    else:
        detailed_test_stats_tab = clinic_service_kpis.get("test_summary_details", {})
        if not detailed_test_stats_tab:
             st.warning("No detailed test summary statistics could be generated. This might be due to missing test data or configuration issues in `app_config.KEY_TEST_TYPES_FOR_ANALYSIS`.")
        else:
            active_test_groups = [k for k,v in detailed_test_stats_tab.items() if v.get('total_conducted_conclusive',0) > 0 or v.get('pending_count',0) > 0 or v.get('rejected_count',0) > 0]
            critical_test_exists_in_config = any(props.get("critical", False) for props in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.values())
            
            test_group_options_tab = []
            if critical_test_exists_in_config : test_group_options_tab.append("All Critical Tests Summary")
            test_group_options_tab.extend(sorted(active_test_groups))

            if not test_group_options_tab:
                 st.info("No test groups with activity or critical tests defined for detailed analysis in this period.")
            else:
                selected_test_group_display = st.selectbox("Focus on Test Group/Type:", options=test_group_options_tab, key="clinic_test_group_select_tab_v2")
                st.markdown("---")

                if selected_test_group_display == "All Critical Tests Summary":
                    st.markdown("###### **Performance Metrics for All Critical Tests (Period Average)**")
                    crit_test_table_data = []
                    for group_disp_name, stats in detailed_test_stats_tab.items(): # Corrected: use detailed_test_stats_tab
                        original_group_key = next((k for k, v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == group_disp_name), None)
                        if original_group_key and app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(original_group_key, {}).get("critical"):
                            crit_test_table_data.append({"Test Group": group_disp_name, "Positivity (%)": stats.get("positive_rate", 0.0), "Avg. TAT (Days)": stats.get("avg_tat_days", 0.0), "% Met TAT Target": stats.get("perc_met_tat_target", 0.0), "Pending Count": stats.get("pending_count", 0), "Rejected Count": stats.get("rejected_count", 0), "Total Conclusive": stats.get("total_conducted_conclusive", 0)})
                    if crit_test_table_data: st.dataframe(pd.DataFrame(crit_test_table_data), use_container_width=True, hide_index=True, column_config={"Positivity (%)": st.column_config.NumberColumn(format="%.1f%%"), "Avg. TAT (Days)":st.column_config.NumberColumn(format="%.1f"),"% Met TAT Target":st.column_config.ProgressColumn(format="%.1f%%",min_value=0,max_value=100)})
                    else: st.caption("No data for critical tests in this period or no critical tests configured.")
                
                elif selected_test_group_display in detailed_test_stats_tab: # Corrected: check against detailed_test_stats_tab
                    stats_selected_group = detailed_test_stats_tab[selected_test_group_display]
                    st.markdown(f"###### **Detailed Metrics for: {selected_test_group_display}**")
                    kpi_cols_test_detail_tab = st.columns(5);
                    with kpi_cols_test_detail_tab[0]: render_kpi_card("Positivity Rate", f"{stats_selected_group.get('positive_rate',0):.1f}%", "➕");
                    with kpi_cols_test_detail_tab[1]: render_kpi_card("Avg. TAT", f"{stats_selected_group.get('avg_tat_days',0):.1f}d", "⏱️");
                    with kpi_cols_test_detail_tab[2]: render_kpi_card("% Met TAT Target", f"{stats_selected_group.get('perc_met_tat_target',0):.1f}%", "🎯");
                    with kpi_cols_test_detail_tab[3]: render_kpi_card("Pending Tests", f"{stats_selected_group.get('pending_count',0)}", "⏳");
                    with kpi_cols_test_detail_tab[4]: render_kpi_card("Rejected Samples", f"{stats_selected_group.get('rejected_count',0)}", "🚫");
                    
                    plot_cols_test_detail_tab = st.columns(2)
                    original_key_for_selected = next((k for k,v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == selected_test_group_display), None)
                    
                    if original_key_for_selected:
                        cfg_selected_test = app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_key_for_selected]
                        actual_test_types_for_plot = cfg_selected_test.get("types_in_group", [original_key_for_selected])
                        if isinstance(actual_test_types_for_plot, str): actual_test_types_for_plot = [actual_test_types_for_plot]
                        target_tat_for_plot = cfg_selected_test.get("target_tat_days", app_config.TARGET_TEST_TURNAROUND_DAYS)

                        with plot_cols_test_detail_tab[0]:
                            st.markdown(f"**Daily Avg. TAT for {selected_test_group_display}**")
                            df_tat_plot_src = filtered_health_df_clinic[(filtered_health_df_clinic['test_type'].isin(actual_test_types_for_plot)) & (filtered_health_df_clinic['test_turnaround_days'].notna()) & (~filtered_health_df_clinic['test_result'].isin(['Pending','Unknown','Rejected Sample', 'Indeterminate']))].copy()
                            if not df_tat_plot_src.empty:
                                tat_trend_plot = get_trend_data(df_tat_plot_src, 'test_turnaround_days', period='D', date_col='encounter_date', agg_func='mean')
                                if not tat_trend_plot.empty: st.plotly_chart(plot_annotated_line_chart(tat_trend_plot, f"Avg. TAT Trend", y_axis_title="Days", target_line=target_tat_for_plot, target_label=f"Target {target_tat_for_plot}d", height=app_config.COMPACT_PLOT_HEIGHT-20, date_format="%d %b"), use_container_width=True)
                                else: st.caption("No aggregated TAT trend data.")
                            else: st.caption("No conclusive tests with TAT data for this group.")
                        with plot_cols_test_detail_tab[1]:
                            st.markdown(f"**Daily Test Volume for {selected_test_group_display}**")
                            df_vol_plot_src = filtered_health_df_clinic[filtered_health_df_clinic['test_type'].isin(actual_test_types_for_plot)].copy()
                            if not df_vol_plot_src.empty:
                                conducted_vol = get_trend_data(df_vol_plot_src[~df_vol_plot_src['test_result'].isin(['Pending','Unknown','Rejected Sample', 'Indeterminate'])], 'patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Conclusive")
                                pending_vol = get_trend_data(df_vol_plot_src[df_vol_plot_src['test_result'] == 'Pending'], 'patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Pending")
                                if not conducted_vol.empty or not pending_vol.empty:
                                    vol_trend_df = pd.concat([conducted_vol, pending_vol], axis=1).fillna(0).reset_index()
                                    date_col_melt = 'encounter_date' if 'encounter_date' in vol_trend_df.columns else ('date' if 'date' in vol_trend_df.columns else vol_trend_df.columns[0])
                                    vol_melt_df = vol_trend_df.melt(id_vars=date_col_melt, value_vars=['Conclusive', 'Pending'], var_name='Status', value_name='Count')
                                    st.plotly_chart(plot_bar_chart(vol_melt_df, x_col=date_col_melt, y_col='Count', color_col='Status', title=f"Daily Volume Trend", barmode='stack', height=app_config.COMPACT_PLOT_HEIGHT-20), use_container_width=True)
                                else: st.caption("No volume data.")
                            else: st.caption(f"No tests matching '{selected_test_group_display}' for volume trend.")
                    else: st.warning(f"Configuration for '{selected_test_group_display}' not found to display trends.")
                else: st.info(f"No activity data found for test group: '{selected_test_group_display}' in this period.")
        
        st.markdown("---"); st.markdown("###### **Overdue Pending Tests (All test types, older than their target TAT + buffer)**")
        op_df_source_clinic = filtered_health_df_clinic.copy()
        date_col_for_pending_calc = 'sample_collection_date' if 'sample_collection_date' in op_df_source_clinic.columns and op_df_source_clinic['sample_collection_date'].notna().any() else 'encounter_date'
        overdue_df_clinic = op_df_source_clinic[(op_df_source_clinic['test_result'] == 'Pending') & (op_df_source_clinic[date_col_for_pending_calc].notna())].copy()
        if not overdue_df_clinic.empty:
            overdue_df_clinic['days_pending_calc'] = (pd.Timestamp('today').normalize() - pd.to_datetime(overdue_df_clinic[date_col_for_pending_calc])).dt.days
            def get_specific_overdue_threshold(test_type_name_or_disp): # Parameter name should match test_type in df (which are keys)
                test_config = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(test_type_name_or_disp) # Direct lookup if test_type_name_or_disp is a key
                if not test_config: # Fallback if test_type column stores display_name (less ideal)
                    original_key = next((k for k,v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == test_type_name_or_disp), None);
                    if original_key: test_config = app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_key]
                buffer_days = 2
                return (test_config.get('target_tat_days', app_config.OVERDUE_PENDING_TEST_DAYS) if test_config else app_config.OVERDUE_PENDING_TEST_DAYS) + buffer_days
            overdue_df_clinic['effective_overdue_days'] = overdue_df_clinic['test_type'].apply(get_specific_overdue_threshold)
            overdue_df_final_display_clinic = overdue_df_clinic[overdue_df_clinic['days_pending_calc'] > overdue_df_clinic['effective_overdue_days']]
            if not overdue_df_final_display_clinic.empty: st.dataframe(overdue_df_final_display_clinic[['patient_id', 'test_type', date_col_for_pending_calc, 'days_pending_calc', 'effective_overdue_days']].sort_values('days_pending_calc', ascending=False).head(10), column_config={date_col_for_pending_calc:st.column_config.DateColumn("Sample/Encounter Date"), "days_pending_calc":st.column_config.NumberColumn("Days Pending",format="%d"), "effective_overdue_days":st.column_config.NumberColumn("Overdue If > (days)",format="%d")}, height=300, use_container_width=True)
            else: st.success(f"✅ No tests pending longer than their target TAT + buffer.")
        else: st.caption("No pending tests to evaluate for overdue status.")

        if 'sample_status' in filtered_health_df_clinic.columns and 'rejection_reason' in filtered_health_df_clinic.columns:
            st.markdown("---"); st.markdown("###### **Sample Rejection Analysis (Period)**")
            rejected_samples_df_tab_clinic = filtered_health_df_clinic[filtered_health_df_clinic['sample_status'] == 'Rejected'].copy()
            if not rejected_samples_df_tab_clinic.empty:
                rejection_reason_counts_clinic = rejected_samples_df_tab_clinic['rejection_reason'].value_counts().reset_index(); rejection_reason_counts_clinic.columns = ['Rejection Reason', 'Count']
                col_rej_donut_cl, col_rej_table_cl = st.columns([0.45, 0.55])
                with col_rej_donut_cl:
                    if not rejection_reason_counts_clinic.empty: st.plotly_chart(plot_donut_chart(rejection_reason_counts_clinic, 'Rejection Reason', 'Count', "Top Sample Rejection Reasons", height=app_config.COMPACT_PLOT_HEIGHT + 20), use_container_width=True)
                    else: st.caption("No rejection reason data.")
                with col_rej_table_cl:
                    st.caption("Rejected Samples List (Top 10 in Period)"); st.dataframe(rejected_samples_df_tab_clinic[['patient_id', 'test_type', 'encounter_date', 'rejection_reason']].head(10), height=280, use_container_width=True)
            else: st.info("✅ No rejected samples recorded in this period.")

with tab_supplies:
    st.subheader("💊 Medical Supply Levels & Consumption Forecast")
    use_ai_forecast = st.checkbox("Use Advanced AI Supply Forecast (Beta)", value=False, key="clinic_ai_supply_forecast_toggle")
    if health_data_available and not health_df_clinic_main.empty and all(c in health_df_clinic_main.columns for c in ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']):
        supply_forecast_df = SupplyForecastingModel().forecast_supply_levels_advanced(health_df_clinic_main, forecast_days_out=30) if use_ai_forecast else get_supply_forecast_data(health_df_clinic_main, forecast_days_out=30)
        if supply_forecast_df is not None and not supply_forecast_df.empty:
            key_drug_items_for_select = sorted(list(supply_forecast_df['item'].unique()))
            if not key_drug_items_for_select: st.info("No forecast data for supply items based on historical data.")
            else:
                default_select_options = [item for item in key_drug_items_for_select if any(sub.lower() in str(item).lower() for sub in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY)]
                default_selection_idx = key_drug_items_for_select.index(default_select_options[0]) if default_select_options and default_select_options[0] in key_drug_items_for_select else 0
                selected_drug_for_forecast = st.selectbox("Select Item for Forecast Details:", key_drug_items_for_select, index=default_selection_idx, key="clinic_supply_item_forecast_selector_v11")
                if selected_drug_for_forecast:
                    item_specific_forecast_df = supply_forecast_df[supply_forecast_df['item'] == selected_drug_for_forecast].copy()
                    if not item_specific_forecast_df.empty:
                        item_specific_forecast_df.sort_values('date', inplace=True)
                        current_info = item_specific_forecast_df.iloc[0]
                        forecast_title = (f"Forecast: {selected_drug_for_forecast}<br><sup_>Stock@Start: {current_info.get('current_stock',0):.0f} | Base Use: {current_info.get('consumption_rate',0):.1f}/d | Est. Stockout: {pd.to_datetime(current_info.get('estimated_stockout_date')).strftime('%d %b %Y') if pd.notna(current_info.get('estimated_stockout_date')) else 'N/A'}</sup>")
                        plot_series = item_specific_forecast_df.set_index('date')['forecast_days']
                        lc_series, uc_series = (item_specific_forecast_df.set_index('date').get('lower_ci'), item_specific_forecast_df.set_index('date').get('upper_ci')) if not use_ai_forecast else (None, None)
                        st.plotly_chart(plot_annotated_line_chart(data_series=plot_series, title=forecast_title, y_axis_title="Forecasted Days of Supply", target_line=app_config.CRITICAL_SUPPLY_DAYS, target_label=f"Critical ({app_config.CRITICAL_SUPPLY_DAYS} Days)", show_ci=(lc_series is not None and not lc_series.empty), lower_bound_series=lc_series, upper_bound_series=uc_series, height=app_config.DEFAULT_PLOT_HEIGHT + 60, show_anomalies=False), use_container_width=True)
                        if use_ai_forecast: st.caption("*Advanced forecast uses a simulated AI model.*")
                    else: st.info(f"No forecast data for {selected_drug_for_forecast}.")
        else: st.warning("Supply forecast could not be generated.")
    elif not health_data_available: st.warning("Supply forecasts require health records data.")
    else: st.error("Health data missing essential columns for supply forecasts.")

with tab_patients:
    st.subheader("🧍 Patient Load & High-Risk Case Identification (Period)")
    if not filtered_health_df_clinic.empty:
        if all(c in filtered_health_df_clinic.columns for c in ['condition', 'encounter_date', 'patient_id']):
            conditions_load_chart = app_config.KEY_CONDITIONS_FOR_TRENDS
            load_src_df = filtered_health_df_clinic[filtered_health_df_clinic['condition'].isin(conditions_load_chart) & (filtered_health_df_clinic['patient_id'].astype(str).str.lower() != 'unknown')].copy()
            if not load_src_df.empty:
                daily_load_summary = load_src_df.groupby([pd.Grouper(key='encounter_date', freq='D'), 'condition'])['patient_id'].nunique().reset_index()
                daily_load_summary.rename(columns={'patient_id': 'unique_patients', 'encounter_date':'date'}, inplace=True)
                if not daily_load_summary.empty: st.plotly_chart(plot_bar_chart(daily_load_summary, x_col='date', y_col='unique_patients', title="Daily Unique Patient Encounters by Key Condition", color_col='condition', barmode='stack', height=app_config.DEFAULT_PLOT_HEIGHT + 70, y_axis_title="Unique Patients/Day", x_axis_title="Date", color_discrete_map=app_config.DISEASE_COLORS, text_auto=False ), use_container_width=True)
                else: st.caption("No patient load data for key conditions in period.")
            else: st.caption("No patients with key conditions in period.")
        else: st.info("Patient Load chart: Missing essential columns.")
        st.markdown("---"); st.markdown("###### **Flagged Patient Cases for Clinical Review (Selected Period)**")
        flagged_patients_clinic_review_df = get_patient_alerts_for_clinic(filtered_health_df_clinic, risk_threshold_moderate=app_config.RISK_THRESHOLDS['moderate'])
        if flagged_patients_clinic_review_df is not None and not flagged_patients_clinic_review_df.empty:
            st.markdown(f"Found **{len(flagged_patients_clinic_review_df)}** unique patient encounters flagged for review.")
            cols_for_alert_table_clinic = ['patient_id', 'encounter_date', 'condition', 'ai_risk_score', 'ai_followup_priority_score', 'alert_reason', 'test_result', 'test_type', 'hiv_viral_load_copies_ml', 'min_spo2_pct', 'priority_score']
            existing_cols_for_display_alerts = [col for col in cols_for_alert_table_clinic if col in flagged_patients_clinic_review_df.columns]
            alerts_display_df = flagged_patients_clinic_review_df[existing_cols_for_display_alerts].copy()
            
            if 'priority_score' in alerts_display_df.columns:
                alerts_display_df_sorted = alerts_display_df.sort_values(by='priority_score', ascending=False)
            else:
                alerts_display_df_sorted = alerts_display_df 

            # --- CONVERSION FOR st.dataframe COMPATIBILITY (JSON Serializable) ---
            df_to_display_streamlit_alerts = alerts_display_df_sorted.head(25).copy()
            for col_alert_disp in df_to_display_streamlit_alerts.columns:
                if df_to_display_streamlit_alerts[col_alert_disp].dtype == 'object' and col_alert_disp != 'encounter_date':
                    df_to_display_streamlit_alerts[col_alert_disp] = df_to_display_streamlit_alerts[col_alert_disp].astype(str).replace(['nan', 'None', '<NA>'], 'N/A', regex=False)
                elif df_to_display_streamlit_alerts[col_alert_disp].dtype == 'object' and col_alert_disp == 'encounter_date':
                     df_to_display_streamlit_alerts[col_alert_disp] = pd.to_datetime(df_to_display_streamlit_alerts[col_alert_disp], errors='coerce')
                # Ensure encounter_date is tz-naive for DateColumn if it's still tz-aware
                if col_alert_disp == 'encounter_date' and pd.api.types.is_datetime64tz_dtype(df_to_display_streamlit_alerts[col_alert_disp]):
                    df_to_display_streamlit_alerts[col_alert_disp] = df_to_display_streamlit_alerts[col_alert_disp].dt.tz_localize(None)


            st.dataframe(df_to_display_streamlit_alerts, use_container_width=True, column_config={ "encounter_date": st.column_config.DateColumn("Encounter Date", format="YYYY-MM-DD"), "ai_risk_score": st.column_config.ProgressColumn("AI Risk", format="%d", min_value=0, max_value=100), "ai_followup_priority_score": st.column_config.ProgressColumn("AI Prio.", format="%d", min_value=0, max_value=100), "priority_score": st.column_config.NumberColumn("Overall Alert Prio.", format="%d"), "alert_reason": st.column_config.TextColumn("Alert Reason(s)", width="large"), "hiv_viral_load_copies_ml": st.column_config.NumberColumn("HIV VL (cp/mL)", format="%.0f"), "min_spo2_pct": st.column_config.NumberColumn("Min SpO2 (%)", format="%d%%"), }, height=450, hide_index=True )
        else: st.info("No specific patient cases flagged for clinical review in period.")
    else: st.info("No health data for selected period for Patient Load or alerts.")

with tab_environment:
    st.subheader("🌿 Clinic Environmental Monitoring - Trends & Details")
    if not filtered_iot_df_clinic.empty:
        env_summary = get_clinic_environmental_summary(filtered_iot_df_clinic)
        st.markdown(f"""**Current Env. Alerts (latest in period):** CO2: {env_summary.get('rooms_co2_alert_latest',0)} rooms > {app_config.CO2_LEVEL_ALERT_PPM}ppm. PM2.5: {env_summary.get('rooms_pm25_alert_latest',0)} rooms > {app_config.PM25_ALERT_UGM3}µg/m³. Noise: {env_summary.get('rooms_noise_alert_latest',0)} rooms > {app_config.NOISE_LEVEL_ALERT_DB}dB.""")
        if env_summary.get('high_occupancy_alert_latest', False): st.warning(f"⚠️ **High Waiting Room Occupancy Detected:** > {app_config.TARGET_WAITING_ROOM_OCCUPANCY} persons.")
        env_trend_cols = st.columns(2)
        with env_trend_cols[0]:
            if 'avg_co2_ppm' in filtered_iot_df_clinic.columns:
                co2_trend = get_trend_data(filtered_iot_df_clinic, 'avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean')
                if not co2_trend.empty: st.plotly_chart(plot_annotated_line_chart(co2_trend, "Hourly Avg. CO2 Levels", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, height=app_config.COMPACT_PLOT_HEIGHT, date_format="%d %b, %H:%M"), use_container_width=True)
                else: st.caption("No CO2 trend data.")
        with env_trend_cols[1]:
            if 'waiting_room_occupancy' in filtered_iot_df_clinic.columns:
                occ_trend = get_trend_data(filtered_iot_df_clinic, 'waiting_room_occupancy', date_col='timestamp', period='H', agg_func='mean')
                if not occ_trend.empty: st.plotly_chart(plot_annotated_line_chart(occ_trend, "Hourly Avg. Waiting Room Occupancy", y_axis_title="Persons", target_line=app_config.TARGET_WAITING_ROOM_OCCUPANCY, height=app_config.COMPACT_PLOT_HEIGHT, date_format="%d %b, %H:%M"), use_container_width=True)
                else: st.caption("No occupancy trend data.")
        st.markdown("---"); st.subheader("Latest Sensor Readings by Room (End of Selected Period)")
        latest_cols = ['clinic_id', 'room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour']
        avail_cols = [col for col in latest_cols if col in filtered_iot_df_clinic.columns]
        if all(c in avail_cols for c in ['timestamp', 'clinic_id', 'room_name']):
            latest_room_reads = filtered_iot_df_clinic.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
            if not latest_room_reads.empty: st.dataframe(latest_room_reads[avail_cols].tail(15), use_container_width=True, height=380, column_config={"timestamp": st.column_config.DatetimeColumn("Last Reading", format="YYYY-MM-DD HH:mm"), "avg_co2_ppm": st.column_config.NumberColumn("CO2", format="%dppm"), "avg_pm25": st.column_config.NumberColumn("PM2.5", format="%.1fµg/m³"), "avg_temp_celsius": st.column_config.NumberColumn("Temp", format="%.1f°C"), "avg_humidity_rh": st.column_config.NumberColumn("Hum.", format="%d%%"), "avg_noise_db": st.column_config.NumberColumn("Noise", format="%ddB"), "waiting_room_occupancy": st.column_config.NumberColumn("Occup.", format="%d P"), "patient_throughput_per_hour": st.column_config.NumberColumn("Thrpt/hr", format="%.1f"), "sanitizer_dispenses_per_hour": st.column_config.NumberColumn("Sanit./hr", format="%.1f")}, hide_index=True)
            else: st.caption("No distinct room sensor readings in period.")
        else: st.caption("Essential IoT columns for room view missing.")
    elif iot_data_available: st.info("No clinic environmental IoT data for selected period.")
