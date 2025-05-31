# test/pages/2_clinic_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
from datetime import date
import numpy as np # Imported for np.nan usage

# Assuming flat import structure for this 'test' setup
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    load_iot_clinic_environment_data,
    get_clinic_summary, # This now returns detailed test stats
    get_clinic_environmental_summary,
    get_trend_data,
    get_supply_forecast_data, # Linear forecast from core
    get_patient_alerts_for_clinic
)
from utils.ai_analytics_engine import (
    apply_ai_models, # To get AI scores
    SupplyForecastingModel # For advanced AI-based supply forecast (optional to use)
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
def load_css_clinic(): # Renamed for clarity
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

all_dates_clinic = []
if health_data_available and 'encounter_date' in health_df_clinic_main.columns:
    all_dates_clinic.extend(pd.to_datetime(health_df_clinic_main['encounter_date'], errors='coerce').dropna())
if iot_data_available and 'timestamp' in iot_df_clinic_main.columns:
    all_dates_clinic.extend(pd.to_datetime(iot_df_clinic_main['timestamp'], errors='coerce').dropna())

min_date_data_clinic = min(all_dates_clinic).date() if all_dates_clinic else date.today() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND * 3)
max_date_data_clinic = max(all_dates_clinic).date() if all_dates_clinic else date.today()

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
    key="clinic_dashboard_date_range_selector_v10",
    help="This date range applies to most charts and Key Performance Indicators (KPIs)."
)

# --- Filter dataframes based on selected date range (Corrected as per Error 1 fix) ---
filtered_health_df_clinic = pd.DataFrame()
filtered_iot_df_clinic = pd.DataFrame()

if health_data_available:
    temp_dates_health = pd.to_datetime(health_df_clinic_main['encounter_date'], errors='coerce')
    health_df_clinic_main['encounter_date_obj'] = pd.NaT
    valid_date_mask_health = temp_dates_health.notna()
    health_df_clinic_main.loc[valid_date_mask_health, 'encounter_date_obj'] = temp_dates_health[valid_date_mask_health].dt.date

    date_filter_mask_health = (
        health_df_clinic_main['encounter_date_obj'].notna() &
        (health_df_clinic_main['encounter_date_obj'] >= selected_start_date_cl) &
        (health_df_clinic_main['encounter_date_obj'] <= selected_end_date_cl)
    )
    filtered_health_df_clinic = health_df_clinic_main[date_filter_mask_health].copy()
    logger.info(f"Clinic Health Data: Filtered from {selected_start_date_cl} to {selected_end_date_cl}, resulting in {len(filtered_health_df_clinic)} encounters.")
    if filtered_health_df_clinic.empty and health_df_clinic_main[valid_date_mask_health].shape[0] > 0 :
        st.info(f"ℹ️ No health encounter data available for the selected period: {selected_start_date_cl.strftime('%d %b %Y')} to {selected_end_date_cl.strftime('%d %b %Y')}.")

if iot_data_available:
    temp_dates_iot = pd.to_datetime(iot_df_clinic_main['timestamp'], errors='coerce')
    iot_df_clinic_main['timestamp_date_obj'] = pd.NaT
    valid_date_mask_iot = temp_dates_iot.notna()
    iot_df_clinic_main.loc[valid_date_mask_iot, 'timestamp_date_obj'] = temp_dates_iot[valid_date_mask_iot].dt.date

    date_filter_mask_iot = (
        iot_df_clinic_main['timestamp_date_obj'].notna() &
        (iot_df_clinic_main['timestamp_date_obj'] >= selected_start_date_cl) &
        (iot_df_clinic_main['timestamp_date_obj'] <= selected_end_date_cl)
    )
    filtered_iot_df_clinic = iot_df_clinic_main[date_filter_mask_iot].copy()
    logger.info(f"Clinic IoT Data: Filtered from {selected_start_date_cl} to {selected_end_date_cl}, resulting in {len(filtered_iot_df_clinic)} records.")
    if filtered_iot_df_clinic.empty and iot_df_clinic_main[valid_date_mask_iot].shape[0] > 0:
         st.info(f"ℹ️ No IoT environmental data available for the selected period: {selected_start_date_cl.strftime('%d %b %Y')} to {selected_end_date_cl.strftime('%d %b %Y')}.")
elif not iot_data_available:
    st.info("ℹ️ IoT environmental data is currently unavailable for this system. Environment monitoring section will be skipped.")

# --- Display KPIs ---
date_range_display_str = f"({selected_start_date_cl.strftime('%d %b %Y')} - {selected_end_date_cl.strftime('%d %b %Y')})"
clinic_service_kpis = get_clinic_summary(filtered_health_df_clinic if not filtered_health_df_clinic.empty else pd.DataFrame())
if not filtered_health_df_clinic.empty:
    logger.debug(f"Clinic Service KPIs calculated for period: {clinic_service_kpis}")
else:
    logger.info("Filtered health data is empty, clinic service KPIs will show defaults or N/A.")

st.subheader(f"Overall Clinic Performance Summary {date_range_display_str}")
kpi_cols_main_clinic = st.columns(4)
with kpi_cols_main_clinic[0]:
    overall_tat_val = clinic_service_kpis.get('overall_avg_test_turnaround', 0.0)
    render_kpi_card("Overall Avg. TAT", f"{overall_tat_val:.1f}d", "⏱️",
                    status="High" if overall_tat_val > (app_config.TARGET_TEST_TURNAROUND_DAYS + 1) else ("Moderate" if overall_tat_val > app_config.TARGET_TEST_TURNAROUND_DAYS else "Low"),
                    help_text=f"Average TAT for all conclusive tests in the period. Target: ≤{app_config.TARGET_TEST_TURNAROUND_DAYS} days.")
with kpi_cols_main_clinic[1]:
    perc_met_tat_val = clinic_service_kpis.get('overall_perc_met_tat', 0.0)
    render_kpi_card("% Critical Tests TAT Met", f"{perc_met_tat_val:.1f}%", "🎯",
                    status="Good High" if perc_met_tat_val >= app_config.TARGET_OVERALL_TESTS_MEETING_TAT_PCT else "Bad Low",
                    help_text=f"Percentage of critical tests meeting their specific TAT targets. Target: ≥{app_config.TARGET_OVERALL_TESTS_MEETING_TAT_PCT}%.")
with kpi_cols_main_clinic[2]:
    pending_crit_val = clinic_service_kpis.get('total_pending_critical_tests', 0)
    render_kpi_card("Pending Critical Tests", str(pending_crit_val), "⏳",
                    status="High" if pending_crit_val > 10 else ("Moderate" if pending_crit_val > 0 else "Low"),
                    help_text="Number of unique patients with critical tests still awaiting results.")
with kpi_cols_main_clinic[3]:
    rejection_rate_val = clinic_service_kpis.get('sample_rejection_rate', 0.0)
    render_kpi_card("Sample Rejection Rate", f"{rejection_rate_val:.1f}%", "🚫",
                    status="High" if rejection_rate_val > app_config.TARGET_SAMPLE_REJECTION_RATE_PCT else "Low",
                    help_text=f"Overall sample rejection rate for all tests. Target: <{app_config.TARGET_SAMPLE_REJECTION_RATE_PCT}%.")

st.markdown("##### Disease-Specific Test Positivity Rates (Selected Period)")
test_details_for_kpis = clinic_service_kpis.get("test_summary_details", {})
kpi_cols_disease_pos = st.columns(4)

tb_gx_key = next((k for k, v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if "genexpert" in v.get("display_name", "").lower()), "Sputum-GeneXpert")
tb_gx_display_name = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(tb_gx_key, {}).get("display_name", "TB GeneXpert")
with kpi_cols_disease_pos[0]:
    tb_pos_rate = test_details_for_kpis.get(tb_gx_display_name, {}).get("positive_rate", 0.0)
    render_kpi_card(f"{tb_gx_display_name} Pos.", f"{tb_pos_rate:.1f}%", "🫁", status="High" if tb_pos_rate > 10 else ("Moderate" if tb_pos_rate > 5 else "Low"))

mal_rdt_key = next((k for k, v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if "rdt-malaria" in k.lower()), "RDT-Malaria")
mal_rdt_display_name = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_rdt_key, {}).get("display_name", "Malaria RDT")
with kpi_cols_disease_pos[1]:
    mal_pos_rate = test_details_for_kpis.get(mal_rdt_display_name, {}).get("positive_rate", 0.0)
    render_kpi_card(f"{mal_rdt_display_name} Pos.", f"{mal_pos_rate:.1f}%", "🦟", status="High" if mal_pos_rate > app_config.TARGET_MALARIA_POSITIVITY_RATE else "Low")

hiv_rapid_key = next((k for k, v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if "hiv-rapid" in k.lower()), "HIV-Rapid")
hiv_rapid_display_name = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(hiv_rapid_key, {}).get("display_name", "HIV Rapid Test")
with kpi_cols_disease_pos[2]:
    hiv_pos_rate = test_details_for_kpis.get(hiv_rapid_display_name, {}).get("positive_rate", 0.0)
    render_kpi_card(f"{hiv_rapid_display_name} Pos.", f"{hiv_pos_rate:.1f}%", "🩸", status="Moderate" if hiv_pos_rate > 2 else "Low")

with kpi_cols_disease_pos[3]:
    drug_stockouts_val = clinic_service_kpis.get('key_drug_stockouts', 0)
    render_kpi_card("Key Drug Stockouts", str(drug_stockouts_val), "💊", status="High" if drug_stockouts_val > 0 else "Low",
                    help_text=f"Number of key drugs with less than {app_config.CRITICAL_SUPPLY_DAYS} days of supply remaining in the period.")

if not filtered_iot_df_clinic.empty:
    st.subheader(f"Clinic Environment Snapshot {date_range_display_str}")
    clinic_env_kpis = get_clinic_environmental_summary(filtered_iot_df_clinic)
    logger.debug(f"Clinic Env KPIs calculated: {clinic_env_kpis}")
    kpi_cols_clinic_environment = st.columns(4)
    with kpi_cols_clinic_environment[0]:
        avg_co2_val = clinic_env_kpis.get('avg_co2_overall', 0.0)
        co2_alert_rooms_val = clinic_env_kpis.get('rooms_co2_alert_latest', 0)
        render_kpi_card("Avg. CO2 (All Rooms)", f"{avg_co2_val:.0f} ppm", "💨",
                        status="High" if co2_alert_rooms_val > 0 else ("Moderate" if avg_co2_val > app_config.CO2_LEVEL_IDEAL_PPM else "Low"),
                        help_text=f"Period average CO2. {co2_alert_rooms_val} room(s) currently > {app_config.CO2_LEVEL_ALERT_PPM}ppm.")
    with kpi_cols_clinic_environment[1]:
        avg_pm25_val = clinic_env_kpis.get('avg_pm25_overall', 0.0)
        pm25_alert_rooms_val = clinic_env_kpis.get('rooms_pm25_alert_latest', 0)
        render_kpi_card("Avg. PM2.5 (All Rooms)", f"{avg_pm25_val:.1f} µg/m³", "🌫️",
                        status="High" if pm25_alert_rooms_val > 0 else ("Moderate" if avg_pm25_val > app_config.PM25_IDEAL_UGM3 else "Low"),
                        help_text=f"Period average PM2.5. {pm25_alert_rooms_val} room(s) currently > {app_config.PM25_ALERT_UGM3}µg/m³.")
    with kpi_cols_clinic_environment[2]:
        avg_occupancy_val = clinic_env_kpis.get('avg_occupancy_overall', 0.0)
        occupancy_alert_val = clinic_env_kpis.get('high_occupancy_alert_latest', False)
        render_kpi_card("Avg. Waiting Room Occupancy", f"{avg_occupancy_val:.1f} persons", "👨‍👩‍👧‍👦",
                        status="High" if occupancy_alert_val else ("Moderate" if avg_occupancy_val > (app_config.TARGET_WAITING_ROOM_OCCUPANCY * 0.75) else "Low"),
                        help_text=f"Average waiting room occupancy. Target < {app_config.TARGET_WAITING_ROOM_OCCUPANCY}. Alert if latest room occupancy is high.")
    with kpi_cols_clinic_environment[3]:
        noise_alert_rooms_val = clinic_env_kpis.get('rooms_noise_alert_latest', 0)
        render_kpi_card("High Noise Alerts", str(noise_alert_rooms_val), "🔊",
                        status="High" if noise_alert_rooms_val > 0 else "Low",
                        help_text=f"Rooms with latest noise levels > {app_config.NOISE_LEVEL_ALERT_DB}dB.")
st.markdown("---")

# --- Tabs for Detailed Analysis ---
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
            critical_test_exists = any(app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_key].get("critical") for original_key in app_config.KEY_TEST_TYPES_FOR_ANALYSIS)
            
            test_group_options_tab = []
            if critical_test_exists : test_group_options_tab.append("All Critical Tests Summary")
            test_group_options_tab.extend(sorted(active_test_groups))

            if not test_group_options_tab:
                 st.info("No test groups with activity found for detailed analysis in this period.")
            else:
                selected_test_group_display = st.selectbox(
                    "Focus on Test Group/Type:", options=test_group_options_tab,
                    key="clinic_test_group_select_tab_v2",
                    help="Select a test group for detailed metrics and trends, or view a summary for all critical tests."
                )
                st.markdown("---")

                if selected_test_group_display == "All Critical Tests Summary":
                    st.markdown("###### **Performance Metrics for All Critical Tests (Period Average)**")
                    crit_test_table_data = []
                    for group_disp_name, stats in detailed_test_stats_tab.items():
                        original_group_key = next((k for k, v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == group_disp_name), None)
                        if original_group_key and app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(original_group_key, {}).get("critical"):
                            crit_test_table_data.append({
                                "Test Group": group_disp_name, "Positivity (%)": stats.get("positive_rate", 0.0),
                                "Avg. TAT (Days)": stats.get("avg_tat_days", 0.0), "% Met TAT Target": stats.get("perc_met_tat_target", 0.0),
                                "Pending Count": stats.get("pending_count", 0), "Rejected Count": stats.get("rejected_count", 0),
                                "Total Conclusive": stats.get("total_conducted_conclusive", 0)
                            })
                    if crit_test_table_data:
                        st.dataframe(pd.DataFrame(crit_test_table_data), use_container_width=True, hide_index=True,
                                     column_config={"Positivity (%)": st.column_config.NumberColumn(format="%.1f%%"),
                                                    "Avg. TAT (Days)": st.column_config.NumberColumn(format="%.1f"),
                                                    "% Met TAT Target": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
                    else: st.caption("No data for critical tests found in this period or no critical tests defined.")

                elif selected_test_group_display in detailed_test_stats_tab:
                    stats_selected_group = detailed_test_stats_tab[selected_test_group_display]
                    st.markdown(f"###### **Detailed Metrics for: {selected_test_group_display}**")
                    
                    kpi_cols_test_detail_tab = st.columns(5)
                    with kpi_cols_test_detail_tab[0]: render_kpi_card("Positivity Rate", f"{stats_selected_group.get('positive_rate',0):.1f}%", "➕")
                    with kpi_cols_test_detail_tab[1]: render_kpi_card("Avg. TAT", f"{stats_selected_group.get('avg_tat_days',0):.1f}d", "⏱️")
                    with kpi_cols_test_detail_tab[2]: render_kpi_card("% Met TAT Target", f"{stats_selected_group.get('perc_met_tat_target',0):.1f}%", "🎯")
                    with kpi_cols_test_detail_tab[3]: render_kpi_card("Pending Tests", f"{stats_selected_group.get('pending_count',0)}", "⏳")
                    with kpi_cols_test_detail_tab[4]: render_kpi_card("Rejected Samples", f"{stats_selected_group.get('rejected_count',0)}", "🚫")

                    plot_cols_test_detail_tab = st.columns(2)
                    original_key_for_selected = next((k for k,v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == selected_test_group_display), None)
                    
                    if original_key_for_selected:
                        # For plots, use the original test_type names if they are grouped
                        actual_test_types_for_plot = app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_key_for_selected].get("types_in_group", [original_key_for_selected])
                        if isinstance(actual_test_types_for_plot, str): actual_test_types_for_plot = [actual_test_types_for_plot] # Ensure list
                        target_tat_for_plot = app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_key_for_selected].get("target_tat_days", app_config.TARGET_TEST_TURNAROUND_DAYS)

                        with plot_cols_test_detail_tab[0]:
                            st.markdown(f"**Daily Avg. TAT for {selected_test_group_display}**")
                            df_tat_plot_src = filtered_health_df_clinic[
                                (filtered_health_df_clinic['test_type'].isin(actual_test_types_for_plot)) & # Use actual test type names for filtering data
                                (filtered_health_df_clinic['test_turnaround_days'].notna()) &
                                (~filtered_health_df_clinic['test_result'].isin(['Pending','Unknown','Rejected Sample', 'Indeterminate']))
                            ].copy()
                            if not df_tat_plot_src.empty:
                                tat_trend_plot = get_trend_data(df_tat_plot_src, 'test_turnaround_days', period='D', date_col='encounter_date', agg_func='mean')
                                if not tat_trend_plot.empty:
                                    st.plotly_chart(plot_annotated_line_chart(tat_trend_plot, f"Avg. TAT Trend", y_axis_title="Days", target_line=target_tat_for_plot, target_label=f"Target {target_tat_for_plot}d", height=app_config.COMPACT_PLOT_HEIGHT-20, date_format="%d %b"), use_container_width=True)
                                else: st.caption("No aggregated TAT trend data for this test group.")
                            else: st.caption("No conclusive tests with TAT data for this group in period.")

                        with plot_cols_test_detail_tab[1]:
                            st.markdown(f"**Daily Test Volume for {selected_test_group_display}**")
                            df_vol_plot_src = filtered_health_df_clinic[filtered_health_df_clinic['test_type'].isin(actual_test_types_for_plot)].copy()
                            if not df_vol_plot_src.empty:
                                conducted_vol = get_trend_data(df_vol_plot_src[~df_vol_plot_src['test_result'].isin(['Pending','Unknown','Rejected Sample', 'Indeterminate'])], 'patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Conclusive")
                                pending_vol = get_trend_data(df_vol_plot_src[df_vol_plot_src['test_result'] == 'Pending'], 'patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Pending")
                                if not conducted_vol.empty or not pending_vol.empty:
                                    vol_trend_df = pd.concat([conducted_vol, pending_vol], axis=1).fillna(0).reset_index()
                                    # Ensure correct date column name for melt
                                    date_col_melt = 'encounter_date' if 'encounter_date' in vol_trend_df.columns else 'date'
                                    vol_melt_df = vol_trend_df.melt(id_vars=date_col_melt, value_vars=['Conclusive', 'Pending'], var_name='Status', value_name='Count')
                                    st.plotly_chart(plot_bar_chart(vol_melt_df, x_col=date_col_melt, y_col='Count', color_col='Status', title=f"Daily Volume Trend", barmode='stack', height=app_config.COMPACT_PLOT_HEIGHT-20), use_container_width=True)
                                else: st.caption("No volume data to plot.")
                            else: st.caption(f"No tests matching '{selected_test_group_display}' found in period for volume trend.")
                    else:
                        st.warning(f"Could not find configuration for '{selected_test_group_display}' in `app_config.KEY_TEST_TYPES_FOR_ANALYSIS` to display trends.")
                else:
                     st.info(f"No activity data found for test group: '{selected_test_group_display}' in this period.")
        
        st.markdown("---"); st.markdown("###### **Overdue Pending Tests (All test types, older than their target TAT + buffer)**")
        op_df_source_clinic = filtered_health_df_clinic.copy()
        date_col_for_pending_calc = 'sample_collection_date' if 'sample_collection_date' in op_df_source_clinic.columns and op_df_source_clinic['sample_collection_date'].notna().any() else 'encounter_date'
        
        overdue_df_clinic = op_df_source_clinic[(op_df_source_clinic['test_result'] == 'Pending') & (op_df_source_clinic[date_col_for_pending_calc].notna())].copy()
        if not overdue_df_clinic.empty:
            overdue_df_clinic['days_pending_calc'] = (pd.Timestamp('today').normalize() - pd.to_datetime(overdue_df_clinic[date_col_for_pending_calc])).dt.days
            
            def get_specific_overdue_threshold(test_type_name_or_disp):
                # Try direct match with key, then by display_name
                test_config = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(test_type_name_or_disp)
                if not test_config:
                    original_key = next((k for k,v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == test_type_name_or_disp), None)
                    if original_key: test_config = app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_key]

                buffer_days = 2
                if test_config: return test_config.get('target_tat_days', app_config.OVERDUE_PENDING_TEST_DAYS) + buffer_days
                return app_config.OVERDUE_PENDING_TEST_DAYS + buffer_days

            overdue_df_clinic['effective_overdue_days'] = overdue_df_clinic['test_type'].apply(get_specific_overdue_threshold)
            overdue_df_final_display_clinic = overdue_df_clinic[overdue_df_clinic['days_pending_calc'] > overdue_df_clinic['effective_overdue_days']]
            
            if not overdue_df_final_display_clinic.empty:
                st.dataframe(overdue_df_final_display_clinic[['patient_id', 'test_type', date_col_for_pending_calc, 'days_pending_calc', 'effective_overdue_days']].sort_values('days_pending_calc', ascending=False).head(10),
                             column_config={date_col_for_pending_calc:st.column_config.DateColumn("Sample/Encounter Date"),
                                            "days_pending_calc":st.column_config.NumberColumn("Days Pending",format="%d"),
                                            "effective_overdue_days":st.column_config.NumberColumn("Overdue If > (days)",format="%d")},
                             height=300, use_container_width=True)
            else:
                st.success(f"✅ No tests pending for longer than their target TAT + buffer.")
        else:
            st.caption("No pending tests found to evaluate for overdue status in this period.")

        if 'sample_status' in filtered_health_df_clinic.columns and 'rejection_reason' in filtered_health_df_clinic.columns:
            st.markdown("---"); st.markdown("###### **Sample Rejection Analysis (Period)**")
            rejected_samples_df_tab_clinic = filtered_health_df_clinic[filtered_health_df_clinic['sample_status'] == 'Rejected'].copy()
            if not rejected_samples_df_tab_clinic.empty:
                rejection_reason_counts_clinic = rejected_samples_df_tab_clinic['rejection_reason'].value_counts().reset_index()
                rejection_reason_counts_clinic.columns = ['Rejection Reason', 'Count']
                
                col_rej_donut_cl, col_rej_table_cl = st.columns([0.45, 0.55])
                with col_rej_donut_cl:
                    if not rejection_reason_counts_clinic.empty:
                         st.plotly_chart(plot_donut_chart(rejection_reason_counts_clinic, 'Rejection Reason', 'Count', "Top Sample Rejection Reasons", height=app_config.COMPACT_PLOT_HEIGHT + 20), use_container_width=True)
                    else: st.caption("No specific rejection reason data available for rejected samples.")
                with col_rej_table_cl:
                    st.caption("Rejected Samples List (Top 10 in Period)")
                    st.dataframe(rejected_samples_df_tab_clinic[['patient_id', 'test_type', 'encounter_date', 'rejection_reason']].head(10), height=280, use_container_width=True)
            else:
                st.info("✅ No rejected samples recorded in this period.")

with tab_supplies:
    st.subheader("💊 Medical Supply Levels & Consumption Forecast")
    use_ai_forecast = st.checkbox("Use Advanced AI Supply Forecast (Beta)", value=False, key="clinic_ai_supply_forecast_toggle")

    if health_data_available and not health_df_clinic_main.empty and all(c in health_df_clinic_main.columns for c in ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']):
        if use_ai_forecast:
            logger.info("Clinic Dashboard: Using AI Supply Forecasting Model.")
            supply_model_ai = SupplyForecastingModel()
            supply_forecast_df = supply_model_ai.forecast_supply_levels_advanced(health_df_clinic_main, forecast_days_out=30)
        else:
            logger.info("Clinic Dashboard: Using Linear Supply Forecasting from core_data_processing.")
            supply_forecast_df = get_supply_forecast_data(health_df_clinic_main, forecast_days_out=30)

        if supply_forecast_df is not None and not supply_forecast_df.empty:
            key_drug_items_for_select = sorted(list(supply_forecast_df['item'].unique()))
            if not key_drug_items_for_select:
                 st.info("No forecast data available for any supply items based on historical data.")
            else:
                default_select_options = [item for item in key_drug_items_for_select if any(sub.lower() in str(item).lower() for sub in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY)]
                default_selection = default_select_options[0] if default_select_options else key_drug_items_for_select[0]

                selected_drug_for_forecast = st.selectbox(
                    "Select Item for Forecast Details:", key_drug_items_for_select,
                    index=key_drug_items_for_select.index(default_selection) if default_selection in key_drug_items_for_select else 0,
                    key="clinic_supply_item_forecast_selector_v11",
                    help="View the forecasted days of supply remaining for the selected item."
                )
                if selected_drug_for_forecast:
                    item_specific_forecast_df = supply_forecast_df[supply_forecast_df['item'] == selected_drug_for_forecast].copy()
                    if not item_specific_forecast_df.empty:
                        item_specific_forecast_df.sort_values('date', inplace=True)
                        current_info_from_forecast = item_specific_forecast_df.iloc[0]
                        forecast_plot_title = (
                            f"Forecast: {selected_drug_for_forecast}<br>"
                            f"<sup_>Stock at Forecast Start: {current_info_from_forecast.get('current_stock',0):.0f} | "
                            f"Base Daily Use (Hist.): {current_info_from_forecast.get('consumption_rate',0):.1f} | " # Use 'consumption_rate' for base
                            f"Est. Stockout: {pd.to_datetime(current_info_from_forecast.get('estimated_stockout_date')).strftime('%d %b %Y') if pd.notna(current_info_from_forecast.get('estimated_stockout_date')) else 'N/A'}</sup>"
                        )
                        plot_data_series = item_specific_forecast_df.set_index('date')['forecast_days']
                        lower_ci = item_specific_forecast_df.set_index('date').get('lower_ci', None)
                        upper_ci = item_specific_forecast_df.set_index('date').get('upper_ci', None)
                        show_ci_plot = (lower_ci is not None and upper_ci is not None and not use_ai_forecast)

                        st.plotly_chart(plot_annotated_line_chart(
                            data_series=plot_data_series, title=forecast_plot_title,
                            y_axis_title="Forecasted Days of Supply",
                            target_line=app_config.CRITICAL_SUPPLY_DAYS, target_label=f"Critical Level ({app_config.CRITICAL_SUPPLY_DAYS} Days)",
                            show_ci=show_ci_plot, lower_bound_series=lower_ci, upper_bound_series=upper_ci,
                            height=app_config.DEFAULT_PLOT_HEIGHT + 60, show_anomalies=False
                        ), use_container_width=True)
                        if use_ai_forecast: st.caption("*Advanced forecast uses a simulated AI model with seasonal and trend components.*")
                    else: st.info(f"No forecast data found for the selected item: {selected_drug_for_forecast}.")
        else:
            st.warning("Supply forecast data could not be generated.")
    elif not health_data_available:
        st.warning("Supply forecasts cannot be generated as health records data is unavailable.")
    else:
        st.error("CRITICAL FOR SUPPLY TAB: Health data missing essential columns for supply forecasts.")

with tab_patients:
    st.subheader("🧍 Patient Load & High-Risk Case Identification (Period)")
    if not filtered_health_df_clinic.empty:
        if all(c in filtered_health_df_clinic.columns for c in ['condition', 'encounter_date', 'patient_id']):
            conditions_for_load_chart = app_config.KEY_CONDITIONS_FOR_TRENDS
            patient_load_df_src = filtered_health_df_clinic[
                filtered_health_df_clinic['condition'].isin(conditions_for_load_chart) &
                (filtered_health_df_clinic['patient_id'].astype(str).str.lower() != 'unknown')
            ].copy()

            if not patient_load_df_src.empty:
                daily_patient_load_summary = patient_load_df_src.groupby(
                    [pd.Grouper(key='encounter_date', freq='D'), 'condition']
                )['patient_id'].nunique().reset_index()
                daily_patient_load_summary.rename(columns={'patient_id': 'unique_patients', 'encounter_date':'date'}, inplace=True)

                if not daily_patient_load_summary.empty:
                    st.plotly_chart(plot_bar_chart(
                        daily_patient_load_summary, x_col='date', y_col='unique_patients',
                        title="Daily Unique Patient Encounters by Key Condition", color_col='condition',
                        barmode='stack', height=app_config.DEFAULT_PLOT_HEIGHT + 70,
                        y_axis_title="Unique Patients per Day", x_axis_title="Date",
                        color_discrete_map=app_config.DISEASE_COLORS, text_auto=False
                    ), use_container_width=True)
                else: st.caption("No patient load data for key conditions in selected period.")
            else: st.caption("No patients with encounters for key conditions in selected period.")
        else:
            st.info("Patient Load chart: Missing 'condition', 'encounter_date', or 'patient_id' columns.")

        st.markdown("---"); st.markdown("###### **Flagged Patient Cases for Clinical Review (Selected Period)**")
        flagged_patients_clinic_review_df = get_patient_alerts_for_clinic(
            filtered_health_df_clinic, risk_threshold_moderate=app_config.RISK_THRESHOLDS['moderate']
        )
        if flagged_patients_clinic_review_df is not None and not flagged_patients_clinic_review_df.empty:
            st.markdown(f"Found **{len(flagged_patients_clinic_review_df)}** unique patient encounters flagged for review.")
            cols_for_alert_table_clinic = ['patient_id', 'encounter_date', 'condition',
                                           'ai_risk_score', 'ai_followup_priority_score',
                                           'alert_reason', 'test_result', 'test_type',
                                           'hiv_viral_load_copies_ml', 'min_spo2_pct', 'priority_score']
            alerts_display_df_clinic = flagged_patients_clinic_review_df[[col for col in cols_for_alert_table_clinic if col in flagged_patients_clinic_review_df.columns]].copy()
            alerts_display_df_clinic_sorted = alerts_display_df_clinic.sort_values(by='priority_score', ascending=False)

            st.dataframe(alerts_display_df_clinic_sorted.head(25), use_container_width=True,
                column_config={
                    "encounter_date": st.column_config.DateColumn("Encounter Date", format="YYYY-MM-DD"),
                    "ai_risk_score": st.column_config.ProgressColumn("AI Risk", format="%d", min_value=0, max_value=100),
                    "ai_followup_priority_score": st.column_config.ProgressColumn("AI Prio.", format="%d", min_value=0, max_value=100),
                    "priority_score": st.column_config.NumberColumn("Overall Alert Prio.", format="%d"),
                    "alert_reason": st.column_config.TextColumn("Alert Reason(s)", width="large"),
                    "hiv_viral_load_copies_ml": st.column_config.NumberColumn("HIV VL (cp/mL)", format="%.0f"),
                    "min_spo2_pct": st.column_config.NumberColumn("Min SpO2 (%)", format="%d%%"),
                }, height=450, hide_index=True )
        else: st.info("No specific patient cases flagged for clinical review in selected period.")
    else:
        st.info("No health data available for selected period for Patient Load or alerts.")

with tab_environment:
    st.subheader("🌿 Clinic Environmental Monitoring - Trends & Details")
    if not filtered_iot_df_clinic.empty:
        env_summary_for_tab = get_clinic_environmental_summary(filtered_iot_df_clinic)
        st.markdown(f"""**Current Environmental Alerts (latest readings in selected period):**
        - **CO2 Alerts:** {env_summary_for_tab.get('rooms_co2_alert_latest',0)} room(s) with CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm.
        - **PM2.5 Alerts:** {env_summary_for_tab.get('rooms_pm25_alert_latest',0)} room(s) with PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³.
        - **Noise Alerts:** {env_summary_for_tab.get('rooms_noise_alert_latest',0)} room(s) with Noise > {app_config.NOISE_LEVEL_ALERT_DB}dB.""")
        if env_summary_for_tab.get('high_occupancy_alert_latest', False):
            st.warning(f"⚠️ **High Waiting Room Occupancy Detected:** At least one area > {app_config.TARGET_WAITING_ROOM_OCCUPANCY} persons.")

        env_trend_plot_cols = st.columns(2)
        with env_trend_plot_cols[0]:
            if 'avg_co2_ppm' in filtered_iot_df_clinic.columns:
                hourly_avg_co2_trend = get_trend_data(filtered_iot_df_clinic, 'avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean')
                if not hourly_avg_co2_trend.empty: st.plotly_chart(plot_annotated_line_chart(hourly_avg_co2_trend, "Hourly Avg. CO2 Levels (All Rooms)", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, target_label="Alert Threshold", height=app_config.COMPACT_PLOT_HEIGHT, show_anomalies=True, date_format="%d %b, %H:%M"), use_container_width=True)
                else: st.caption("No CO2 trend data for selected period.")
            else: st.caption("CO2 data ('avg_co2_ppm') missing.")
        with env_trend_plot_cols[1]:
            if 'waiting_room_occupancy' in filtered_iot_df_clinic.columns:
                hourly_avg_occupancy_trend = get_trend_data(filtered_iot_df_clinic, 'waiting_room_occupancy', date_col='timestamp', period='H', agg_func='mean')
                if not hourly_avg_occupancy_trend.empty: st.plotly_chart(plot_annotated_line_chart(hourly_avg_occupancy_trend, "Hourly Avg. Waiting Room Occupancy", y_axis_title="Persons", target_line=app_config.TARGET_WAITING_ROOM_OCCUPANCY, target_label="Target Occupancy", height=app_config.COMPACT_PLOT_HEIGHT, show_anomalies=True, date_format="%d %b, %H:%M"), use_container_width=True)
                else: st.caption("No occupancy trend data for selected period.")
            else: st.caption("Occupancy data ('waiting_room_occupancy') missing.")

        st.markdown("---"); st.subheader("Latest Sensor Readings by Room (End of Selected Period)")
        latest_room_cols_display = ['clinic_id', 'room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour']
        available_latest_cols_env = [col for col in latest_room_cols_display if col in filtered_iot_df_clinic.columns]
        
        if all(c in available_latest_cols_env for c in ['timestamp', 'clinic_id', 'room_name']):
            latest_room_sensor_readings = filtered_iot_df_clinic.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
            if not latest_room_sensor_readings.empty:
                st.dataframe(latest_room_sensor_readings[available_latest_cols_env].tail(15), use_container_width=True, height=380,
                    column_config={"timestamp": st.column_config.DatetimeColumn("Last Reading At", format="YYYY-MM-DD HH:mm"),
                                   "avg_co2_ppm": st.column_config.NumberColumn("CO2 (ppm)", format="%d ppm"),
                                   "avg_pm25": st.column_config.NumberColumn("PM2.5 (µg/m³)", format="%.1f µg/m³"),
                                   "avg_temp_celsius": st.column_config.NumberColumn("Temp (°C)", format="%.1f°C"),
                                   "avg_humidity_rh": st.column_config.NumberColumn("Humidity (%RH)", format="%d%%"),
                                   "avg_noise_db": st.column_config.NumberColumn("Noise (dB)", format="%d dB"),
                                   "waiting_room_occupancy": st.column_config.NumberColumn("Occupancy", format="%d persons"),
                                   "patient_throughput_per_hour": st.column_config.NumberColumn("Throughput (/hr)", format="%.1f"),
                                   "sanitizer_dispenses_per_hour": st.column_config.NumberColumn("Sanitizer Use (/hr)", format="%.1f")},
                    hide_index=True)
            else: st.caption("No distinct room sensor readings in selected period.")
        else: st.caption("Essential IoT columns (timestamp, clinic_id, room_name) missing for detailed room view.")
    elif iot_data_available:
         st.info("No clinic environmental IoT data found for the selected period.")
    # If iot_data_available is False, initial info message covers it.
