# health_hub/pages/2_clinic_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
import numpy as np 
from config import app_config 
from utils.core_data_processing import (
    load_health_records, load_iot_clinic_environment_data, 
    get_clinic_summary, get_clinic_environmental_summary,    
    get_trend_data, get_supply_forecast_data, get_patient_alerts_for_clinic 
)
from utils.ui_visualization_helpers import (
    render_kpi_card, plot_donut_chart, plot_annotated_line_chart, plot_bar_chart, _create_empty_figure
)
import plotly.express as px 

# --- Page Configuration and Styling ---
st.set_page_config(page_title="Clinic Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__) 

@st.cache_resource 
def load_css(): 
    if os.path.exists(app_config.STYLE_CSS_PATH):
        with open(app_config.STYLE_CSS_PATH) as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        logger.info("Clinic Dashboard: CSS loaded successfully.")
    else: logger.warning(f"Clinic Dashboard: CSS file not found at {app_config.STYLE_CSS_PATH}.")
load_css()

# --- Data Loading ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS) 
def get_clinic_dashboard_data():
    logger.info("Clinic Dashboard: Loading health records and IoT data...")
    health_df = load_health_records()
    iot_df = load_iot_clinic_environment_data()
    logger.debug(f"Health DF loaded shape: {health_df.shape if health_df is not None else 'None'}")
    logger.debug(f"IoT DF loaded shape: {iot_df.shape if iot_df is not None else 'None'}")
    return health_df, iot_df
health_df_clinic_main, iot_df_clinic_main = get_clinic_dashboard_data()

# --- Main Page Rendering ---
critical_data_missing_flag = False 
if health_df_clinic_main is None or health_df_clinic_main.empty:
    st.warning("⚠️ **Health records data is unavailable.** Dashboard features for patient services, testing, and supplies will be limited.")
    logger.warning("Clinic Dashboard: health_df_clinic_main is None or empty.")
    health_df_clinic_main = pd.DataFrame(columns=['date', 'item', 'condition', 'patient_id', 'test_type', 'test_result', 'test_turnaround_days', 'stock_on_hand', 'consumption_rate_per_day', 'ai_risk_score', 'sample_status', 'rejection_reason']) # Added sample status/reason
    critical_data_missing_flag = True
if iot_df_clinic_main is None or iot_df_clinic_main.empty:
    st.info("ℹ️ IoT environmental data is unavailable. Clinic environment section will be skipped.")
    logger.info("Clinic Dashboard: iot_df_clinic_main is None or empty.")
    iot_df_clinic_main = pd.DataFrame(columns=['timestamp', 'avg_co2_ppm', 'avg_pm25', 'waiting_room_occupancy', 'clinic_id', 'room_name'])
if critical_data_missing_flag and iot_df_clinic_main.empty : # Only stop if health data is missing AND critical IoT parts are missing
     st.error("🚨 **CRITICAL Error:** Essential data (Health records and potentially IoT) are unavailable. Clinic Dashboard cannot be displayed.")
     st.stop()

st.title("🏥 Clinic Operations & Environmental Dashboard")
st.markdown("**Monitoring Service Efficiency, Quality of Care, Resource Management, and Facility Environment**")
st.markdown("---")

# --- Sidebar ---
if os.path.exists(app_config.APP_LOGO): st.sidebar.image(app_config.APP_LOGO, use_column_width='auto'); st.sidebar.markdown("---")
else: logger.warning(f"Sidebar logo not found at {app_config.APP_LOGO}")
st.sidebar.header("🗓️ Clinic Filters") 
all_potential_dates_cl = []
default_min_date_cl = pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND * 3); default_max_date_cl = pd.Timestamp('today').date()
def safe_extract_timestamps(df, col_name, log_df_name="DF"):
    _timestamps = []
    if df is not None and col_name in df.columns and not df.empty:
        s_dates = pd.Series(df[col_name]); dt_s = pd.to_datetime(s_dates, errors='coerce'); valid_ts = dt_s.dropna()
        if not valid_ts.empty: _timestamps.extend(valid_ts.tolist()); logger.debug(f"{log_df_name}: Extracted {len(valid_ts)} timestamps from '{col_name}'.")
    return _timestamps
all_potential_dates_cl.extend(safe_extract_timestamps(health_df_clinic_main, 'date', "HealthDF_Clinic"))
all_potential_dates_cl.extend(safe_extract_timestamps(iot_df_clinic_main, 'timestamp', "IoTDF_Clinic"))
valid_ts_list_cl = [d for d in all_potential_dates_cl if isinstance(d, pd.Timestamp)]

min_date_data_cl = default_min_date_cl; max_date_data_cl = default_max_date_cl
default_start_val_cl_sidebar = default_min_date_cl; default_end_val_cl_sidebar = default_max_date_cl

if valid_ts_list_cl:
    try:
        combined_ts_series_cl = pd.Series(valid_ts_list_cl).drop_duplicates().sort_values(ignore_index=True)
        if not combined_ts_series_cl.empty:
            min_date_ts_cl = combined_ts_series_cl.iloc[0]; max_date_ts_cl = combined_ts_series_cl.iloc[-1]
            min_date_data_cl = min_date_ts_cl.date(); max_date_data_clinic = max_date_ts_cl.date() # Corrected max var name
            default_end_val_cl_sidebar = max_date_data_clinic 
            default_start_val_cl_sidebar = max_date_data_clinic - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
            if default_start_val_cl_sidebar < min_date_data_cl: default_start_val_cl_sidebar = min_date_data_cl
            logger.info(f"Clinic Sidebar: Date filter range determined: {min_date_data_cl} to {max_date_data_clinic}. Default: {default_start_val_cl_sidebar} to {default_end_val_cl_sidebar}")
        else: logger.warning("Clinic Sidebar: Combined TS series empty. Using fallback.")
    except Exception as e_date_cl: logger.error(f"Clinic Sidebar: Error determining date range: {e_date_cl}. Fallback.", exc_info=True)
else: logger.warning("Clinic Sidebar: No valid TS for date filter. Fallback.")

# Ensure robust clamping of default dates
if default_start_val_cl_sidebar < min_date_data_cl: default_start_val_cl_sidebar = min_date_data_cl
if default_end_val_cl_sidebar > max_date_data_clinic: default_end_val_cl_sidebar = max_date_data_clinic # Use corrected max var
if default_start_val_cl_sidebar > default_end_val_cl_sidebar: default_start_val_cl_sidebar = default_end_val_cl_sidebar

selected_start_date_cl, selected_end_date_cl = st.sidebar.date_input("Select Date Range:", value=[default_start_val_cl_sidebar, default_end_val_cl_sidebar], min_value=min_date_data_cl, max_value=max_date_data_clinic, key="clinic_date_range_final_v9", help="Applies to most KPIs and charts.")

# Filter dataframes based on selected date range more robustly
filtered_health_df_clinic = pd.DataFrame(columns=health_df_clinic_main.columns); filtered_iot_df_clinic = pd.DataFrame(columns=iot_df_clinic_main.columns)
if selected_start_date_cl and selected_end_date_cl:
    if health_df_clinic_main is not None and 'date' in health_df_clinic_main.columns and not health_df_clinic_main.empty:
        df_h = health_df_clinic_main.copy(); df_h['date'] = pd.to_datetime(df_h['date'], errors='coerce'); df_h.dropna(subset=['date'], inplace=True)
        if not df_h.empty: df_h['date_obj'] = df_h['date'].dt.date; filtered_health_df_clinic = df_h[(df_h['date_obj'] >= selected_start_date_cl) & (df_h['date_obj'] <= selected_end_date_cl) & (df_h['date_obj'].notna())].copy()
    if iot_df_clinic_main is not None and 'timestamp' in iot_df_clinic_main.columns and not iot_df_clinic_main.empty:
        df_i = iot_df_clinic_main.copy(); df_i['timestamp'] = pd.to_datetime(df_i['timestamp'], errors='coerce'); df_i.dropna(subset=['timestamp'], inplace=True)
        if not df_i.empty: df_i['date_obj'] = df_i['timestamp'].dt.date; filtered_iot_df_clinic = df_i[(df_i['date_obj'] >= selected_start_date_cl) & (df_i['date_obj'] <= selected_end_date_cl) & (df_i['date_obj'].notna())].copy()

date_range_display_str = f"({selected_start_date_cl.strftime('%d %b %Y')} - {selected_end_date_cl.strftime('%d %b %Y')})" if selected_start_date_cl and selected_end_date_cl else "(Date range not set)"
st.subheader(f"Overall Clinic Performance Summary {date_range_display_str}")
if filtered_health_df_clinic.empty and not critical_data_missing_flag : st.info("No health data in selected period for service metrics.")
clinic_service_kpis = get_clinic_summary(filtered_health_df_clinic); logger.debug(f"Main KPIs from get_clinic_summary: {clinic_service_kpis}")

kpi_cols_main_clinic = st.columns(4)
with kpi_cols_main_clinic[0]: overall_tat_val = clinic_service_kpis.get('overall_avg_test_turnaround',0.0); render_kpi_card("Overall Avg. TAT", f"{overall_tat_val:.1f}d", "⏱️", status="High" if overall_tat_val > (app_config.TARGET_TEST_TURNAROUND_DAYS +1) else "Moderate" if overall_tat_val > app_config.TARGET_TEST_TURNAROUND_DAYS else "Low", help_text="Avg TAT for all conclusive tests.")
with kpi_cols_main_clinic[1]: perc_met_tat_val = clinic_service_kpis.get('overall_perc_met_tat',0.0); render_kpi_card("% Critical Tests TAT Met", f"{perc_met_tat_val:.1f}%", "🎯",status="Good High" if perc_met_tat_val >= app_config.TARGET_OVERALL_TESTS_MEETING_TAT_PCT else "Bad Low", help_text=f"Critical tests meeting TAT. Target: ≥{app_config.TARGET_OVERALL_TESTS_MEETING_TAT_PCT}%")
with kpi_cols_main_clinic[2]: pending_crit_val = clinic_service_kpis.get('total_pending_critical_tests',0); render_kpi_card("Pending Critical Tests", str(pending_crit_val),"⏳", status="High" if pending_crit_val > 10 else "Moderate", help_text="Critical tests awaiting results.")
with kpi_cols_main_clinic[3]: rejection_rate_val = clinic_service_kpis.get('sample_rejection_rate',0.0); render_kpi_card("Sample Rejection Rate", f"{rejection_rate_val:.1f}%", "🚫",status="High" if rejection_rate_val > app_config.TARGET_SAMPLE_REJECTION_RATE_PCT else "Low",help_text=f"Overall sample rejection rate. Target: <{app_config.TARGET_SAMPLE_REJECTION_RATE_PCT}%")

# Disease Specific KPIs (moved down for better flow)
st.markdown("##### Disease-Specific Test Positivity Rates (Selected Period)")
kpi_cols_disease_pos = st.columns(4) # Example for 4 key diseases
test_details_for_kpis = clinic_service_kpis.get("test_summary_details", {})

with kpi_cols_disease_pos[0]: 
    tb_pos_rate = test_details_for_kpis.get(app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("Sputum-GeneXpert",{}).get("display_name", "TB GeneXpert"), {}).get("positive_rate",0.0)
    render_kpi_card("TB (GeneXpert) Pos.", f"{tb_pos_rate:.1f}%", "🫁", status="High" if tb_pos_rate > 10 else "Moderate")
with kpi_cols_disease_pos[1]: 
    mal_pos_rate = test_details_for_kpis.get(app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("RDT-Malaria",{}).get("display_name", "Malaria RDT"), {}).get("positive_rate",0.0)
    render_kpi_card("Malaria (RDT) Pos.", f"{mal_pos_rate:.1f}%", "🦟", status="High" if mal_pos_rate > app_config.TARGET_MALARIA_POSITIVITY_RATE else "Low")
with kpi_cols_disease_pos[2]: 
    hiv_pos_rate = test_details_for_kpis.get(app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get("HIV-Rapid",{}).get("display_name", "HIV Rapid Test"), {}).get("positive_rate",0.0)
    render_kpi_card("HIV (Rapid) Pos.", f"{hiv_pos_rate:.1f}%", "🦠", status="Moderate" if hiv_pos_rate > 2 else "Low") # Example pos threshold for HIV rapid
with kpi_cols_disease_pos[3]:
    drug_stockouts_val = clinic_service_kpis.get('key_drug_stockouts',0); render_kpi_card("Key Drug Stockouts", str(drug_stockouts_val), "💊", status="High" if drug_stockouts_val > 0 else "Low", help_text=f"Key drugs with <{app_config.CRITICAL_SUPPLY_DAYS} days supply.")

if not filtered_iot_df_clinic.empty:
    st.subheader(f"Clinic Environment Snapshot {date_range_display_str}") # Moved env KPIs here too
    clinic_env_kpis = get_clinic_environmental_summary(filtered_iot_df_clinic); logger.debug(f"Clinic Env KPIs: {clinic_env_kpis}")
    kpi_cols_env_clinic = st.columns(4)
    with kpi_cols_env_clinic[0]: avg_co2 = clinic_env_kpis.get('avg_co2_overall',0); co2_alert = clinic_env_kpis.get('rooms_co2_alert_latest',0); render_kpi_card("Avg. CO2",f"{avg_co2:.0f} ppm","💨",status="High" if co2_alert > 0 else "Low", help_text=f"Period Avg. {co2_alert} room(s) > {app_config.CO2_LEVEL_ALERT_PPM}ppm now.")
    with kpi_cols_env_clinic[1]: avg_pm25 = clinic_env_kpis.get('avg_pm25_overall',0); pm25_alert = clinic_env_kpis.get('rooms_pm25_alert_latest',0); render_kpi_card("Avg. PM2.5",f"{avg_pm25:.1f} µg/m³","🌫️",status="High" if pm25_alert > 0 else "Low",help_text=f"Period Avg. {pm25_alert} room(s) > {app_config.PM25_ALERT_UGM3}µg/m³ now.")
    with kpi_cols_env_clinic[2]: avg_occ = clinic_env_kpis.get('avg_occupancy_overall',0); occ_alert = clinic_env_kpis.get('high_occupancy_alert_latest',False); render_kpi_card("Avg. Occupancy",f"{avg_occ:.1f} ppl","👨‍👩‍👧‍👦",status="High" if occ_alert else "Low",help_text=f"Avg Waiting Room Occupancy. Target < {app_config.TARGET_WAITING_ROOM_OCCUPANCY}. Alert if any high.")
    with kpi_cols_env_clinic[3]: noise_alert = clinic_env_kpis.get('rooms_noise_alert_latest',0); render_kpi_card("Noise Alerts",str(noise_alert),"🔊",status="High" if noise_alert > 0 else "Low",help_text=f"Rooms > {app_config.NOISE_LEVEL_ALERT_DB}dB now.")
elif iot_df_clinic_main is not None and not iot_df_clinic_main.empty: st.info("No IoT data for selected period for Environmental KPIs.")
st.markdown("---")

tab_titles_clinic = ["🔬 Detailed Testing Insights", "💊 Supply Chain", "🧍 Patient Alerts", "🌿 Clinic Environment Details"]
tab_tests, tab_supplies, tab_patients, tab_environment = st.tabs(tab_titles_clinic)

with tab_tests:
    st.subheader("🔬 In-depth Laboratory Testing Performance & Trends")
    if filtered_health_df_clinic.empty: st.info("No health data available for the selected period to display detailed testing insights."); 
    else:
        detailed_test_stats_tab = clinic_service_kpis.get("test_summary_details", {}) # Already calculated
        test_group_options_tab = ["All Critical Tests"] + sorted(list(detailed_test_stats_tab.keys())) if detailed_test_stats_tab else ["All Critical Tests"]
        selected_test_group_display = st.selectbox("Focus on Test Group/Type:", options=test_group_options_tab, key="clinic_test_group_select_tab", help="Select a test group for detailed metrics and trends.")
        
        st.markdown("---")
        data_for_selected_group_exists = False

        if selected_test_group_display == "All Critical Tests":
            st.markdown("###### **Performance Metrics for All Critical Tests**")
            crit_test_table_data = []
            for group_disp_name, stats in detailed_test_stats.items():
                original_group_key = next((k for k, v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == group_disp_name), None)
                if original_group_key and app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_group_key].get("critical"):
                    crit_test_table_data.append({"Test Group": group_disp_name, "Positivity (%)": stats.get("positive_rate",0.0), "Avg. TAT (Days)": stats.get("avg_tat_days",0.0), "% Met TAT Target": stats.get("perc_met_tat_target",0.0), "Pending": stats.get("pending_count",0), "Rejected": stats.get("rejected_count",0)})
            if crit_test_table_data: st.dataframe(pd.DataFrame(crit_test_table_data), use_container_width=True, hide_index=True, column_config={"Positivity (%)": st.column_config.NumberColumn(format="%.1f%%"), "Avg. TAT (Days)":st.column_config.NumberColumn(format="%.1f"),"% Met TAT Target":st.column_config.ProgressColumn(format="%.1f%%",min_value=0,max_value=100)})
            else: st.caption("No data for critical tests in this period.")
        
        elif selected_test_group_display in detailed_test_stats:
            stats_selected_group = detailed_test_stats[selected_test_group_display]
            st.markdown(f"###### **Detailed Metrics for: {selected_test_group_display}**")
            kpi_cols_test_detail = st.columns(5);
            with kpi_cols_test_detail[0]: render_kpi_card("Positivity", f"{stats_selected_group.get('positive_rate',0):.1f}%", "➕");
            with kpi_cols_test_detail[1]: render_kpi_card("Avg. TAT", f"{stats_selected_group.get('avg_tat_days',0):.1f}d", "⏱️");
            with kpi_cols_test_detail[2]: render_kpi_card("% Met TAT", f"{stats_selected_group.get('perc_met_tat_target',0):.1f}%", "🎯");
            with kpi_cols_test_detail[3]: render_kpi_card("Pending", f"{stats_selected_group.get('pending_count',0)}", "⏳");
            with kpi_cols_test_detail[4]: render_kpi_card("Rejected", f"{stats_selected_group.get('rejected_count',0)}", "🚫");
            
            data_for_selected_group_exists = True # Mark that we have specific group data
            
            plot_cols_test_detail = st.columns(2)
            with plot_cols_test_detail[0]: # TAT Trend for selected test group
                st.markdown(f"**Daily Avg. TAT for {selected_test_group_display}**")
                # Find original key test types for this display name
                original_group_key_plot = next((k for k,v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == selected_test_group_display), None)
                if original_group_key_plot:
                    test_types_for_plot = app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_group_key_plot].get("types", [original_group_key_plot])
                    if isinstance(test_types_for_plot, str): test_types_for_plot = [test_types_for_plot]
                    
                    df_tat_plot_src = filtered_health_df_clinic[filtered_health_df_clinic['test_type'].isin(test_types_for_plot) & filtered_health_df_clinic['test_turnaround_days'].notna() & (~filtered_health_df_clinic['test_result'].isin(['Pending','Unknown','Rejected Sample']))].copy()
                    if not df_tat_plot_src.empty:
                        tat_trend_plot = get_trend_data(df_tat_plot_src, 'test_turnaround_days', period='D', date_col='date', agg_func='mean')
                        if not tat_trend_plot.empty:
                            target_tat_for_plot = app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_group_key_plot].get("target_tat_days", app_config.TARGET_TEST_TURNAROUND_DAYS)
                            st.plotly_chart(plot_annotated_line_chart(tat_trend_plot, f"Avg. TAT Trend", y_axis_title="Days", target_line=target_tat_for_plot, height=app_config.COMPACT_PLOT_HEIGHT-20, date_format="%d %b"), use_container_width=True)
                        else: st.caption("No TAT trend to plot.")
                    else: st.caption("No TAT data for this test group in period.")
                else: st.caption(f"Configuration for '{selected_test_group_display}' not found for TAT trend.")

            with plot_cols_test_detail[1]: # Volume Trend
                st.markdown(f"**Daily Test Volume for {selected_test_group_display}**")
                if original_group_key_plot: # Reuse from above
                    df_vol_plot_src = filtered_health_df_clinic[filtered_health_df_clinic['test_type'].isin(test_types_for_plot)].copy()
                    if not df_vol_plot_src.empty:
                        conducted = get_trend_data(df_vol_plot_src[~df_vol_plot_src['test_result'].isin(['Pending','Unknown','Rejected Sample'])], 'patient_id', date_col='date', period='D', agg_func='count').rename("Conducted")
                        pending = get_trend_data(df_vol_plot_src[df_vol_plot_src['test_result'] == 'Pending'], 'patient_id', date_col='date', period='D', agg_func='count').rename("Pending")
                        if not conducted.empty or not pending.empty:
                            vol_trend_df = pd.concat([conducted, pending], axis=1).fillna(0).reset_index()
                            vol_melt_df = vol_trend_df.melt(id_vars='date', value_vars=['Conducted', 'Pending'], var_name='Status', value_name='Count')
                            st.plotly_chart(plot_bar_chart(vol_melt_df, x_col='date', y_col='Count', color_col='Status', title=f"Volume Trend", height=app_config.COMPACT_PLOT_HEIGHT-20), use_container_width=True)
                        else: st.caption("No volume data to plot.")
                    else: st.caption(f"No tests of type {selected_test_group_display} to plot volume.")


        if not data_for_selected_group_exists and selected_test_group_display != "All Critical Tests":
             st.info(f"No specific summary found for test group: '{selected_test_group_display}'. This might be due to no tests of this type in the period, or a configuration mismatch for its display name in KEY_TEST_TYPES_FOR_ANALYSIS.")

        st.markdown("---"); st.markdown("###### **Overdue Pending Tests (All tests in current date range)**")
        op_df_source = filtered_health_df_clinic # Base for overdue tests
        overdue_df = op_df_source[(op_df_source['test_result'] == 'Pending') & (op_df_source['test_date'].notna())].copy()
        if not overdue_df.empty:
            overdue_df['days_pending_calc'] = (pd.Timestamp('today').normalize() - overdue_df['test_date']).dt.days
            overdue_df_final_display = overdue_df[overdue_df['days_pending_calc'] > app_config.OVERDUE_PENDING_TEST_DAYS]
            if not overdue_df_final_display.empty: st.dataframe(overdue_df_final_display[['patient_id', 'test_type', 'test_date', 'days_pending_calc']].sort_values('days_pending_calc', ascending=False).head(10), column_config={"test_date":st.column_config.DateColumn("Sample Date"), "days_pending_calc":st.column_config.NumberColumn("Days Pending",format="%d")}, height=250, use_container_width=True)
            else: st.success(f"✅ No tests pending for >{app_config.OVERDUE_PENDING_TEST_DAYS} days.")
        else: st.caption("No pending tests to evaluate for overdue status.")

        if 'sample_status' in filtered_health_df_clinic.columns and 'rejection_reason' in filtered_health_df_clinic.columns:
            st.markdown("---"); st.markdown("###### **Sample Rejection Analysis**")
            rejected_samples_df_tab = filtered_health_df_clinic[filtered_health_df_clinic['sample_status'] == 'Rejected'].copy()
            if not rejected_samples_df_tab.empty:
                rejection_reason_counts = rejected_samples_df_tab['rejection_reason'].value_counts().reset_index()
                rejection_reason_counts.columns = ['Rejection Reason', 'Count']
                col_rej_donut, col_rej_table = st.columns([0.4, 0.6])
                with col_rej_donut: 
                    if not rejection_reason_counts.empty:
                         st.plotly_chart(plot_donut_chart(rejection_reason_counts, 'Rejection Reason', 'Count', "Sample Rejection Reasons", height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                    else: st.caption("No rejection reason data.")
                with col_rej_table:
                    st.caption("Rejected Samples List (Top 10)")
                    st.dataframe(rejected_samples_df_tab[['patient_id', 'test_type', 'date', 'rejection_reason']].head(10), height=260, use_container_width=True)
            else: st.info("✅ No rejected samples recorded in this period.")


with tab_supplies: # Remains largely as before
    # ... (Supply tab content) ...
    st.subheader("Medical Supply Levels & Consumption Forecast")
    if health_df_clinic_main is not None and not health_df_clinic_main.empty and all(c in health_df_clinic_main.columns for c in ['item', 'date', 'stock_on_hand', 'consumption_rate_per_day']):
        supply_forecast_df = get_supply_forecast_data(health_df_clinic_main, forecast_days_out=28)
        if supply_forecast_df is not None and not supply_forecast_df.empty:
            key_drug_items_for_select = sorted([item for item in supply_forecast_df['item'].unique() if any(sub.lower() in str(item).lower() for sub in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY)])
            if not key_drug_items_for_select: st.info("No forecast data available for the defined key disease drugs.")
            else:
                selected_drug_for_forecast = st.selectbox("Select Key Drug for Forecast Details:", key_drug_items_for_select, key="clinic_supply_item_forecast_selector_final_v9", help="View the forecasted days of supply remaining for the selected drug.")
                if selected_drug_for_forecast:
                    item_specific_forecast_df = supply_forecast_df[supply_forecast_df['item'] == selected_drug_for_forecast].copy()
                    if not item_specific_forecast_df.empty:
                        item_specific_forecast_df.sort_values('date', inplace=True)
                        current_day_info = item_specific_forecast_df.iloc[0] 
                        forecast_plot_title = (f"Forecast: {selected_drug_for_forecast}<br><sup_>Current Stock: {current_day_info.get('current_stock',0):.0f} units | Est. Daily Use: {current_day_info.get('consumption_rate',0):.1f} units/day | Est. Stockout: {pd.to_datetime(current_day_info.get('estimated_stockout_date')).strftime('%d %b %Y') if pd.notna(current_day_info.get('estimated_stockout_date')) else 'N/A'}</sup>")
                        st.plotly_chart(plot_annotated_line_chart(item_specific_forecast_df.set_index('date')['forecast_days'], title=forecast_plot_title, y_axis_title="Days of Supply Remaining", target_line=app_config.CRITICAL_SUPPLY_DAYS, target_label=f"Critical Level ({app_config.CRITICAL_SUPPLY_DAYS} Days)", show_ci=True, lower_bound_series=item_specific_forecast_df.set_index('date')['lower_ci'], upper_bound_series=item_specific_forecast_df.set_index('date')['upper_ci'], height=app_config.DEFAULT_PLOT_HEIGHT + 60, show_anomalies=False), use_container_width=True)
                    else: st.info(f"No forecast data for {selected_drug_for_forecast}.")
        else: st.warning("Supply forecast data could not be generated.")
    else: st.error("CRITICAL: Base health data unusable or missing essential columns for supply forecasts.")

with tab_patients: # Remains largely as before
    # ... (Patient tab content) ...
    st.subheader("Patient Load & High-Risk Case Identification")
    if not filtered_health_df_clinic.empty and all(c in filtered_health_df_clinic.columns for c in ['condition', 'date', 'patient_id']):
        conditions_for_load_chart = app_config.KEY_CONDITIONS_FOR_TRENDS; patient_load_df_src = filtered_health_df_clinic[filtered_health_df_clinic['condition'].isin(conditions_for_load_chart) & (filtered_health_df_clinic['patient_id'] != 'Unknown')].copy()
        if not patient_load_df_src.empty:
            daily_patient_load_summary = patient_load_df_src.groupby([pd.Grouper(key='date', freq='D'), 'condition'])['patient_id'].nunique().reset_index(); daily_patient_load_summary.rename(columns={'patient_id': 'unique_patients'}, inplace=True)
            if not daily_patient_load_summary.empty: st.plotly_chart(plot_bar_chart(daily_patient_load_summary, x_col='date', y_col='unique_patients', title="Daily Unique Patient Visits by Key Condition", color_col='condition', barmode='stack', height=app_config.DEFAULT_PLOT_HEIGHT + 70, y_axis_title="Unique Patients per Day", x_axis_title="Date", color_discrete_map=app_config.DISEASE_COLORS, text_auto=False ), use_container_width=True)
            else: st.caption("No patient load data for key conditions in period for chart.")
        else: st.caption("No patients with key conditions in period.")
    else: st.info("No health data available for selected period for Patient Load chart.")
    st.markdown("---"); st.markdown("###### **Flagged Patient Cases for Clinical Review (Selected Period)**")
    if not filtered_health_df_clinic.empty:
        flagged_patients_clinic_review_df = get_patient_alerts_for_clinic(filtered_health_df_clinic, risk_threshold_moderate=app_config.RISK_THRESHOLDS['moderate'])
        if flagged_patients_clinic_review_df is not None and not flagged_patients_clinic_review_df.empty:
            st.markdown(f"Found **{len(flagged_patients_clinic_review_df)}** patient cases flagged for review in period."); cols_for_alert_table_clinic = ['patient_id', 'condition', 'ai_risk_score', 'alert_reason', 'test_result', 'test_type', 'hiv_viral_load', 'priority_score', 'date']
            alerts_display_df_clinic = flagged_patients_clinic_review_df[[col for col in cols_for_alert_table_clinic if col in flagged_patients_clinic_review_df.columns]].copy()
            st.dataframe(alerts_display_df_clinic.head(25), use_container_width=True, column_config={ "ai_risk_score": st.column_config.ProgressColumn("AI Risk", format="%d", min_value=0, max_value=100, width="medium"), "date": st.column_config.DateColumn("Latest Record Date", format="YYYY-MM-DD"), "alert_reason": st.column_config.TextColumn("Alert Reason(s)", width="large"), "priority_score": st.column_config.NumberColumn("Priority", format="%d"), "hiv_viral_load": st.column_config.NumberColumn("HIV VL", format="%.0f copies/mL")}, height=450, hide_index=True )
        else: st.info("No specific patient cases flagged for clinical review in selected period.")
    else: st.info("No health data for selected period to generate patient alerts.")

with tab_environment: # Remains largely as before
    # ... (Environment tab content) ...
    st.subheader("Clinic Environmental Monitoring - Trends & Details")
    if not filtered_iot_df_clinic.empty and 'timestamp' in filtered_iot_df_clinic.columns:
        env_summary_for_tab = get_clinic_environmental_summary(filtered_iot_df_clinic)
        st.markdown(f"""**Current Environmental Alerts (latest readings in period):**
        - **CO2 Alerts:** {env_summary_for_tab.get('rooms_co2_alert_latest',0)} room(s) with CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm.
        - **PM2.5 Alerts:** {env_summary_for_tab.get('rooms_pm25_alert_latest',0)} room(s) with PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³.
        - **Noise Alerts:** {env_summary_for_tab.get('rooms_noise_alert_latest',0)} room(s) with Noise > {app_config.NOISE_LEVEL_ALERT_DB}dB.""")
        if env_summary_for_tab.get('high_occupancy_alert_latest', False): st.warning(f"⚠️ **High Waiting Room Occupancy Detected:** At least one area > {app_config.TARGET_WAITING_ROOM_OCCUPANCY} persons (latest reading).")
        env_trend_plot_cols = st.columns(2)
        with env_trend_plot_cols[0]:
            if 'avg_co2_ppm' in filtered_iot_df_clinic.columns:
                hourly_avg_co2_trend = get_trend_data(filtered_iot_df_clinic, 'avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean')
                if not hourly_avg_co2_trend.empty: st.plotly_chart(plot_annotated_line_chart(hourly_avg_co2_trend, "Hourly Avg. CO2 Levels (All Rooms)", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, target_label="Alert Threshold", height=app_config.COMPACT_PLOT_HEIGHT, show_anomalies=True, date_format="%d %b, %H:%M"), use_container_width=True)
                else: st.caption("No CO2 trend data.")
            else: st.caption("CO2 data ('avg_co2_ppm') missing for trend.")
        with env_trend_plot_cols[1]:
            if 'waiting_room_occupancy' in filtered_iot_df_clinic.columns:
                hourly_avg_occupancy_trend = get_trend_data(filtered_iot_df_clinic, 'waiting_room_occupancy', date_col='timestamp', period='H', agg_func='mean')
                if not hourly_avg_occupancy_trend.empty: st.plotly_chart(plot_annotated_line_chart(hourly_avg_occupancy_trend, "Hourly Avg. Waiting Room Occupancy", y_axis_title="Persons", target_line=app_config.TARGET_WAITING_ROOM_OCCUPANCY, target_label="Target Occupancy", height=app_config.COMPACT_PLOT_HEIGHT, show_anomalies=True, date_format="%d %b, %H:%M"), use_container_width=True)
                else: st.caption("No occupancy trend data.")
            else: st.caption("Occupancy data ('waiting_room_occupancy') missing for trend.")
        st.markdown("---"); st.subheader("Latest Sensor Readings by Room (End of Selected Period)")
        latest_room_cols_display = ['clinic_id', 'room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy']
        available_latest_cols = [col for col in latest_room_cols_display if col in filtered_iot_df_clinic.columns]
        if all(c in available_latest_cols for c in ['timestamp', 'clinic_id', 'room_name']):
            if not filtered_iot_df_clinic.empty:
                latest_room_sensor_readings = filtered_iot_df_clinic.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
                if not latest_room_sensor_readings.empty: st.dataframe(latest_room_sensor_readings[available_latest_cols].tail(15), use_container_width=True, height=380, column_config={"timestamp": st.column_config.DatetimeColumn("Last Reading At", format="YYYY-MM-DD HH:mm"), "avg_co2_ppm": st.column_config.NumberColumn("CO2 (ppm)", format="%d ppm"), "avg_pm25": st.column_config.NumberColumn("PM2.5 (µg/m³)", format="%.1f µg/m³"), "avg_temp_celsius": st.column_config.NumberColumn("Temperature (°C)", format="%.1f°C"), "avg_humidity_rh": st.column_config.NumberColumn("Humidity (%RH)", format="%d%%"), "avg_noise_db": st.column_config.NumberColumn("Noise Level (dB)", format="%d dB"), "waiting_room_occupancy": st.column_config.NumberColumn("Occupancy", format="%d persons")}, hide_index=True)
                else: st.caption("No detailed room sensor readings available for this period.")
            else:  st.caption("IoT data for selected period is empty.")
        else: st.caption(f"Essential columns for detailed room readings missing. Need: 'timestamp', 'clinic_id', 'room_name'. Available: {', '.join(available_latest_cols)}")
    else: st.info("No clinic environmental data for this tab (no data loaded or none in selected range), or 'timestamp' is problematic.")
