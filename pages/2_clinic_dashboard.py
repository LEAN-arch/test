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
    render_kpi_card, plot_donut_chart, plot_annotated_line_chart, plot_bar_chart 
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="Clinic Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__) 

@st.cache_resource 
def load_css(): 
    if os.path.exists(app_config.STYLE_CSS_PATH):
        with open(app_config.STYLE_CSS_PATH) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info("Clinic Dashboard: CSS loaded successfully.")
    else:
        logger.warning(f"Clinic Dashboard: CSS file not found at {app_config.STYLE_CSS_PATH}. Default Streamlit styles will be used.")
load_css()

# --- Data Loading ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS) 
def get_clinic_dashboard_data():
    logger.info("Clinic Dashboard: Attempting to load health records and IoT data...")
    health_df = load_health_records()
    iot_df = load_iot_clinic_environment_data()

    logger.debug(f"Health DF loaded shape: {health_df.shape if health_df is not None else 'None'}, Dtypes present: {'date' in health_df.columns if health_df is not None else False}")
    if health_df is not None and not health_df.empty and 'date' in health_df.columns: logger.debug(f"Health DF 'date' sample: {health_df['date'].dropna().head(2).to_string()}")
    
    logger.debug(f"IoT DF loaded shape: {iot_df.shape if iot_df is not None else 'None'}, Dtypes present: {'timestamp' in iot_df.columns if iot_df is not None else False}")
    if iot_df is not None and not iot_df.empty and 'timestamp' in iot_df.columns: logger.debug(f"IoT DF 'timestamp' sample: {iot_df['timestamp'].dropna().head(2).to_string()}")

    if health_df.empty: logger.warning("Clinic Dashboard: Health records are empty or failed to load after core processing.")
    else: logger.info(f"Clinic Dashboard: Successfully loaded and processed {len(health_df)} health records.")
    if iot_df.empty: logger.warning("Clinic Dashboard: IoT data is empty or failed to load after core processing.")
    else: logger.info(f"Clinic Dashboard: Successfully loaded and processed {len(iot_df)} IoT records.")
    return health_df, iot_df

health_df_clinic_main, iot_df_clinic_main = get_clinic_dashboard_data()

# --- Main Page Rendering ---
critical_data_missing_flag = False 
if health_df_clinic_main is None or health_df_clinic_main.empty:
    st.warning("⚠️ **Health records data is currently unavailable.** Some dashboard features related to patient services, testing, and supply chain will be limited or show no data.")
    logger.warning("Clinic Dashboard cannot display full health-related metrics: health_df_clinic_main is None or empty.")
    health_df_clinic_main = pd.DataFrame(columns=['date', 'item', 'condition', 'patient_id', 'test_type', 'test_result', 'test_turnaround_days', 'stock_on_hand', 'consumption_rate_per_day', 'ai_risk_score'])
    critical_data_missing_flag = True

if iot_df_clinic_main is None or iot_df_clinic_main.empty:
    st.info("ℹ️ IoT environmental data is unavailable. Clinic environment monitoring section will not be displayed.")
    logger.info("Clinic Dashboard: iot_df_clinic_main is None or empty. Environmental metrics will be skipped.")
    iot_df_clinic_main = pd.DataFrame(columns=['timestamp', 'avg_co2_ppm', 'avg_pm25', 'waiting_room_occupancy', 'sanitizer_dispenses_per_hour', 'avg_noise_db', 'clinic_id', 'room_name', 'zone_id'])

if critical_data_missing_flag and iot_df_clinic_main.empty:
     st.error("🚨 **CRITICAL Error:** Both Health records and essential IoT data are unavailable. Clinic Dashboard cannot be displayed.")
     logger.critical("Clinic Dashboard cannot render: both health_df and iot_df are unusable/empty.")
     st.stop()


st.title("🏥 Clinic Operations & Environmental Dashboard")
st.markdown("**Monitoring Service Efficiency, Quality of Care, Resource Management, and Facility Environment**")
st.markdown("---")

# --- Sidebar Filters & Date Range Setup ---
st.sidebar.image("assets//DNA-DxBrand.png", width=200)
st.sidebar.header("🗓️ Clinic Filters") 

all_potential_dates = [] 
default_min_date = pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND * 3)
default_max_date = pd.Timestamp('today').date()

def safe_extract_dates_from_df(df, col_name, df_name_log="DataFrame"):
    _extracted_ts = []
    if df is not None and col_name in df.columns and not df.empty:
        date_like_column = df[col_name]
        if not isinstance(date_like_column, pd.Series): date_like_column = pd.Series(date_like_column)
        
        # Convert to datetime64[ns] if not already, this handles mixed types better and standardizes to Timestamp
        dt_series = pd.to_datetime(date_like_column, errors='coerce')
        valid_timestamps = dt_series.dropna() # Remove NaT
        if not valid_timestamps.empty:
            _extracted_ts.extend(valid_timestamps.tolist())
            logger.debug(f"{df_name_log}: Extracted {len(valid_timestamps)} valid timestamps from '{col_name}'. Sample: {_extracted_ts[:3]}")
        else: logger.debug(f"{df_name_log}: Column '{col_name}' had no valid timestamps after coercion and dropna.")
    else: logger.debug(f"{df_name_log}: Column '{col_name}' not found or DataFrame is empty.")
    return _extracted_ts

all_potential_dates.extend(safe_extract_dates_from_df(health_df_clinic_main, 'date', "HealthDF"))
all_potential_dates.extend(safe_extract_dates_from_df(iot_df_clinic_main, 'timestamp', "IoTDF"))

# Filter for pd.Timestamp instances (should be redundant if safe_extract_dates works well, but safety first)
all_valid_timestamps_final = [d for d in all_potential_dates if isinstance(d, pd.Timestamp)]

min_date_data_clinic = default_min_date
max_date_data_clinic = default_max_date
default_start_val_clinic = default_min_date 
default_end_val_clinic = default_max_date   

if all_valid_timestamps_final:
    try:
        # Create a Series from the list of pd.Timestamp objects, drop duplicates, sort.
        combined_series_clinic = pd.Series(all_valid_timestamps_final).drop_duplicates().sort_values(ignore_index=True)
        logger.debug(f"Clinic Date Filter: Combined Series - dtype: {combined_series_clinic.dtype}, empty: {combined_series_clinic.empty}, head:\n{combined_series_clinic.head().to_string() if not combined_series_clinic.empty else 'Series is empty'}")

        if not combined_series_clinic.empty:
            min_date_ts_clinic = combined_series_clinic.iloc[0] # Min after sort
            max_date_ts_clinic = combined_series_clinic.iloc[-1] # Max after sort
            
            min_date_data_clinic = min_date_ts_clinic.date()
            max_date_data_clinic = max_date_ts_clinic.date()
            
            default_end_val_clinic = max_date_data_clinic 
            default_start_val_clinic = max_date_data_clinic - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
            if default_start_val_clinic < min_date_data_clinic:
                default_start_val_clinic = min_date_data_clinic
            logger.info(f"Clinic Dashboard: Date filter range determined: {min_date_data_clinic} to {max_date_data_clinic}. Defaulting to: {default_start_val_clinic} to {default_end_val_clinic}")
        else: 
            logger.warning("Clinic Dashboard: Combined timestamp series for date filter is empty after all processing. Using wide fallback dates.")
    except Exception as e_date_range_clinic:
        logger.error(f"Clinic Dashboard: CRITICAL ERROR determining date range: {e_date_range_clinic}. Using wide fallback.", exc_info=True)
else: 
    logger.warning("Clinic Dashboard: No valid Timestamps extracted from any data source for date filter. Using wide fallback dates.")

if default_start_val_clinic < min_date_data_clinic : default_start_val_clinic = min_date_data_clinic
if default_end_val_clinic > max_date_data_clinic : default_end_val_clinic = max_date_data_clinic
if default_start_val_clinic > default_end_val_clinic : default_start_val_clinic = default_end_val_clinic


selected_start_date_cl, selected_end_date_cl = st.sidebar.date_input(
    "Select Date Range for Analysis:",
    value=[default_start_val_clinic, default_end_val_clinic],
    min_value=min_date_data_clinic,
    max_value=max_date_data_clinic,
    key="clinic_dashboard_date_range_selector_v15_final", 
    help="This date range applies to most charts and Key Performance Indicators (KPIs)."
)

# --- Filter dataframes based on selected date range ---
filtered_health_df_clinic = pd.DataFrame(columns=health_df_clinic_main.columns if health_df_clinic_main is not None else [])
if selected_start_date_cl and selected_end_date_cl and health_df_clinic_main is not None and 'date' in health_df_clinic_main.columns and not health_df_clinic_main.empty:
    temp_health_df_filter = health_df_clinic_main.copy() 
    if not pd.api.types.is_datetime64_ns_dtype(temp_health_df_filter['date']):
         temp_health_df_filter['date'] = pd.to_datetime(temp_health_df_filter['date'], errors='coerce')
    
    temp_health_df_filter.dropna(subset=['date'], inplace=True) 
    if not temp_health_df_filter.empty:
        temp_health_df_filter['date_obj_for_filter'] = temp_health_df_filter['date'].dt.date
        health_mask = (temp_health_df_filter['date_obj_for_filter'] >= selected_start_date_cl) & \
                      (temp_health_df_filter['date_obj_for_filter'] <= selected_end_date_cl) & \
                      (temp_health_df_filter['date_obj_for_filter'].notna()) 
        filtered_health_df_clinic = temp_health_df_filter[health_mask].copy()
        logger.info(f"Clinic Health Data: Filtered from {selected_start_date_cl} to {selected_end_date_cl}, resulting in {len(filtered_health_df_clinic)} rows.")
        if filtered_health_df_clinic.empty:
            logger.warning(f"Clinic Health Data Filter resulted in 0 rows. Original data had {len(health_df_clinic_main)} rows. Date range: {selected_start_date_cl} to {selected_end_date_cl}. First 5 health_df dates: {health_df_clinic_main['date'].dropna().head().to_list() if not health_df_clinic_main.empty and 'date' in health_df_clinic_main else 'N/A'}")


filtered_iot_df_clinic = pd.DataFrame(columns=iot_df_clinic_main.columns if iot_df_clinic_main is not None else [])
if selected_start_date_cl and selected_end_date_cl and iot_df_clinic_main is not None and 'timestamp' in iot_df_clinic_main.columns and not iot_df_clinic_main.empty:
    temp_iot_df_filter = iot_df_clinic_main.copy()
    if not pd.api.types.is_datetime64_ns_dtype(temp_iot_df_filter['timestamp']):
        temp_iot_df_filter['timestamp'] = pd.to_datetime(temp_iot_df_filter['timestamp'], errors='coerce')
    
    temp_iot_df_filter.dropna(subset=['timestamp'], inplace=True)
    if not temp_iot_df_filter.empty:
        temp_iot_df_filter['date_obj_for_filter'] = temp_iot_df_filter['timestamp'].dt.date
        iot_mask = (temp_iot_df_filter['date_obj_for_filter'] >= selected_start_date_cl) & \
                   (temp_iot_df_filter['date_obj_for_filter'] <= selected_end_date_cl) & \
                   (temp_iot_df_filter['date_obj_for_filter'].notna())
        filtered_iot_df_clinic = temp_iot_df_filter[iot_mask].copy()
        logger.info(f"Clinic IoT Data: Filtered from {selected_start_date_cl} to {selected_end_date_cl}, resulting in {len(filtered_iot_df_clinic)} rows.")
        if filtered_iot_df_clinic.empty:
            logger.warning(f"Clinic IoT Data Filter resulted in 0 rows. Original data had {len(iot_df_clinic_main)} rows. Date range: {selected_start_date_cl} to {selected_end_date_cl}. First 5 iot_df timestamps: {iot_df_clinic_main['timestamp'].dropna().head().to_list() if not iot_df_clinic_main.empty and 'timestamp' in iot_df_clinic_main else 'N/A'}")

# --- Display KPIs ---
date_range_display_str = f"({selected_start_date_cl.strftime('%d %b %Y')} - {selected_end_date_cl.strftime('%d %b %Y')})" if selected_start_date_cl and selected_end_date_cl else "(Date range not fully set)"

st.subheader(f"Key Disease Service Metrics {date_range_display_str}")
if filtered_health_df_clinic.empty and not critical_data_missing_flag: # Only show if health data was expected
    logger.warning("Cannot calculate Clinic Service KPIs as filtered_health_df_clinic is empty for the selected date range.")
    st.info("No health data available for the selected period to display service metrics.")
clinic_service_kpis = get_clinic_summary(filtered_health_df_clinic)
logger.debug(f"Clinic Service KPIs calculated for period: {clinic_service_kpis}")


kpi_cols_clinic_services = st.columns(5)
with kpi_cols_clinic_services[0]:
    tb_pos_rate = clinic_service_kpis.get('tb_sputum_positivity', 0.0)
    render_kpi_card("TB Positivity Rate", f"{tb_pos_rate:.1f}%", "🔬",
                    status="High" if tb_pos_rate > 15 else ("Moderate" if tb_pos_rate > 5 else "Low"),
                    help_text="Percentage of sputum/GeneXpert tests positive for TB in the selected period.")
with kpi_cols_clinic_services[1]:
    mal_pos_rate = clinic_service_kpis.get('malaria_positivity', 0.0)
    render_kpi_card("Malaria Positivity", f"{mal_pos_rate:.1f}%", "🦟",
                    status="High" if mal_pos_rate > app_config.TARGET_MALARIA_POSITIVITY_RATE else "Low",
                    help_text=f"Malaria test (RDT/Microscopy) positivity rate. Target: <{app_config.TARGET_MALARIA_POSITIVITY_RATE}%.")
with kpi_cols_clinic_services[2]:
    avg_tat = clinic_service_kpis.get('avg_test_turnaround_all_tests', 0.0)
    render_kpi_card("Avg. Test TAT", f"{avg_tat:.1f} days", "⏱️",
                    status="High" if avg_tat > app_config.TARGET_TEST_TURNAROUND_DAYS + 1 else ("Moderate" if avg_tat > app_config.TARGET_TEST_TURNAROUND_DAYS else "Low"),
                    help_text=f"Average turnaround time for all conclusive tests. Target: ≤{app_config.TARGET_TEST_TURNAROUND_DAYS} days.")
with kpi_cols_clinic_services[3]:
    hiv_tests_count = clinic_service_kpis.get('hiv_tests_conclusive_period', 0)
    render_kpi_card("HIV Tests Conducted", str(hiv_tests_count), 
                    "🧪",  # <<< USING TWO EMOJIS AS A SINGLE STRING
                    # icon_is_html=True, # NOT needed for direct emojis
                    status="Low" if hiv_tests_count < 20 else "Moderate", 
                    help_text="Number of unique patients with conclusive HIV test results in the period.")
with kpi_cols_clinic_services[4]:
    drug_stockouts_count = clinic_service_kpis.get('key_drug_stockouts', 0)
    render_kpi_card("Key Drug Stockouts", str(drug_stockouts_count), "💊",
                    status="High" if drug_stockouts_count > 0 else "Low",
                    help_text=f"Number of key disease drugs with < {app_config.CRITICAL_SUPPLY_DAYS} days of supply remaining.")

if not filtered_iot_df_clinic.empty:
    st.subheader(f"Clinic Environment Snapshot {date_range_display_str}")
    clinic_env_kpis = get_clinic_environmental_summary(filtered_iot_df_clinic)
    logger.debug(f"Clinic Env KPIs calculated: {clinic_env_kpis}")

    kpi_cols_clinic_environment = st.columns(4)
    with kpi_cols_clinic_environment[0]:
        avg_co2_val = clinic_env_kpis.get('avg_co2_overall', 0.0)
        co2_alert_rooms_val = clinic_env_kpis.get('rooms_co2_alert_latest', 0)
        render_kpi_card("Avg. CO2 (All Rooms)", f"{avg_co2_val:.0f} ppm", "💨",
                        status="High" if co2_alert_rooms_val > 0 else "Low",
                        help_text=f"Period average CO2. {co2_alert_rooms_val} room(s) currently > {app_config.CO2_LEVEL_ALERT_PPM}ppm.")
    with kpi_cols_clinic_environment[1]:
        avg_pm25_val = clinic_env_kpis.get('avg_pm25_overall', 0.0)
        pm25_alert_rooms_val = clinic_env_kpis.get('rooms_pm25_alert_latest', 0)
        render_kpi_card("Avg. PM2.5 (All Rooms)", f"{avg_pm25_val:.1f} µg/m³", "🌫️",
                        status="High" if pm25_alert_rooms_val > 0 else "Low",
                        help_text=f"Period average PM2.5. {pm25_alert_rooms_val} room(s) currently > {app_config.PM25_ALERT_UGM3}µg/m³.")
    with kpi_cols_clinic_environment[2]:
        avg_occupancy_val = clinic_env_kpis.get('avg_occupancy_overall', 0.0)
        occupancy_alert_val = clinic_env_kpis.get('high_occupancy_alert_latest', False)
        render_kpi_card("Avg. Occupancy", f"{avg_occupancy_val:.1f} persons", "👨‍👩‍👧‍👦",
                        status="High" if occupancy_alert_val else "Low",
                        help_text=f"Average waiting room occupancy. Target < {app_config.TARGET_WAITING_ROOM_OCCUPANCY}. Alert if any room's latest reading is high.")
    with kpi_cols_clinic_environment[3]:
        avg_noise_alert_rooms_val = clinic_env_kpis.get('rooms_noise_alert_latest', 0)
        render_kpi_card("Noise Alerts", str(avg_noise_alert_rooms_val), "🔊",
                        status="High" if avg_noise_alert_rooms_val > 0 else "Low",
                        help_text=f"Rooms with latest noise levels > {app_config.NOISE_LEVEL_ALERT_DB}dB.")
elif iot_df_clinic_main is not None and not iot_df_clinic_main.empty: # If main IoT data exists but filtered is empty
    st.info("No IoT data for the selected period. Environmental KPIs cannot be displayed.")
# else: # If iot_df_clinic_main itself was empty, message handled by st.info at top

st.markdown("---")

tab_titles_clinic = ["🔬 Disease Testing Insights", "💊 Supply Chain Management", "🧍 Patient Focus & Alerts", "🌿 Clinic Environment Details"]
tab_tests, tab_supplies, tab_patients, tab_environment = st.tabs(tab_titles_clinic)

with tab_tests:
    st.subheader("Disease-Specific Testing Performance and Trends")
    if not filtered_health_df_clinic.empty and all(c in filtered_health_df_clinic.columns for c in ['test_type', 'test_result', 'patient_id', 'date']):
        col_test_dist, col_test_tat_trend = st.columns([0.45, 0.55])
        with col_test_dist:
            st.markdown("###### **TB Test Result Distribution**")
            tb_tests_df_viz = filtered_health_df_clinic[filtered_health_df_clinic['test_type'].str.contains("Sputum|GeneXpert", case=False, na=False)].copy()
            if not tb_tests_df_viz.empty:
                tb_results_dist_summary = tb_tests_df_viz.dropna(subset=['test_result']).groupby('test_result')['patient_id'].nunique().reset_index()
                tb_results_dist_summary.columns = ['Test Result', 'Unique Patients']
                tb_results_dist_summary_conclusive = tb_results_dist_summary[~tb_results_dist_summary['Test Result'].isin(['Unknown', 'N/A', 'nan', 'Pending'])]
                if not tb_results_dist_summary_conclusive.empty:
                    st.plotly_chart(plot_donut_chart(tb_results_dist_summary_conclusive, 'Test Result', 'Unique Patients',
                                                    title="TB Test Results (Conclusive)", height=app_config.COMPACT_PLOT_HEIGHT + 20,
                                                    color_discrete_map={"Positive": app_config.RISK_STATUS_COLORS["High"], "Negative": app_config.RISK_STATUS_COLORS["Low"]}),
                                   use_container_width=True)
                else: st.caption("No conclusive TB test result data for donut chart in the selected period.")
            else: st.caption("No TB tests found in the selected period to display distribution.")

        with col_test_tat_trend:
            st.markdown("###### **Daily Average Test Turnaround Time (TAT)**")
            if 'test_turnaround_days' in filtered_health_df_clinic.columns and 'date' in filtered_health_df_clinic.columns: 
                tat_trend_data_src = filtered_health_df_clinic[
                    filtered_health_df_clinic['test_turnaround_days'].notna() &
                    (~filtered_health_df_clinic['test_result'].isin(['Pending', 'N/A', 'Unknown']))
                ].copy()
                if not tat_trend_data_src.empty:
                    daily_avg_tat_trend = get_trend_data(tat_trend_data_src,'test_turnaround_days', period='D', date_col='date', agg_func='mean')
                    if not daily_avg_tat_trend.empty:
                        st.plotly_chart(plot_annotated_line_chart(
                            daily_avg_tat_trend, "Daily Avg. Test Turnaround Time", y_axis_title="Days",
                            target_line=app_config.TARGET_TEST_TURNAROUND_DAYS, target_label="Target TAT",
                            height=app_config.COMPACT_PLOT_HEIGHT + 20, show_anomalies=True, date_format="%d %b"
                        ), use_container_width=True)
                    else: st.caption("No aggregated TAT data available for trend in the selected period.")
                else: st.caption("No conclusive tests with TAT data found in the selected period for trend.")
            else: st.caption("Test Turnaround Time data ('test_turnaround_days') or result date ('date') missing for TAT trend analysis.")
    else:
        st.info("No health data available for the selected period to display Disease Testing Insights.")

with tab_supplies:
    st.subheader("Medical Supply Levels & Consumption Forecast")
    if health_df_clinic_main is not None and not health_df_clinic_main.empty and \
       all(c in health_df_clinic_main.columns for c in ['item', 'date', 'stock_on_hand', 'consumption_rate_per_day']):

        supply_forecast_df = get_supply_forecast_data(health_df_clinic_main, forecast_days_out=28) # Using main df for historical rate

        if supply_forecast_df is not None and not supply_forecast_df.empty:
            key_drug_items_for_select = sorted([
                item for item in supply_forecast_df['item'].unique()
                if any(sub.lower() in str(item).lower() for sub in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY)
            ])

            if not key_drug_items_for_select:
                st.info("No forecast data available for the defined key disease drugs.")
            else:
                selected_drug_for_forecast = st.selectbox(
                    "Select Key Drug for Forecast Details:", key_drug_items_for_select,
                    key="clinic_supply_item_forecast_selector_final_v7", 
                    help="View the forecasted days of supply remaining for the selected drug."
                )
                if selected_drug_for_forecast:
                    item_specific_forecast_df = supply_forecast_df[supply_forecast_df['item'] == selected_drug_for_forecast].copy()
                    if not item_specific_forecast_df.empty:
                        item_specific_forecast_df.sort_values('date', inplace=True)
                        
                        current_day_info = item_specific_forecast_df.iloc[0] 
                        forecast_plot_title = (
                            f"Forecast: {selected_drug_for_forecast}<br>"
                            f"<sup_>Current Stock: {current_day_info.get('current_stock',0):.0f} units | "
                            f"Est. Daily Use: {current_day_info.get('consumption_rate',0):.1f} units/day | "
                            f"Est. Stockout: {pd.to_datetime(current_day_info.get('estimated_stockout_date')).strftime('%d %b %Y') if pd.notna(current_day_info.get('estimated_stockout_date')) else 'N/A'}</sup>"
                        )
                        
                        st.plotly_chart(plot_annotated_line_chart(
                            item_specific_forecast_df.set_index('date')['forecast_days'],
                            title=forecast_plot_title,
                            y_axis_title="Days of Supply Remaining",
                            target_line=app_config.CRITICAL_SUPPLY_DAYS,
                            target_label=f"Critical Level ({app_config.CRITICAL_SUPPLY_DAYS} Days)",
                            show_ci=True,
                            lower_bound_series=item_specific_forecast_df.set_index('date')['lower_ci'],
                            upper_bound_series=item_specific_forecast_df.set_index('date')['upper_ci'],
                            height=app_config.DEFAULT_PLOT_HEIGHT + 60,
                            show_anomalies=False
                        ), use_container_width=True)
                    else: st.info(f"No forecast data found for the selected drug: {selected_drug_for_forecast}.")
        else:
            st.warning("Supply forecast data could not be generated. This may be due to missing consumption rates or stock levels.")
    else:
        st.error("CRITICAL FOR SUPPLY TAB: Cannot generate supply forecasts. Base health data is unusable or missing essential columns (item, date, stock_on_hand, consumption_rate_per_day).")

with tab_patients:
    st.subheader("Patient Load & High-Risk Case Identification")
    if not filtered_health_df_clinic.empty and all(c in filtered_health_df_clinic.columns for c in ['condition', 'date', 'patient_id']):
        conditions_for_load_chart = app_config.KEY_CONDITIONS_FOR_TRENDS
        patient_load_df_src = filtered_health_df_clinic[
            filtered_health_df_clinic['condition'].isin(conditions_for_load_chart) &
            (filtered_health_df_clinic['patient_id'] != 'Unknown')
        ].copy()

        if not patient_load_df_src.empty:
            daily_patient_load_summary = patient_load_df_src.groupby(
                [pd.Grouper(key='date', freq='D'), 'condition']
            )['patient_id'].nunique().reset_index()
            daily_patient_load_summary.rename(columns={'patient_id': 'unique_patients'}, inplace=True)

            if not daily_patient_load_summary.empty:
                st.plotly_chart(plot_bar_chart(
                    daily_patient_load_summary, x_col='date', y_col='unique_patients',
                    title="Daily Unique Patient Visits by Key Condition", color_col='condition',
                    barmode='stack', height=app_config.DEFAULT_PLOT_HEIGHT + 70,
                    y_axis_title="Unique Patients per Day", x_axis_title="Date",
                    color_discrete_map=app_config.DISEASE_COLORS,
                    text_auto=False 
                ), use_container_width=True)
            else: st.caption("No patient load data for key conditions found in the selected period to display the chart.")
        else: st.caption("No patients with key conditions recorded in the selected period.")
    else:
        st.info("No health data available for the selected period for Patient Load chart display.")

    st.markdown("---"); st.markdown("###### **Flagged Patient Cases for Clinical Review (Selected Period)**")
    if not filtered_health_df_clinic.empty:
        flagged_patients_clinic_review_df = get_patient_alerts_for_clinic(
            filtered_health_df_clinic,
            risk_threshold_moderate=app_config.RISK_THRESHOLDS['moderate']
        )
        if flagged_patients_clinic_review_df is not None and not flagged_patients_clinic_review_df.empty:
            st.markdown(f"Found **{len(flagged_patients_clinic_review_df)}** patient cases flagged for review based on high risk, recent critical positive tests, or overdue critical tests within the period.")
            cols_for_alert_table_clinic = ['patient_id', 'condition', 'ai_risk_score', 'alert_reason', 'test_result', 'test_type', 'hiv_viral_load', 'priority_score', 'date']
            alerts_display_df_clinic = flagged_patients_clinic_review_df[[col for col in cols_for_alert_table_clinic if col in flagged_patients_clinic_review_df.columns]].copy()
            st.dataframe(alerts_display_df_clinic.head(25), use_container_width=True,
                column_config={ "ai_risk_score": st.column_config.ProgressColumn("AI Risk", format="%d", min_value=0, max_value=100, width="medium"),
                    "date": st.column_config.DateColumn("Latest Record Date", format="YYYY-MM-DD"),
                    "alert_reason": st.column_config.TextColumn("Alert Reason(s)", width="large", help="Reasons why this patient case is flagged."),
                    "priority_score": st.column_config.NumberColumn("Priority", help="Calculated alert priority.", format="%d"),
                    "hiv_viral_load": st.column_config.NumberColumn("HIV VL", format="%.0f copies/mL", help="HIV Viral Load if applicable.")},
                height=450, hide_index=True )
        else: st.info("No specific patient cases flagged for clinical review in the selected period based on current criteria.")
    else: st.info("No health data available for the selected period to generate patient alerts.")

with tab_environment:
    st.subheader("Clinic Environmental Monitoring - Trends & Details")
    if not filtered_iot_df_clinic.empty and 'timestamp' in filtered_iot_df_clinic.columns:
        env_summary_for_tab = get_clinic_environmental_summary(filtered_iot_df_clinic)
        st.markdown(f"""**Current Environmental Alerts (based on latest readings in period):**
        - **CO2 Alerts:** {env_summary_for_tab.get('rooms_co2_alert_latest',0)} room(s) with CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm.
        - **PM2.5 Alerts:** {env_summary_for_tab.get('rooms_pm25_alert_latest',0)} room(s) with PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³.
        - **Noise Alerts:** {env_summary_for_tab.get('rooms_noise_alert_latest',0)} room(s) with Noise > {app_config.NOISE_LEVEL_ALERT_DB}dB.""")
        if env_summary_for_tab.get('high_occupancy_alert_latest', False):
            st.warning(f"⚠️ **High Waiting Room Occupancy Detected:** At least one area has occupancy > {app_config.TARGET_WAITING_ROOM_OCCUPANCY} persons (latest reading).")

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
                if not latest_room_sensor_readings.empty:
                    st.dataframe(latest_room_sensor_readings[available_latest_cols].tail(15), use_container_width=True, height=380,
                        column_config={"timestamp": st.column_config.DatetimeColumn("Last Reading At", format="YYYY-MM-DD HH:mm"), "avg_co2_ppm": st.column_config.NumberColumn("CO2 (ppm)", format="%d ppm"), "avg_pm25": st.column_config.NumberColumn("PM2.5 (µg/m³)", format="%.1f µg/m³"), "avg_temp_celsius": st.column_config.NumberColumn("Temperature (°C)", format="%.1f°C"), "avg_humidity_rh": st.column_config.NumberColumn("Humidity (%RH)", format="%d%%"), "avg_noise_db": st.column_config.NumberColumn("Noise Level (dB)", format="%d dB"), "waiting_room_occupancy": st.column_config.NumberColumn("Occupancy", format="%d persons")}, hide_index=True)
                else: st.caption("No detailed room sensor readings available for this period.")
            else:  st.caption("IoT data for selected period is empty.")
        else: st.caption(f"Essential columns for detailed room readings missing. Need: 'timestamp', 'clinic_id', 'room_name'. Available: {', '.join(available_latest_cols)}")
    else: st.info("No clinic environmental data for this tab (no data loaded or none in selected range), or 'timestamp' is problematic.")
