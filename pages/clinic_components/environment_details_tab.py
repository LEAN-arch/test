# test/pages/clinic_components/environment_details_tab.py
import streamlit as st
import pandas as pd # Required if using pd.to_datetime or other pd functions
from config import app_config
from utils.core_data_processing import get_clinic_environmental_summary, get_trend_data
from utils.ui_visualization_helpers import plot_annotated_line_chart
import logging # Good practice

logger = logging.getLogger(__name__) # Good practice

def render_environment_details(filtered_iot_df_clinic, iot_data_was_loaded_initially: bool):
    st.subheader("🌿 Clinic Environmental Monitoring - Trends & Detailed Readings")
    
    if filtered_iot_df_clinic.empty:
        if iot_data_was_loaded_initially:
            st.info("No clinic environmental IoT data found for the selected period to display details and trends.")
        # If iot_data_was_loaded_initially is False, main page handles this message
        return

    env_summary_for_tab_display = get_clinic_environmental_summary(filtered_iot_df_clinic)
    
    st.markdown("##### Current Environmental Alerts (based on latest readings in selected period):")
    alert_messages = [
        f"- **CO2 Alerts:** {env_summary_for_tab_display.get('rooms_co2_alert_latest',0)} room(s) with CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm.",
        f"- **PM2.5 Alerts:** {env_summary_for_tab_display.get('rooms_pm25_alert_latest',0)} room(s) with PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³.",
        f"- **Noise Alerts:** {env_summary_for_tab_display.get('rooms_noise_alert_latest',0)} room(s) with Noise > {app_config.NOISE_LEVEL_ALERT_DB}dB."
    ]
    if env_summary_for_tab_display.get('high_occupancy_alert_latest', False):
        # CORRECTED LINE:
        alert_messages.append(f"- ⚠️ **High Waiting Room Occupancy Detected:** At least one area had occupancy > {app_config.TARGET_WAITING_ROOM_OCCUPANCY} persons (latest reading in period).")
    
    for msg in alert_messages:
        st.markdown(msg)
    st.markdown("---")

    st.markdown("##### Hourly Trends for Key Environmental Metrics")
    env_trend_plot_cols = st.columns(2)
    with env_trend_plot_cols[0]:
        if 'avg_co2_ppm' in filtered_iot_df_clinic.columns:
            iot_co2_trend_df = filtered_iot_df_clinic.copy()
            if not pd.api.types.is_datetime64_ns_dtype(iot_co2_trend_df['timestamp']):
                iot_co2_trend_df['timestamp'] = pd.to_datetime(iot_co2_trend_df['timestamp'], errors='coerce')
            iot_co2_trend_df.dropna(subset=['timestamp'], inplace=True)

            if not iot_co2_trend_df.empty:
                hourly_avg_co2_trend = get_trend_data(iot_co2_trend_df, 'avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean')
                if not hourly_avg_co2_trend.empty: 
                    st.plotly_chart(plot_annotated_line_chart(hourly_avg_co2_trend, "Hourly Avg. CO2 Levels (All Rooms)", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, target_label="Alert Threshold", height=app_config.COMPACT_PLOT_HEIGHT, show_anomalies=True, date_format="%d %b, %H:%M"), use_container_width=True)
                else: st.caption("No CO2 trend data to plot for the selected period.")
            else: st.caption("No valid timestamp data for CO2 trend.")
        else: st.caption("CO2 data ('avg_co2_ppm') missing for trend visualization.")
        
    with env_trend_plot_cols[1]:
        if 'waiting_room_occupancy' in filtered_iot_df_clinic.columns:
            iot_occ_trend_df = filtered_iot_df_clinic.copy()
            if not pd.api.types.is_datetime64_ns_dtype(iot_occ_trend_df['timestamp']):
                iot_occ_trend_df['timestamp'] = pd.to_datetime(iot_occ_trend_df['timestamp'], errors='coerce')
            iot_occ_trend_df.dropna(subset=['timestamp'], inplace=True)

            if not iot_occ_trend_df.empty:
                hourly_avg_occupancy_trend = get_trend_data(iot_occ_trend_df, 'waiting_room_occupancy', date_col='timestamp', period='H', agg_func='mean')
                if not hourly_avg_occupancy_trend.empty: 
                    st.plotly_chart(plot_annotated_line_chart(hourly_avg_occupancy_trend, "Hourly Avg. Waiting Room Occupancy", y_axis_title="Persons", target_line=app_config.TARGET_WAITING_ROOM_OCCUPANCY, target_label="Target Occupancy", height=app_config.COMPACT_PLOT_HEIGHT, show_anomalies=True, date_format="%d %b, %H:%M", y_is_count=False), use_container_width=True)
                else: st.caption("No waiting room occupancy trend data for the selected period.")
            else: st.caption("No valid timestamp data for occupancy trend.")
        else: st.caption("Occupancy data ('waiting_room_occupancy') column missing for trend.")

    st.markdown("---"); st.subheader("Latest Sensor Readings by Room (End of Selected Period)")
    latest_room_cols_display_env = ['clinic_id', 'room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour']
    available_latest_cols_env_tab = [col for col in latest_room_cols_display_env if col in filtered_iot_df_clinic.columns]
    
    if all(c in available_latest_cols_env_tab for c in ['timestamp', 'clinic_id', 'room_name']):
        latest_room_sensor_readings_df = filtered_iot_df_clinic.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
        
        if not latest_room_sensor_readings_df.empty:
            df_to_display_env_latest_tab = latest_room_sensor_readings_df[available_latest_cols_env_tab].tail(15).copy()
            for col in df_to_display_env_latest_tab.columns: # Ensure data is JSON serializable
                if col == 'timestamp':
                    df_to_display_env_latest_tab[col] = pd.to_datetime(df_to_display_env_latest_tab[col], errors='coerce')
                    if pd.api.types.is_datetime64tz_dtype(df_to_display_env_latest_tab[col]):
                         df_to_display_env_latest_tab[col] = df_to_display_env_latest_tab[col].dt.tz_localize(None)
                elif df_to_display_env_latest_tab[col].dtype == 'object':
                    df_to_display_env_latest_tab[col] = df_to_display_env_latest_tab[col].fillna('N/A').astype(str).replace(['nan','None','<NA>'],'N/A',regex=False)
            
            st.dataframe(df_to_display_env_latest_tab, use_container_width=True, height=380,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Last Reading", format="YYYY-MM-DD HH:mm"),
                    "avg_co2_ppm": st.column_config.NumberColumn("CO2", format="%dppm"),
                    "avg_pm25": st.column_config.NumberColumn("PM2.5", format="%.1fµg/m³"),
                    "avg_temp_celsius": st.column_config.NumberColumn("Temp", format="%.1f°C"),
                    "avg_humidity_rh": st.column_config.NumberColumn("Hum.", format="%d%%"),
                    "avg_noise_db": st.column_config.NumberColumn("Noise", format="%ddB"),
                    "waiting_room_occupancy": st.column_config.NumberColumn("Occup.", format="%d P", help="Avg persons in waiting area if 'waiting_room' in name"),
                    "patient_throughput_per_hour": st.column_config.NumberColumn("Thrpt/hr", format="%.1f"),
                    "sanitizer_dispenses_per_hour": st.column_config.NumberColumn("Sanit./hr", format="%.1f")
                }, hide_index=True)
        else: 
            st.caption("No distinct room sensor readings available for the latest point in this period after filtering.")
    else: 
        st.caption(f"Essential columns (timestamp, clinic_id, room_name) are missing from IoT data for the selected period, cannot display detailed room readings.")
