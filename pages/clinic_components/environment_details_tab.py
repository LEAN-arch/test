# test/pages/clinic_components/environment_details_tab.py
import streamlit as st
import pandas as pd
from config import app_config
from utils.core_data_processing import get_clinic_environmental_summary, get_trend_data
from utils.ui_visualization_helpers import plot_annotated_line_chart

def render_environment_details(filtered_iot_df_clinic): # No longer needs date_range_display_str here, main title handled by tab
    st.subheader("🌿 Clinic Environmental Monitoring - Trends & Details") # Redundant if tab already has similar title
    
    if filtered_iot_df_clinic.empty:
        # Check if iot_data_available was True in main page, to distinguish "no data in period" from "no data at all"
        if 'iot_df_clinic_main' in st.session_state and st.session_state.iot_df_clinic_main is not None and not st.session_state.iot_df_clinic_main.empty:
            st.info("No clinic environmental IoT data found for the selected period.")
        else:
            st.info("Clinic environmental IoT data is currently unavailable for the system.")
        return

    # Environmental alerts based on latest readings *in the filtered period*
    env_summary_for_tab_display = get_clinic_environmental_summary(filtered_iot_df_clinic)
    st.markdown(f"""**Environmental Alerts (Latest in Period):**
    - CO2: {env_summary_for_tab_display.get('rooms_co2_alert_latest',0)} room(s) > {app_config.CO2_LEVEL_ALERT_PPM}ppm.
    - PM2.5: {env_summary_for_tab_display.get('rooms_pm25_alert_latest',0)} room(s) > {app_config.PM25_ALERT_UGM3}µg/m³.
    - Noise: {env_summary_for_tab_display.get('rooms_noise_alert_latest',0)} room(s) > {app_config.NOISE_LEVEL_ALERT_DB}dB.""")
    if env_summary_for_tab_display.get('high_occupancy_alert_latest', False):
        st.warning(f"⚠️ **High Waiting Room Occupancy Detected:** > {app_config.TARGET_WAITING_ROOM_OCCUPANCY} persons (latest in period).")

    env_trend_plot_cols = st.columns(2)
    with env_trend_plot_cols[0]:
        if 'avg_co2_ppm' in filtered_iot_df_clinic.columns:
            hourly_avg_co2_trend = get_trend_data(filtered_iot_df_clinic, 'avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean')
            if not hourly_avg_co2_trend.empty: st.plotly_chart(plot_annotated_line_chart(hourly_avg_co2_trend, "Hourly Avg. CO2 Levels (All Rooms)", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, target_label="Alert Threshold", height=app_config.COMPACT_PLOT_HEIGHT, show_anomalies=True, date_format="%d %b, %H:%M"), use_container_width=True)
            else: st.caption("No CO2 trend data for period.")
        else: st.caption("CO2 data ('avg_co2_ppm') missing.")
    with env_trend_plot_cols[1]:
        if 'waiting_room_occupancy' in filtered_iot_df_clinic.columns:
            hourly_avg_occupancy_trend = get_trend_data(filtered_iot_df_clinic, 'waiting_room_occupancy', date_col='timestamp', period='H', agg_func='mean')
            if not hourly_avg_occupancy_trend.empty: st.plotly_chart(plot_annotated_line_chart(hourly_avg_occupancy_trend, "Hourly Avg. Waiting Room Occupancy", y_axis_title="Persons", target_line=app_config.TARGET_WAITING_ROOM_OCCUPANCY, target_label="Target Occupancy", height=app_config.COMPACT_PLOT_HEIGHT, show_anomalies=True, date_format="%d %b, %H:%M"), use_container_width=True)
            else: st.caption("No occupancy trend data for period.")
        else: st.caption("Occupancy data ('waiting_room_occupancy') missing.")

    st.markdown("---"); st.subheader("Latest Sensor Readings by Room (End of Selected Period)")
    latest_room_cols_display = ['clinic_id', 'room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour']
    available_latest_cols_env = [col for col in latest_room_cols_display if col in filtered_iot_df_clinic.columns]
    
    if all(c in available_latest_cols_env for c in ['timestamp', 'clinic_id', 'room_name']):
        latest_room_sensor_readings = filtered_iot_df_clinic.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
        if not latest_room_sensor_readings.empty:
            # Prepare for st.dataframe (serialization fix for object columns)
            df_to_display_env_latest = latest_room_sensor_readings[available_latest_cols_env].tail(15).copy()
            for col in df_to_display_env_latest.columns:
                if col == 'timestamp':
                    df_to_display_env_latest[col] = pd.to_datetime(df_to_display_env_latest[col], errors='coerce')
                    if pd.api.types.is_datetime64tz_dtype(df_to_display_env_latest[col]):
                         df_to_display_env_latest[col] = df_to_display_env_latest[col].dt.tz_localize(None)
                elif df_to_display_env_latest[col].dtype == 'object':
                    df_to_display_env_latest[col] = df_to_display_env_latest[col].fillna('N/A').astype(str).replace(['nan','None','<NA>'],'N/A',regex=False)
            
            st.dataframe(df_to_display_env_latest, use_container_width=True, height=380,
                column_config={"timestamp": st.column_config.DatetimeColumn("Last Reading", format="YYYY-MM-DD HH:mm"),
                               "avg_co2_ppm": st.column_config.NumberColumn("CO2", format="%dppm"),
                               "avg_pm25": st.column_config.NumberColumn("PM2.5", format="%.1fµg/m³"),
                               "avg_temp_celsius": st.column_config.NumberColumn("Temp", format="%.1f°C"),
                               "avg_humidity_rh": st.column_config.NumberColumn("Hum.", format="%d%%"),
                               "avg_noise_db": st.column_config.NumberColumn("Noise", format="%ddB"),
                               "waiting_room_occupancy": st.column_config.NumberColumn("Occup.", format="%d P"),
                               "patient_throughput_per_hour": st.column_config.NumberColumn("Thrpt/hr", format="%.1f"),
                               "sanitizer_dispenses_per_hour": st.column_config.NumberColumn("Sanit./hr", format="%.1f")},
                hide_index=True)
        else: st.caption("No distinct room sensor readings available for the latest point in this period.")
    else: st.caption(f"Essential columns for detailed room readings (timestamp, clinic_id, room_name) are missing from IoT data for selected period.")
