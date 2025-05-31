# test/pages/clinic_components/environment_details_tab.py
import streamlit as st
import pandas as pd
from config import app_config # For thresholds and plot heights
from utils.core_data_processing import get_clinic_environmental_summary, get_trend_data
from utils.ui_visualization_helpers import plot_annotated_line_chart

def render_environment_details(filtered_iot_df_clinic, iot_data_was_loaded_initially: bool):
    # Subheader for this tab section should be handled by the main 2_clinic_dashboard.py file
    # st.subheader("🌿 Clinic Environmental Monitoring - Trends & Details") # Usually here if it's a main section
    
    if filtered_iot_df_clinic.empty:
        if iot_data_was_loaded_initially:
            st.info("No clinic environmental IoT data found for the selected period to display details.")
        # else: # If IoT data was never loaded, main page shows a more general message
            # st.info("Clinic environmental IoT data is currently unavailable for the system.")
        return

    # Environmental alerts based on latest readings *in the filtered period*
    env_summary_for_tab_details = get_clinic_environmental_summary(filtered_iot_df_clinic)
    st.markdown(f"""**Key Environmental Indicators (Latest in Period):**
    - CO2 Alerts: {env_summary_for_tab_details.get('rooms_co2_alert_latest',0)} room(s) with CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm.
    - PM2.5 Alerts: {env_summary_for_tab_details.get('rooms_pm25_alert_latest',0)} room(s) with PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³.
    - Noise Alerts: {env_summary_for_tab_details.get('rooms_noise_alert_latest',0)} room(s) with Noise > {app_config.NOISE_LEVEL_ALERT_DB}dB.
    """)
    if env_summary_for_tab_details.get('high_occupancy_alert_latest', False):
        st.warning(f"⚠️ **High Waiting Room Occupancy Detected:** At least one area had occupancy > {app_config.TARGET_WAITING_ROOM_OCCUP**File 18: `test/pages/clinic_components/environment_details_tab.py` (Final Version)**

```python
# test/pages/clinic_components/environment_details_tab.py
import streamlit as stANCY} persons (latest reading in period).")

    st.markdown("##### Hourly Trends for Key Environmental Metrics")
    env_trend_plot_cols_details = st.columns(2)
    with env_trend_plot_cols_details[0]:
        if 'avg_co2_ppm' in filtered_iot_df_clinic.columns:
            hourly_avg_co2_trend_details = get_trend_data(filtered_iot_df_clinic, 'avg_co2_ppm', date_col='timestamp', period='H', agg_func='mean')
            if not hourly_avg_co2_trend_details.empty: 
                st.plotly_chart(plot_annotated_line_chart(
                    hourly_avg_co2_trend_details, "Hourly Avg. CO2 Levels (All Rooms in Period)", 
                    y_axis_title="CO2 (ppm)", 
                    target_line=app_config.CO2_LEVEL_ALERT_PPM, target_label="Alert Threshold", 
                    height=app_config.COMPACT_PLOT_HEIGHT, show_anomalies=True, date_format="%d %b, %H:%M"
                ), use_container_width=True)
            else: st.caption("No CO
import pandas as pd
from config import app_config
from utils.core_data_processing import get_clinic_environmental_summary, get_trend_data
from utils.ui_visualization_helpers import plot_annotated_line_chart

def render_environment_details(filtered_iot_df_clinic, iot_data_was_loaded_initially: bool): # Added iot_data_was_loaded_initially
    # Main subheader is part of the Clinic Dashboard main page or this component can add one.
    # For consistency, let's assume main subheader handled by tab name and this gives specifics.
    st.subheader("🌿 Clinic Environmental Monitoring - Trends & Detailed Readings")
    
    if filtered_iot_df_clinic.empty:
        if iot_data_was_loaded_initially: # If overall IoT data exists, but not for this specific period/filter
            st.info("No clinic environmental IoT data found for the selected period to display details and trends.")
        # If iot_data_was_loaded_initially is False, main dashboard page has already shown a general "IoT unavailable" message.
        return

    # Environmental alerts based on latest readings *in the filtered period*
    env_summary_for_tab_display = get_clinic_environmental_summary(filtered_iot_df_clinic)
    
    st.markdown("##### Current Environmental Alerts (based2 trend data for the selected period.")
        else: st.caption("CO2 data ('avg_co2_ppm') column missing for trend.")
    with env_trend_plot_cols_details[1]:
        if 'waiting_room_occupancy' in filtered_iot_df_clinic.columns:
            hourly_avg_occupancy_trend_details = get_trend_data(filtered_iot_df_clinic, 'waiting_room_occupancy', date_col='timestamp', period='H', agg_func='mean')
            if not hourly_avg_occupancy_trend_details.empty: 
                st.plotly_chart(plot_annotated_line_chart(
                    hourly_avg_occupancy_trend_details, "Hourly Avg. Waiting Room Occupancy (Period)", 
                    y_axis_title="Persons", 
                    target_line=app_config.TARGET_WAITING_ROOM_OCCUPANCY, target_label="Target Occupancy", 
                    height=app_config.COMPACT_PLOT_HEIGHT, show_anomalies=True, date_format="%d %b, %H:%M",
                    y_is_count=False # Occupancy average can be fractional
                ), use_container_width=True)
            else: st.caption("No waiting room occupancy trend data for the selected period.")
        else: st.caption("Occupancy data ('waiting_room_occupancy') column missing for trend.")

    st.markdown("---"); st.subheader("Latest Sensor Readings by Room (End of Selected Period)")
    latest_room_cols_display_env = [
        'clinic_id', 'room_name', 'timestamp', 'avg_co2_ppm', 'avg_pm25', 
        'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 
        'waiting_room_occupancy', 'patient_throughput_per_hour', 
        'sanitizer_dispenses_per_hour'
    ]
    available_latest_cols_env_tab = [col for col in latest_room_cols_display_env if col in filtered_iot_df_clinic.columns]
    
    if all on latest readings in selected period):")
    alert_messages = [
        f"- **CO2 Alerts:** {env_summary_for_tab_display.get('rooms_co2_alert_latest',0)} room(s) with CO2 > {app_config.CO2_LEVEL_ALERT_PPM}ppm.",
        f"- **PM2.5 Alerts:** {env_summary_for_tab_display.get('rooms_pm25_alert_latest',0)} room(s) with PM2.5 > {app_config.PM25_ALERT_UGM3}µg/m³.",
        f"- **Noise Alerts:** {env_summary_for_tab_display.get('rooms_noise_alert_latest',0)} room(s) with Noise > {app_config.NOISE_LEVEL_ALERT_DB}dB."
    ]
    if env_summary_for_tab_display.get('high_occupancy_alert_latest', False):
        alert_messages.append(f"- ⚠️ **High Waiting Room Occupancy Detected:** At least one area > {app_config.TARGET_WAITING_ROOM_OCCUPANCY} persons.")
    
    for msg in alert_messages:
        st.markdown(msg)
    st.markdown("---")

    # IoT Trend Charts
    st.markdown("##### Environmental Trends (Period Average)")
    env_trend_plot_cols = st.columns(2)
    with env_trend_plot_cols[0]:
        if 'avg_co2_ppm' in filtered_iot_df_clinic.columns:
            # Ensure 'timestamp' is datetime64[ns] before passing to get_trend_data
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
                    st.plotly_chart(plot_annotated_line_chart(hourly_avg_occupancy_trend, "Hourly Avg. Waiting Room Occupancy", y_axis_title="Persons", target_line=app_config.TARGET_WAITING_ROOM_OCCUPANCY, target_label="Target Occupancy", height=app_config.COMPACT_(c in available_latest_cols_env_tab for c in ['timestamp', 'clinic_id', 'room_name']):
        # Sort by timestamp to get the latest unique reading for each room
        latest_room_sensor_readings_df = filtered_iot_df_clinic.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
        
        if not latest_room_sensor_readings_df.empty:
            # Prepare for st.dataframe by converting types if necessary
            df_to_display_env_latest_tab = latest_room_sensor_readings_df[available_latest_cols_env_tab].tail(15).copy() # Display last 15 rooms with data
            for col in df_to_display_env_latest_tab.columns:
                if col == 'timestamp':
                    df_to_display_env_latest_tab[col] = pd.to_datetime(df_to_display_env_latest_tab[col], errors='coerce')
                    if pd.api.types.is_datetime64tz_dtype(df_to_display_env_latest_tab[col]):
                         df_to_display_env_latest_tab[col] = df_to_display_env_latest_tab[col].dt.tz_localize(None)
                elif df_to_display_env_latest_tab[col].dtype == 'object': # Convert other object columns to string
                    df_to_display_env_latest_tab[col] = df_to_display_env_latest_tab[col].fillna('N/A').astype(str).replace(['nan','None','<NA>'],'N/A',regex=False)
            
            st.dataframe(df_to_display_env_latest_tab, use_container_width=True, height=380,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Last Reading At", format="YYYY-MM-DD HH:mm"),
                    "avg_co2_ppm": st.column_config.NumberColumn("CO2 (ppm)", format="%d ppm"),
                    "avg_pm25": st.column_config.NumberColumn("PM2.5 (µg/m³)", format="%.1f µg/m³"),
                    "avg_temp_celsius": st.column_config.NumberColumn("Temperature (°C)", format="%.1f°C"),
                    "avg_humidity_rh": st.column_config.NumberColumn("Humidity (%RH)", format="%d%%"),
                    "avg_noise_db": st.column_config.NumberColumn("Noise Level (dB)", format="%d dB"),
                    "waiting_room_occupancy": st.column_config.NumberColumn("Occupancy", format="%d persons"),
                    "patient_throughput_per_hour": st.column_config.NumberColumn("Throughput (/hr)", format="%.1f"),
                    "sanitizer_dispenses_per_hour": st.column_config.NumberColumn("Sanitizer Use (/hr)", format="%.1f")
                }, hide_index=True)
        else: 
            st.caption("No distinct room sensor readings available for the latest point in this period after filtering.")
    else: 
        st.caption(f"Essential columns (timestamp, clinic_id, room_name) are missing from IoT data for the selected period, cannot display detailed room readings.")
