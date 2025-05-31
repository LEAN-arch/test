# test/pages/clinic_components/environmental_kpis.py
import streamlit as st
import pandas as pd # For pd.notna
import numpy as np # For np.nan
from config import app_config
from utils.ui_visualization_helpers import render_kpi_card
from utils.core_data_processing import get_clinic_environmental_summary

def render_clinic_environmental_kpis(filtered_iot_df_clinic, date_range_display_str, iot_data_was_loaded_initially: bool):
    # This component is responsible for the Environment Snapshot KPIs section
    
    if filtered_iot_df_clinic.empty:
        if iot_data_was_loaded_initially: # If IoT data exists in general but not for this period
            st.info(f"No IoT environmental data found for the period {date_range_display_str} to display snapshot KPIs.")
        # If iot_data_was_loaded_initially is False, the main page has already shown a general "IoT unavailable" message
        return

    st.subheader(f"Clinic Environment Snapshot {date_range_display_str}")
    clinic_env_kpis = get_clinic_environmental_summary(filtered_iot_df_clinic)
    
    kpi_cols_env = st.columns(4)
    with kpi_cols_env[0]:
        avg_co2 = clinic_env_kpis.get('avg_co2_overall', np.nan)
        co2_alert_rooms = clinic_env_kpis.get('rooms_co2_alert_latest', 0)
        co2_status = "Neutral"
        if pd.notna(avg_co2):
            co2_status = "High" if co2_alert_rooms > 0 else ("Moderate" if avg_co2 > app_config.CO2_LEVEL_IDEAL_PPM else "Low")
        render_kpi_card("Avg. CO2 (All Rooms)", f"{avg_co2:.0f} ppm" if pd.notna(avg_co2) else "N/A", "💨", 
                        status=co2_status, 
                        help_text=f"Period average CO2. {co2_alert_rooms} room(s) currently > {app_config.CO2_LEVEL_ALERT_PPM}ppm.")
    with kpi_cols_env[1]:
        avg_pm25 = clinic_env_kpis.get('avg_pm25_overall', np.nan)
        pm25_alert_rooms = clinic_env_kpis.get('rooms_pm25_alert_latest', 0)
        pm25_status = "Neutral"
        if pd.notna(avg_pm25):
            pm25_status = "High" if pm25_alert_rooms > 0 else ("Moderate" if avg_pm25 > app_config.PM25_IDEAL_UGM3 else "Low")
        render_kpi_card("Avg. PM2.5 (All Rooms)", f"{avg_pm25:.1f} µg/m³" if pd.notna(avg_pm25) else "N/A", "🌫️", 
                        status=pm25_status, 
                        help_text=f"Period average PM2.5. {pm25_alert_rooms} room(s) currently > {app_config.PM25_ALERT_UGM3}µg/m³.")
    with kpi_cols_env[2]:
        avg_occ = clinic_env_kpis.get('avg_occupancy_overall', np.nan)
        occ_alert = clinic_env_kpis.get('high_occupancy_alert_latest', False)
        occ_status = "Neutral"
        if pd.notna(avg_occ):
            occ_status = "High" if occ_alert else ("Moderate" if avg_occ > (app_config.TARGET_WAITING_ROOM_OCCUPANCY * 0.75) else "Low")
        render_kpi_card("Avg. Waiting Room Occupancy", f"{avg_occ:.1f} ppl" if pd.notna(avg_occ) else "N/A", "👨‍👩‍👧‍👦", 
                        status=occ_status, 
                        help_text=f"Average waiting room occupancy. Target < {app_config.TARGET_WAITING_ROOM_OCCUPANCY}.")
    with kpi_cols_env[3]:
        noise_alert_rooms = clinic_env_kpis.get('rooms_noise_alert_latest', 0)
        render_kpi_card("High Noise Alerts", str(noise_alert_rooms), "🔊", 
                        status="High" if noise_alert_rooms > 0 else "Low", 
                        help_text=f"Number of rooms with latest noise levels > {app_config.NOISE_LEVEL_ALERT_DB}dB.")
    st.markdown("---") # Separator after this section
