# test/pages/clinic_components/environmental_kpis.py
import streamlit as st
from config import app_config
from utils.ui_visualization_helpers import render_kpi_card
from utils.core_data_processing import get_clinic_environmental_summary # In case it's called here directly with different filter

def render_clinic_environmental_kpis(filtered_iot_df_clinic, date_range_display_str):
    if not filtered_iot_df_clinic.empty:
        st.subheader(f"Clinic Environment Snapshot {date_range_display_str}")
        clinic_env_kpis = get_clinic_environmental_summary(filtered_iot_df_clinic)
        
        kpi_cols_env = st.columns(4)
        with kpi_cols_env[0]:
            avg_co2 = clinic_env_kpis.get('avg_co2_overall', 0.0)
            co2_alert = clinic_env_kpis.get('rooms_co2_alert_latest', 0)
            render_kpi_card("Avg. CO2", f"{avg_co2:.0f} ppm", "💨", status="High" if co2_alert > 0 else ("Moderate" if avg_co2 > app_config.CO2_LEVEL_IDEAL_PPM else "Low"), help_text=f"Period Avg. {co2_alert} room(s) > {app_config.CO2_LEVEL_ALERT_PPM}ppm now.")
        with kpi_cols_env[1]:
            avg_pm25 = clinic_env_kpis.get('avg_pm25_overall', 0.0)
            pm25_alert = clinic_env_kpis.get('rooms_pm25_alert_latest', 0)
            render_kpi_card("Avg. PM2.5", f"{avg_pm25:.1f} µg/m³", "🌫️", status="High" if pm25_alert > 0 else ("Moderate" if avg_pm25 > app_config.PM25_IDEAL_UGM3 else "Low"), help_text=f"Period Avg. {pm25_alert} room(s) > {app_config.PM25_ALERT_UGM3}µg/m³ now.")
        with kpi_cols_env[2]:
            avg_occ = clinic_env_kpis.get('avg_occupancy_overall', 0.0)
            occ_alert = clinic_env_kpis.get('high_occupancy_alert_latest', False)
            render_kpi_card("Avg. Occupancy", f"{avg_occ:.1f} ppl", "👨‍👩‍👧‍👦", status="High" if occ_alert else ("Moderate" if avg_occ > (app_config.TARGET_WAITING_ROOM_OCCUPANCY*0.75) else "Low"), help_text=f"Avg Waiting Room Occupancy. Target < {app_config.TARGET_WAITING_ROOM_OCCUPANCY}.")
        with kpi_cols_env[3]:
            noise_alert = clinic_env_kpis.get('rooms_noise_alert_latest', 0)
            render_kpi_card("Noise Alerts", str(noise_alert), "🔊", status="High" if noise_alert > 0 else "Low", help_text=f"Rooms > {app_config.NOISE_LEVEL_ALERT_DB}dB now.")
        st.markdown("---")
    elif 'iot_df_clinic_main' in st.session_state and st.session_state.iot_df_clinic_main is not None and not st.session_state.iot_df_clinic_main.empty:
        st.info("No IoT data for selected period for Environmental KPIs.")
    # else: implies no IoT data loaded at all, handled by main page
