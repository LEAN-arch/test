# test/pages/2_clinic_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
from datetime import date, timedelta
import numpy as np

from config import app_config
from utils.core_data_processing import (
    load_health_records, load_iot_clinic_environment_data,
    get_clinic_summary, get_clinic_environmental_summary, # Main summary functions
    # get_trend_data, get_supply_forecast_data, get_patient_alerts_for_clinic are now mostly used within components
)
from utils.ai_analytics_engine import apply_ai_models # For AI scores on loaded data
# Import components
from pages.clinic_components import kpi_display, environmental_kpis, epi_module, \
                                    testing_insights_tab, supply_chain_tab, \
                                    patient_focus_tab, environment_details_tab


st.set_page_config(page_title="Clinic Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_clinic(): # Definition unchanged
    css_path = app_config.STYLE_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path) as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else: logger.warning(f"Clinic CSS file not found: {css_path}.")
load_css_clinic()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading clinic operational data...")
def get_clinic_dashboard_data_enriched(): # Definition unchanged
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    iot_df = load_iot_clinic_environment_data(file_path=app_config.IOT_CLINIC_ENVIRONMENT_CSV)
    if health_df_raw.empty: return pd.DataFrame(), iot_df if iot_df is not None else pd.DataFrame()
    health_df_ai_enriched = apply_ai_models(health_df_raw)
    return health_df_ai_enriched if not health_df_ai_enriched.empty else health_df_raw, iot_df if iot_df is not None else pd.DataFrame()

health_df_clinic_main, iot_df_clinic_main = get_clinic_dashboard_data_enriched()

health_data_available = health_df_clinic_main is not None and not health_df_clinic_main.empty
iot_data_available = iot_df_clinic_main is not None and not iot_df_clinic_main.empty

if not health_data_available:
    st.error("🚨 Critical Error: Health records data unavailable. Clinic Dashboard features will be limited."); 
    # Don't stop if IoT might still be useful for some parts or if other non-health components could render

st.title("🏥 Clinic Operations & Environmental Dashboard")
st.markdown("**Service Efficiency, Quality of Care, Resource Management, Environment, & Local Epidemiology**") # Updated subtitle
st.markdown("---")

# Sidebar (Date Range Logic - unchanged from previous correction)
# ... (min_date_data_clinic, max_date_data_clinic, selected_start_date_cl, selected_end_date_cl logic remains here)
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, use_column_width='auto'); st.sidebar.markdown("---")
st.sidebar.header("🗓️ Clinic Filters")
# (Copy the robust date range selection logic from the previously corrected version of this file)
all_potential_timestamps_clinic = []
if health_data_available and 'encounter_date' in health_df_clinic_main.columns:
    encounter_dates_ts = pd.to_datetime(health_df_clinic_main['encounter_date'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(encounter_dates_ts): encounter_dates_ts = encounter_dates_ts.dt.tz_localize(None)
    all_potential_timestamps_clinic.extend(encounter_dates_ts.dropna())
if iot_data_available and 'timestamp' in iot_df_clinic_main.columns:
    iot_timestamps_ts = pd.to_datetime(iot_df_clinic_main['timestamp'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(iot_timestamps_ts): iot_timestamps_ts = iot_timestamps_ts.dt.tz_localize(None)
    all_potential_timestamps_clinic.extend(iot_timestamps_ts.dropna())
all_valid_timestamps_clinic = [ts for ts in all_potential_timestamps_clinic if isinstance(ts, pd.Timestamp)]
min_date_data_clinic = min(all_valid_timestamps_clinic).date() if all_valid_timestamps_clinic else date.today() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND * 3)
max_date_data_clinic = max(all_valid_timestamps_clinic).date() if all_valid_timestamps_clinic else date.today()
if min_date_data_clinic > max_date_data_clinic: min_date_data_clinic = max_date_data_clinic
default_end_val_clinic = max_date_data_clinic
default_start_val_clinic = max_date_data_clinic - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if default_start_val_clinic < min_date_data_clinic: default_start_val_clinic = min_date_data_clinic
selected_start_date_cl, selected_end_date_cl = st.sidebar.date_input("Select Date Range:",value=[default_start_val_clinic, default_end_val_clinic],min_value=min_date_data_clinic,max_value=max_date_data_clinic,key="clinic_date_range_v13")
if selected_start_date_cl > selected_end_date_cl: st.sidebar.error("Start date must be before end date."); selected_start_date_cl = selected_end_date_cl


# Filter dataframes
# ... (Data filtering logic for filtered_health_df_clinic and filtered_iot_df_clinic - unchanged)
# (Copy the robust filtering logic from the previously corrected version of this file)
if health_data_available:
    health_df_clinic_main['encounter_date'] = pd.to_datetime(health_df_clinic_main['encounter_date'], errors='coerce')
    health_df_clinic_main['encounter_date_obj'] = health_df_clinic_main['encounter_date'].dt.date
    valid_dates_for_filtering_health = health_df_clinic_main['encounter_date_obj'].notna()
    df_health_to_filter = health_df_clinic_main[valid_dates_for_filtering_health].copy()
    filtered_health_df_clinic = df_health_to_filter[(df_health_to_filter['encounter_date_obj'] >= selected_start_date_cl) & (df_health_to_filter['encounter_date_obj'] <= selected_end_date_cl)].copy() if not df_health_to_filter.empty else pd.DataFrame(columns=health_df_clinic_main.columns)
if iot_data_available:
    iot_df_clinic_main['timestamp'] = pd.to_datetime(iot_df_clinic_main['timestamp'], errors='coerce')
    iot_df_clinic_main['timestamp_date_obj'] = iot_df_clinic_main['timestamp'].dt.date
    valid_dates_for_filtering_iot = iot_df_clinic_main['timestamp_date_obj'].notna()
    df_iot_to_filter = iot_df_clinic_main[valid_dates_for_filtering_iot].copy()
    filtered_iot_df_clinic = df_iot_to_filter[(df_iot_to_filter['timestamp_date_obj'] >= selected_start_date_cl) & (df_iot_to_filter['timestamp_date_obj'] <= selected_end_date_cl)].copy() if not df_iot_to_filter.empty else pd.DataFrame(columns=iot_df_clinic_main.columns)


date_range_display_str = f"({selected_start_date_cl.strftime('%d %b %Y')} - {selected_end_date_cl.strftime('%d %b %Y')})"
# Calculate main clinic service KPIs from the filtered health data
clinic_service_kpis = get_clinic_summary(filtered_health_df_clinic if not filtered_health_df_clinic.empty else pd.DataFrame(columns=(health_df_clinic_main.columns if health_data_available else [])))

# Render KPIs sections
kpi_display.render_main_clinic_kpis(clinic_service_kpis, date_range_display_str)
kpi_display.render_disease_specific_kpis(clinic_service_kpis)
environmental_kpis.render_clinic_environmental_kpis(filtered_iot_df_clinic, date_range_display_str)

# Render NEW Epidemiology Module section
# This module should define its own subheader or rely on main page's flow.
if health_data_available: # Only render if base health data was available for context
    epi_module.render_clinic_epi_module(filtered_health_df_clinic, date_range_display_str) # Pass filtered data for the period
else:
    st.warning("Epidemiology module cannot be displayed as health data is unavailable.")
st.markdown("---")


# Tabbed interface using component functions
tab_titles_clinic = ["🔬 Testing Insights", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment Details"]
tab_tests, tab_supplies, tab_patients_alerts, tab_environment_detail = st.tabs(tab_titles_clinic)

with tab_tests:
    # The component function should now contain all the logic previously in this tab
    testing_insights_tab.render_testing_insights(filtered_health_df_clinic, clinic_service_kpis)

with tab_supplies:
    supply_chain_tab.render_supply_chain(health_df_clinic_main, filtered_health_df_clinic) # Main for history, filtered for context

with tab_patients_alerts:
    patient_focus_tab.render_patient_focus(filtered_health_df_clinic)

with tab_environment_detail:
    environment_details_tab.render_environment_details(filtered_iot_df_clinic)
