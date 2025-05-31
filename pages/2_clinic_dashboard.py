# test/pages/2_clinic_dashboard.py
import streamlit as st
import pandas as pd
import os
import logging
from datetime import date, timedelta # Ensure timedelta is imported
import numpy as np

from config import app_config
from utils.core_data_processing import (
    load_health_records, load_iot_clinic_environment_data,
    get_clinic_summary, # This summary is key for multiple components
    # get_clinic_environmental_summary is now called within environmental_kpis and environment_details_tab
    # get_trend_data, get_supply_forecast_data, get_patient_alerts_for_clinic are now primarily used within specific components
)
from utils.ai_analytics_engine import apply_ai_models
# Import all components for the clinic dashboard
from pages.clinic_components import kpi_display, environmental_kpis, epi_module, \
                                    testing_insights_tab, supply_chain_tab, \
                                    patient_focus_tab, environment_details_tab

st.set_page_config(page_title="Clinic Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_clinic():
    css_path = app_config.STYLE_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path) as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        logger.info("Clinic Dashboard CSS loaded.")
    else: logger.warning(f"Clinic CSS file not found at {css_path}.")
load_css_clinic()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading clinic operational data...")
def get_clinic_dashboard_data_enriched():
    logger.info("Clinic Dashboard: Loading and enriching all data...")
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    iot_df_raw = load_iot_clinic_environment_data(file_path=app_config.IOT_CLINIC_ENVIRONMENT_CSV)
    
    health_df_for_display = pd.DataFrame() # Initialize
    if not health_df_raw.empty:
        health_df_ai_enriched = apply_ai_models(health_df_raw)
        health_df_for_display = health_df_ai_enriched if not health_df_ai_enriched.empty else health_df_raw
    else:
        logger.error("Clinic Dashboard: Base health records failed to load.")
        
    # iot_df does not need AI enrichment for its current use.
    iot_df_for_display = iot_df_raw if iot_df_raw is not None else pd.DataFrame()

    if health_df_for_display.empty : logger.warning("Clinic Dashboard: Health data (raw or AI enriched) is empty.")
    logger.info(f"Clinic Dashboard: Data loaded. Health records: {len(health_df_for_display)}, IoT records: {len(iot_df_for_display)}")
    return health_df_for_display, iot_df_for_display

health_df_clinic_main, iot_df_clinic_main = get_clinic_dashboard_data_enriched()

health_data_available = health_df_clinic_main is not None and not health_df_clinic_main.empty
iot_data_available = iot_df_clinic_main is not None and not iot_df_clinic_main.empty

if not health_data_available:
    st.error("🚨 Critical Error: Health records data unavailable. Clinic Dashboard features will be significantly limited.")
    logger.critical("Clinic Dashboard: health_df_clinic_main is empty. Most sections cannot render.")

st.title("🏥 Clinic Operations & Environmental Dashboard")
st.markdown("**Service Efficiency, Quality of Care, Resource Management, Environment, & Local Epidemiology**")
st.markdown("---")

# --- Sidebar Filters ---
if os.path.exists(app_config.APP_LOGO): st.sidebar.image(app_config.APP_LOGO, use_column_width='auto'); st.sidebar.markdown("---")
st.sidebar.header("🗓️ Clinic Filters")

# Date Range Setup (robust version)
min_date_overall_clinic = date.today() - timedelta(days=365*2) # Wider default if no data
max_date_overall_clinic = date.today()
all_timestamps_for_range_clinic = []
if health_data_available and 'encounter_date' in health_df_clinic_main.columns:
    ts_health = pd.to_datetime(health_df_clinic_main['encounter_date'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(ts_health): ts_health = ts_health.dt.tz_localize(None)
    all_timestamps_for_range_clinic.extend(ts_health.dropna())
if iot_data_available and 'timestamp' in iot_df_clinic_main.columns:
    ts_iot = pd.to_datetime(iot_df_clinic_main['timestamp'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(ts_iot): ts_iot = ts_iot.dt.tz_localize(None)
    all_timestamps_for_range_clinic.extend(ts_iot.dropna())

if all_timestamps_for_range_clinic:
    min_date_overall_clinic = min(all_timestamps_for_range_clinic).date()
    max_date_overall_clinic = max(all_timestamps_for_range_clinic).date()
if min_date_overall_clinic > max_date_overall_clinic: min_date_overall_clinic = max_date_overall_clinic

default_end_val_sidebar_clinic = max_date_overall_clinic
default_start_val_sidebar_clinic = default_end_val_sidebar_clinic - timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if default_start_val_sidebar_clinic < min_date_overall_clinic: default_start_val_sidebar_clinic = min_date_overall_clinic

selected_start_date_cl, selected_end_date_cl = st.sidebar.date_input(
    "Select Date Range for Analysis:", value=[default_start_val_sidebar_clinic, default_end_val_sidebar_clinic],
    min_value=min_date_overall_clinic, max_value=max_date_overall_clinic,
    key="clinic_date_range_selector_v13" # Incremented
)
if selected_start_date_cl > selected_end_date_cl: st.sidebar.error("Start date must be before end date."); selected_start_date_cl = selected_end_date_cl

# --- Filter dataframes for selected period ---
filtered_health_df_clinic = pd.DataFrame()
if health_data_available:
    # Ensure correct conversion before filtering
    health_df_clinic_main['encounter_date_obj'] = pd.to_datetime(health_df_clinic_main['encounter_date'], errors='coerce').dt.date
    health_df_to_filter_period = health_df_clinic_main[health_df_clinic_main['encounter_date_obj'].notna()]
    if not health_df_to_filter_period.empty:
        filtered_health_df_clinic = health_df_to_filter_period[
            (health_df_to_filter_period['encounter_date_obj'] >= selected_start_date_cl) &
            (health_df_to_filter_period['encounter_date_obj'] <= selected_end_date_cl)
        ].copy()
    if filtered_health_df_clinic.empty and not health_df_to_filter_period.empty: # Data existed but not in range
        st.info(f"ℹ️ No health encounter data available for period: {selected_start_date_cl.strftime('%d %b %Y')} - {selected_end_date_cl.strftime('%d %b %Y')}.")

filtered_iot_df_clinic = pd.DataFrame()
if iot_data_available:
    iot_df_clinic_main['timestamp_date_obj'] = pd.to_datetime(iot_df_clinic_main['timestamp'], errors='coerce').dt.date
    iot_df_to_filter_period = iot_df_clinic_main[iot_df_clinic_main['timestamp_date_obj'].notna()]
    if not iot_df_to_filter_period.empty:
        filtered_iot_df_clinic = iot_df_to_filter_period[
            (iot_df_to_filter_period['timestamp_date_obj'] >= selected_start_date_cl) &
            (iot_df_to_filter_period['timestamp_date_obj'] <= selected_end_date_cl)
        ].copy()
    if filtered_iot_df_clinic.empty and not iot_df_to_filter_period.empty:
         st.info(f"ℹ️ No IoT environmental data for period: {selected_start_date_cl.strftime('%d %b %Y')} - {selected_end_date_cl.strftime('%d %b %Y')}.")
elif not iot_data_available : st.info("ℹ️ IoT environmental data is currently unavailable for this system.")


# --- Display Main Sections (KPIs, Epi Module, then Tabs) ---
date_range_display_str = f"({selected_start_date_cl.strftime('%d %b %Y')} - {selected_end_date_cl.strftime('%d %b %Y')})"
clinic_service_kpis = get_clinic_summary(filtered_health_df_clinic if not filtered_health_df_clinic.empty else pd.DataFrame(columns=(health_df_clinic_main.columns if health_data_available else [])))

# Render Main KPIs
if health_data_available:
    kpi_display.render_main_clinic_kpis(clinic_service_kpis, date_range_display_str)
    kpi_display.render_disease_specific_kpis(clinic_service_kpis)
else:
    st.warning("Main clinic performance KPIs cannot be displayed as health data is unavailable for the selected period.")

# Render Environmental KPIs (only if IoT data for period)
environmental_kpis.render_clinic_environmental_kpis(filtered_iot_df_clinic, date_range_display_str, iot_data_available)
st.markdown("---")

# Render NEW Epidemiology Module section
if health_data_available: # Epidemiology module depends on health data
    epi_module.render_clinic_epi_module(filtered_health_df_clinic, date_range_display_str)
else:
    st.warning("Clinic Epidemiology module cannot be displayed as health data is unavailable for the selected period.")
st.markdown("---")

# Tabbed interface using component functions
tab_titles_clinic = ["🔬 Testing Insights", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment Details"]
tab_tests_disp, tab_supplies_disp, tab_patients_alerts_disp, tab_environment_detail_disp = st.tabs(tab_titles_clinic)

with tab_tests_disp:
    testing_insights_tab.render_testing_insights(filtered_health_df_clinic, clinic_service_kpis)

with tab_supplies_disp:
    # Supply chain component needs main_df for historical rates and filtered for context of period.
    supply_chain_tab.render_supply_chain(health_df_clinic_main if health_data_available else pd.DataFrame(), filtered_health_df_clinic)

with tab_patients_alerts_disp:
    patient_focus_tab.render_patient_focus(filtered_health_df_clinic)

with tab_environment_detail_disp:
    environment_details_tab.render_environment_details(filtered_iot_df_clinic, iot_data_available) # Pass iot_data_available flag
