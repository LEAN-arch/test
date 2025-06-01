# test/pages/2_clinic_dashboard.py
import streamlit as st
import pandas as pd
import os
import sys 
import logging
from datetime import date, timedelta 
import numpy as np

# --- Explicitly add project root to sys.path for robust imports ---
PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_PATH)

# --- Page Configuration ---
st.set_page_config(
    page_title="Clinic Dashboard - Health Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project-specific absolute imports:
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    load_iot_clinic_environment_data,
    get_clinic_summary,
    get_clinic_environmental_summary
)
from utils.ai_analytics_engine import (
    apply_ai_models
)

# Absolute imports for components from the 'pages' package
from pages.clinic_components import kpi_display
from pages.clinic_components import environmental_kpis
from pages.clinic_components import epi_module
from pages.clinic_components import testing_insights_tab
from pages.clinic_components import supply_chain_tab
from pages.clinic_components import patient_focus_tab
from pages.clinic_components import environment_details_tab


logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_clinic():
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
    logger.info("Clinic Dashboard: Loading and enriching health records and IoT data...")
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    iot_df_raw = load_iot_clinic_environment_data(file_path=app_config.IOT_CLINIC_ENVIRONMENT_CSV)
    
    health_df_for_display = pd.DataFrame() 
    if not health_df_raw.empty:
        health_df_ai_enriched = apply_ai_models(health_df_raw)
        health_df_for_display = health_df_ai_enriched if not health_df_ai_enriched.empty else health_df_raw
    else: logger.error("Clinic Dashboard: Base health records failed to load or were empty.")
        
    iot_df_for_display = iot_df_raw if iot_df_raw is not None else pd.DataFrame()

    if health_df_for_display.empty : logger.warning("Clinic Dashboard: Health data (raw or AI enriched) is empty after processing.")
    logger.info(f"Clinic Dashboard: Data loaded. Health records: {len(health_df_for_display)}, IoT records: {len(iot_df_for_display)}")
    return health_df_for_display, iot_df_for_display

health_df_clinic_main, iot_df_clinic_main = get_clinic_dashboard_data_enriched()

health_data_available = health_df_clinic_main is not None and not health_df_clinic_main.empty
iot_data_available = iot_df_clinic_main is not None and not iot_df_clinic_main.empty

# --- Main Page Title for Clinic Dashboard ---
if not health_data_available:
    st.error("🚨 **Critical Error:** Health records data unavailable. Clinic Dashboard features will be significantly limited.")
    logger.critical("Clinic Dashboard: health_df_clinic_main is empty. Most sections cannot render.")
    # Still render title for consistency
    
st.title("🏥 Clinic Operations & Environmental Dashboard") # CORRECT CLINIC TITLE
st.markdown("**Service Efficiency, Quality of Care, Resource Management, Environment, & Local Epidemiology**") # CORRECT CLINIC SUBTITLE
st.markdown("---")
# !!! IMPORTANT: Any duplicated "DHO Dashboard" titles that were here have been removed. !!!

# --- Sidebar Filters & Date Range Setup ---
if os.path.exists(app_config.APP_LOGO): st.sidebar.image(app_config.APP_LOGO, width=100); st.sidebar.markdown("---")
else: logger.warning(f"Sidebar logo not found on Clinic Dashboard at {app_config.APP_LOGO}")
st.sidebar.header("🗓️ Clinic Filters")

min_date_overall_clinic = date.today() - timedelta(days=365*2); max_date_overall_clinic = date.today()
all_timestamps_for_range_clinic = []
if health_data_available and 'encounter_date' in health_df_clinic_main.columns:
    ts_health_clinic = pd.to_datetime(health_df_clinic_main['encounter_date'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(ts_health_clinic): ts_health_clinic = ts_health_clinic.dt.tz_localize(None)
    all_timestamps_for_range_clinic.extend(ts_health_clinic.dropna())
if iot_data_available and 'timestamp' in iot_df_clinic_main.columns:
    ts_iot_clinic = pd.to_datetime(iot_df_clinic_main['timestamp'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(ts_iot_clinic): ts_iot_clinic = ts_iot_clinic.dt.tz_localize(None)
    all_timestamps_for_range_clinic.extend(ts_iot_clinic.dropna())
if all_timestamps_for_range_clinic: 
    min_date_overall_clinic = min(all_timestamps_for_range_clinic).date()
    max_date_overall_clinic = max(all_timestamps_for_range_clinic).date()
if min_date_overall_clinic > max_date_overall_clinic: min_date_overall_clinic = max_date_overall_clinic
default_end_val_sidebar_clinic = max_date_overall_clinic; default_start_val_sidebar_clinic = default_end_val_sidebar_clinic - timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if default_start_val_sidebar_clinic < min_date_overall_clinic: default_start_val_sidebar_clinic = min_date_overall_clinic
selected_start_date_cl, selected_end_date_cl = st.sidebar.date_input("Select Date Range:", value=[default_start_val_sidebar_clinic, default_end_val_sidebar_clinic],min_value=min_date_overall_clinic, max_value=max_date_overall_clinic,key="clinic_date_range_v14")
if selected_start_date_cl > selected_end_date_cl: selected_start_date_cl = selected_end_date_cl

# --- Filter dataframes for selected period ---
filtered_health_df_clinic = pd.DataFrame(columns=(health_df_clinic_main.columns if health_data_available else []))
if health_data_available and 'encounter_date' in health_df_clinic_main.columns: # Check column exists before .dt
    health_df_clinic_main.loc[:, 'encounter_date'] = pd.to_datetime(health_df_clinic_main['encounter_date'], errors='coerce')
    health_df_clinic_main.loc[:, 'encounter_date_obj'] = health_df_clinic_main['encounter_date'].dt.date
    valid_dates_for_filtering_health = health_df_clinic_main['encounter_date_obj'].notna()
    df_health_to_filter_period = health_df_clinic_main[valid_dates_for_filtering_health].copy()
    if not df_health_to_filter_period.empty: filtered_health_df_clinic = df_health_to_filter_period[ (df_health_to_filter_period['encounter_date_obj'] >= selected_start_date_cl) & (df_health_to_filter_period['encounter_date_obj'] <= selected_end_date_cl) ].copy()

filtered_iot_df_clinic = pd.DataFrame(columns=(iot_df_clinic_main.columns if iot_data_available else []))
if iot_data_available and 'timestamp' in iot_df_clinic_main.columns: # Check column exists
    iot_df_clinic_main.loc[:,'timestamp'] = pd.to_datetime(iot_df_clinic_main['timestamp'], errors='coerce')
    iot_df_clinic_main.loc[:,'timestamp_date_obj'] = iot_df_clinic_main['timestamp'].dt.date
    valid_dates_for_filtering_iot = iot_df_clinic_main['timestamp_date_obj'].notna()
    df_iot_to_filter_period = iot_df_clinic_main[valid_dates_for_filtering_iot].copy()
    if not df_iot_to_filter_period.empty: filtered_iot_df_clinic = df_iot_to_filter_period[ (df_iot_to_filter_period['timestamp_date_obj'] >= selected_start_date_cl) & (df_iot_to_filter_period['timestamp_date_obj'] <= selected_end_date_cl) ].copy()

date_range_display_str = f"({selected_start_date_cl.strftime('%d %b %Y')} - {selected_end_date_cl.strftime('%d %b %Y')})"
base_cols_for_empty_summary_clinic = health_df_clinic_main.columns if health_data_available else ['test_type', 'test_result', 'test_turnaround_days', 'sample_status', 'encounter_date']
clinic_service_kpis = get_clinic_summary(filtered_health_df_clinic if not filtered_health_df_clinic.empty else pd.DataFrame(columns=base_cols_for_empty_summary_clinic))

# Display messages if filtered data is empty
if health_data_available and filtered_health_df_clinic.empty and (health_df_clinic_main['encounter_date_obj'].notna() & ((health_df_clinic_main['encounter_date_obj'] >= selected_start_date_cl) & (health_df_clinic_main['encounter_date_obj'] <= selected_end_date_cl))).any():
    # This means main data had relevant dates but filter for *this period* yielded nothing.
    st.info(f"ℹ️ No health encounter data for the selected period: {date_range_display_str}.")
if iot_data_available and filtered_iot_df_clinic.empty and (iot_df_clinic_main['timestamp_date_obj'].notna() & ((iot_df_clinic_main['timestamp_date_obj'] >= selected_start_date_cl) & (iot_df_clinic_main['timestamp_date_obj'] <= selected_end_date_cl))).any():
    st.info(f"ℹ️ No IoT data for the selected period: {date_range_display_str}.")


# --- Display Main Sections using Components ---
if health_data_available:
    kpi_display.render_main_clinic_kpis(clinic_service_kpis, date_range_display_str)
    kpi_display.render_disease_specific_kpis(clinic_service_kpis)
else: st.warning("Main clinic performance KPIs cannot be displayed as health data is unavailable.")
environmental_kpis.render_clinic_environmental_kpis(filtered_iot_df_clinic, date_range_display_str, iot_data_available)
st.markdown("---")
if health_data_available : # Epi module needs health data
    epi_module.render_clinic_epi_module(filtered_health_df_clinic, date_range_display_str) # Pass the period-filtered health data
else: st.warning("Clinic Epidemiology module requires health data.")
st.markdown("---")

# --- Tabbed Interface ---
# test/pages/2_clinic_dashboard.py
# ... (imports, data loading, KPI sections, epi_module call) ...
st.markdown("---") # Separator before tabs

# --- Tabbed Interface using component functions ---
tab_titles_for_clinic_dashboard = [ # Using a consistent name for the list of titles
    "🔬 Testing Insights", 
    "💊 Supply Chain", 
    "🧍 Patient Focus", 
    "🌿 Environment Details"
]

# Unpack tabs with consistent names
tab_tests_display, tab_supplies_display, tab_patients_display, tab_environment_display = st.tabs(
    tab_titles_for_clinic_dashboard 
) # Line ~159

# Use the EXACT same names in the 'with' statements
with tab_tests_display: # CORRECTED
    testing_insights_tab.render_testing_insights(filtered_health_df_clinic, clinic_service_kpis)

with tab_supplies_display: # CORRECTED (Line ~174 causing the NameError)
    supply_chain_tab.render_supply_chain(
        health_df_clinic_main if health_data_available else pd.DataFrame(columns=(health_df_clinic_main.columns if health_data_available else [])), 
        filtered_health_df_clinic
    )

with tab_patients_display: # CORRECTED
    patient_focus_tab.render_patient_focus(filtered_health_df_clinic)

with tab_environment_display: # CORRECTED
    environment_details_tab.render_environment_details(filtered_iot_df_clinic, iot_data_available)
