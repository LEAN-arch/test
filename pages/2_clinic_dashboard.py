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

# Project-specific absolute imports (now should work reliably):
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    load_iot_clinic_environment_data,
    get_clinic_summary,
    get_clinic_environmental_summary # <<< CORRECTED: Space removed
)
from utils.ai_analytics_engine import (
    apply_ai_models
)

# Project-specific relative imports for components
# Requires appropriate __init__.py files in 'pages' and 'pages/clinic_components'
from .clinic_components import kpi_display
from .clinic_components import environmental_kpis
from .clinic_components import epi_module
from .clinic_components import testing_insights_tab
from .clinic_components import supply_chain_tab
from .clinic_components import patient_focus_tab
from .clinic_components import environment_details_tab

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
    iot_df_raw = load_iot_clinic_environment_data(file_path=app_config.IOT_CLINIC_ENVIRONMENT_CSV) # Load IoT data
    
    health_df_for_display = pd.DataFrame() # Initialize
    if not health_df_raw.empty:
        health_df_ai_enriched = apply_ai_models(health_df_raw) # Apply AI enrichment
        health_df_for_display = health_df_ai_enriched if not health_df_ai_enriched.empty else health_df_raw
    else:
        logger.error("Clinic Dashboard: Base health records failed to load or were empty.")
        
    iot_df_for_display = iot_df_raw if iot_df_raw is not None else pd.DataFrame()

    if health_df_for_display.empty:
        logger.warning("Clinic Dashboard: Health data (raw or AI enriched) is empty after processing.")
    logger.info(f"Clinic Dashboard: Data loaded. Health records: {len(health_df_for_display)}, IoT records: {len(iot_df_for_display)}")
    return health_df_for_display, iot_df_for_display

health_df_clinic_main, iot_df_clinic_main = get_clinic_dashboard_data_enriched()

# --- Main Page Rendering ---
health_data_available = health_df_clinic_main is not None and not health_df_clinic_main.empty
iot_data_available = iot_df_clinic_main is not None and not iot_df_clinic_main.empty

if not health_data_available:
    st.error("🚨 **Critical Error:** Health records data is unavailable or failed to process. Most Clinic Dashboard features will be significantly limited.")
    logger.critical("Clinic Dashboard: health_df_clinic_main is empty. Most sections cannot render.")
    # Allow app to proceed to render sidebar and potentially some static text or IoT-only components if they existed
    # st.stop() would halt everything.

st.title("🏥 Clinic Operations & Environmental Dashboard")
st.markdown("**Service Efficiency, Quality of Care, Resource Management, Environment, & Local Epidemiology**")
st.markdown("---")

# --- Sidebar Filters & Date Range Setup ---
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, use_column_width='auto'); st.sidebar.markdown("---")
st.sidebar.header("🗓️ Clinic Filters")

# Determine date range from available data (Corrected from previous fixes)
min_date_overall_clinic = date.today() - timedelta(days=365*2)
max_date_overall_clinic = date.today()
all_timestamps_for_range_clinic = []
if health_data_available and 'encounter_date' in health_df_clinic_main.columns:
    ts_health_clinic = pd.to_datetime(health_df_clinic_main['encounter_date'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(ts_health_clinic): ts_health_clinic = ts_health_clinic.dt.tz_localize(None)
    all_timestamps_for_range_clinic.extend(ts_health_clinic.dropna())
if iot_data_available and 'timestamp' in iot_df_clinic_main.columns:
    ts_iot_clinic = pd.to_datetime(iot_df_clinic_main['timestamp'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(ts_iot_clinic): ts_iot_clinic = ts_iot_clinic.dt.tz_localize(None)
    all_timestamps_for_range_clinic.extend(ts_iot_clinic.dropna())

if all_timestamps_for_range_clinic: # Check if list is not empty
    min_date_overall_clinic = min(all_timestamps_for_range_clinic).date()
    max_date_overall_clinic = max(all_timestamps_for_range_clinic).date()
if min_date_overall_clinic > max_date_overall_clinic: min_date_overall_clinic = max_date_overall_clinic

default_end_val_sidebar_clinic = max_date_overall_clinic
default_start_val_sidebar_clinic = default_end_val_sidebar_clinic - timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if default_start_val_sidebar_clinic < min_date_overall_clinic: default_start_val_sidebar_clinic = min_date_overall_clinic

selected_start_date_cl, selected_end_date_cl = st.sidebar.date_input(
    "Select Date Range for Analysis:", value=[default_start_val_sidebar_clinic, default_end_val_sidebar_clinic],
    min_value=min_date_overall_clinic, max_value=max_date_overall_clinic,
    key="clinic_date_range_selector_v14" # Incremented key
)
if selected_start_date_cl > selected_end_date_cl: st.sidebar.error("Start date must be before end date."); selected_start_date_cl = selected_end_date_cl

# --- Filter dataframes for selected period (Corrected and Refined) ---
filtered_health_df_clinic = pd.DataFrame()
if health_data_available:
    health_df_clinic_main['encounter_date'] = pd.to_datetime(health_df_clinic_main['encounter_date'], errors='coerce')
    health_df_clinic_main['encounter_date_obj'] = health_df_clinic_main['encounter_date'].dt.date
    valid_dates_for_filtering_health = health_df_clinic_main['encounter_date_obj'].notna()
    df_health_to_filter_period = health_df_clinic_main[valid_dates_for_filtering_health].copy()
    if not df_health_to_filter_period.empty:
        filtered_health_df_clinic = df_health_to_filter_period[
            (df_health_to_filter_period['encounter_date_obj'] >= selected_start_date_cl) &
            (df_health_to_filter_period['encounter_date_obj'] <= selected_end_date_cl)
        ].copy()
    else: filtered_health_df_clinic = pd.DataFrame(columns=health_df_clinic_main.columns)
    if filtered_health_df_clinic.empty and not df_health_to_filter_period.empty :
        st.info(f"ℹ️ No health encounter data available for period: {selected_start_date_cl.strftime('%d %b %Y')} - {selected_end_date_cl.strftime('%d %b %Y')}.")

filtered_iot_df_clinic = pd.DataFrame()
if iot_data_available:
    iot_df_clinic_main['timestamp'] = pd.to_datetime(iot_df_clinic_main['timestamp'], errors='coerce')
    iot_df_clinic_main['timestamp_date_obj'] = iot_df_clinic_main['timestamp'].dt.date
    valid_dates_for_filtering_iot = iot_df_clinic_main['timestamp_date_obj'].notna()
    df_iot_to_filter_period = iot_df_clinic_main[valid_dates_for_filtering_iot].copy()
    if not df_iot_to_filter_period.empty:
        filtered_iot_df_clinic = df_iot_to_filter_period[
            (df_iot_to_filter_period['timestamp_date_obj'] >= selected_start_date_cl) &
            (df_iot_to_filter_period['timestamp_date_obj'] <= selected_end_date_cl)
        ].copy()
    else: filtered_iot_df_clinic = pd.DataFrame(columns=iot_df_clinic_main.columns)
    if filtered_iot_df_clinic.empty and not df_iot_to_filter_period.empty:
         st.info(f"ℹ️ No IoT environmental data for period: {selected_start_date_cl.strftime('%d %b %Y')} - {selected_end_date_cl.strftime('%d %b %Y')}.")
elif not iot_data_available : st.info("ℹ️ IoT environmental data is currently unavailable for this system.")

# --- Calculate main clinic service KPIs for the filtered health data ---
base_health_cols_for_empty = health_df_clinic_main.columns if health_data_available else []
clinic_service_kpis = get_clinic_summary(filtered_health_df_clinic if not filtered_health_df_clinic.empty else pd.DataFrame(columns=base_health_cols_for_empty))

# --- Display Main Sections (KPIs, Epi Module, then Tabs) ---
date_range_display_str = f"({selected_start_date_cl.strftime('%d %b %Y')} - {selected_end_date_cl.strftime('%d %b %Y')})"

# Render Main KPIs (only if health data was loaded initially, component will handle empty filtered_health_df_clinic)
if health_data_available:
    kpi_display.render_main_clinic_kpis(clinic_service_kpis, date_range_display_str)
    kpi_display.render_disease_specific_kpis(clinic_service_kpis)
else:
    st.warning("Main clinic performance KPIs cannot be displayed as essential health data is unavailable.")

# Render Environmental KPIs (component handles empty filtered_iot_df_clinic)
environmental_kpis.render_clinic_environmental_kpis(filtered_iot_df_clinic, date_range_display_str, iot_data_available) # Pass iot_data_available flag
st.markdown("---")

# Render NEW Epidemiology Module section
if health_data_available:
    epi_module.render_clinic_epi_module(filtered_health_df_clinic, date_range_display_str)
else:
    st.warning("Clinic Epidemiology module cannot be displayed as health data is unavailable for the selected period.")
st.markdown("---")

# --- Tabbed interface using component functions ---
tab_titles_clinic_new = ["🔬 Testing Insights", "💊 Supply Chain", "🧍 Patient Focus", "🌿 Environment Details"] # Consistent tab names
tab_tests_comp, tab_supplies_comp, tab_patients_alerts_comp, tab_environment_detail_comp = st.tabs(tab_titles_clinic_new)

with tab_tests_comp:
    testing_insights_tab.render_testing_insights(filtered_health_df_clinic, clinic_service_kpis)

with tab_supplies_comp:
    # Supply chain needs main_df for historical rates, filtered for context unless AI uses main only
    supply_chain_tab.render_supply_chain(health_df_clinic_main, filtered_health_df_clinic)

with tab_patients_alerts_comp:
    patient_focus_tab.render_patient_focus(filtered_health_df_clinic)

with tab_environment_detail_comp:
    environment_details_tab.render_environment_details(filtered_iot_df_clinic, iot_data_available)
