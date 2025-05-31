# test/pages/4_population_analytics_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from datetime import date, timedelta
import plotly.express as px # For scatter plot or more complex direct plots

from config import app_config
from utils.core_data_processing import (
    load_health_records,
    load_zone_data,
    get_trend_data
)
from utils.ai_analytics_engine import apply_ai_models
from utils.ui_visualization_helpers import (
    plot_bar_chart,
    plot_donut_chart,
    plot_annotated_line_chart
)

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Population Dashboard - Health Hub", # << RENAMED HERE
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_pop_dashboard(): # Renamed function to reflect page name change
    css_path = app_config.STYLE_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info("Population Dashboard: CSS loaded successfully.") # Updated log
    else:
        logger.warning(f"Population Dashboard: CSS file not found at {css_path}.") # Updated log
load_css_pop_dashboard()

# --- Data Loading ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading population data...") # Updated spinner text
def get_population_dashboard_data(): # Renamed function
    logger.info("Population Dashboard: Loading health records and zone data...") # Updated log
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    health_df = apply_ai_models(health_df_raw) if not health_df_raw.empty else pd.DataFrame()
    
    zone_gdf = load_zone_data()
    if zone_gdf is not None and not zone_gdf.empty and hasattr(zone_gdf, 'geometry') and zone_gdf.geometry.name in zone_gdf.columns:
        zone_attributes_df = pd.DataFrame(zone_gdf.drop(columns=[zone_gdf.geometry.name], errors='ignore'))
        logger.info(f"Loaded {len(zone_attributes_df)} zone attributes for SDOH context.")
    else:
        logger.warning("Zone attributes data not available. SDOH visualizations by zone will be limited.")
        zone_attributes_df = pd.DataFrame(columns=['zone_id', 'name', 'population', 'socio_economic_index'])

    if health_df.empty:
        logger.error("Population Dashboard: Failed to load or process health records.") # Updated log
    return health_df, zone_attributes_df

health_df_pop, zone_attr_df_pop = get_population_dashboard_data() # Renamed local variable

if health_df_pop.empty:
    st.error("🚨 **Data Error:** Could not load health records. Population Dashboard cannot be displayed.") # Updated message
    logger.critical("Population Dashboard: health_df_pop is empty.") # Updated log
    st.stop()

st.title("📊 Population Dashboard") # << RENAMED HERE
st.markdown("""
    Explore demographic distributions, epidemiological patterns, clinical trends, 
    and health system factors across the population. Use the filters to narrow your analysis.
""") # Markdown remains suitable
st.markdown("---")

# --- Sidebar Filters ---
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, width=230) # Using the decreased size
    st.sidebar.markdown("---")
else:
    logger.warning(f"Sidebar logo not found on Population Dashboard at {app_config.APP_LOGO}") # Updated log

st.sidebar.header("🔎 Population Filters") # Updated sidebar header
# ... (Rest of the sidebar filters logic from previous version - unchanged except for keys/context if needed) ...
min_date_pop = health_df_pop['encounter_date'].min().date() if 'encounter_date' in health_df_pop and not health_df_pop['encounter_date'].dropna().empty else date.today() - timedelta(days=365)
max_date_pop = health_df_pop['encounter_date'].max().date() if 'encounter_date' in health_df_pop and not health_df_pop['encounter_date'].dropna().empty else date.today()
if min_date_pop > max_date_pop: min_date_pop = max_date_pop
default_pop_start_date = min_date_pop; default_pop_end_date = max_date_pop
selected_start_date_pop, selected_end_date_pop = st.sidebar.date_input(
    "Select Date Range for Analysis:", value=[default_pop_start_date, default_pop_end_date],
    min_value=min_date_pop, max_value=max_date_pop, key="pop_dashboard_date_range_v1" # New key
)
if selected_start_date_pop > selected_end_date_pop: st.sidebar.error("Start date must be before end date."); selected_start_date_pop = selected_end_date_pop

# Filter data
health_df_pop['encounter_date_obj'] = pd.to_datetime(health_df_pop['encounter_date'], errors='coerce').dt.date
analytics_df_base = health_df_pop[
    (health_df_pop['encounter_date_obj'].notna()) &
    (health_df_pop['encounter_date_obj'] >= selected_start_date_pop) &
    (health_df_pop['encounter_date_obj'] <= selected_end_date_pop)
].copy()

if analytics_df_base.empty: st.warning(f"No health data for selected period: {selected_start_date_pop.strftime('%d %b %Y')} - {selected_end_date_pop.strftime('%d %b %Y')}."); st.stop()

conditions_list_pop_dash = ["All Conditions"] + sorted(analytics_df_base['condition'].dropna().unique().tolist())
selected_condition_filter_pop_dash = st.sidebar.selectbox("Filter by Condition:", options=conditions_list_pop_dash, index=0, key="pop_dashboard_condition_filter_v1")
analytics_df = analytics_df_base.copy()
if selected_condition_filter_pop_dash != "All Conditions": analytics_df = analytics_df[analytics_df['condition'] == selected_condition_filter_pop_dash]

zones_list_pop_dash = ["All Zones"] + sorted(analytics_df_base['zone_id'].dropna().unique().tolist())
selected_zone_filter_pop_dash = st.sidebar.selectbox("Filter by Zone:", options=zones_list_pop_dash, index=0, key="pop_dashboard_zone_filter_v1")
if selected_zone_filter_pop_dash != "All Zones": analytics_df = analytics_df[analytics_df['zone_id'] == selected_zone_filter_pop_dash]

analytics_df_display = analytics_df.copy()
if analytics_df.empty :
    st.warning(f"No data for '{selected_condition_filter_pop_dash}' in '{selected_zone_filter_pop_dash}'. Showing broader data for period if available.")
    analytics_df_display = analytics_df_base.copy()
    if selected_zone_filter_pop_dash != "All Zones": analytics_df_display = analytics_df_display[analytics_df_display['zone_id'] == selected_zone_filter_pop_dash]
    if analytics_df_display.empty: analytics_df_display = analytics_df_base.copy() # Last resort if zone filter also empty


# --- Tabbed Interface (content unchanged, only needs analytics_df_display) ---
tab_epi_overview, tab_demographics_sdoh, tab_clinical_dx, tab_systems_equity = st.tabs([
    "📈 Epidemiological Overview", "🧑‍⚕️ Demographics & SDOH", "🧬 Clinical & Diagnostics", "🌍 Systems & Equity"
])

with tab_epi_overview:
    st.header(f"Epidemiological Overview for '{selected_condition_filter_pop_dash}' in '{selected_zone_filter_pop_dash}'")
    # ... (Epi overview content from previous version, ensuring it uses analytics_df_display) ...
    if analytics_df_display.empty: st.info("No data for epidemiological overview.")
    else:
        # (Previous logic using analytics_df_display)
        pass # Placeholder for brevity

with tab_demographics_sdoh:
    st.header("Demographics & Social Determinants of Health (SDOH)")
    # ... (Demographics/SDOH content from previous version, using analytics_df_display and zone_attr_df_pop) ...
    if analytics_df_display.empty: st.info("No data for Demographics/SDOH.")
    else:
        # (Previous logic using analytics_df_display)
        pass

with tab_clinical_dx:
    st.header("Clinical & Diagnostic Data Patterns")
    # ... (Clinical/Dx content from previous version, using analytics_df_display) ...
    if analytics_df_display.empty: st.info("No data for Clinical/Dx patterns.")
    else:
        # (Previous logic using analytics_df_display)
        pass

with tab_systems_equity:
    st.header("Health Systems Context & Equity Insights")
    # ... (Systems/Equity content from previous version, using analytics_df_display and zone_attr_df_pop) ...
    if analytics_df_display.empty: st.info("No data for Systems/Equity insights.")
    else:
        # (Previous logic using analytics_df_display)
        pass


# --- Footer ---
st.markdown("---"); st.caption(app_config.APP_FOOTER)
