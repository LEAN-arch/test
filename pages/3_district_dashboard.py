# test/pages/3_district_dashboard.py
import streamlit as st
import pandas as pd
import geopandas as gpd # Used for type hinting potentially
import os
import logging
import numpy as np
from datetime import date, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="District Dashboard - Health Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project-specific absolute imports
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    load_zone_data,
    load_iot_clinic_environment_data,
    enrich_zone_geodata_with_health_aggregates,
    get_district_summary_kpis,
    hash_geodataframe # Used for caching GDF
)
from utils.ai_analytics_engine import apply_ai_models
# ui_visualization_helpers are imported by components

# Project-specific relative imports for components
# Requires:
# test/pages/__init__.py (empty)
# test/pages/district_components/__init__.py (empty)
from pages.district_components import kpi_display_district
from pages.district_components import map_display_district
from pages.district_components import trends_tab_district
from pages.district_components import comparison_tab_district
from pages.district_components import intervention_tab_district

logger = logging.getLogger(__name__)
# CSS Loading (as before)
@st.cache_resource
def load_css_district():
    css_path = app_config.STYLE_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path) as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else: logger.warning(f"District CSS file not found at {css_path}.")
load_css_district()

# Data Loading (as before, using the corrected version from prior fixes)
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, hash_funcs={gpd.GeoDataFrame: hash_geodataframe}, show_spinner="Loading district-level data...")
def get_district_dashboard_data_enriched_modular(): # Renamed for clarity in modular context
    # ... (Full data loading logic from previous get_district_dashboard_data_enriched)
    logger.info("District Dashboard: Attempting to load all necessary data sources and enrich...")
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    if health_df_raw.empty:
        logger.error("District Dashboard: Base health records failed to load.")
        return pd.DataFrame(), gpd.GeoDataFrame(columns=['zone_id', 'name', 'geometry'], crs=app_config.DEFAULT_CRS), pd.DataFrame()
    health_df_ai_enriched = apply_ai_models(health_df_raw)
    health_df_for_enrichment = health_df_ai_enriched if not health_df_ai_enriched.empty else health_df_raw
    zone_gdf_base = load_zone_data() # Uses default paths from app_config
    iot_df = load_iot_clinic_environment_data()
    if zone_gdf_base is None or zone_gdf_base.empty:
        return health_df_for_enrichment, gpd.GeoDataFrame(columns=['zone_id', 'name', 'population', 'geometry'], crs=app_config.DEFAULT_CRS), iot_df if iot_df is not None else pd.DataFrame()
    enriched_zone_gdf_final = enrich_zone_geodata_with_health_aggregates(zone_gdf_base, health_df_for_enrichment, iot_df)
    # Fallback logic if enrichment results in empty GDF
    if (enriched_zone_gdf_final is None or enriched_zone_gdf_final.empty) and (zone_gdf_base is not None and not zone_gdf_base.empty):
        logger.warning("District Dashboard: Enrichment resulted in empty GDF, falling back to base zone data.")
        # Ensure base GDF has columns expected by components (with defaults)
        for col_check in ['avg_risk_score', 'population', 'name', 'facility_coverage_score', 'active_tb_cases', 'total_active_key_infections', 'prevalence_per_1000', 'avg_daily_steps_zone', 'zone_avg_co2', 'num_clinics']:
            if col_check not in zone_gdf_base.columns:
                 zone_gdf_base[col_check] = 0.0 if col_check not in ['name','num_clinics'] else ("Unknown Zone" if col_check == 'name' else 0)
        return health_df_for_enrichment, zone_gdf_base, iot_df if iot_df is not None else pd.DataFrame()
    return health_df_for_enrichment, enriched_zone_gdf_final, iot_df if iot_df is not None else pd.DataFrame()

health_records_district_main, district_gdf_main_enriched, iot_records_district_main = get_district_dashboard_data_enriched_modular()

# Page Title & Subtitle (as before)
st.title("🗺️ District Health Officer (DHO) Dashboard")
st.markdown("**Strategic Population Health Insights, Resource Allocation, and Environmental Well-being Monitoring**")
st.markdown("---")

# Sidebar (Date Range Logic for Trends - Corrected and Unchanged)
# ... (Sidebar logic from previous full version for date selectors) ...
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, use_column_width='auto'); st.sidebar.markdown("---")
st.sidebar.header("🗓️ District Filters")
# (Robust date range selection logic from previous version, copied here)
all_potential_timestamps_dist_trend = []
if health_records_district_main is not None and not health_records_district_main.empty and 'encounter_date' in health_records_district_main.columns:
    encounter_dates_ts_dist = pd.to_datetime(health_records_district_main['encounter_date'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(encounter_dates_ts_dist): encounter_dates_ts_dist = encounter_dates_ts_dist.dt.tz_localize(None)
    all_potential_timestamps_dist_trend.extend(encounter_dates_ts_dist.dropna())
if iot_records_district_main is not None and not iot_records_district_main.empty and 'timestamp' in iot_records_district_main.columns:
    iot_timestamps_ts_dist = pd.to_datetime(iot_records_district_main['timestamp'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(iot_timestamps_ts_dist): iot_timestamps_ts_dist = iot_timestamps_ts_dist.dt.tz_localize(None)
    all_potential_timestamps_dist_trend.extend(iot_timestamps_ts_dist.dropna())
all_valid_timestamps_dist = [ts for ts in all_potential_timestamps_dist_trend if isinstance(ts, pd.Timestamp)]
min_date_for_trends_dist = min(all_valid_timestamps_dist).date() if all_valid_timestamps_dist else date.today() - timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND * 6)
max_date_for_trends_dist = max(all_valid_timestamps_dist).date() if all_valid_timestamps_dist else date.today()
if min_date_for_trends_dist > max_date_for_trends_dist: min_date_for_trends_dist = max_date_for_trends_dist
default_end_val_dist_trends = max_date_for_trends_dist
default_start_val_dist_trends = default_end_val_dist_trends - timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if default_start_val_dist_trends < min_date_for_trends_dist: default_start_val_dist_trends = min_date_for_trends_dist
selected_start_date_dist_trends, selected_end_date_dist_trends = st.sidebar.date_input("Select Date Range for Trend Analysis:", value=[default_start_val_dist_trends, default_end_val_dist_trends], min_value=min_date_for_trends_dist, max_value=max_date_for_trends_dist, key="district_trends_date_selector_v11")
if selected_start_date_dist_trends > selected_end_date_dist_trends: st.sidebar.error("Trend start date must be before end date."); selected_start_date_dist_trends = selected_end_date_dist_trends

# Filter health and IoT data for the selected trend period
# ... (Data filtering for trends - Corrected and Unchanged from previous version) ...
filtered_health_for_trends_dist = pd.DataFrame()
if health_records_district_main is not None and not health_records_district_main.empty:
    health_records_district_main['encounter_date'] = pd.to_datetime(health_records_district_main['encounter_date'], errors='coerce')
    health_records_district_main['encounter_date_obj_trend'] = health_records_district_main['encounter_date'].dt.date # Keep distinct name if 'encounter_date_obj' used elsewhere
    valid_dates_filter_health_trend = health_records_district_main['encounter_date_obj_trend'].notna()
    df_health_trend_to_filter = health_records_district_main[valid_dates_filter_health_trend].copy()
    if not df_health_trend_to_filter.empty: filtered_health_for_trends_dist = df_health_trend_to_filter[(df_health_trend_to_filter['encounter_date_obj_trend'] >= selected_start_date_dist_trends) & (df_health_trend_to_filter['encounter_date_obj_trend'] <= selected_end_date_dist_trends)].copy()
filtered_iot_for_trends_dist = pd.DataFrame()
if iot_records_district_main is not None and not iot_records_district_main.empty:
    iot_records_district_main['timestamp'] = pd.to_datetime(iot_records_district_main['timestamp'], errors='coerce')
    iot_records_district_main['timestamp_date_obj_trend'] = iot_records_district_main['timestamp'].dt.date
    valid_dates_filter_iot_trend = iot_records_district_main['timestamp_date_obj_trend'].notna()
    df_iot_trend_to_filter = iot_records_district_main[valid_dates_filter_iot_trend].copy()
    if not df_iot_trend_to_filter.empty: filtered_iot_for_trends_dist = df_iot_trend_to_filter[(df_iot_trend_to_filter['timestamp_date_obj_trend'] >= selected_start_date_dist_trends) & (df_iot_trend_to_filter['timestamp_date_obj_trend'] <= selected_end_date_dist_trends)].copy()


# --- KPIs Section --- (using component)
district_overall_kpis = get_district_summary_kpis(district_gdf_main_enriched)
kpi_display_district.render_district_kpis(district_overall_kpis, district_gdf_main_enriched) # Pass GDF for total zones calc if needed
st.markdown("---")

# --- Interactive Map Section --- (using component)
map_display_district.render_district_map(district_gdf_main_enriched)
st.markdown("---")

# --- Tabs for Trends, Comparison, and Interventions ---
tab_titles_dist = ["📈 District-Wide Trends", "📊 Zonal Comparative Analysis", "🎯 Intervention Planning Insights"]
tab_dist_trends_disp, tab_dist_comparison_disp, tab_dist_interventions_disp = st.tabs(tab_titles_dist)

with tab_dist_trends_disp:
    trends_tab_district.render_district_trends_tab(filtered_health_for_trends_dist, filtered_iot_for_trends_dist, selected_start_date_dist_trends, selected_end_date_dist_trends)

with tab_dist_comparison_disp:
    comparison_tab_district.render_zonal_comparison_tab(district_gdf_main_enriched) # comparison based on latest enriched GDF

with tab_dist_interventions_disp:
    intervention_tab_district.render_intervention_planning_tab(district_gdf_main_enriched)
