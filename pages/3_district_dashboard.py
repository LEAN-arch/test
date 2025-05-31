# test/pages/3_district_dashboard.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import logging
import numpy as np
from datetime import date

# Assuming flat import structure for this 'test' setup
from config import app_config
from utils.core_data_processing import (
    load_health_records,
    load_zone_data,
    load_iot_clinic_environment_data,
    enrich_zone_geodata_with_health_aggregates,
    get_district_summary_kpis,
    get_trend_data,
    hash_geodataframe
)
from utils.ai_analytics_engine import apply_ai_models
from utils.ui_visualization_helpers import (
    render_kpi_card,
    plot_layered_choropleth_map,
    plot_annotated_line_chart,
    plot_bar_chart
)

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="District Dashboard - Health Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_css_district():
    css_path = app_config.STYLE_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info("District Dashboard: CSS loaded successfully.")
    else:
        logger.warning(f"District Dashboard: CSS file not found at {css_path}.")
load_css_district()

# --- Data Loading and Enrichment (Corrected) ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, hash_funcs={gpd.GeoDataFrame: hash_geodataframe}, show_spinner="Loading district-level data...")
def get_district_dashboard_data_enriched():
    logger.info("District Dashboard: Attempting to load all necessary data sources and enrich...")
    health_df_raw = load_health_records(file_path=app_config.HEALTH_RECORDS_CSV)
    if health_df_raw.empty:
        logger.error("District Dashboard: Base health records failed to load. Cannot proceed with AI enrichment or GDF enrichment.")
        return pd.DataFrame(), gpd.GeoDataFrame(columns=['zone_id', 'name', 'geometry'], crs=app_config.DEFAULT_CRS), pd.DataFrame()

    health_df_ai_enriched = apply_ai_models(health_df_raw)
    health_df_for_enrichment = health_df_ai_enriched if not health_df_ai_enriched.empty else health_df_raw
    if health_df_ai_enriched.empty and not health_df_raw.empty:
        logger.warning("District Dashboard: AI enrichment failed for health data, using raw (cleaned) data for GDF enrichment.")

    zone_gdf_base = load_zone_data(
        attributes_path=app_config.ZONE_ATTRIBUTES_CSV,
        geometries_path=app_config.ZONE_GEOMETRIES_GEOJSON
    )
    iot_df = load_iot_clinic_environment_data(file_path=app_config.IOT_CLINIC_ENVIRONMENT_CSV)

    if zone_gdf_base is None or zone_gdf_base.empty or 'geometry' not in zone_gdf_base.columns:
        logger.error("CRITICAL - District Dashboard: Base zone geographic data (zone_gdf_base) is unusable.")
        return health_df_for_enrichment, \
               gpd.GeoDataFrame(columns=['zone_id', 'name', 'population', 'geometry'], crs=app_config.DEFAULT_CRS), \
               iot_df if iot_df is not None else pd.DataFrame()

    logger.info("District Dashboard: Enriching zone geographic data with health and IoT aggregates...")
    enriched_zone_gdf_final = enrich_zone_geodata_with_health_aggregates(
        zone_gdf_base, health_df_for_enrichment, iot_df
    )

    if (enriched_zone_gdf_final is None or enriched_zone_gdf_final.empty) and (zone_gdf_base is not None and not zone_gdf_base.empty):
        logger.warning("District Dashboard: Enrichment of zone GDF resulted in an empty GDF. Falling back to base zone data with default aggregates.")
        for col_check in ['avg_risk_score', 'population', 'name', 'facility_coverage_score', 'active_tb_cases', 'total_active_key_infections', 'prevalence_per_1000', 'avg_daily_steps_zone', 'zone_avg_co2', 'num_clinics']:
             if col_check not in zone_gdf_base.columns:
                 zone_gdf_base[col_check] = 0.0 if col_check not in ['name','num_clinics'] else ("Unknown Zone" if col_check == 'name' else 0)
        return health_df_for_enrichment, zone_gdf_base, iot_df if iot_df is not None else pd.DataFrame()

    logger.info("District Dashboard: Data loading and enrichment process complete.")
    return health_df_for_enrichment, enriched_zone_gdf_final, iot_df if iot_df is not None else pd.DataFrame()

health_records_district_main, district_gdf_main_enriched, iot_records_district_main = get_district_dashboard_data_enriched()

# --- Main Page Structure ---
if district_gdf_main_enriched is None or district_gdf_main_enriched.empty or \
   'geometry' not in district_gdf_main_enriched.columns or \
   (hasattr(district_gdf_main_enriched.geometry, 'is_empty') and district_gdf_main_enriched.geometry.is_empty.all()):
    st.error("🚨 **CRITICAL GIS Data Error:** Geographic zone data (enriched GDF) is missing, invalid, or empty. The District Dashboard cannot be rendered effectively. Please verify 'zone_geometries.geojson', 'zone_attributes.csv', and their processing.")
    logger.critical("District Dashboard HALTED: district_gdf_main_enriched is unusable for map.")
else:
     logger.info(f"District GDF loaded for dashboard: {len(district_gdf_main_enriched)} zones.")

st.title("🗺️ District Health Officer (DHO) Dashboard")
st.markdown("**Strategic Population Health Insights, Resource Allocation, and Environmental Well-being Monitoring**")
st.markdown("---")

# --- Sidebar (Corrected Date Range Logic) ---
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, use_column_width='auto')
    st.sidebar.markdown("---")
st.sidebar.header("🗓️ District Filters")

all_potential_timestamps_dist_trend = []
if health_records_district_main is not None and not health_records_district_main.empty and 'encounter_date' in health_records_district_main.columns:
    encounter_dates_ts_dist = pd.to_datetime(health_records_district_main['encounter_date'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(encounter_dates_ts_dist):
        encounter_dates_ts_dist = encounter_dates_ts_dist.dt.tz_localize(None)
    all_potential_timestamps_dist_trend.extend(encounter_dates_ts_dist.dropna())

if iot_records_district_main is not None and not iot_records_district_main.empty and 'timestamp' in iot_records_district_main.columns:
    iot_timestamps_ts_dist = pd.to_datetime(iot_records_district_main['timestamp'], errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(iot_timestamps_ts_dist):
        iot_timestamps_ts_dist = iot_timestamps_ts_dist.dt.tz_localize(None)
    all_potential_timestamps_dist_trend.extend(iot_timestamps_ts_dist.dropna())

all_valid_timestamps_dist = [ts for ts in all_potential_timestamps_dist_trend if isinstance(ts, pd.Timestamp)]

if all_valid_timestamps_dist:
    min_ts_for_trends_dist = min(all_valid_timestamps_dist)
    max_ts_for_trends_dist = max(all_valid_timestamps_dist)
    min_date_for_trends_dist = min_ts_for_trends_dist.date()
    max_date_for_trends_dist = max_ts_for_trends_dist.date()
    logger.info(f"District trend date range from data: {min_date_for_trends_dist} to {max_date_for_trends_dist}")
else:
    logger.warning("No valid dates found in health or IoT data for District dashboard trends. Using default fallback.")
    min_date_for_trends_dist = date.today() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND * 6)
    max_date_for_trends_dist = date.today()

if min_date_for_trends_dist > max_date_for_trends_dist:
    min_date_for_trends_dist = max_date_for_trends_dist

default_end_val_dist_trends = max_date_for_trends_dist
default_start_val_dist_trends = max_date_for_trends_dist - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
if default_start_val_dist_trends < min_date_for_trends_dist:
    default_start_val_dist_trends = min_date_for_trends_dist

selected_start_date_dist_trends, selected_end_date_dist_trends = st.sidebar.date_input(
    "Select Date Range for Trend Analysis:",
    value=[default_start_val_dist_trends, default_end_val_dist_trends],
    min_value=min_date_for_trends_dist, max_value=max_date_for_trends_dist,
    key="district_trends_date_selector_v10",
    help="This date range applies to time-series trend charts for health and environmental data."
)

filtered_health_for_trends_dist = pd.DataFrame()
if health_records_district_main is not None and not health_records_district_main.empty:
    temp_health_dates_trend = pd.to_datetime(health_records_district_main['encounter_date'], errors='coerce')
    health_records_district_main['encounter_date_obj_trend'] = pd.NaT # Use distinct obj col name
    valid_health_date_mask_trend = temp_health_dates_trend.notna()
    health_records_district_main.loc[valid_health_date_mask_trend, 'encounter_date_obj_trend'] = temp_health_dates_trend[valid_health_date_mask_trend].dt.date
    health_date_filter_mask_trend = (
        health_records_district_main['encounter_date_obj_trend'].notna() &
        (health_records_district_main['encounter_date_obj_trend'] >= selected_start_date_dist_trends) &
        (health_records_district_main['encounter_date_obj_trend'] <= selected_end_date_dist_trends)
    )
    filtered_health_for_trends_dist = health_records_district_main[health_date_filter_mask_trend].copy()

filtered_iot_for_trends_dist = pd.DataFrame()
if iot_records_district_main is not None and not iot_records_district_main.empty:
    temp_iot_dates_trend = pd.to_datetime(iot_records_district_main['timestamp'], errors='coerce')
    iot_records_district_main['timestamp_date_obj_trend'] = pd.NaT # Use distinct obj col name
    valid_iot_date_mask_trend = temp_iot_dates_trend.notna()
    iot_records_district_main.loc[valid_iot_date_mask_trend, 'timestamp_date_obj_trend'] = temp_iot_dates_trend[valid_iot_date_mask_trend].dt.date
    iot_date_filter_mask_trend = (
        iot_records_district_main['timestamp_date_obj_trend'].notna() &
        (iot_records_district_main['timestamp_date_obj_trend'] >= selected_start_date_dist_trends) &
        (iot_records_district_main['timestamp_date_obj_trend'] <= selected_end_date_dist_trends)
    )
    filtered_iot_for_trends_dist = iot_records_district_main[iot_date_filter_mask_trend].copy()

# --- KPIs Section ---
st.subheader("District-Wide Key Performance Indicators (Latest Aggregated Zonal Data)")
if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty:
    district_overall_kpis = get_district_summary_kpis(district_gdf_main_enriched)
    logger.debug(f"District Overall KPIs: {district_overall_kpis}")
    kpi_cols_row1_dist = st.columns(4)
    with kpi_cols_row1_dist[0]:
        avg_pop_risk_val = district_overall_kpis.get('avg_population_risk', 0.0)
        render_kpi_card("Avg. Population AI Risk", f"{avg_pop_risk_val:.1f}", "🎯", status="High" if avg_pop_risk_val >= app_config.RISK_THRESHOLDS['high'] else ("Moderate" if avg_pop_risk_val >= app_config.RISK_THRESHOLDS['moderate'] else "Low"), help_text="Population-weighted average AI risk score across all zones.")
    with kpi_cols_row1_dist[1]:
        facility_coverage_val = district_overall_kpis.get('overall_facility_coverage', 0.0)
        render_kpi_card("Facility Coverage Score", f"{facility_coverage_val:.1f}%", "🏥", status="Good High" if facility_coverage_val >= 80 else ("Moderate" if facility_coverage_val >= app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD else "Bad Low"), help_text="Population-weighted score reflecting access and capacity of health facilities.")
    with kpi_cols_row1_dist[2]:
        high_risk_zones_num = district_overall_kpis.get('zones_high_risk_count', 0)
        total_zones_val = len(district_gdf_main_enriched) if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty else 1
        perc_high_risk_zones = (high_risk_zones_num / total_zones_val) * 100 if total_zones_val > 0 else 0.0
        render_kpi_card("High AI Risk Zones", f"{high_risk_zones_num} ({perc_high_risk_zones:.0f}%)", "⚠️", status="High" if perc_high_risk_zones > 25 else ("Moderate" if high_risk_zones_num > 0 else "Low"), help_text=f"Zones with average AI risk score ≥ {app_config.RISK_THRESHOLDS['district_zone_high_risk']}.")
    with kpi_cols_row1_dist[3]:
        district_prevalence_val = district_overall_kpis.get('key_infection_prevalence_district_per_1000', 0.0)
        render_kpi_card("Overall Key Inf. Prevalence", f"{district_prevalence_val:.1f} /1k Pop", "📈", status="High" if district_prevalence_val > 50 else ("Moderate" if district_prevalence_val > 20 else "Low"), help_text="Combined prevalence of key infectious diseases per 1,000 population.")

    st.markdown("##### Key Disease Burdens & District Wellness / Environment")
    kpi_cols_row2_dist = st.columns(4)
    with kpi_cols_row2_dist[0]:
        tb_total_burden = district_overall_kpis.get('district_tb_burden_total', 0)
        render_kpi_card("Total Active TB Cases", str(tb_total_burden), "🫁", status="High" if tb_total_burden > (len(district_gdf_main_enriched) * app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty else 20) else ("Moderate" if tb_total_burden > 0 else "Low"), help_text="Total active TB cases across the district.")
    with kpi_cols_row2_dist[1]:
        malaria_total_burden = district_overall_kpis.get('district_malaria_burden_total',0)
        render_kpi_card("Total Active Malaria Cases", str(malaria_total_burden), "🦟", status="High" if malaria_total_burden > (len(district_gdf_main_enriched) * app_config.INTERVENTION_MALARIA_BURDEN_HIGH_THRESHOLD if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty else 50) else ("Moderate" if malaria_total_burden > 0 else "Low"), help_text="Total active Malaria cases across the district.")
    with kpi_cols_row2_dist[2]:
        avg_steps_district = district_overall_kpis.get('population_weighted_avg_steps', 0.0)
        render_kpi_card("Avg. Patient Steps (Pop. Weighted)", f"{avg_steps_district:,.0f}", "👣", status="Good High" if avg_steps_district >= app_config.TARGET_DAILY_STEPS else ("Moderate" if avg_steps_district >= app_config.TARGET_DAILY_STEPS * 0.7 else "Bad Low"), help_text=f"Population-weighted average daily steps. Target: {app_config.TARGET_DAILY_STEPS:,.0f} steps.")
    with kpi_cols_row2_dist[3]:
        avg_co2_district_val = district_overall_kpis.get('avg_clinic_co2_district',0.0)
        render_kpi_card("Avg. Clinic CO2 (District)", f"{avg_co2_district_val:.0f} ppm", "💨", status="High" if avg_co2_district_val > app_config.CO2_LEVEL_ALERT_PPM else ("Moderate" if avg_co2_district_val > app_config.CO2_LEVEL_IDEAL_PPM else "Low"), help_text="District average of zonal mean CO2 levels in clinics.")
else:
    st.warning("District-Wide KPIs cannot be displayed: Enriched zone geographic data (GDF) is unavailable.")
st.markdown("---")

st.subheader("🗺️ Interactive Health & Environment Map of the District")
if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty and 'geometry' in district_gdf_main_enriched.columns and not district_gdf_main_enriched.geometry.is_empty.all():
    map_metric_options_config_dist = {
        "Avg. AI Risk Score": {"col": "avg_risk_score", "colorscale": "Reds_r", "format": "{:.1f}"},
        "Total Active Key Infections": {"col": "total_active_key_infections", "colorscale": "OrRd_r", "format": "{:.0f}"},
        "Prevalence per 1,000 (Key Inf.)": {"col": "prevalence_per_1000", "colorscale": "YlOrRd_r", "format": "{:.1f}"},
        "Facility Coverage Score": {"col": "facility_coverage_score", "colorscale": "Greens", "format": "{:.1f}%"},
        "Active TB Cases": {"col": "active_tb_cases", "colorscale": "Purples_r", "format": "{:.0f}"},
        "Active Malaria Cases": {"col": "active_malaria_cases", "colorscale": "Oranges_r", "format": "{:.0f}"},
        "Avg. Patient Steps (Zone)": {"col": "avg_daily_steps_zone", "colorscale": "Cividis", "format": "{:,.0f}"},
        "Avg. Zone CO2 (Clinics)": {"col": "zone_avg_co2", "colorscale": "Blues_r", "format": "{:.0f} ppm"},
        "Population": {"col": "population", "colorscale": "Viridis", "format": "{:,.0f}"},
        "Number of Clinics": {"col": "num_clinics", "colorscale": "Blues", "format":"{:.0f}"},
        "Socio-Economic Index": {"col": "socio_economic_index", "colorscale": "Tealgrn", "format": "{:.2f}"}
    }
    gdf_for_area_calc = district_gdf_main_enriched.copy()
    if 'population' in gdf_for_area_calc.columns:
        try:
            if gdf_for_area_calc.crs and not gdf_for_area_calc.crs.is_geographic:
                gdf_for_area_calc['area_sqkm_calc'] = gdf_for_area_calc.geometry.area / 1_000_000
            elif gdf_for_area_calc.crs and gdf_for_area_calc.crs.is_geographic:
                utm_crs = gdf_for_area_calc.estimate_utm_crs()
                if utm_crs: gdf_for_area_calc['area_sqkm_calc'] = gdf_for_area_calc.to_crs(utm_crs).geometry.area / 1_000_000
                else: gdf_for_area_calc['area_sqkm_calc'] = np.nan
            else: gdf_for_area_calc['area_sqkm_calc'] = np.nan
            
            if 'area_sqkm_calc' in gdf_for_area_calc.columns and gdf_for_area_calc['area_sqkm_calc'].notna().any():
                if 'population_density' not in district_gdf_main_enriched.columns: district_gdf_main_enriched['population_density'] = 0.0 
                district_gdf_main_enriched['population_density'] = gdf_for_area_calc.apply(lambda r: (r['population'] / r['area_sqkm_calc']) if pd.notna(r.get('area_sqkm_calc')) and r.get('area_sqkm_calc',0)>0 and pd.notna(r.get('population')) else 0.0, axis=1).fillna(0.0)
                map_metric_options_config_dist["Population Density (Pop/SqKm)"] = {"col": "population_density", "colorscale": "Plasma_r", "format": "{:,.1f}"}
        except Exception as e_map_area: logger.error(f"Map area/pop density calculation failed: {e_map_area}", exc_info=True)

    available_map_metrics_for_select = { disp_name: details for disp_name, details in map_metric_options_config_dist.items() if details["col"] in district_gdf_main_enriched.columns and district_gdf_main_enriched[details["col"]].notna().any() }
    if available_map_metrics_for_select:
        selected_map_metric_display_name = st.selectbox("Select Metric to Visualize on Map:", list(available_map_metrics_for_select.keys()), key="district_interactive_map_metric_selector_v7")
        selected_map_metric_config = available_map_metrics_for_select.get(selected_map_metric_display_name)
        if selected_map_metric_config:
            map_val_col = selected_map_metric_config["col"]; map_colorscale = selected_map_metric_config["colorscale"]
            hover_cols_for_map = ['name', map_val_col, 'population', 'num_clinics', 'facility_coverage_score']
            final_hover_cols_map = list(dict.fromkeys([col for col in hover_cols_for_map if col in district_gdf_main_enriched.columns and col != map_val_col])) # Don't repeat val_col if also in general list
            map_figure = plot_layered_choropleth_map(gdf=district_gdf_main_enriched, value_col=map_val_col, title=f"District Map: {selected_map_metric_display_name}", id_col='zone_id', color_continuous_scale=map_colorscale, hover_cols=final_hover_cols_map, height=app_config.MAP_PLOT_HEIGHT, mapbox_style=app_config.MAPBOX_STYLE)
            st.plotly_chart(map_figure, use_container_width=True)
    else: st.warning("No metrics available for map visualization. Check data enrichment.")
else: st.error("🚨 District map cannot be displayed: Enriched zone geographic data is unusable or unavailable.")
st.markdown("---")

tab_dist_trends, tab_dist_comparison, tab_dist_interventions = st.tabs(["📈 District-Wide Trends", "📊 Zonal Comparative Analysis", "🎯 Intervention Planning Insights"])

with tab_dist_trends:
    st.header("📈 District-Wide Health & Environmental Trends")
    if (filtered_health_for_trends_dist.empty and (iot_records_district_main is None or filtered_iot_for_trends_dist.empty)):
        st.info(f"No health or environmental data for selected trend period: {selected_start_date_dist_trends.strftime('%d %b %Y')} to {selected_end_date_dist_trends.strftime('%d %b %Y')}.")
    else:
        st.markdown(f"Displaying trends from **{selected_start_date_dist_trends.strftime('%d %b %Y')}** to **{selected_end_date_dist_trends.strftime('%d %b %Y')}**.")
        st.subheader("Key Disease Incidence Trends (New Cases Identified per Week)")
        cols_disease_trends_dist = st.columns(2)
        # ... (Full Trend Tab logic from previous response, ensuring date_col='encounter_date' for health data trends) ...
        with cols_disease_trends_dist[0]: # TB Trend
            if not filtered_health_for_trends_dist.empty and all(c in filtered_health_for_trends_dist.columns for c in ['condition', 'patient_id', 'encounter_date']):
                tb_trends_src = filtered_health_for_trends_dist[filtered_health_for_trends_dist['condition'].str.contains("TB", case=False, na=False)].copy()
                if not tb_trends_src.empty:
                    tb_trends_src['is_first_tb_encounter'] = ~tb_trends_src.sort_values('encounter_date').duplicated(subset=['patient_id', 'condition'], keep='first') # Identify first record for patient+TB
                    new_tb_cases_df = tb_trends_src[tb_trends_src['is_first_tb_encounter']]
                    weekly_tb_trend = get_trend_data(new_tb_cases_df, value_col='patient_id', date_col='encounter_date', period='W', agg_func='count') # Count new cases
                    if not weekly_tb_trend.empty: st.plotly_chart(plot_annotated_line_chart(weekly_tb_trend, "Weekly New TB Patients Identified", y_axis_title="New TB Patients", height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                    else: st.caption("No new TB patient trend data for this period.")
                else: st.caption("No TB encounter data in this period for trends.")
            else: st.caption("TB trend data cannot be generated (missing data/columns).")
        with cols_disease_trends_dist[1]: # Malaria Trend
            if not filtered_health_for_trends_dist.empty and all(c in filtered_health_for_trends_dist.columns for c in ['condition', 'patient_id', 'encounter_date']):
                malaria_trends_src = filtered_health_for_trends_dist[filtered_health_for_trends_dist['condition'].str.contains("Malaria", case=False, na=False)].copy()
                if not malaria_trends_src.empty:
                    malaria_trends_src['is_first_malaria_encounter'] = ~malaria_trends_src.sort_values('encounter_date').duplicated(subset=['patient_id', 'condition'], keep='first')
                    new_malaria_cases_df = malaria_trends_src[malaria_trends_src['is_first_malaria_encounter']]
                    weekly_malaria_trend = get_trend_data(new_malaria_cases_df, value_col='patient_id', date_col='encounter_date', period='W', agg_func='count')
                    if not weekly_malaria_trend.empty: st.plotly_chart(plot_annotated_line_chart(weekly_malaria_trend, "Weekly New Malaria Patients Identified", y_axis_title="New Malaria Patients", height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                    else: st.caption("No new Malaria patient trend data for this period.")
                else: st.caption("No Malaria encounter data in this period for trends.")
            else: st.caption("Malaria trend data unavailable.")

        st.subheader("Population Wellness & Clinic Environmental Trends"); cols_wellness_env_dist = st.columns(2)
        with cols_wellness_env_dist[0]:
            if not filtered_health_for_trends_dist.empty and 'avg_daily_steps' in filtered_health_for_trends_dist.columns:
                steps_trends_dist = get_trend_data(filtered_health_for_trends_dist, 'avg_daily_steps', date_col='encounter_date', period='W', agg_func='mean')
                if not steps_trends_dist.empty: st.plotly_chart(plot_annotated_line_chart(steps_trends_dist, "Weekly Avg. Patient Daily Steps (District)", y_axis_title="Average Steps", target_line=app_config.TARGET_DAILY_STEPS, target_label=f"Target {app_config.TARGET_DAILY_STEPS} Steps", height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                else: st.caption("No steps trend data for this period.")
            else: st.caption("Avg. daily steps data missing for trends.")
        with cols_wellness_env_dist[1]:
            if not filtered_iot_for_trends_dist.empty and 'avg_co2_ppm' in filtered_iot_for_trends_dist.columns:
                co2_trends_dist_iot = get_trend_data(filtered_iot_for_trends_dist, 'avg_co2_ppm', date_col='timestamp', period='D', agg_func='mean')
                if not co2_trends_dist_iot.empty: st.plotly_chart(plot_annotated_line_chart(co2_trends_dist_iot, "Daily Avg. CO2 (All Monitored Clinics)", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, target_label=f"Alert >{app_config.CO2_LEVEL_ALERT_PPM}ppm", height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                else: st.caption("No clinic CO2 trend data from IoT for this period.")
            else: st.caption("Clinic CO2 data missing or no IoT data for this period for trends.")

with tab_dist_comparison:
    st.header("📊 Zonal Comparative Analysis (Based on Latest Aggregates)")
    # ... (Full Zonal Comparison Tab logic from previous complete response, no changes needed for date bug) ...
    if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty and 'geometry' in district_gdf_main_enriched.columns:
        st.markdown("Compare zones using aggregated health, resource, environmental, and socio-economic metrics from the latest available data.")
        # Reuse map_metric_options_config_dist as it contains relevant cols and formats
        comp_table_metrics_dict_dist = { name: details for name, details in map_metric_options_config_dist.items() if details["col"] in district_gdf_main_enriched.columns and district_gdf_main_enriched[details["col"]].notna().any()}
        if comp_table_metrics_dict_dist:
            st.subheader("Zonal Statistics Table"); cols_for_comp_table_display = ['name'] + [d['col'] for d in comp_table_metrics_dict_dist.values()]
            df_for_comp_table_display = district_gdf_main_enriched[[col for col in cols_for_comp_table_display if col in district_gdf_main_enriched.columns]].copy()
            df_for_comp_table_display.rename(columns={'name':'Zone'}, inplace=True)
            if 'Zone' in df_for_comp_table_display.columns: df_for_comp_table_display.set_index('Zone', inplace=True)
            style_formats_comp_dist = {details["col"]: details.get("format", "{:.1f}") for _, details in comp_table_metrics_dict_dist.items() if details["col"] in df_for_comp_table_display.columns}
            styler_obj_comp_dist = df_for_comp_table_display.style.format(style_formats_comp_dist)
            for metric_display_name, details_style_comp in comp_table_metrics_dict_dist.items():
                col_name_to_style = details_style_comp["col"]
                if col_name_to_style in df_for_comp_table_display.columns:
                    colorscale_name = details_style_comp.get("colorscale", "Blues")
                    try: styler_obj_comp_dist = styler_obj_comp_dist.background_gradient(subset=[col_name_to_style], cmap=colorscale_name, axis=0)
                    except Exception as e_style: logger.warning(f"Could not apply background_gradient for {col_name_to_style} with cmap {colorscale_name}: {e_style}")
            st.dataframe(styler_obj_comp_dist, use_container_width=True, height=min(len(df_for_comp_table_display) * 45 + 60, 650))
            st.subheader("Visual Comparison Chart"); selected_bar_metric_name_dist_comp_viz = st.selectbox("Select Metric for Bar Chart Comparison:", list(comp_table_metrics_dict_dist.keys()), key="district_comp_barchart_v8")
            selected_bar_details_dist_comp_viz = comp_table_metrics_dict_dist.get(selected_bar_metric_name_dist_comp_viz)
            if selected_bar_details_dist_comp_viz:
                bar_col_for_comp_viz = selected_bar_details_dist_comp_viz["col"]; text_format_bar = selected_bar_details_dist_comp_viz.get("format", ",.1f")
                sort_asc_bar_viz = "_r" not in selected_bar_details_dist_comp_viz.get("colorscale", "")
                st.plotly_chart(plot_bar_chart(district_gdf_main_enriched, x_col='name', y_col=bar_col_for_comp_viz, title=f"{selected_bar_metric_name_dist_comp_viz} by Zone", x_axis_title="Zone Name", height=app_config.DEFAULT_PLOT_HEIGHT + 150, sort_values_by=bar_col_for_comp_viz, ascending=sort_asc_bar_viz, text_auto=True, text_format=text_format_bar), use_container_width=True)
        else: st.info("No metrics available for Zonal Comparison. Check GDF enrichment.")
    else: st.info("Zonal comparison requires enriched geographic zone data.")


with tab_dist_interventions:
    st.header("🎯 Intervention Planning Insights")
    # ... (Full Intervention Planning Tab logic from previous complete response)
    if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty and 'geometry' in district_gdf_main_enriched.columns:
        st.markdown("Identify zones for targeted interventions based on customizable criteria related to health risks, disease burdens, resource accessibility, and environmental factors from the latest aggregated data.")
        criteria_lambdas_intervention_dist = {
            f"High Avg. AI Risk (Score ≥ {app_config.RISK_THRESHOLDS['district_zone_high_risk']})": lambda df: df.get('avg_risk_score', pd.Series(dtype=float)) >= app_config.RISK_THRESHOLDS['district_zone_high_risk'],
            f"Low Facility Coverage (< {app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD}%)": lambda df: df.get('facility_coverage_score', pd.Series(dtype=float)) < app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD,
            f"High Key Inf. Prevalence (Top {100 - app_config.INTERVENTION_PREVALENCE_HIGH_PERCENTILE*100:.0f}%)": lambda df: df.get('prevalence_per_1000', pd.Series(dtype=float)) >= df.get('prevalence_per_1000', pd.Series(dtype=float)).quantile(app_config.INTERVENTION_PREVALENCE_HIGH_PERCENTILE) if 'prevalence_per_1000' in df and df['prevalence_per_1000'].notna().any() and len(df['prevalence_per_1000'].dropna()) > 1 else pd.Series([False]*len(df), index=df.index),
            f"High TB Burden (Abs. Cases > {app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD} per zone)": lambda df: df.get('active_tb_cases', pd.Series(dtype=float)) > app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD,
            f"High Avg. Clinic CO2 (> {app_config.CO2_LEVEL_IDEAL_PPM}ppm in zone)": lambda df: df.get('zone_avg_co2', pd.Series(dtype=float)) > app_config.CO2_LEVEL_IDEAL_PPM
        }
        available_criteria_for_intervention_dist = {}
        for name_crit_int, func_crit_int in criteria_lambdas_intervention_dist.items():
            try:
                test_apply_df = district_gdf_main_enriched.head(1) if not district_gdf_main_enriched.empty else pd.DataFrame(columns=district_gdf_main_enriched.columns)
                if test_apply_df.empty and not district_gdf_main_enriched.columns.empty : # If GDF has columns but is empty, create test df with those cols
                    test_apply_df = pd.DataFrame(columns=district_gdf_main_enriched.columns)
                elif test_apply_df.empty and district_gdf_main_enriched.columns.empty : # Worst case, create dummy for basic lambda check
                     test_apply_df = pd.DataFrame(columns=['avg_risk_score','facility_coverage_score','prevalence_per_1000','active_tb_cases','zone_avg_co2'])

                func_crit_int(test_apply_df) # Test applicability
                available_criteria_for_intervention_dist[name_crit_int] = func_crit_int
            except Exception as e_crit_test: logger.debug(f"Intervention criterion '{name_crit_int}' not available: {e_crit_test}")
        
        if not available_criteria_for_intervention_dist:
            st.warning("No intervention criteria can be applied. Relevant data columns may be missing from the enriched zone data.")
        else:
            default_selection_interv = list(available_criteria_for_intervention_dist.keys())[0:min(2, len(available_criteria_for_intervention_dist))]
            selected_criteria_names_interv = st.multiselect( "Select Criteria to Identify Priority Zones (Zones meeting ANY selected criteria will be shown):", options=list(available_criteria_for_intervention_dist.keys()), default=default_selection_interv, key="district_intervention_criteria_multiselect_v5")
            if not selected_criteria_names_interv: st.info("Please select at least one criterion above to identify priority zones.")
            else:
                final_intervention_mask_dist = pd.Series([False] * len(district_gdf_main_enriched), index=district_gdf_main_enriched.index)
                for crit_name_selected_interv in selected_criteria_names_interv:
                    crit_func_selected_interv = available_criteria_for_intervention_dist[crit_name_selected_interv]
                    try:
                        current_crit_mask_interv = crit_func_selected_interv(district_gdf_main_enriched)
                        if isinstance(current_crit_mask_interv, pd.Series) and current_crit_mask_interv.dtype == 'bool': final_intervention_mask_dist = final_intervention_mask_dist | current_crit_mask_interv.fillna(False)
                        else: logger.warning(f"Intervention criterion '{crit_name_selected_interv}' did not produce a valid boolean Series. Type: {type(current_crit_mask_interv)}")
                    except Exception as e_crit_apply_interv: logger.error(f"Error applying intervention criterion '{crit_name_selected_interv}': {e_crit_apply_interv}", exc_info=True); st.warning(f"Could not apply criterion: {crit_name_selected_interv}. Error: {str(e_crit_apply_interv)[:100]}...")
                priority_zones_df_for_interv = district_gdf_main_enriched[final_intervention_mask_dist].copy()
                if not priority_zones_df_for_interv.empty:
                    st.markdown(f"###### Identified **{len(priority_zones_df_for_interv)}** Zone(s) Meeting Selected Intervention Criteria:")
                    cols_intervention_table_display = ['name', 'population', 'avg_risk_score', 'total_active_key_infections', 'prevalence_per_1000', 'facility_coverage_score', 'zone_avg_co2', 'active_tb_cases', 'active_malaria_cases']
                    actual_cols_interv_table_display = [col for col in cols_intervention_table_display if col in priority_zones_df_for_interv.columns]
                    sort_by_list_interv_display = []; sort_asc_list_interv_display = []
                    if 'avg_risk_score' in actual_cols_interv_table_display: sort_by_list_interv_display.append('avg_risk_score'); sort_asc_list_interv_display.append(False)
                    if 'prevalence_per_1000' in actual_cols_interv_table_display: sort_by_list_interv_display.append('prevalence_per_1000'); sort_asc_list_interv_display.append(False)
                    if 'facility_coverage_score' in actual_cols_interv_table_display: sort_by_list_interv_display.append('facility_coverage_score'); sort_asc_list_interv_display.append(True)
                    interv_df_display_sorted_final = priority_zones_df_for_interv.sort_values(by=sort_by_list_interv_display, ascending=sort_asc_list_interv_display) if sort_by_list_interv_display else priority_zones_df_for_interv
                    st.dataframe(interv_df_display_sorted_final[actual_cols_interv_table_display], use_container_width=True, hide_index=True, height=min(400, len(interv_df_display_sorted_final)*38 + 58),
                        column_config={
                            "name": st.column_config.TextColumn("Zone Name"), "population": st.column_config.NumberColumn("Population", format="%,.0f"),
                            "avg_risk_score": st.column_config.ProgressColumn("Avg. AI Risk Score", format="%.1f", min_value=0, max_value=100),
                            "total_active_key_infections": st.column_config.NumberColumn("Total Key Infections", format="%.0f"),
                            "prevalence_per_1000": st.column_config.NumberColumn("Prevalence (/1k Pop.)", format="%.1f"),
                            "facility_coverage_score": st.column_config.NumberColumn("Facility Coverage (%)", format="%.1f%%"),
                            "zone_avg_co2": st.column_config.NumberColumn("Avg. Clinic CO2 (ppm)", format="%.0f ppm"),
                            "active_tb_cases": st.column_config.NumberColumn("TB Cases", format="%.0f"),
                            "active_malaria_cases": st.column_config.NumberColumn("Malaria Cases", format="%.0f"),
                        })
                else: st.success("✅ No zones currently meet the selected high-priority criteria based on the available aggregated data.")
    else: st.info("Intervention planning insights require successfully loaded and enriched geographic zone data.")
