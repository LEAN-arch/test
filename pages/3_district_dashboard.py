# health_hub/pages/3_district_dashboard.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import logging
import numpy as np 
from config import app_config 
from utils.core_data_processing import (
    load_health_records, load_zone_data, load_iot_clinic_environment_data, 
    enrich_zone_geodata_with_health_aggregates, 
    get_district_summary_kpis, get_trend_data, hash_geodataframe 
)
from utils.ui_visualization_helpers import (
    render_kpi_card, plot_layered_choropleth_map, plot_annotated_line_chart, 
    plot_bar_chart
)

# --- Page Configuration and Styling ---
st.set_page_config(page_title="District Dashboard - Health Hub", layout="wide", initial_sidebar_state="expanded")
logger = logging.getLogger(__name__) 

@st.cache_resource 
def load_css(_app_config_param): # Function expects an argument
    if os.path.exists(_app_config_param.STYLE_CSS_PATH): 
        with open(_app_config_param.STYLE_CSS_PATH) as f: 
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info("District Dashboard: CSS loaded successfully.")
    else:
        logger.warning(f"District Dashboard: CSS file not found at {_app_config_param.STYLE_CSS_PATH}. Default Streamlit styles will be used.")

load_css(app_config) # Pass the imported app_config module

# --- Data Loading (Cached for performance) ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, hash_funcs={
    pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None,
    gpd.GeoDataFrame: hash_geodataframe 
})
def get_district_dashboard_data():
    logger.info("District Dashboard: Attempting to load all necessary data sources...")
    health_df = load_health_records()
    zone_gdf_base = load_zone_data() 
    iot_df = load_iot_clinic_environment_data()

    if zone_gdf_base is None or zone_gdf_base.empty or 'geometry' not in zone_gdf_base.columns:
        logger.error("CRITICAL - District Dashboard: Base zone geographic data (zone_gdf_base) could not be loaded, is empty, or lacks geometry. Map and most zonal analyses will be unavailable.")
        return (health_df if health_df is not None else pd.DataFrame(),
                gpd.GeoDataFrame(columns=['zone_id', 'name', 'population', 'geometry'], crs=app_config.DEFAULT_CRS),
                iot_df if iot_df is not None else pd.DataFrame())

    logger.info("District Dashboard: Enriching zone geographic data with health and IoT aggregates...")
    enriched_zone_gdf = enrich_zone_geodata_with_health_aggregates(
        zone_gdf_base, health_df, iot_df
    )
    
    if enriched_zone_gdf.empty and (zone_gdf_base is not None and not zone_gdf_base.empty):
        logger.warning("District Dashboard: Enrichment of zone GDF resulted in an empty GDF. Falling back to base zone data for map shapes if possible.")
        for col_check in ['avg_risk_score', 'population', 'name']: 
             if col_check not in zone_gdf_base.columns: zone_gdf_base[col_check] = 0 if col_check != 'name' else "Unknown Zone"
        return health_df if health_df is not None else pd.DataFrame(), zone_gdf_base, iot_df if iot_df is not None else pd.DataFrame()

    logger.info("District Dashboard: Data loading and enrichment process complete.")
    return health_df if health_df is not None else pd.DataFrame(), \
           enriched_zone_gdf, \
           iot_df if iot_df is not None else pd.DataFrame()


health_records_district_main, district_gdf_main_enriched, iot_records_district_main = get_district_dashboard_data()

# --- Main Page Structure ---
if district_gdf_main_enriched is None or district_gdf_main_enriched.empty or 'geometry' not in district_gdf_main_enriched.columns or district_gdf_main_enriched.geometry.is_empty.all():
    st.error("🚨 **CRITICAL Error:** Geographic zone data is missing, invalid, or empty. The District Dashboard cannot be rendered. Please verify 'zone_geometries.geojson' and its link with 'zone_attributes.csv'.")
    logger.critical("District Dashboard HALTED: district_gdf_main_enriched is unusable.")
    st.stop()

st.title("🗺️ District Health Officer (DHO) Dashboard")
st.markdown("**Strategic Population Health Insights, Resource Allocation, and Environmental Well-being Monitoring**")
st.markdown("---")

# --- Sidebar ---
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, use_column_width='auto')
    st.sidebar.markdown("---")

st.sidebar.header("🗓️ District Filters")
all_potential_dates_dist_raw = [] 
default_min_date_dist = pd.Timestamp('today').date() - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND * 6)
default_max_date_dist = pd.Timestamp('today').date()

def safe_extract_dates_from_df_dist(df, col_name, df_name_log="DataFrame"): 
    _extracted_ts_dist = []
    if df is not None and col_name in df.columns and not df.empty:
        date_like_column_dist = df[col_name]
        if not isinstance(date_like_column_dist, pd.Series): date_like_column_dist = pd.Series(date_like_column_dist)
        dt_series_dist = pd.to_datetime(date_like_column_dist, errors='coerce')
        valid_timestamps_dist = dt_series_dist.dropna() 
        if not valid_timestamps_dist.empty:
            _extracted_ts_dist.extend(valid_timestamps_dist.tolist())
            logger.debug(f"{df_name_log}: Extracted {len(valid_timestamps_dist)} valid timestamps from '{col_name}'. Sample: {_extracted_ts_dist[:3]}")
        else: logger.debug(f"{df_name_log}: Column '{col_name}' had no valid timestamps after coercion and dropna.")
    else: logger.debug(f"{df_name_log}: Column '{col_name}' not found or DataFrame is empty.")
    return _extracted_ts_dist

all_potential_dates_dist_raw.extend(safe_extract_dates_from_df_dist(health_records_district_main, 'date', "DistrictHealthDF"))
all_potential_dates_dist_raw.extend(safe_extract_dates_from_df_dist(iot_records_district_main, 'timestamp', "DistrictIoTDF"))

all_valid_timestamps_dist_final = [d for d in all_potential_dates_dist_raw if isinstance(d, pd.Timestamp)]
min_date_for_trends_dist = default_min_date_dist; max_date_for_trends_dist = default_max_date_dist
default_start_val_dist_trends = default_min_date_dist; default_end_val_dist_trends = default_max_date_dist   

if all_valid_timestamps_dist_final:
    try:
        combined_series_dist = pd.Series(all_valid_timestamps_dist_final).drop_duplicates().sort_values(ignore_index=True)
        if not combined_series_dist.empty:
            min_date_ts_dist = combined_series_dist.iloc[0] 
            max_date_ts_dist = combined_series_dist.iloc[-1] 
            min_date_for_trends_dist = min_date_ts_dist.date()
            max_date_for_trends_dist = max_date_ts_dist.date()
            default_end_val_dist_trends = max_date_for_trends_dist
            default_start_val_dist_trends = max_date_for_trends_dist - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND - 1)
            if default_start_val_dist_trends < min_date_for_trends_dist: default_start_val_dist_trends = min_date_for_trends_dist
            logger.info(f"District Dashboard: Date filter range for trends determined: {min_date_for_trends_dist} to {max_date_for_trends_dist}.")
        else: logger.warning("District Dashboard: Combined timestamp series empty for trends. Using fallback dates.")
    except Exception as e_min_max_dist_trends: logger.error(f"District Dashboard: CRITICAL ERROR determining trend date range: {e_min_max_dist_trends}. Using fallback.", exc_info=True)
else: logger.warning("District Dashboard: No valid Timestamps for trend date filter. Using wide fallback dates.")

if default_start_val_dist_trends < min_date_for_trends_dist : default_start_val_dist_trends = min_date_for_trends_dist
if default_end_val_dist_trends > max_date_for_trends_dist : default_end_val_dist_trends = max_date_for_trends_dist
if default_start_val_dist_trends > default_end_val_dist_trends : default_start_val_dist_trends = default_end_val_dist_trends

selected_start_date_dist_trends, selected_end_date_dist_trends = st.sidebar.date_input(
    "Select Date Range for Trend Analysis:", value=[default_start_val_dist_trends, default_end_val_dist_trends],
    min_value=min_date_for_trends_dist, max_value=max_date_for_trends_dist, key="district_trends_date_selector_final_v7", # Key incremented
    help="This date range applies to time-series trend charts for health and environmental data."
)

# Filter health_records and IoT records for the selected trend period
filtered_health_for_trends = pd.DataFrame(columns=health_records_district_main.columns if health_records_district_main is not None else [])
if selected_start_date_dist_trends and selected_end_date_dist_trends and health_records_district_main is not None and 'date' in health_records_district_main.columns and not health_records_district_main.empty:
    temp_health_trends = health_records_district_main.copy()
    if not pd.api.types.is_datetime64_ns_dtype(temp_health_trends['date']): 
        temp_health_trends['date'] = pd.to_datetime(temp_health_trends['date'], errors='coerce')
    temp_health_trends.dropna(subset=['date'], inplace=True)
    if not temp_health_trends.empty:
        # Ensure date_obj_filter_trends is correctly typed as date objects
        date_col_is_date_type_trends_h = False
        if 'date_obj_filter_trends' in temp_health_trends.columns and not temp_health_trends.empty and temp_health_trends['date_obj_filter_trends'].notna().any(): 
            first_valid_h_trends = temp_health_trends['date_obj_filter_trends'].dropna().iloc[0] if not temp_health_trends['date_obj_filter_trends'].dropna().empty else None
            if first_valid_h_trends is not None and isinstance(first_valid_h_trends, pd.Timestamp.date().__class__):
                date_col_is_date_type_trends_h = True
        if not date_col_is_date_type_trends_h:
            temp_health_trends['date_obj_filter_trends'] = temp_health_trends['date'].dt.date
        
        mask_health_trends = (temp_health_trends['date_obj_filter_trends'] >= selected_start_date_dist_trends) & \
                             (temp_health_trends['date_obj_filter_trends'] <= selected_end_date_dist_trends) & \
                             (temp_health_trends['date_obj_filter_trends'].notna())
        filtered_health_for_trends = temp_health_trends[mask_health_trends].copy()
        logger.debug(f"District Health Trends: Filtered to {len(filtered_health_for_trends)} rows for {selected_start_date_dist_trends} to {selected_end_date_dist_trends}")


filtered_iot_for_trends = pd.DataFrame(columns=iot_records_district_main.columns if iot_records_district_main is not None else [])
if selected_start_date_dist_trends and selected_end_date_dist_trends and iot_records_district_main is not None and 'timestamp' in iot_records_district_main.columns and not iot_records_district_main.empty:
    temp_iot_trends = iot_records_district_main.copy()
    if not pd.api.types.is_datetime64_ns_dtype(temp_iot_trends['timestamp']): 
        temp_iot_trends['timestamp'] = pd.to_datetime(temp_iot_trends['timestamp'], errors='coerce')
    temp_iot_trends.dropna(subset=['timestamp'], inplace=True)
    if not temp_iot_trends.empty:
        date_col_is_date_type_trends_i = False
        if 'date_obj_filter_trends' in temp_iot_trends.columns and not temp_iot_trends.empty and temp_iot_trends['date_obj_filter_trends'].notna().any(): 
            first_valid_i_trends = temp_iot_trends['date_obj_filter_trends'].dropna().iloc[0] if not temp_iot_trends['date_obj_filter_trends'].dropna().empty else None
            if first_valid_i_trends is not None and isinstance(first_valid_i_trends, pd.Timestamp.date().__class__): # Corrected variable name
                date_col_is_date_type_trends_i = True
        if not date_col_is_date_type_trends_i:
            temp_iot_trends['date_obj_filter_trends'] = temp_iot_trends['timestamp'].dt.date

        mask_iot_trends = (temp_iot_trends['date_obj_filter_trends'] >= selected_start_date_dist_trends) & \
                          (temp_iot_trends['date_obj_filter_trends'] <= selected_end_date_dist_trends) & \
                          (temp_iot_trends['date_obj_filter_trends'].notna())
        filtered_iot_for_trends = temp_iot_trends[mask_iot_trends].copy()
        logger.debug(f"District IoT Trends: Filtered to {len(filtered_iot_for_trends)} rows for {selected_start_date_dist_trends} to {selected_end_date_dist_trends}")


# --- KPIs Section ---
st.subheader("District-Wide Key Performance Indicators (Aggregated Zonal Data)")
if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty:
    district_overall_kpis = get_district_summary_kpis(district_gdf_main_enriched)
    logger.debug(f"District Overall KPIs: {district_overall_kpis}")

    kpi_cols_row1_dist = st.columns(4) 
    with kpi_cols_row1_dist[0]:
        avg_pop_risk_val = district_overall_kpis.get('avg_population_risk', 0.0)
        render_kpi_card("Avg. Population Risk", f"{avg_pop_risk_val:.1f}", "🎯",
                        status="High" if avg_pop_risk_val > app_config.RISK_THRESHOLDS['high'] else "Moderate" if avg_pop_risk_val > app_config.RISK_THRESHOLDS['moderate'] else "Low",
                        help_text="Population-weighted average AI risk score across all zones.")
    with kpi_cols_row1_dist[1]:
        facility_coverage_val = district_overall_kpis.get('overall_facility_coverage', 0.0)
        render_kpi_card("Facility Coverage", f"{facility_coverage_val:.1f}%", "🏥",
                        status="Bad Low" if facility_coverage_val < app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD else "Moderate" if facility_coverage_val < 80 else "Good High",
                        help_text="Population-weighted score reflecting access and capacity of health facilities.")
    with kpi_cols_row1_dist[2]:
        high_risk_zones_num = district_overall_kpis.get('zones_high_risk_count', 0)
        total_zones_val = len(district_gdf_main_enriched) if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty else 1 
        perc_high_risk_zones = (high_risk_zones_num / total_zones_val) * 100 if total_zones_val > 0 else 0
        render_kpi_card("High-Risk Zones", f"{high_risk_zones_num} ({perc_high_risk_zones:.0f}%)", "⚠️",
                        status="High" if perc_high_risk_zones > 25 else "Moderate" if high_risk_zones_num > 0 else "Low", 
                        help_text=f"Number (and %) of zones with average risk score ≥ {app_config.RISK_THRESHOLDS['district_zone_high_risk']}.")
    with kpi_cols_row1_dist[3]:
        district_prevalence_val = district_overall_kpis.get('key_infection_prevalence_district_per_1000', 0.0)
        render_kpi_card("Overall Prevalence", f"{district_prevalence_val:.1f} /1k Pop", "📈", 
                        status="High" if district_prevalence_val > 50 else ("Moderate" if district_prevalence_val > 10 else "Low"), 
                        help_text="Combined prevalence of key infectious diseases per 1,000 population in the district.")

    st.markdown("##### Key Disease Burdens & District Wellness / Environment")
    kpi_cols_row2_dist = st.columns(4)
    with kpi_cols_row2_dist[0]:
        tb_total_burden = district_overall_kpis.get('district_tb_burden_total', 0)
        render_kpi_card(
            title="Active TB Cases", 
            value=str(tb_total_burden), 
            icon="🫁",  # <<< CHANGED TO DIRECT EMOJI
            # icon_is_html=True, # NOT NEEDED for direct emoji
            status="High" if tb_total_burden > (len(district_gdf_main_enriched) * app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty else 50) else "Moderate", 
            help_text="Total active TB cases identified across the district (latest aggregates)."
        )
    with kpi_cols_row2_dist[1]:
        malaria_total_burden = district_overall_kpis.get('district_malaria_burden_total',0)
        render_kpi_card(
            title="Active Malaria Cases", 
            value=str(malaria_total_burden), 
            icon="🦟",  # <<< CHANGED TO DIRECT EMOJI
            # icon_is_html=True, # NOT NEEDED for direct emoji
            status="High" if malaria_total_burden > (len(district_gdf_main_enriched) * app_config.INTERVENTION_MALARIA_BURDEN_HIGH_THRESHOLD if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty else 100) else "Moderate",
            help_text="Total active Malaria cases identified across the district (latest aggregates)."
        )
    with kpi_cols_row2_dist[2]:
        avg_steps_district = district_overall_kpis.get('population_weighted_avg_steps', 0.0)
        render_kpi_card("Avg. Patient Steps", f"{avg_steps_district:,.0f}", "👣", 
                        status="Bad Low" if avg_steps_district < (app_config.TARGET_DAILY_STEPS * 0.7) else "Moderate" if avg_steps_district < app_config.TARGET_DAILY_STEPS else "Good High",
                        help_text=f"Population-weighted average daily steps. Target: {app_config.TARGET_DAILY_STEPS:,.0f} steps.")
    with kpi_cols_row2_dist[3]:
        avg_co2_district_val = district_overall_kpis.get('avg_clinic_co2_district',0.0)
        render_kpi_card("Avg. Clinic CO2", f"{avg_co2_district_val:.0f} ppm", "💨", 
                        status="High" if avg_co2_district_val > app_config.CO2_LEVEL_ALERT_PPM else "Moderate" if avg_co2_district_val > app_config.CO2_LEVEL_IDEAL_PPM else "Low",
                        help_text="District average of zonal mean CO2 levels in clinics (unweighted average of zonal means).")
else:
    st.warning("District-Wide KPIs cannot be displayed: Enriched zone geographic data is unavailable. Please check data loading and processing steps.")
st.markdown("---") 

st.subheader("🗺️ Interactive Health & Environment Map of the District")
if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty and 'geometry' in district_gdf_main_enriched.columns and not district_gdf_main_enriched.geometry.is_empty.all():
    map_metric_options_config_dist = {"Avg. AI Risk Score": {"col": "avg_risk_score", "colorscale": "Reds_r"}, "Total Key Infections": {"col": "total_active_key_infections", "colorscale": "OrRd_r"}, "Prevalence per 1,000 (Key Inf.)": {"col": "prevalence_per_1000", "colorscale": "YlOrRd_r"}, "Facility Coverage Score": {"col": "facility_coverage_score", "colorscale": "Greens"}, "Active TB Cases": {"col": "active_tb_cases", "colorscale": "Purples_r"}, "Active Malaria Cases": {"col": "active_malaria_cases", "colorscale": "Oranges_r"}, "HIV Positive Cases (Agg.)": {"col": "hiv_positive_cases", "colorscale": "Magenta_r"}, "Avg. Patient Steps (Zone)": {"col": "avg_daily_steps_zone", "colorscale": "Cividis"}, "Avg. Zone CO2 (Clinics)": {"col": "zone_avg_co2", "colorscale": "Sunsetdark_r"}, "Number of Clinics": {"col": "num_clinics", "colorscale": "Blues"}, "Socio-Economic Index": {"col": "socio_economic_index", "colorscale": "Tealgrn"}}
    if 'population' in district_gdf_main_enriched.columns:
        try: 
            gdf_proj_area_map = district_gdf_main_enriched.copy() # Create a copy for reprojection
            if gdf_proj_area_map.crs and not gdf_proj_area_map.crs.is_geographic: # If already projected
                pass # Area can be calculated directly
            elif gdf_proj_area_map.crs and gdf_proj_area_map.crs.is_geographic: # If geographic, try to project
                utm_crs = gdf_proj_area_map.estimate_utm_crs()
                if utm_crs:
                    gdf_proj_area_map = gdf_proj_area_map.to_crs(utm_crs)
                else: # Could not estimate UTM, skip area calculation
                    logger.warning("Map: Could not estimate UTM CRS for area calculation. Population density will be NaN.")
                    district_gdf_main_enriched.loc[:,'area_sqkm'] = np.nan # Assign NaN to the original GDF
            else: # No CRS information
                 logger.warning("Map: No CRS information on GeoDataFrame. Population density will be NaN.")
                 district_gdf_main_enriched.loc[:,'area_sqkm'] = np.nan

            # Calculate area only if it has a projected CRS and geometry
            if hasattr(gdf_proj_area_map, 'crs') and gdf_proj_area_map.crs is not None and not gdf_proj_area_map.crs.is_geographic:
                 district_gdf_main_enriched.loc[:, 'area_sqkm'] = gdf_proj_area_map.geometry.area / 1_000_000
                 district_gdf_main_enriched.loc[:, 'population_density'] = district_gdf_main_enriched.apply(
                    lambda r: r['population'] / r['area_sqkm'] if pd.notna(r.get('area_sqkm')) and r.get('area_sqkm',0)>0 and pd.notna(r.get('population')) else 0, 
                    axis=1
                )
                 map_metric_options_config_dist["Population Density (Pop/SqKm)"] = {"col": "population_density", "colorscale": "Plasma_r"}
            elif 'population_density' not in district_gdf_main_enriched.columns: # Ensure column exists even if not calculated
                district_gdf_main_enriched.loc[:,'population_density'] = np.nan

        except Exception as e_map_area_calc_new: logger.warning(f"Map: Could not calculate area/pop density for map metric options: {e_map_area_calc_new}", exc_info=True); district_gdf_main_enriched.loc[:,'population_density'] = np.nan

    available_map_metrics_for_select = { disp_name: details for disp_name, details in map_metric_options_config_dist.items() if details["col"] in district_gdf_main_enriched.columns and district_gdf_main_enriched[details["col"]].notna().any() }
    if available_map_metrics_for_select:
        selected_map_metric_display_name = st.selectbox("Select Metric to Visualize on Map:", list(available_map_metrics_for_select.keys()), key="district_interactive_map_metric_selector_final_v6", help="Choose a metric for spatial visualization.")
        selected_map_metric_config = available_map_metrics_for_select.get(selected_map_metric_display_name)
        if selected_map_metric_config:
            map_val_col = selected_map_metric_config["col"]; map_colorscale = selected_map_metric_config["colorscale"]; hover_cols_for_map = ['name', 'population', map_val_col] 
            if 'num_clinics' in district_gdf_main_enriched.columns and map_val_col != 'num_clinics': hover_cols_for_map.append('num_clinics')
            if 'facility_coverage_score' in district_gdf_main_enriched.columns and map_val_col != 'facility_coverage_score': hover_cols_for_map.append('facility_coverage_score')
            final_hover_cols_map = list(dict.fromkeys([col for col in hover_cols_for_map if col in district_gdf_main_enriched.columns]))
            map_figure = plot_layered_choropleth_map(gdf=district_gdf_main_enriched, value_col=map_val_col, title=f"District Map: {selected_map_metric_display_name}", id_col='zone_id', featureidkey_prefix='properties', color_continuous_scale=map_colorscale, hover_cols=final_hover_cols_map, height=app_config.MAP_PLOT_HEIGHT, mapbox_style=app_config.MAPBOX_STYLE)
            st.plotly_chart(map_figure, use_container_width=True)
        else: st.info("Please select a metric from the dropdown to display on the map.")
    else: st.warning("No metrics with valid data are currently available for map visualization in the enriched GeoDataFrame.")
else: st.error("🚨 District map cannot be displayed: Enriched zone geographic data is unusable.")
st.markdown("---") 
tab_dist_trends, tab_dist_comparison, tab_dist_interventions = st.tabs(["📈 District-Wide Trends", "📊 Zonal Comparative Analysis", "🎯 Intervention Planning Insights"])
with tab_dist_trends:
    st.header("📈 District-Wide Health & Environmental Trends")
    if (filtered_health_for_trends.empty and filtered_iot_for_trends.empty): st.info(f"No health or environmental data available for the selected trend period ({selected_start_date_dist_trends.strftime('%d %b %Y')} to {selected_end_date_dist_trends.strftime('%d %b %Y')}).")
    else:
        st.markdown(f"Displaying trends from **{selected_start_date_dist_trends.strftime('%d %b %Y')}** to **{selected_end_date_dist_trends.strftime('%d %b %Y')}**."); st.subheader("Key Disease Incidence Trends (New Cases Identified per Week)")
        cols_disease_trends = st.columns(2)
        with cols_disease_trends[0]:
            if not filtered_health_for_trends.empty and 'condition' in filtered_health_for_trends.columns and 'patient_id' in filtered_health_for_trends.columns:
                tb_trends_src = filtered_health_for_trends[filtered_health_for_trends['condition'] == 'TB']; weekly_tb_trend = get_trend_data(tb_trends_src, 'patient_id', date_col='date', period='W', agg_func='nunique')
                if not weekly_tb_trend.empty: st.plotly_chart(plot_annotated_line_chart(weekly_tb_trend, "Weekly New TB Patients Identified", y_axis_title="New TB Patients", height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                else: st.caption("No TB trend data for this period.")
            else: st.caption("TB trend data cannot be generated.")
        with cols_disease_trends[1]:
            if not filtered_health_for_trends.empty and 'condition' in filtered_health_for_trends.columns and 'patient_id' in filtered_health_for_trends.columns:
                malaria_trends_src = filtered_health_for_trends[filtered_health_for_trends['condition'] == 'Malaria']; weekly_malaria_trend = get_trend_data(malaria_trends_src, 'patient_id', date_col='date', period='W', agg_func='nunique')
                if not weekly_malaria_trend.empty: st.plotly_chart(plot_annotated_line_chart(weekly_malaria_trend, "Weekly New Malaria Patients Identified", y_axis_title="New Malaria Patients", height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                else: st.caption("No Malaria trend data for this period.")
            else: st.caption("Malaria trend data unavailable.")
        st.subheader("Population Wellness & Environmental Trends"); cols_wellness_env = st.columns(2)
        with cols_wellness_env[0]:
            if not filtered_health_for_trends.empty and 'avg_daily_steps' in filtered_health_for_trends.columns:
                steps_trends_dist = get_trend_data(filtered_health_for_trends, 'avg_daily_steps', date_col='date', period='W', agg_func='mean')
                if not steps_trends_dist.empty: st.plotly_chart(plot_annotated_line_chart(steps_trends_dist, "Weekly Avg. Patient Daily Steps", y_axis_title="Average Steps", target_line=app_config.TARGET_DAILY_STEPS, height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                else: st.caption("No steps trend data for this period.")
            else: st.caption("Avg. daily steps data missing for trends.")
        with cols_wellness_env[1]:
            if not filtered_iot_for_trends.empty and 'avg_co2_ppm' in filtered_iot_for_trends.columns:
                co2_trends_dist_iot = get_trend_data(filtered_iot_for_trends, 'avg_co2_ppm', date_col='timestamp', period='D', agg_func='mean')
                if not co2_trends_dist_iot.empty: st.plotly_chart(plot_annotated_line_chart(co2_trends_dist_iot, "Daily Avg. CO2 (All Monitored Clinics)", y_axis_title="CO2 (ppm)", target_line=app_config.CO2_LEVEL_ALERT_PPM, height=app_config.COMPACT_PLOT_HEIGHT), use_container_width=True)
                else: st.caption("No CO2 trend data from clinics for this period.")
            else: st.caption("Clinic CO2 data missing for trends.")

with tab_dist_comparison:
    st.header("📊 Zonal Comparative Analysis")
    if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty and 'geometry' in district_gdf_main_enriched.columns:
        st.markdown("Compare zones using aggregated health, resource, environmental, and socio-economic metrics. Data is based on latest aggregations.")
        comp_table_metrics_dict = {name: details for name, details in map_metric_options_config_dist.items() if details["col"] in district_gdf_main_enriched.columns and district_gdf_main_enriched[details["col"]].notna().any()}
        if 'Population Density (Pop/SqKm)' in comp_table_metrics_dict and 'format' not in comp_table_metrics_dict['Population Density (Pop/SqKm)']: comp_table_metrics_dict['Population Density (Pop/SqKm)']['format'] = "{:,.1f}" 
        if comp_table_metrics_dict:
            st.subheader("Zonal Statistics Table"); cols_for_comp_table_display = ['name'] + [d['col'] for d in comp_table_metrics_dict.values()]
            df_for_comp_table_display = district_gdf_main_enriched[[col for col in cols_for_comp_table_display if col in district_gdf_main_enriched.columns]].copy()
            df_for_comp_table_display.rename(columns={'name':'Zone'}, inplace=True)
            if 'Zone' in df_for_comp_table_display.columns: df_for_comp_table_display.set_index('Zone', inplace=True)
            style_formats_comp = {details["col"]: details.get("format", "{:.1f}") for _, details in comp_table_metrics_dict.items() if "format" in details and details["col"] in df_for_comp_table_display.columns}
            styler_obj_comp = df_for_comp_table_display.style.format(style_formats_comp)
            for _, details_style_comp in comp_table_metrics_dict.items():
                col_name_to_style = details_style_comp["col"]
                if col_name_to_style in df_for_comp_table_display.columns:
                    cmap_gradient = 'Reds_r' if "_r" in details_style_comp.get("colorscale", "Reds_r").lower() else 'Greens'; 
                    if "greens" in details_style_comp.get("colorscale","").lower() : cmap_gradient = 'Greens'
                    elif "reds" in details_style_comp.get("colorscale","").lower() : cmap_gradient = 'Reds'
                    try: styler_obj_comp = styler_obj_comp.background_gradient(subset=[col_name_to_style], cmap=cmap_gradient, axis=0)
                    except: pass 
            st.dataframe(styler_obj_comp, use_container_width=True, height=min(len(df_for_comp_table_display) * 45 + 60, 600))
            st.subheader("Visual Comparison Chart"); selected_bar_metric_name_dist_comp_viz = st.selectbox("Select Metric for Bar Chart Comparison:", list(comp_table_metrics_dict.keys()), key="district_comp_barchart_final_v7") # Incremented key
            selected_bar_details_dist_comp_viz = comp_table_metrics_dict.get(selected_bar_metric_name_dist_comp_viz)
            if selected_bar_details_dist_comp_viz:
                bar_col_for_comp_viz = selected_bar_details_dist_comp_viz["col"]; text_format_bar_comp_viz = selected_bar_details_dist_comp_viz.get("format", "{:.1f}").replace('{','').replace('}','').split(':')[-1]; sort_asc_bar_viz = "_r" not in selected_bar_details_dist_comp_viz.get("colorscale", "") 
                st.plotly_chart(plot_bar_chart(district_gdf_main_enriched, x_col='name', y_col=bar_col_for_comp_viz, title=f"{selected_bar_metric_name_dist_comp_viz} by Zone", x_axis_title="Zone Name", height=app_config.DEFAULT_PLOT_HEIGHT + 150, sort_values_by=bar_col_for_comp_viz, ascending=sort_asc_bar_viz, text_auto=True, text_format=text_format_bar_comp_viz), use_container_width=True)
        else: st.info("No metrics available for Zonal Comparison table/chart.")
    else: st.info("Zonal comparison requires enriched geographic data.")

with tab_dist_interventions:
    st.header("🎯 Intervention Planning Insights")
    if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty and 'geometry' in district_gdf_main_enriched.columns:
        st.markdown("Identify zones for targeted interventions based on customizable criteria related to health risks, disease burdens, resource accessibility, and environmental factors.")
        criteria_lambdas_intervention_dist = {f"High Avg. Risk (≥ {app_config.RISK_THRESHOLDS['district_zone_high_risk']})": lambda df: df.get('avg_risk_score', pd.Series(dtype=float)) >= app_config.RISK_THRESHOLDS['district_zone_high_risk'], f"Low Facility Coverage (< {app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD}%)": lambda df: df.get('facility_coverage_score', pd.Series(dtype=float)) < app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD, f"High Key Inf. Prevalence (Top {100-app_config.INTERVENTION_PREVALENCE_HIGH_PERCENTILE*100:.0f}%)": lambda df: df.get('prevalence_per_1000', pd.Series(dtype=float)) >= df.get('prevalence_per_1000', pd.Series(dtype=float)).quantile(app_config.INTERVENTION_PREVALENCE_HIGH_PERCENTILE) if 'prevalence_per_1000' in df and df['prevalence_per_1000'].notna().any() else pd.Series([False]*len(df), index=df.index), f"High TB Burden (Abs. > {app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD})": lambda df: df.get('active_tb_cases', pd.Series(dtype=float)) > app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD, f"High Avg. Clinic CO2 (> {app_config.CO2_LEVEL_IDEAL_PPM}ppm)": lambda df: df.get('zone_avg_co2', pd.Series(dtype=float)) > app_config.CO2_LEVEL_IDEAL_PPM }
        available_criteria_for_intervention_dist = {}
        for name_crit_int, func_crit_int in criteria_lambdas_intervention_dist.items():
            try: func_crit_int(district_gdf_main_enriched.head(1) if not district_gdf_main_enriched.empty else pd.DataFrame(columns=district_gdf_main_enriched.columns)); available_criteria_for_intervention_dist[name_crit_int] = func_crit_int
            except : pass 
        if not available_criteria_for_intervention_dist: st.warning("Intervention criteria cannot be applied; relevant data columns may be missing from the enriched zone data.")
        else:
            selected_criteria_names_interv = st.multiselect( "Select Criteria to Identify Priority Zones (Zones meeting ANY selected criteria will be shown):", options=list(available_criteria_for_intervention_dist.keys()), default=list(available_criteria_for_intervention_dist.keys())[0:min(2, len(available_criteria_for_intervention_dist))] if available_criteria_for_intervention_dist else [], key="district_intervention_criteria_multiselect_final_v4", help="Choose one or more criteria. Zones satisfying any of these will be listed." )
            if not selected_criteria_names_interv: st.info("Please select at least one criterion above to identify priority zones for potential interventions.")
            else:
                final_intervention_mask_dist = pd.Series([False] * len(district_gdf_main_enriched), index=district_gdf_main_enriched.index)
                for crit_name_selected_interv in selected_criteria_names_interv:
                    crit_func_selected_interv = available_criteria_for_intervention_dist[crit_name_selected_interv]
                    try:
                        current_crit_mask_interv = crit_func_selected_interv(district_gdf_main_enriched)
                        if isinstance(current_crit_mask_interv, pd.Series) and current_crit_mask_interv.dtype == 'bool': final_intervention_mask_dist = final_intervention_mask_dist | current_crit_mask_interv.fillna(False)
                        else: logger.warning(f"Intervention criterion '{crit_name_selected_interv}' did not produce a valid boolean Series.")
                    except Exception as e_crit_apply_interv: logger.error(f"Error applying intervention criterion '{crit_name_selected_interv}': {e_crit_apply_interv}", exc_info=True); st.warning(f"Could not apply criterion: {crit_name_selected_interv}. Error: {e_crit_apply_interv}")
                priority_zones_df_for_interv = district_gdf_main_enriched[final_intervention_mask_dist].copy()
                if not priority_zones_df_for_interv.empty:
                    st.markdown(f"###### Identified **{len(priority_zones_df_for_interv)}** Zone(s) Meeting Selected Intervention Criteria:")
                    cols_intervention_table_display = ['name', 'population', 'avg_risk_score', 'total_active_key_infections', 'prevalence_per_1000', 'facility_coverage_score', 'zone_avg_co2']
                    actual_cols_interv_table_display = [col for col in cols_intervention_table_display if col in priority_zones_df_for_interv.columns]
                    sort_by_list_interv_display = []; sort_asc_list_interv_display = []
                    if 'avg_risk_score' in actual_cols_interv_table_display: sort_by_list_interv_display.append('avg_risk_score'); sort_asc_list_interv_display.append(False)
                    if 'prevalence_per_1000' in actual_cols_interv_table_display: sort_by_list_interv_display.append('prevalence_per_1000'); sort_asc_list_interv_display.append(False)
                    if 'facility_coverage_score' in actual_cols_interv_table_display: sort_by_list_interv_display.append('facility_coverage_score'); sort_asc_list_interv_display.append(True)
                    interv_df_display_sorted_final = priority_zones_df_for_interv.sort_values(by=sort_by_list_interv_display, ascending=sort_asc_list_interv_display) if sort_by_list_interv_display else priority_zones_df_for_interv
                    st.dataframe(interv_df_display_sorted_final[actual_cols_interv_table_display], use_container_width=True, hide_index=True, column_config={"name": st.column_config.TextColumn("Zone Name", help="Administrative zone name."), "population": st.column_config.NumberColumn("Population", format="%,.0f"), "avg_risk_score": st.column_config.ProgressColumn("Avg. Risk Score", format="%.1f", min_value=0, max_value=100), "total_active_key_infections": st.column_config.NumberColumn("Total Key Infections", format="%.0f"), "prevalence_per_1000": st.column_config.NumberColumn("Prevalence (/1k Pop.)", format="%.1f"), "facility_coverage_score": st.column_config.NumberColumn("Facility Coverage (%)", format="%.1f%%"), "zone_avg_co2": st.column_config.NumberColumn("Avg. Clinic CO2 (ppm)", format="%.0f ppm")})
                else: st.success("✅ No zones currently meet the selected high-priority criteria based on the available aggregated data.")
    else: st.info("Intervention planning insights require successfully loaded and enriched geographic zone data. Please check data sources and processing.")
