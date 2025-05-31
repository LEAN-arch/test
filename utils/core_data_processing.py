# health_hub/utils/core_data_processing.py
import pandas as pd
import geopandas as gpd
import os
import logging
import streamlit as st
import numpy as np
from config import app_config 

logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
                    format=app_config.LOG_FORMAT, datefmt=app_config.LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

# --- Custom Hash Function for GeoDataFrames ---
def hash_geodataframe(gdf): # pragma: no cover
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        return None 
    try:
        gdf_sorted = gdf.sort_index().reindex(sorted(gdf.columns), axis=1)
        geom_col_name = gdf_sorted.geometry.name
        data_hash_part = gdf_sorted.drop(columns=[geom_col_name], errors='ignore').to_parquet()
        geometry_hash_part = b""
        if geom_col_name in gdf_sorted.columns and not gdf_sorted[geom_col_name].empty:
            valid_geoms = gdf_sorted[geom_col_name][gdf_sorted[geom_col_name].is_valid & ~gdf_sorted[geom_col_name].is_empty]
            if not valid_geoms.empty:
                geometry_hash_part = valid_geoms.to_wkb().values.tobytes()
        crs_hash_part = gdf_sorted.crs.to_wkt() if gdf_sorted.crs else None
        return (data_hash_part, geometry_hash_part, crs_hash_part)
    except Exception as e:
        logger.error(f"Error hashing GeoDataFrame (falling back to basic hash of head): {e}", exc_info=True)
        return (str(gdf.drop(columns=[gdf.geometry.name], errors='ignore').head(3).to_dict()), str(gdf.geometry.to_wkt().head(3).to_dict()), str(gdf.crs.to_wkt() if gdf.crs else None))

# --- Data Loading Functions ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def load_health_records():
    file_path = app_config.HEALTH_RECORDS_CSV
    logger.info(f"Attempting to load health records from: {file_path}")
    # Updated default_empty_health_df_cols to include new sample rejection columns
    default_empty_health_df_cols = ["date", "zone_id", "patient_id", "condition", "ai_risk_score", 'item', 'stock_on_hand', 'consumption_rate_per_day', 'test_type', 'test_result', 'test_turnaround_days', 'sample_status', 'rejection_reason']
    default_empty_health_df = pd.DataFrame(columns=default_empty_health_df_cols)
    try:
        if not os.path.exists(file_path):
            logger.error(f"Health records file not found: {file_path}")
            st.error(f"Data file '{os.path.basename(file_path)}' not found. Please check 'data_sources/' directory and configuration.")
            return default_empty_health_df

        df = pd.read_csv(file_path, low_memory=False) 
        if df.empty:
            logger.warning(f"Health records file '{file_path}' is empty.")
            return default_empty_health_df

        date_cols_to_parse = ['date', 'referral_date', 'test_date']
        for col in date_cols_to_parse:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                logger.warning(f"Date column '{col}' not found in health_records.csv.")
        
        if 'date' not in df.columns or df['date'].isnull().all():
            logger.error("Critical 'date' column is missing or all values are invalid in health_records.csv.")
            st.error("Health records are missing a valid 'date' column. Processing cannot continue reliably.")
            return default_empty_health_df

        required_cols = ["zone_id", "patient_id", "condition", "ai_risk_score"] 
        missing_req = [col for col in required_cols if col not in df.columns or df[col].isnull().all()] 
        if missing_req:
            logger.error(f"Missing or entirely null critical required columns in health records: {missing_req}.")
            st.error(f"Health records are missing critical data in columns: {missing_req}.")
            return default_empty_health_df

        numeric_cols_expected = ['test_turnaround_days', 'quantity_dispensed', 'stock_on_hand', 'consumption_rate_per_day', 'ai_risk_score', 'avg_daily_steps', 'avg_spo2', 'min_spo2_pct', 'max_skin_temp_celsius', 'fall_detected_today', 'age', 'chw_visit', 'tb_contact_traced', 'resting_heart_rate', 'avg_hrv', 'avg_sleep_duration_hrs', 'sleep_score_pct', 'stress_level_score', 'hiv_viral_load']
        string_cols_expected = ['patient_id', 'condition', 'test_type', 'test_result', 'item', 'zone_id', 'clinic_id', 'physician_id', 'notes', 'hpv_status', 'referral_status', 'gender', 'sample_status', 'rejection_reason'] # Added new sample cols

        for col in numeric_cols_expected:
            if col not in df.columns: df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')

        common_null_strs = ['', 'nan', 'None', 'NONE', 'Null', 'NULL', '<NA>', 'N/A', ' '] # Added single space
        for col in string_cols_expected:
            if col not in df.columns: df[col] = 'Unknown'
            df[col] = df[col].astype(str).fillna('Unknown')
            # Replace common null/empty representations and whitespace-only with 'Unknown'
            df[col] = df[col].str.strip() # Remove leading/trailing whitespace first
            df.loc[df[col].isin(common_null_strs), col] = 'Unknown'
        
        df['condition'] = df['condition'].replace('Healthy Checkup', 'Wellness Visit')
        # Clean test_result further, map N/A specifically, handle potential case issues if needed
        df['test_result'] = df['test_result'].str.strip().replace({'Positive ': 'Positive', ' Negative': 'Negative', 'N/A':'Unknown', 'nan':'Unknown'})
        
        df.dropna(subset=['patient_id', 'zone_id', 'date'], inplace=True)
        df = df[~df['patient_id'].isin(['Unknown'])]
        df = df[~df['zone_id'].isin(['Unknown'])]
        
        if df.empty:
            logger.warning("Health records DataFrame became empty after cleaning essential IDs or dates.")
            return default_empty_health_df

        logger.info(f"Successfully loaded and preprocessed health records from {file_path} ({len(df)} rows).")
        return df
    except pd.errors.ParserError as pe:
        logger.error(f"Parsing error loading health records from {file_path}: {pe}", exc_info=True)
        st.error(f"Error parsing health records CSV. Please check file format, especially around line numbers indicated in logs. Error: {pe}")
        return default_empty_health_df
    except pd.errors.EmptyDataError: 
        logger.error(f"Health records file is empty: {file_path}")
        st.error(f"Health records file '{os.path.basename(file_path)}' is empty.")
        return default_empty_health_df
    except Exception as e: 
        logger.error(f"Unexpected error loading health records from {file_path}: {e}", exc_info=True)
        st.error(f"An unexpected error occurred while loading health records: {e}")
        return default_empty_health_df

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, hash_funcs={gpd.GeoDataFrame: hash_geodataframe})
def load_zone_data():
    attributes_path = app_config.ZONE_ATTRIBUTES_CSV
    geometries_path = app_config.ZONE_GEOMETRIES_GEOJSON
    logger.info(f"Loading zone attributes from: {attributes_path}")
    logger.info(f"Loading zone geometries from: {geometries_path}")

    if not os.path.exists(attributes_path): logger.error(f"Zone attributes file not found: {attributes_path}"); st.error(f"Zone attributes file ('{os.path.basename(attributes_path)}') not found."); return None 
    if not os.path.exists(geometries_path): logger.error(f"Zone geometries file not found: {geometries_path}"); st.error(f"Zone geometries file ('{os.path.basename(geometries_path)}') not found."); return None

    try:
        zone_attributes_df = pd.read_csv(attributes_path); 
        if zone_attributes_df.empty: logger.error(f"Zone attributes file '{os.path.basename(attributes_path)}' empty."); st.error(f"Zone attributes file '{os.path.basename(attributes_path)}' empty."); return None
        required_attr_cols = ["zone_id", "zone_display_name", "population", "socio_economic_index", "num_clinics", "avg_travel_time_clinic_min"]
        missing_attrs = [col for col in required_attr_cols if col not in zone_attributes_df.columns]; 
        if missing_attrs: logger.error(f"Zone_attr CSV missing: {missing_attrs}."); st.error(f"Zone_attr CSV missing: {missing_attrs}."); return None
        zone_geometries_gdf = gpd.read_file(geometries_path); 
        if zone_geometries_gdf.empty: logger.error(f"Zone_geom file '{os.path.basename(geometries_path)}' empty/no features."); st.error(f"Zone_geom file '{os.path.basename(geometries_path)}' empty/invalid."); return None
        if "zone_id" not in zone_geometries_gdf.columns: logger.error("Zone_geom missing 'zone_id' property."); st.error("Zone_geom missing 'zone_id'."); return None
        if zone_geometries_gdf.crs is None: logger.warning(f"Zone_geom no CRS. Assuming {app_config.DEFAULT_CRS}."); zone_geometries_gdf = zone_geometries_gdf.set_crs(app_config.DEFAULT_CRS, allow_override=True)
        elif zone_geometries_gdf.crs.to_string().upper() != app_config.DEFAULT_CRS.upper(): logger.info(f"Reprojecting Zone_geom from {zone_geometries_gdf.crs.to_string()} to {app_config.DEFAULT_CRS}."); zone_geometries_gdf = zone_geometries_gdf.to_crs(app_config.DEFAULT_CRS)
        zone_attributes_df['zone_id'] = zone_attributes_df['zone_id'].astype(str); zone_geometries_gdf['zone_id'] = zone_geometries_gdf['zone_id'].astype(str)
        merged_gdf = zone_geometries_gdf.merge(zone_attributes_df, on="zone_id", how="left", suffixes=('_geom', '_attr')); logger.debug(f"Cols in merged_gdf after initial merge: {merged_gdf.columns.tolist()}")
        if 'zone_display_name' in merged_gdf.columns: merged_gdf.rename(columns={'zone_display_name': 'name'}, inplace=True); logger.info("Renamed 'zone_display_name' to 'name'.")
        elif 'name_geom' in merged_gdf.columns and 'name' not in merged_gdf.columns : merged_gdf.rename(columns={'name_geom': 'name'}, inplace=True); logger.info("Used 'name_geom' as 'name'.")
        elif 'name' not in merged_gdf.columns: logger.warning("No 'name' or 'zone_display_name'. Creating 'name' from 'zone_id'."); merged_gdf['name'] = "Zone_" + merged_gdf['zone_id'].astype(str)
        if 'name' in merged_gdf.columns and merged_gdf['name'].isnull().any(): unmatched_zones = merged_gdf[merged_gdf['name'].isnull()]['zone_id'].unique().tolist(); logger.warning(f"Zones with null names: {unmatched_zones}. Using zone_id fallback."); merged_gdf.loc[merged_gdf['name'].isnull(), 'name'] = "Zone_" + merged_gdf.loc[merged_gdf['name'].isnull(), 'zone_id'].astype(str)
        merged_gdf.drop(columns=[c for c in ['name_geom', 'name_attr'] if c in merged_gdf.columns], errors='ignore', inplace=True); logger.debug(f"Cols in merged_gdf after name handling: {merged_gdf.columns.tolist()}")
        numeric_attr_cols = ["population", "socio_economic_index", "num_clinics", "avg_travel_time_clinic_min"]
        for col_b in num_attr_cols:
            suf_attr = f"{col_b}_attr"; suf_geom = f"{col_b}_geom"
            if suf_attr in merged_gdf.columns: merged_gdf[col_b] = merged_gdf[suf_attr]
            elif suf_geom in merged_gdf.columns and col_b not in merged_gdf.columns: merged_gdf[col_b] = merged_gdf[suf_geom]
            if col_b in merged_gdf.columns: merged_gdf[col_b] = pd.to_numeric(merged_gdf[col_b], errors='coerce').fillna(0) 
            else: logger.warning(f"Num attr col '{col_b}' missing. Init 0."); merged_gdf[col_b] = 0.0
            merged_gdf.drop(columns=[suf_attr, suf_geom], errors='ignore', inplace=True)
        logger.info(f"Loaded/merged zone data ({len(merged_gdf)} features)."); return merged_gdf
    except Exception as e: logger.error(f"Err loading/merging zone data: {type(e).__name__} - {e}", exc_info=True); st.error(f"Err loading zone data: {e}"); return None

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
def load_iot_clinic_environment_data():
    file_path = app_config.IOT_CLINIC_ENVIRONMENT_CSV; logger.info(f"Loading IoT data from: {file_path}")
    def_empty_cols = ['timestamp', 'clinic_id', 'room_name', 'zone_id', 'avg_co2_ppm', 'avg_pm25', 'waiting_room_occupancy', 'sanitizer_dispenses_per_hour']; def_empty_df = pd.DataFrame(columns=def_empty_cols)
    try:
        if not os.path.exists(file_path): logger.warning(f"IoT file not found: {file_path}. Env metrics unavailable."); return def_empty_df
        df = pd.read_csv(file_path); 
        if df.empty: logger.warning(f"IoT file empty: {file_path}."); return def_empty_df
        if 'timestamp' not in df.columns: logger.error("IoT missing 'timestamp'. Unusable."); st.error("IoT missing 'timestamp'."); return def_empty_df 
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce'); df.dropna(subset=['timestamp'], inplace=True); 
        if df.empty : return def_empty_df 
        req_iot_cols = ['clinic_id', 'room_name', 'avg_co2_ppm']; missing_iot = [c for c in req_iot_cols if c not in df.columns]; 
        if missing_iot: logger.warning(f"IoT missing: {missing_iot}. Metrics may be incomplete.")
        if 'zone_id' not in df.columns or df['zone_id'].isnull().all():
            if 'clinic_id' in df.columns: df['zone_id'] = df['clinic_id'].astype(str).map({ 'C01': 'ZoneA', 'C02': 'ZoneB', 'C03': 'ZoneC', 'C04': 'ZoneD', 'C05':'ZoneE', 'C06':'ZoneF' }).fillna('UnknownZoneByClinic'); logger.info("Derived 'zone_id' for IoT from 'clinic_id'.")
            else: logger.warning("IoT missing 'zone_id' & 'clinic_id'. Defaulting 'zone_id'."); df['zone_id'] = 'UnknownZoneDirect'
        df['zone_id'] = df['zone_id'].astype(str).fillna('UnknownZoneFill')
        num_iot_cols = ['avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour']
        for col in num_iot_cols:
            if col not in df.columns: df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')
        str_iot_cols = ['clinic_id', 'room_name']; 
        for col in str_iot_cols:
            if col in df.columns: df[col] = df[col].astype(str).fillna('Unknown'); df.loc[df[col].isin(['', 'nan', 'None']), col] = 'Unknown'
            else: df[col] = 'Unknown'
        logger.info(f"Loaded IoT data ({len(df)} rows)."); return df
    except pd.errors.EmptyDataError: logger.warning(f"IoT file empty: {file_path}."); return def_empty_df
    except Exception as e: logger.error(f"Err loading IoT data: {e}", exc_info=True); st.error(f"Err loading IoT data: {e}"); return def_empty_df

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, hash_funcs={ gpd.GeoDataFrame: hash_geodataframe, pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None })
def enrich_zone_geodata_with_health_aggregates(zone_gdf_base, health_df_input, iot_df_input=None):
    default_enriched_cols = ['zone_id', 'name', 'geometry', 'population', 'avg_risk_score', 'active_tb_cases', 'active_malaria_cases', 'hiv_positive_cases', 'pneumonia_cases', 'anemia_cases', 'sti_cases', 'total_active_key_infections', 'prevalence_per_1000', 'facility_coverage_score', 'chw_visits_in_zone', 'avg_daily_steps_zone', 'avg_spo2_zone', 'zone_avg_co2', 'num_clinics', 'socio_economic_index']
    empty_enriched_gdf = gpd.GeoDataFrame(columns=default_enriched_cols, crs=app_config.DEFAULT_CRS)
    if zone_gdf_base is None or zone_gdf_base.empty: return empty_enriched_gdf
    enriched_gdf = zone_gdf_base.copy(); 
    if 'zone_id' not in enriched_gdf.columns: logger.error("CRIT: 'zone_id' missing in zone_gdf_base for enrichment."); return enriched_gdf 
    enriched_gdf['zone_id'] = enriched_gdf['zone_id'].astype(str)
    health_agg_cols_defaults = {'avg_risk_score': np.nan, 'active_tb_cases': 0, 'active_malaria_cases': 0, 'hiv_positive_cases': 0, 'pneumonia_cases': 0, 'anemia_cases': 0, 'sti_cases': 0, 'chw_visits_in_zone': 0, 'avg_daily_steps_zone': np.nan, 'avg_spo2_zone': np.nan, 'avg_skin_temp_zone': np.nan, 'total_falls_detected_zone': 0, 'hpv_screenings_done': 0}
    for col, default_val in health_agg_cols_defaults.items(): enriched_gdf[col] = default_val 
    if health_df_input is not None and not health_df_input.empty and 'zone_id' in health_df_input.columns:
        health_df = health_df_input.copy(); health_df['zone_id'] = health_df['zone_id'].astype(str)
        zone_health_summary_list = []
        if 'zone_id' in health_df and health_df['zone_id'].notna().any():
            for zone_val, group_df in health_df.groupby('zone_id'):
                summary = {'zone_id': zone_val}
                summary['avg_risk_score'] = group_df.get('ai_risk_score', pd.Series(dtype=float)).mean()
                if 'condition' not in group_df.columns: group_df['condition'] = pd.Series(dtype=str) # Ensure 'condition' exists for .isin/.str
                else: group_df['condition'] = group_df['condition'].astype(str) 
                
                summary['active_tb_cases'] = group_df[group_df['condition'].isin(['TB'])]['patient_id'].nunique()
                summary['active_malaria_cases'] = group_df[group_df['condition'].isin(['Malaria'])]['patient_id'].nunique()
                summary['hiv_positive_cases'] = group_df[group_df['condition'].isin(['HIV-Positive'])]['patient_id'].nunique()
                summary['pneumonia_cases'] = group_df[group_df['condition'].isin(['Pneumonia'])]['patient_id'].nunique()
                summary['anemia_cases'] = group_df[group_df['condition'].isin(['Anemia'])]['patient_id'].nunique()
                summary['sti_cases'] = group_df[group_df['condition'].str.startswith('STI-', na=False)]['patient_id'].nunique()
                summary['chw_visits_in_zone'] = pd.to_numeric(group_df.get('chw_visit'), errors='coerce').sum()
                summary['avg_daily_steps_zone'] = group_df.get('avg_daily_steps', pd.Series(dtype=float)).mean()
                summary['avg_spo2_zone'] = group_df.get('avg_spo2', pd.Series(dtype=float)).mean()
                summary['avg_skin_temp_zone'] = group_df.get('max_skin_temp_celsius', pd.Series(dtype=float)).mean() 
                summary['total_falls_detected_zone'] = pd.to_numeric(group_df.get('fall_detected_today'), errors='coerce').sum()
                summary['hpv_screenings_done'] = (group_df.get('test_type', pd.Series(dtype=str)) == 'PapSmear').sum()
                zone_health_summary_list.append(summary)
            if zone_health_summary_list: zone_health_summary = pd.DataFrame(zone_health_summary_list)
            else: zone_health_summary = pd.DataFrame(columns=['zone_id'] + list(health_agg_cols_defaults.keys()))
        else: zone_health_summary = pd.DataFrame(columns=['zone_id'] + list(health_agg_cols_defaults.keys()))
        for col_to_merge in zone_health_summary.columns:
            if col_to_merge != 'zone_id': enriched_gdf = enriched_gdf.drop(columns=[col_to_merge], errors='ignore') 
        enriched_gdf = enriched_gdf.merge(zone_health_summary, on='zone_id', how='left')
        for col, default_val in health_agg_cols_defaults.items():
            if col in enriched_gdf.columns: enriched_gdf[col] = enriched_gdf[col].fillna(default_val)
    iot_agg_cols_defaults = {'zone_avg_co2': np.nan, 'zone_max_co2': np.nan, 'zone_avg_temp': np.nan, 'zone_avg_pm25':np.nan, 'zone_avg_occupancy': np.nan}
    for col, default_val in iot_agg_cols_defaults.items(): enriched_gdf[col] = default_val
    if iot_df_input is not None and not iot_df_input.empty and 'zone_id' in iot_df_input.columns:
        iot_df = iot_df_input.copy(); iot_df['zone_id'] = iot_df['zone_id'].astype(str)
        latest_iot_readings_per_room = iot_df.sort_values('timestamp').drop_duplicates(subset=['zone_id', 'clinic_id', 'room_name'], keep='last')
        iot_zone_summary = latest_iot_readings_per_room.groupby('zone_id').agg(zone_avg_co2=('avg_co2_ppm', 'mean'), zone_max_co2=('max_co2_ppm', 'max'), zone_avg_temp=('avg_temp_celsius', 'mean'), zone_avg_pm25=('avg_pm25', 'mean'), zone_avg_occupancy=('waiting_room_occupancy', 'mean')).reset_index()
        for col_to_merge_iot in iot_zone_summary.columns:
            if col_to_merge_iot != 'zone_id': enriched_gdf = enriched_gdf.drop(columns=[col_to_merge_iot], errors='ignore')
        enriched_gdf = enriched_gdf.merge(iot_zone_summary, on='zone_id', how='left')
        for col, default_val in iot_agg_cols_defaults.items():
            if col in enriched_gdf.columns: enriched_gdf[col] = enriched_gdf[col].fillna(default_val)
    key_infection_cols_list = ['active_tb_cases', 'active_malaria_cases', 'hiv_positive_cases', 'pneumonia_cases', 'sti_cases']
    for col in key_infection_cols_list:
        if col not in enriched_gdf.columns: enriched_gdf[col] = 0
        else: enriched_gdf[col] = pd.to_numeric(enriched_gdf[col], errors='coerce').fillna(0)
    enriched_gdf['total_active_key_infections'] = enriched_gdf[key_infection_cols_list].sum(axis=1)
    enriched_gdf['population'] = pd.to_numeric(enriched_gdf.get('population', 0), errors='coerce').fillna(0)
    enriched_gdf['prevalence_per_1000'] = enriched_gdf.apply(lambda row: (row['total_active_key_infections'] / row['population']) * 1000 if row.get('population',0) > 0 else 0.0, axis=1).fillna(0.0)
    if 'avg_travel_time_clinic_min' in enriched_gdf.columns and 'num_clinics' in enriched_gdf.columns:
        enriched_gdf['avg_travel_time_clinic_min'] = pd.to_numeric(enriched_gdf['avg_travel_time_clinic_min'], errors='coerce').fillna(enriched_gdf['avg_travel_time_clinic_min'].max() if enriched_gdf['avg_travel_time_clinic_min'].notna().any() else 60) 
        enriched_gdf['num_clinics'] = pd.to_numeric(enriched_gdf['num_clinics'], errors='coerce').fillna(0)
        min_travel = enriched_gdf['avg_travel_time_clinic_min'].min(); max_travel = enriched_gdf['avg_travel_time_clinic_min'].max()
        if pd.notna(max_travel) and pd.notna(min_travel) and max_travel > min_travel: enriched_gdf['travel_score'] = 100 * (1 - (enriched_gdf['avg_travel_time_clinic_min'].fillna(max_travel) - min_travel) / (max_travel - min_travel))
        elif enriched_gdf['avg_travel_time_clinic_min'].notna().any(): enriched_gdf['travel_score'] = 50.0 
        else: enriched_gdf['travel_score'] = 0.0
        enriched_gdf['clinics_per_1k_pop'] = enriched_gdf.apply(lambda r: (r.get('num_clinics',0)/r.get('population',0))*1000 if r.get('population',0)>0 else 0, axis=1)
        min_density = enriched_gdf['clinics_per_1k_pop'].min(); max_density = enriched_gdf['clinics_per_1k_pop'].max()
        if pd.notna(max_density) and pd.notna(min_density) and max_density > min_density: enriched_gdf['clinic_density_score'] = 100 * (enriched_gdf['clinics_per_1k_pop'].fillna(min_density) - min_density) / (max_density - min_density)
        elif enriched_gdf['clinics_per_1k_pop'].notna().any(): enriched_gdf['clinic_density_score'] = 50.0
        else: enriched_gdf['clinic_density_score'] = 0.0
        enriched_gdf['facility_coverage_score'] = (enriched_gdf['travel_score'].fillna(0) * 0.5 + enriched_gdf['clinic_density_score'].fillna(0) * 0.5)
    else: enriched_gdf['facility_coverage_score'] = 0.0 
    final_numeric_cols_to_check_enrich = list(health_agg_cols_defaults.keys()) + list(iot_agg_cols_defaults.keys()) + ['total_active_key_infections', 'prevalence_per_1000', 'facility_coverage_score', 'population', 'num_clinics', 'socio_economic_index'] 
    for col in final_numeric_cols_to_check_enrich:
        if col in enriched_gdf.columns: enriched_gdf[col] = pd.to_numeric(enriched_gdf[col], errors='coerce').fillna(0.0)
        elif col not in ['geometry', 'name', 'zone_id']: enriched_gdf[col] = 0.0
    logger.info(f"Zone geodata successfully enriched. Resulting GDF shape: {enriched_gdf.shape}, Columns: {enriched_gdf.columns.tolist()}")
    return enriched_gdf

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_overall_kpis(df_health_records, date_filter_end=None):
    default_kpis_overall = {"total_patients": 0, "avg_patient_risk": 0.0, "active_tb_cases_current": 0, "malaria_rdt_positive_rate_period": 0.0, "hiv_newly_diagnosed_period": 0, "pending_critical_referrals_current": 0, "avg_test_turnaround_period": 0.0, "critical_supply_shortages_current":0, "anemia_prevalence_women_period": 0.0}
    if df_health_records is None or df_health_records.empty: return default_kpis_overall
    df = df_health_records.copy(); 
    if 'date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date']): df['date'] = pd.to_datetime(df.get('date'), errors='coerce') 
    if 'date' not in df.columns or df['date'].isnull().all(): return default_kpis_overall
    df.dropna(subset=['date'], inplace=True); 
    if df.empty: return default_kpis_overall
    current_data_snapshot_date = pd.to_datetime(date_filter_end if date_filter_end else df['date'].max()).normalize()
    df_upto_snapshot = df[df['date'] <= current_data_snapshot_date].copy()
    if df_upto_snapshot.empty: return default_kpis_overall
    latest_patient_records = df_upto_snapshot.sort_values('date').drop_duplicates(subset=['patient_id'], keep='last')
    total_patients = latest_patient_records['patient_id'].nunique(); avg_patient_risk = latest_patient_records.get('ai_risk_score', pd.Series(dtype=float)).mean()
    active_tb_cases_current = latest_patient_records[(latest_patient_records.get('condition', pd.Series(dtype=str)) == 'TB') & (latest_patient_records.get('referral_status', 'Unknown') != 'Completed')]['patient_id'].nunique()
    pending_critical_referrals_current = latest_patient_records[(latest_patient_records.get('referral_status', 'Unknown') == 'Pending') & (latest_patient_records.get('condition', pd.Series(dtype=str)).isin(app_config.KEY_CONDITIONS_FOR_TRENDS))]['patient_id'].nunique()
    latest_item_stock_levels = df_upto_snapshot.sort_values('date').drop_duplicates(subset=['item'], keep='last').copy() 
    latest_item_stock_levels['days_of_supply'] = latest_item_stock_levels.apply(lambda r: (r.get('stock_on_hand',0) / r.get('consumption_rate_per_day',0)) if pd.notna(r.get('consumption_rate_per_day')) and r.get('consumption_rate_per_day', 0) > 0 and pd.notna(r.get('stock_on_hand')) else np.nan, axis=1) 
    critical_supply_shortages_current = latest_item_stock_levels[(latest_item_stock_levels['days_of_supply'].notna()) & (latest_item_stock_levels['days_of_supply'] <= app_config.CRITICAL_SUPPLY_DAYS) & (latest_item_stock_levels.get('item', pd.Series(dtype=str)).str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False))]['item'].nunique()
    period_start_date = current_data_snapshot_date - pd.Timedelta(days=app_config.DEFAULT_DATE_RANGE_DAYS_TREND -1); df_period = df_upto_snapshot[df_upto_snapshot['date'] >= period_start_date].copy()
    malaria_rdt_positive_rate_period = 0.0; hiv_newly_diagnosed_period = 0; avg_test_turnaround_period = 0.0; anemia_prevalence_women_period = 0.0
    if not df_period.empty:
        malaria_tests_period = df_period[df_period.get('test_type', pd.Series(dtype=str)).isin(['RDT-Malaria', 'Microscopy-Malaria'])]; malaria_pos_period = malaria_tests_period[malaria_tests_period.get('test_result', pd.Series(dtype=str)) == 'Positive'].shape[0]; malaria_conclusive_period = malaria_tests_period[~malaria_tests_period.get('test_result', pd.Series(dtype=str)).isin(['Pending', 'N/A', 'Unknown'])].shape[0]
        if malaria_conclusive_period > 0: malaria_rdt_positive_rate_period = (malaria_pos_period / malaria_conclusive_period) * 100
        hiv_newly_diagnosed_period = df_period[(df_period.get('condition', pd.Series(dtype=str)) == 'HIV-Positive') & (df_period.get('test_result', pd.Series(dtype=str)) == 'Positive') & (df_period.get('test_type', pd.Series(dtype=str)).str.contains("HIV", case=False, na=False))]['patient_id'].nunique()
        completed_tests_period = df_period[df_period.get('test_turnaround_days', pd.Series(dtype=float)).notna() & (~df_period.get('test_result', pd.Series(dtype=str)).isin(['Pending', 'N/A', 'Unknown']))]; avg_test_turnaround_period = completed_tests_period.get('test_turnaround_days', pd.Series(dtype=float)).mean()
        women_tested_anemia = df_period[(df_period.get('gender','Unknown') == 'Female') & (pd.to_numeric(df_period.get('age',0), errors='coerce').fillna(0).between(15,49)) & (df_period.get('test_type', pd.Series(dtype=str)) == 'Hemoglobin Test') & (~df_period.get('test_result', pd.Series(dtype=str)).isin(['Pending', 'N/A', 'Unknown']))]; anemia_low_hb = women_tested_anemia[women_tested_anemia.get('test_result', pd.Series(dtype=str)) == 'Low'].shape[0]
        if not women_tested_anemia.empty: anemia_prevalence_women_period = (anemia_low_hb / women_tested_anemia.shape[0]) * 100
    return {"total_patients": total_patients, "avg_patient_risk": avg_patient_risk if pd.notna(avg_patient_risk) else 0.0, "active_tb_cases_current": active_tb_cases_current, "malaria_rdt_positive_rate_period": malaria_rdt_positive_rate_period, "hiv_newly_diagnosed_period": hiv_newly_diagnosed_period, "pending_critical_referrals_current": pending_critical_referrals_current, "avg_test_turnaround_period": avg_test_turnaround_period if pd.notna(avg_test_turnaround_period) else 0.0, "critical_supply_shortages_current": critical_supply_shortages_current, "anemia_prevalence_women_period": anemia_prevalence_women_period}

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_chw_summary(df_chw_day_view): 
    default_summary = {"visits_today": 0, "tb_contacts_to_trace_today": 0, "sti_symptomatic_referrals_today": 0, "avg_patient_risk_visited_today":0.0, "patients_low_spo2_visited_today": 0, "patients_fever_visited_today": 0, "avg_patient_steps_visited_today":0.0, "high_risk_followups_today":0}
    if df_chw_day_view is None or df_chw_day_view.empty: return default_summary
    df = df_chw_day_view.copy(); numeric_cols_chw = ['chw_visit', 'tb_contact_traced', 'ai_risk_score', 'avg_spo2', 'min_spo2_pct', 'max_skin_temp_celsius', 'avg_daily_steps', 'fall_detected_today']
    for col in numeric_cols_chw: df[col] = pd.to_numeric(df.get(col), errors='coerce')
    visits_today = int(df.get('chw_visit', pd.Series(dtype=float)).sum()); tb_contacts_to_trace_today = df[(df.get('condition', pd.Series(dtype=str)) == 'TB') & (df.get('tb_contact_traced', pd.Series(dtype=float)).fillna(1) == 0)].shape[0]; sti_symptomatic_referrals_today = df[(df.get('condition', pd.Series(dtype=str)).str.startswith("STI-", na=False)) & (df.get('referral_status', 'Unknown') == 'Pending')].shape[0]
    patients_visited_df = df[df.get('chw_visit', pd.Series(dtype=float)) == 1]; avg_patient_risk_visited_today = patients_visited_df.get('ai_risk_score', pd.Series(dtype=float)).mean()
    patients_low_spo2_visited_today = 0; patients_fever_visited_today = 0; avg_patient_steps_visited_today = 0.0
    if not patients_visited_df.empty:
        spo2_col_to_use = 'min_spo2_pct' if 'min_spo2_pct' in patients_visited_df.columns and patients_visited_df['min_spo2_pct'].notna().any() else 'avg_spo2'
        if spo2_col_to_use in patients_visited_df.columns and patients_visited_df[spo2_col_to_use].notna().any() : patients_low_spo2_visited_today = patients_visited_df[patients_visited_df[spo2_col_to_use] < app_config.SPO2_LOW_THRESHOLD_PCT].shape[0]
        if 'max_skin_temp_celsius' in patients_visited_df.columns and patients_visited_df['max_skin_temp_celsius'].notna().any(): patients_fever_visited_today = patients_visited_df[patients_visited_df['max_skin_temp_celsius'] >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C].shape[0]
        if 'avg_daily_steps' in patients_visited_df.columns and patients_visited_df['avg_daily_steps'].notna().any(): avg_patient_steps_visited_today = patients_visited_df['avg_daily_steps'].mean()
    high_risk_followups_today = df[ (df.get('ai_risk_score', 0) >= app_config.RISK_THRESHOLDS['chw_alert_high']) | (df.get('min_spo2_pct', 100) < app_config.SPO2_CRITICAL_THRESHOLD_PCT) | (df.get('fall_detected_today', 0) > 0) ]['patient_id'].nunique() 
    return {"visits_today": visits_today, "tb_contacts_to_trace_today": tb_contacts_to_trace_today, "sti_symptomatic_referrals_today": sti_symptomatic_referrals_today, "avg_patient_risk_visited_today": avg_patient_risk_visited_today if pd.notna(avg_patient_risk_visited_today) else 0.0, "patients_low_spo2_visited_today": patients_low_spo2_visited_today, "patients_fever_visited_today": patients_fever_visited_today, "avg_patient_steps_visited_today": avg_patient_steps_visited_today if pd.notna(avg_patient_steps_visited_today) else 0.0, "high_risk_followups_today": high_risk_followups_today}

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_patient_alerts_for_chw(df_chw_day_view, risk_threshold_moderate=None): 
    if df_chw_day_view is None or df_chw_day_view.empty: return pd.DataFrame()
    df = df_chw_day_view.copy(); alert_logic_cols = {'ai_risk_score': 'float', 'min_spo2_pct': 'float', 'max_skin_temp_celsius': 'float', 'fall_detected_today': 'int', 'tb_contact_traced': 'int', 'condition': 'str', 'referral_status': 'str', 'patient_id': 'str', 'zone_id': 'str', 'date': 'datetime'}
    for col, dtype in alert_logic_cols.items():
        if col not in df.columns: df[col] = np.nan if dtype == 'float' else (0 if dtype == 'int' else ('Unknown' if dtype=='str' else pd.NaT))
        if dtype == 'float': df[col] = pd.to_numeric(df[col], errors='coerce')
        elif dtype == 'int': df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        elif dtype == 'datetime' and col in df.columns and df[col] is not None : df[col] = pd.to_datetime(df.get(col), errors='coerce') 
        else: df[col] = df[col].astype(str).fillna('Unknown')
    risk_mod = risk_threshold_moderate if risk_threshold_moderate is not None else app_config.RISK_THRESHOLDS['chw_alert_moderate']; risk_high = app_config.RISK_THRESHOLDS['chw_alert_high']; spo2_crit = app_config.SPO2_CRITICAL_THRESHOLD_PCT; spo2_low = app_config.SPO2_LOW_THRESHOLD_PCT; fever_thresh = app_config.SKIN_TEMP_FEVER_THRESHOLD_C
    s_false = pd.Series([False]*len(df), index=df.index) 
    cond_fall = df.get('fall_detected_today', s_false.copy()) > 0; cond_critical_spo2 = df.get('min_spo2_pct', s_false.copy()) < spo2_crit; cond_fever = df.get('max_skin_temp_celsius', s_false.copy()) >= fever_thresh; cond_high_risk_score = df.get('ai_risk_score', s_false.copy()) >= risk_high
    cond_low_spo2_mod = (df.get('min_spo2_pct', s_false.copy()) >= spo2_crit) & (df.get('min_spo2_pct', s_false.copy()) < spo2_low); cond_mod_risk_score = (df.get('ai_risk_score', s_false.copy()) >= risk_mod) & (df.get('ai_risk_score', s_false.copy()) < risk_high)
    cond_tb_trace_needed = (df.get('condition', s_false.copy()) == 'TB') & (df.get('tb_contact_traced', s_false.copy()) == 0); cond_key_referral_pending = (df.get('referral_status', s_false.copy()) == 'Pending') & (df.get('condition', s_false.copy()).isin(app_config.KEY_CONDITIONS_FOR_TRENDS))
    alertable_conditions_mask = (cond_fall | cond_critical_spo2 | cond_fever | cond_high_risk_score | cond_low_spo2_mod | cond_mod_risk_score | cond_tb_trace_needed | cond_key_referral_pending)
    alerts_df = df[alertable_conditions_mask].copy()
    if alerts_df.empty: return pd.DataFrame()
    def determine_chw_alert_reason_and_priority(row):
        reasons = []; priority = 0 
        if row.get('fall_detected_today', 0) > 0: reasons.append(f"Fall ({int(row['fall_detected_today'])})"); priority += 100
        if pd.notna(row.get('min_spo2_pct')):
            if row['min_spo2_pct'] < spo2_crit: reasons.append(f"Critical SpO2 ({row['min_spo2_pct']:.0f}%)"); priority += 90
            elif row['min_spo2_pct'] < spo2_low: reasons.append(f"Low SpO2 ({row['min_spo2_pct']:.0f}%)"); priority += 50
        if pd.notna(row.get('max_skin_temp_celsius')) and row['max_skin_temp_celsius'] >= fever_thresh: reasons.append(f"Fever ({row['max_skin_temp_celsius']:.1f}°C)"); priority += 70
        if pd.notna(row.get('ai_risk_score')):
            if row['ai_risk_score'] >= risk_high: reasons.append(f"High Risk ({row['ai_risk_score']:.0f})"); priority += 80
            elif row['ai_risk_score'] >= risk_mod: reasons.append(f"Mod. Risk ({row['ai_risk_score']:.0f})"); priority += 40
        if row.get('condition') == 'TB' and row.get('tb_contact_traced', 1) == 0: reasons.append("TB Contact Trace"); priority += 60
        if row.get('referral_status') == 'Pending' and row.get('condition') in app_config.KEY_CONDITIONS_FOR_TRENDS: reasons.append(f"Referral ({row.get('condition')})"); priority += 30
        return "; ".join(reasons) if reasons else "Review Case", priority
    alerts_df[['alert_reason', 'priority_score']] = alerts_df.apply(lambda row: pd.Series(determine_chw_alert_reason_and_priority(row)), axis=1)
    alerts_df.sort_values(by=['priority_score', 'ai_risk_score'], ascending=[False, False], inplace=True); alerts_df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True)
    output_cols_chw_alerts = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'alert_reason', 'referral_status', 'min_spo2_pct', 'max_skin_temp_celsius', 'fall_detected_today', 'priority_score', 'date']
    return alerts_df[[col for col in output_cols_chw_alerts if col in alerts_df.columns]]

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_clinic_summary(df_clinic_period_view): 
    default_summary = {
        "tb_sputum_positivity": 0.0, "malaria_positivity": 0.0, 
        "sti_critical_tests_pending": 0, "hiv_tests_conclusive_period": 0, 
        "key_drug_stockouts": 0, "avg_test_turnaround_all_tests":0.0, 
        "hpv_screening_coverage_proxy":0.0, "avg_patient_risk_clinic":0.0,
        "test_summary_details": {}, "sample_rejection_rate":0.0,
        "overall_perc_met_tat": 0.0, "total_pending_critical_tests": 0
    }
    if df_clinic_period_view is None or df_clinic_period_view.empty: 
        logger.debug("get_clinic_summary received empty/None DataFrame. Returning default summary.")
        return default_summary
        
    df = df_clinic_period_view.copy(); 
    str_cols = ['test_type', 'test_result', 'item', 'condition', 'patient_id', 'sample_status'] # Added sample_status
    num_cols = ['stock_on_hand', 'consumption_rate_per_day', 'test_turnaround_days', 'ai_risk_score']
    for col in str_cols: df[col] = df.get(col, pd.Series(dtype='str')).astype(str).fillna('Unknown').replace(['', 'nan', 'None', 'N/A'], 'Unknown') # Ensure N/A also becomes Unknown for consistency
    for col in num_cols: df[col] = pd.to_numeric(df.get(col), errors='coerce')

    test_summary_details = {} # To store detailed stats per test group
    
    # Use KEY_TEST_TYPES_FOR_ANALYSIS from app_config for dynamic processing
    for group_key, group_config in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items():
        test_types_in_group = group_config.get("types", [group_key]) # If "types" not specified, assume group_key is the test_type
        if isinstance(test_types_in_group, str): test_types_in_group = [test_types_in_group] # Ensure it's a list
            
        current_tests_df = df[df.get('test_type', pd.Series(dtype=str)).isin(test_types_in_group)]
        group_name_display = group_config.get("display_name", group_key.replace("_", " "))

        group_stats = {"total_done_in_period": 0, "conclusive_count": 0, "positive_count": 0, "positive_rate": 0.0, "pending_count": 0, "avg_tat_days": 0.0, "perc_met_tat_target": 0.0, "rejected_count": 0}
        
        if not current_tests_df.empty:
            # Use unique patient_id for counts where it makes sense (positivity, conclusive)
            positive_patients = current_tests_df[current_tests_df['test_result'] == 'Positive']['patient_id'].nunique()
            
            conclusive_mask = ~current_tests_df['test_result'].isin(['Pending', 'Unknown', 'Rejected Sample'])
            conclusive_tests_df = current_tests_df[conclusive_mask]
            conclusive_patients_count = conclusive_tests_df['patient_id'].nunique()
            
            pending_patients = current_tests_df[current_tests_df['test_result'] == 'Pending']['patient_id'].nunique()
            rejected_count = current_tests_df[current_tests_df['test_result'] == 'Rejected Sample'].shape[0] # Count of rejected samples for this test type

            tat_valid_tests_df = conclusive_tests_df[conclusive_tests_df['test_turnaround_days'].notna()]
            avg_tat_for_group = tat_valid_tests_df['test_turnaround_days'].mean()
            # In utils/core_data_processing.py, inside get_clinic_summary, before the problematic line
logger.debug(f"Accessing app_config.TARGET_TEST_TURNAROUND_DAYS. Type of app_config: {type(app_config)}")
logger.debug(f"Does app_config have TARGET_TEST_TURNAROUND_DAYS? {'TARGET_TEST_TURNAROUND_DAYS' in dir(app_config)}")
if 'TARGET_TEST_TURNAROUND_DAYS' in dir(app_config):
    logger.debug(f"Value of app_config.TARGET_TEST_TURNAROUND_DAYS: {app_config.TARGET_TEST_TURNAROUND_DAYS}")
else:
    logger.error("CRITICAL DEBUG: TARGET_TEST_TURNAROUND_DAYS is NOT an attribute of the imported app_config module!")
    # You could even list all attributes to see what IS there:
    # logger.debug(f"Available attributes in app_config: {dir(app_config)}")

target_tat_for_group = config_details.get("target_tat_days", app_config.TARGET_TEST_TURNAROUND_DAYS) # Line 390
            target_tat = group_config.get("target_tat_days", app_config.TARGET_TEST_TURNAROUND_DAYS) # Use specific or global TAT
            met_target_tests_count = tat_valid_tests_df[tat_valid_tests_df['test_turnaround_days'] <= target_tat].shape[0]
            
            group_stats["total_done_in_period"] = current_tests_df.shape[0] # All tests of this type recorded
            group_stats["conclusive_count"]= conclusive_patients_count 
            group_stats["positive_count"] = positive_patients
            group_stats["positive_rate"]= (positive_patients / conclusive_patients_count) * 100 if conclusive_patients_count > 0 else 0.0
            group_stats["pending_count"]= pending_patients
            group_stats["avg_tat_days"]= avg_tat_for_group if pd.notna(avg_tat_for_group) else 0.0
            group_stats["perc_met_tat_target"]= (met_target_tests_count / tat_valid_tests_df.shape[0]) * 100 if not tat_valid_tests_df.empty else 0.0
            group_stats["rejected_count"] = rejected_count
        
        test_summary_details[group_name_display] = group_stats # Use display name as key
    
    # --- Overall KPIs derived from detailed stats or calculated broadly ---
    tb_sputum_positivity = test_summary_details.get("TB Sputum/GeneXpert", {}).get("positive_rate", 0.0) # Match a potential display name
    if not tb_sputum_positivity and "TB_Tests" in test_summary_details: tb_sputum_positivity = test_summary_details["TB_Tests"].get("positive_rate", 0.0)

    malaria_positivity = test_summary_details.get("Malaria RDT/Microscopy", {}).get("positive_rate", 0.0)
    if not malaria_positivity and "Malaria_Tests" in test_summary_details: malaria_positivity = test_summary_details["Malaria_Tests"].get("positive_rate", 0.0)

    hiv_tests_conclusive_period = test_summary_details.get("HIV Rapid/VL", {}).get("conclusive_count",0)
    if not hiv_tests_conclusive_period and "HIV_Tests" in test_summary_details: hiv_tests_conclusive_period = test_summary_details["HIV_Tests"].get("conclusive_count",0)

    sti_critical_tests_pending = sum(test_summary_details.get(test_group, {}).get("pending_count", 0) 
                                    for test_group, cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() 
                                    if cfg.get("disease_group") == "STI" and cfg.get("critical"))


    df['days_of_supply'] = df.apply(lambda r: (r.get('stock_on_hand',0)/r.get('consumption_rate_per_day',0)) if pd.notna(r.get('stock_on_hand')) and pd.notna(r.get('consumption_rate_per_day')) and r.get('consumption_rate_per_day',0)>0 else np.nan, axis=1)
    latest_stock_in_period = df.sort_values('date').drop_duplicates(subset=['item'], keep='last')
    key_drug_stockouts = latest_stock_in_period[(latest_stock_in_period['days_of_supply'].notna()) & (latest_stock_in_period['days_of_supply'] <= app_config.CRITICAL_SUPPLY_DAYS) & (latest_stock_in_period.get('item', pd.Series(dtype=str)).str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False))]['item'].nunique()
    
    all_conclusive_w_tat_overall = df[(~df.get('test_result',pd.Series(dtype=str)).isin(['Pending', 'Unknown', 'Rejected Sample'])) & (df.get('test_turnaround_days',pd.Series(dtype=float)).notna())]
    avg_test_turnaround_all_tests = all_conclusive_w_tat_overall.get('test_turnaround_days',pd.Series(dtype=float)).mean()
    all_critical_tests_with_tat = all_conclusive_w_tat_overall[all_conclusive_w_tat_overall['test_type'].isin(app_config.CRITICAL_TESTS_LIST)]
    
    overall_met_target_count = 0
    if not all_critical_tests_with_tat.empty:
        for test_type in all_critical_tests_with_tat['test_type'].unique():
            target_for_type = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(test_type, {}).get("target_tat_days", app_config.TARGET_TEST_TURNAROUND_DAYS)
            overall_met_target_count += all_critical_tests_with_tat[
                (all_critical_tests_with_tat['test_type'] == test_type) &
                (all_critical_tests_with_tat['test_turnaround_days'] <= target_for_type)
            ].shape[0]
    overall_perc_met_tat = (overall_met_target_count / all_critical_tests_with_tat.shape[0]) * 100 if not all_critical_tests_with_tat.empty else 0.0

    
    hpv_screenings_done = test_summary_details.get("PapSmear",{}).get("total_done_in_period", 0) # Or however PapSmear group is named
    total_unique_patients = df['patient_id'].nunique(); 
    hpv_screening_coverage_proxy = (hpv_screenings_done / total_unique_patients) * 100 if total_unique_patients > 0 else 0.0
    avg_patient_risk_clinic = df.drop_duplicates(subset=['patient_id']).get('ai_risk_score',pd.Series(dtype=float)).mean()

    total_tests_with_status = df[df['sample_status'] != 'Unknown'].shape[0] # Denominator for rejection rate
    total_rejected_samples = df[df['sample_status'] == 'Rejected'].shape[0]
    sample_rejection_rate = (total_rejected_samples / total_tests_with_status) * 100 if total_tests_with_status > 0 else 0.0

    return {"tb_sputum_positivity": tb_sputum_positivity, "malaria_positivity": malaria_positivity, 
            "sti_critical_tests_pending": sti_critical_tests_pending, "hiv_tests_conclusive_period": hiv_tests_conclusive_period, 
            "key_drug_stockouts": key_drug_stockouts, 
            "avg_test_turnaround_all_tests": avg_test_turnaround_all if pd.notna(avg_test_turnaround_all) else 0.0, 
            "hpv_screening_coverage_proxy": hpv_screening_coverage_proxy, 
            "avg_patient_risk_clinic": avg_patient_risk_clinic if pd.notna(avg_patient_risk_clinic) else 0.0,
            "test_summary_details": test_summary_details, 
            "sample_rejection_rate": sample_rejection_rate,
            "overall_perc_met_tat": overall_perc_met_tat,
            "total_pending_critical_tests": sum(details.get("pending_count", 0) for test, details in test_summary_details.items() if test in app_config.CRITICAL_TESTS_LIST or any(t in test for t in app_config.CRITICAL_TESTS_LIST) ) # Approximate
           }

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_clinic_environmental_summary(df_iot_clinic_period_view): 
    default_summary = {"avg_co2_overall": 0.0, "rooms_co2_alert_latest": 0, "avg_pm25_overall": 0.0, "rooms_pm25_alert_latest": 0, "avg_occupancy_overall":0.0, "high_occupancy_alert_latest": False, "avg_sanitizer_use_hr_overall":0.0, "rooms_noise_alert_latest":0}
    if df_iot_clinic_period_view is None or df_iot_clinic_period_view.empty: return default_summary
    df = df_iot_clinic_period_view.copy(); iot_numeric_cols = ['avg_co2_ppm', 'avg_pm25', 'waiting_room_occupancy', 'sanitizer_dispenses_per_hour', 'avg_noise_db', 'max_co2_ppm']
    for col in iot_numeric_cols: df[col] = pd.to_numeric(df.get(col), errors='coerce')
    avg_co2_overall = df.get('avg_co2_ppm', pd.Series(dtype=float)).mean(); avg_pm25_overall = df.get('avg_pm25', pd.Series(dtype=float)).mean(); avg_occupancy_overall = df.get('waiting_room_occupancy', pd.Series(dtype=float)).mean(); avg_sanitizer_use_hr_overall = df.get('sanitizer_dispenses_per_hour', pd.Series(dtype=float)).mean()
    rooms_co2_alert_latest = 0; rooms_pm25_alert_latest = 0; high_occupancy_alert_latest = False; rooms_noise_alert_latest = 0
    if 'timestamp' in df.columns and 'clinic_id' in df.columns and 'room_name' in df.columns and not df.empty : 
        latest_readings_per_room = df.sort_values('timestamp').drop_duplicates(subset=['clinic_id', 'room_name'], keep='last')
        if not latest_readings_per_room.empty:
            if 'avg_co2_ppm' in latest_readings_per_room.columns and latest_readings_per_room['avg_co2_ppm'].notna().any(): rooms_co2_alert_latest = latest_readings_per_room[latest_readings_per_room['avg_co2_ppm'] > app_config.CO2_LEVEL_ALERT_PPM].shape[0]
            if 'avg_pm25' in latest_readings_per_room.columns and latest_readings_per_room['avg_pm25'].notna().any(): rooms_pm25_alert_latest = latest_readings_per_room[latest_readings_per_room['avg_pm25'] > app_config.PM25_ALERT_UGM3].shape[0]
            if 'waiting_room_occupancy' in latest_readings_per_room.columns and latest_readings_per_room['waiting_room_occupancy'].notna().any(): high_occupancy_alert_latest = (latest_readings_per_room['waiting_room_occupancy'] > app_config.TARGET_WAITING_ROOM_OCCUPANCY).any()
            if 'avg_noise_db' in latest_readings_per_room.columns and latest_readings_per_room['avg_noise_db'].notna().any(): rooms_noise_alert_latest = latest_readings_per_room[latest_readings_per_room['avg_noise_db'] > app_config.NOISE_LEVEL_ALERT_DB].shape[0]
    return {"avg_co2_overall": avg_co2_overall if pd.notna(avg_co2_overall) else 0.0, "rooms_co2_alert_latest": rooms_co2_alert_latest, "avg_pm25_overall": avg_pm25_overall if pd.notna(avg_pm25_overall) else 0.0, "rooms_pm25_alert_latest": rooms_pm25_alert_latest, "avg_occupancy_overall": avg_occupancy_overall if pd.notna(avg_occupancy_overall) else 0.0, "high_occupancy_alert_latest": bool(high_occupancy_alert_latest), "avg_sanitizer_use_hr_overall": avg_sanitizer_use_hr_overall if pd.notna(avg_sanitizer_use_hr_overall) else 0.0, "rooms_noise_alert_latest": rooms_noise_alert_latest}

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_patient_alerts_for_clinic(df_clinic_period_view, risk_threshold_moderate=None):
    if df_clinic_period_view is None or df_clinic_period_view.empty: return pd.DataFrame()
    df = df_clinic_period_view.copy()
    if 'date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date']): df['date'] = pd.to_datetime(df.get('date'), errors='coerce')
    if 'date' not in df.columns or df['date'].isnull().all(): return pd.DataFrame()
    df.dropna(subset=['date'], inplace=True); 
    if df.empty: return pd.DataFrame()
    df['test_date'] = pd.to_datetime(df.get('test_date'), errors='coerce'); df['ai_risk_score'] = pd.to_numeric(df.get('ai_risk_score'), errors='coerce')
    str_cols_clinic_alert = ['test_result', 'condition', 'referral_status', 'patient_id', 'zone_id', 'test_type']
    for col in str_cols_clinic_alert: df[col] = df.get(col, pd.Series(dtype='str')).astype(str).fillna('Unknown').replace(['', 'nan', 'None'], 'Unknown')
    df['hiv_viral_load'] = pd.to_numeric(df.get('hiv_viral_load'), errors='coerce')
    risk_mod_clinic = risk_threshold_moderate if risk_threshold_moderate is not None else app_config.RISK_THRESHOLDS['moderate']; risk_high_clinic = app_config.RISK_THRESHOLDS['high']
    current_snapshot_date_clinic = df['date'].max().normalize(); recent_positive_lookback_days = 7; overdue_test_lookback_days = 10 
    latest_records_in_period = df.sort_values('date').drop_duplicates(subset=['patient_id', 'condition'], keep='last').copy()
    s_false_latest = pd.Series([False]*len(latest_records_in_period), index=latest_records_in_period.index) 
    latest_records_in_period.loc[:, 'cond_clinic_high_risk'] = latest_records_in_period.get('ai_risk_score', s_false_latest.copy()) >= risk_high_clinic
    latest_records_in_period.loc[:, 'cond_clinic_recent_critical_positive'] = ((latest_records_in_period.get('test_result', s_false_latest.copy()) == 'Positive') & (latest_records_in_period.get('condition', s_false_latest.copy()).isin(app_config.KEY_CONDITIONS_FOR_TRENDS)) & (latest_records_in_period.get('test_date', pd.Series(pd.NaT, index=latest_records_in_period.index)).notna()) & (latest_records_in_period.get('test_date', pd.Series(pd.NaT, index=latest_records_in_period.index)) >= (current_snapshot_date_clinic - pd.Timedelta(days=recent_positive_lookback_days))))
    latest_records_in_period.loc[:, 'cond_clinic_overdue_critical_test'] = ((latest_records_in_period.get('test_result', s_false_latest.copy()) == 'Pending') & (latest_records_in_period.get('test_type', s_false_latest.copy()).isin(app_config.CRITICAL_TESTS_LIST)) & (latest_records_in_period.get('test_date', pd.Series(pd.NaT, index=latest_records_in_period.index)).notna()) & (latest_records_in_period.get('test_date', pd.Series(pd.NaT, index=latest_records_in_period.index)) < (current_snapshot_date_clinic - pd.Timedelta(days=overdue_test_lookback_days)))) # Used CRITICAL_TESTS_LIST
    latest_records_in_period.loc[:, 'cond_clinic_hiv_high_vl'] = ((latest_records_in_period.get('condition', s_false_latest.copy()) == 'HIV-Positive') & (latest_records_in_period.get('hiv_viral_load', pd.Series(dtype=float, index=latest_records_in_period.index)).notna()) & (latest_records_in_period.get('hiv_viral_load', pd.Series(dtype=float, index=latest_records_in_period.index)) > 1000) ) 
    alert_mask_clinic = (latest_records_in_period.get('cond_clinic_high_risk', s_false_latest.copy()) | latest_records_in_period.get('cond_clinic_recent_critical_positive', s_false_latest.copy()) | latest_records_in_period.get('cond_clinic_overdue_critical_test', s_false_latest.copy()) | latest_records_in_period.get('cond_clinic_hiv_high_vl', s_false_latest.copy())) 
    alerts_df_clinic = latest_records_in_period[alert_mask_clinic].copy()
    if alerts_df_clinic.empty: return pd.DataFrame()
    def determine_clinic_alert_reason_and_priority(row): 
        reasons = []; priority_score = 0
        if row.get('cond_clinic_hiv_high_vl', False): reasons.append(f"High HIV VL ({row.get('hiv_viral_load',0):.0f})"); priority_score += 100 
        if row.get('cond_clinic_high_risk', False): reasons.append(f"High Risk ({row.get('ai_risk_score',0):.0f})"); priority_score += row.get('ai_risk_score', 0)
        if row.get('cond_clinic_recent_critical_positive', False): reasons.append(f"Recent Crit. Positive ({row.get('condition')})"); priority_score += 70
        if row.get('cond_clinic_overdue_critical_test', False): days_pending = (current_snapshot_date_clinic - row.get('test_date', current_snapshot_date_clinic)).days; reasons.append(f"Overdue Crit. Test ({row.get('test_type')}, {days_pending}d)"); priority_score += 60
        return "; ".join(reasons) if reasons else "Review Case", priority_score
    alerts_df_clinic[['alert_reason', 'priority_score']] = alerts_df_clinic.apply(lambda row: pd.Series(determine_clinic_alert_reason_and_priority(row)), axis=1)
    alerts_df_clinic.sort_values(by=['priority_score', 'date'], ascending=[False, False], inplace=True)
    output_cols_clinic_alerts = ['patient_id', 'zone_id', 'condition', 'ai_risk_score', 'test_result', 'test_type', 'referral_status', 'alert_reason', 'hiv_viral_load', 'priority_score', 'date']
    return alerts_df_clinic[[col for col in output_cols_clinic_alerts if col in alerts_df_clinic.columns]]

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_trend_data(df_input, value_col, date_col='date', period='D', agg_func='mean'):
    default_index_name = date_col if df_input is not None and date_col in df_input.columns else 'date'
    empty_series = pd.Series(dtype='float64', name=value_col).rename_axis(default_index_name)
    if df_input is None or df_input.empty: return empty_series
    df = df_input.copy() 
    if date_col not in df.columns: return empty_series
    if not pd.api.types.is_datetime64_ns_dtype(df[date_col]): df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col], inplace=True); 
    if df.empty: return empty_series
    if value_col not in df.columns:
        if agg_func == 'count': logger.debug(f"Trend count: Using row count as '{value_col}' not found.")
        else: logger.warning(f"Trend data: Value col '{value_col}' missing for agg '{agg_func}'."); return empty_series
    elif agg_func in ['mean', 'sum', 'median', 'std']:
        if not pd.api.types.is_numeric_dtype(df[value_col]): df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df.dropna(subset=[value_col], inplace=True) 
    if df.empty: return empty_series
    try:
        df_indexed = df.set_index(date_col).sort_index() 
        if value_col in df_indexed.columns: trend_series = df_indexed[value_col].resample(period).agg(agg_func)
        elif agg_func == 'count': trend_series = df_indexed.resample(period).size(); trend_series.name = 'count'
        else: return empty_series
        fill_val = 0 if agg_func in ['count', 'sum', 'nunique'] else np.nan 
        return trend_series.fillna(fill_val)
    except Exception as e: logger.error(f"Error in get_trend_data for '{value_col}': {e}", exc_info=True); return empty_series

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_parquet() if not df.empty and isinstance(df, pd.DataFrame) else None})
def get_supply_forecast_data(df_health_records_full, forecast_days_out=21): 
    cols = ['date', 'item', 'forecast_days', 'lower_ci', 'upper_ci', 'current_stock', 'consumption_rate', 'estimated_stockout_date']
    empty_forecast_df = pd.DataFrame(columns=cols)
    if df_health_records_full is None or df_health_records_full.empty: return empty_forecast_df
    df = df_health_records_full.copy(); required_cols = ['date', 'item', 'stock_on_hand', 'consumption_rate_per_day']
    if not all(col in df.columns for col in required_cols): return empty_forecast_df
    df['date'] = pd.to_datetime(df['date'], errors='coerce'); df['item'] = df['item'].astype(str).fillna('Unknown').replace(['', 'nan', 'None'], 'Unknown')
    df['stock_on_hand'] = pd.to_numeric(df['stock_on_hand'], errors='coerce'); df['consumption_rate_per_day'] = pd.to_numeric(df['consumption_rate_per_day'], errors='coerce')
    df.dropna(subset=['date', 'item', 'stock_on_hand'], inplace=True); df = df[~df['item'].isin(['Unknown'])]; df = df[df['stock_on_hand'] >= 0] 
    if df.empty: return empty_forecast_df
    latest_supplies = df.sort_values('date').drop_duplicates(subset=['item'], keep='last').copy()
    default_min_consumption = 0.1
    latest_supplies['consumption_rate_per_day'] = latest_supplies.apply(lambda row: row['consumption_rate_per_day'] if pd.notna(row['consumption_rate_per_day']) and row['consumption_rate_per_day'] > 0 else (default_min_consumption if row['stock_on_hand'] > 0 else 0), axis=1)
    forecast_list = []; today_for_forecast = pd.Timestamp('today').normalize() 
    for _, row in latest_supplies.iterrows():
        item_name = row['item']; current_stock_amount = row['stock_on_hand']; consumption_rate = row['consumption_rate_per_day']
        days_supply_now = current_stock_amount / consumption_rate if consumption_rate > 0 else (np.inf if current_stock_amount > 0 else 0)
        estimated_stockout_date = today_for_forecast + pd.Timedelta(days=days_supply_now) if consumption_rate > 0 and current_stock_amount > 0 and days_supply_now != np.inf else pd.NaT
        forecast_list.append({'date': today_for_forecast, 'item': item_name, 'forecast_days': days_supply_now, 'lower_ci': days_supply_now, 'upper_ci': days_supply_now, 'current_stock': current_stock_amount, 'consumption_rate': consumption_rate, 'estimated_stockout_date': estimated_stockout_date})
        if consumption_rate <= 0: 
            for i in range(1, forecast_days_out + 1): forecast_list.append({'date': today_for_forecast + pd.Timedelta(days=i), 'item': item_name, 'forecast_days': days_supply_now, 'lower_ci': days_supply_now, 'upper_ci': days_supply_now, 'current_stock': current_stock_amount, 'consumption_rate': consumption_rate, 'estimated_stockout_date': estimated_stockout_date})
            continue 
        for i in range(1, forecast_days_out + 1):
            forecast_date_iter = today_for_forecast + pd.Timedelta(days=i); forecasted_stock_level = max(0, current_stock_amount - (consumption_rate * i)); days_of_supply_fc = forecasted_stock_level / consumption_rate if consumption_rate > 0 else (np.inf if forecasted_stock_level > 0 else 0)
            consumption_ci_factor = 0.25; lower_cons = max(0.01, consumption_rate * (1 - consumption_ci_factor)); upper_cons = consumption_rate * (1 + consumption_ci_factor)
            stock_at_upper_cons = max(0, current_stock_amount - (upper_cons * i)); lower_ci_days_fc = stock_at_upper_cons / upper_cons if upper_cons > 0 else (np.inf if stock_at_upper_cons > 0 else 0)
            stock_at_lower_cons = max(0, current_stock_amount - (lower_cons * i)); upper_ci_days_fc = stock_at_lower_cons / lower_cons if lower_cons > 0 else (np.inf if stock_at_lower_cons > 0 else 0)
            forecast_list.append({'date': forecast_date_iter, 'item': item_name, 'forecast_days': days_of_supply_fc, 'lower_ci': lower_ci_days_fc, 'upper_ci': upper_ci_days_fc, 'current_stock': forecasted_stock_level, 'consumption_rate': consumption_rate, 'estimated_stockout_date': estimated_stockout_date})
            if forecasted_stock_level == 0 and days_of_supply_fc == 0: break 
    return pd.DataFrame(forecast_list) if forecast_list else empty_forecast_df

@st.cache_data(hash_funcs={gpd.GeoDataFrame: hash_geodataframe})
def get_district_summary_kpis(enriched_zone_gdf):
    default_kpis = {"avg_population_risk": 0.0, "overall_facility_coverage": 0.0, "zones_high_risk_count": 0, "district_tb_burden_total":0, "district_malaria_burden_total":0, "avg_clinic_co2_district":0.0, "population_weighted_avg_steps": 0.0, "population_weighted_avg_spo2": 0.0, "key_infection_prevalence_district_per_1000": 0.0}
    if enriched_zone_gdf is None or enriched_zone_gdf.empty: return default_kpis
    gdf = enriched_zone_gdf.copy(); kpi_calc_cols = {'population': 0.0, 'avg_risk_score': 0.0, 'facility_coverage_score': 0.0, 'active_tb_cases': 0.0, 'active_malaria_cases': 0.0, 'zone_avg_co2': 0.0, 'avg_daily_steps_zone': 0.0, 'avg_spo2_zone': 0.0, 'total_active_key_infections':0.0}
    for col, default_val in kpi_calc_cols.items():
        if col not in gdf.columns: gdf[col] = default_val
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce').fillna(default_val)
    total_district_population = gdf['population'].sum()
    if total_district_population == 0: district_tb_total = gdf['active_tb_cases'].sum(); district_mal_total = gdf['active_malaria_cases'].sum(); return {**default_kpis, "district_tb_burden_total": int(district_tb_total), "district_malaria_burden_total": int(district_mal_total)}
    avg_pop_risk = (gdf['avg_risk_score'] * gdf['population']).sum() / total_district_population; overall_facility_coverage = (gdf['facility_coverage_score'] * gdf['population']).sum() / total_district_population
    zones_high_risk_count = gdf[gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk']].shape[0]; district_tb_burden_total = gdf['active_tb_cases'].sum(); district_malaria_burden_total = gdf['active_malaria_cases'].sum()
    avg_clinic_co2_district_series = gdf['zone_avg_co2'][gdf['zone_avg_co2'] > 0] 
    avg_clinic_co2_district = avg_clinic_co2_district_series.mean() if not avg_clinic_co2_district_series.empty and avg_clinic_co2_district_series.notna().any() else 0.0
    pop_weighted_avg_steps = (gdf['avg_daily_steps_zone'] * gdf['population']).sum() / total_district_population; pop_weighted_avg_spo2 = (gdf['avg_spo2_zone'] * gdf['population']).sum() / total_district_population
    total_key_infections_district = gdf['total_active_key_infections'].sum(); key_infection_prevalence_district_per_1000 = (total_key_infections_district / total_district_population) * 1000 if total_district_population > 0 else 0.0
    return {"avg_population_risk": avg_pop_risk if pd.notna(avg_pop_risk) else 0.0, "overall_facility_coverage": overall_facility_coverage if pd.notna(overall_facility_coverage) else 0.0, "zones_high_risk_count": int(zones_high_risk_count), "district_tb_burden_total": int(district_tb_burden_total), "district_malaria_burden_total": int(district_malaria_burden_total), "avg_clinic_co2_district": avg_clinic_co2_district if pd.notna(avg_clinic_co2_district) else 0.0, "population_weighted_avg_steps": pop_weighted_avg_steps if pd.notna(pop_weighted_avg_steps) else 0.0, "population_weighted_avg_spo2": pop_weighted_avg_spo2 if pd.notna(pop_weighted_avg_spo2) else 0.0, "key_infection_prevalence_district_per_1000": key_infection_prevalence_district_per_1000}
