# test/utils/core_data_processing.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import logging
from config import app_config
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Helper Functions ---
def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

def _convert_to_numeric(series: pd.Series, default_value: Any = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').fillna(default_value)

def hash_geodataframe(gdf: gpd.GeoDataFrame) -> Optional[str]:
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame): return None
    try:
        # Ensure geometry column exists and is valid before attempting to access/hash it
        geom_col_name = gdf.geometry.name
        if geom_col_name not in gdf.columns or gdf[geom_col_name].is_empty.all():
             non_geom_cols = gdf.columns.tolist() # Hash all if no valid geometry
        else:
            non_geom_cols = gdf.drop(columns=[geom_col_name], errors='ignore').columns.tolist()

        df_hash = pd.util.hash_pandas_object(gdf[non_geom_cols], index=True).sum()
        geom_hash = pd.util.hash_array(gdf.geometry.to_wkt().values).sum() if geom_col_name in gdf.columns and not gdf[geom_col_name].is_empty.all() and hasattr(gdf.geometry, 'to_wkt') else 0
        return f"{df_hash}-{geom_hash}"
    except Exception as e:
        logger.warning(f"Could not hash GeoDataFrame: {e}"); return None

def _robust_merge_agg(
    left_df: pd.DataFrame, right_df: pd.DataFrame, target_col_name: str,
    on_col: str = 'zone_id', default_fill_value: Any = 0.0
) -> pd.DataFrame:
    """Robustly merges an aggregated right_df into left_df."""
    if target_col_name not in left_df.columns:
        left_df[target_col_name] = default_fill_value # Ensure target column exists in left_df

    if right_df.empty or on_col not in right_df.columns: # If right_df is empty or doesn't have the 'on' col
        # Ensure target_col_name is filled if it was newly added
        left_df[target_col_name] = left_df.get(target_col_name, default_fill_value)
        left_df[target_col_name].fillna(default_fill_value, inplace=True)
        return left_df

    value_col_in_right_df_candidates = [col for col in right_df.columns if col != on_col]
    if not value_col_in_right_df_candidates: # If right_df only has the 'on' col
        logger.debug(f"_robust_merge_agg: Right DF for target '{target_col_name}' has no value column besides '{on_col}'.")
        left_df[target_col_name] = left_df.get(target_col_name, default_fill_value)
        left_df[target_col_name].fillna(default_fill_value, inplace=True)
        return left_df
    
    value_col_in_right_df = value_col_in_right_df_candidates[0] # Take the first value column
    # Create a more unique temporary column name to avoid collisions
    temp_agg_col_name = f"__{target_col_name}_temp_agg_{np.random.randint(1000,9999)}__"
    
    # Make a copy for renaming to avoid modifying the original right_df
    right_df_renamed = right_df[[on_col, value_col_in_right_df]].copy()
    right_df_renamed.rename(columns={value_col_in_right_df: temp_agg_col_name}, inplace=True)
    
    merged_df = left_df.merge(right_df_renamed, on=on_col, how='left')
    
    if temp_agg_col_name in merged_df.columns:
        # Update target_col_name: if temp_agg is notna, use it, else use existing target_col_name
        merged_df[target_col_name] = np.where(
            merged_df[temp_agg_col_name].notna(),
            merged_df[temp_agg_col_name],
            merged_df[target_col_name] # Keep existing value if merge brought NaN
        )
        merged_df.drop(columns=[temp_agg_col_name], inplace=True)
    
    # Final fillna on the target column itself
    merged_df[target_col_name].fillna(default_fill_value, inplace=True)
    return merged_df

# --- Data Loading and Basic Cleaning Functions ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading health records...")
def load_health_records(file_path: str = None) -> pd.DataFrame:
    file_path = file_path or app_config.HEALTH_RECORDS_CSV
    logger.info(f"Attempting to load health records from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Health records file not found: {file_path}")
        st.error(f"🚨 **Critical Data Error:** Health records file '{os.path.basename(file_path)}' not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df = _clean_column_names(df)
        logger.info(f"Successfully loaded {len(df)} records from {file_path}.")
        date_cols = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date', 'referral_date', 'referral_outcome_date']
        for col in date_cols:
            if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
        numeric_cols = ['test_turnaround_days', 'quantity_dispensed', 'item_stock_agg_zone', 'consumption_rate_per_day', 'ai_risk_score', 'ai_followup_priority_score', 'vital_signs_bp_systolic', 'vital_signs_bp_diastolic', 'vital_signs_temperature_celsius', 'min_spo2_pct', 'max_skin_temp_celsius', 'avg_spo2', 'avg_daily_steps', 'resting_heart_rate', 'avg_hrv', 'avg_sleep_duration_hrs', 'sleep_score_pct', 'stress_level_score', 'fall_detected_today', 'age', 'chw_visit', 'tb_contact_traced', 'patient_latitude', 'patient_longitude']
        for col in numeric_cols:
            if col in df.columns: df[col] = _convert_to_numeric(df[col])
        if 'hiv_viral_load_copies_ml' in df.columns: df['hiv_viral_load_copies_ml'] = pd.to_numeric(df['hiv_viral_load_copies_ml'], errors='coerce')
        string_like_cols = ['encounter_id', 'patient_id', 'encounter_type', 'condition', 'diagnosis_code_icd10', 'test_type', 'test_result', 'item', 'zone_id', 'clinic_id', 'physician_id', 'notes', 'patient_reported_symptoms', 'gender', 'screening_hpv_status', 'key_chronic_conditions_summary', 'medication_adherence_self_report', 'referral_status', 'referral_reason', 'referred_to_facility_id', 'referral_outcome', 'sample_status', 'rejection_reason']
        for col in string_like_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str).str.strip()
                df.loc[df[col].isin(['', 'nan', 'None', 'N/A', '#N/A', 'np.nan']), col] = "Unknown"
            else: df[col] = "Unknown"
        required_cols = ['patient_id', 'encounter_date', 'condition', 'test_type', 'test_result', 'zone_id', 'ai_risk_score']
        for r_col in required_cols:
            if r_col not in df.columns:
                logger.warning(f"Required column '{r_col}' not found in health records. Adding as empty.")
                if 'date' in r_col: df[r_col] = pd.Series(dtype='datetime64[ns]')
                elif 'score' in r_col or 'risk' in r_col : df[r_col] = pd.Series(dtype='float64')
                else: df[r_col] = pd.Series(dtype='object').fillna("Unknown")
        logger.info("Health records cleaning complete.")
        return df
    except Exception as e:
        logger.error(f"Error loading/processing health records from {file_path}: {e}", exc_info=True)
        st.error(f"Failed to load/process health records: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading IoT environmental data...")
def load_iot_clinic_environment_data(file_path: str = None) -> pd.DataFrame:
    file_path = file_path or app_config.IOT_CLINIC_ENVIRONMENT_CSV
    logger.info(f"Attempting to load IoT data from: {file_path}")
    if not os.path.exists(file_path):
        logger.warning(f"IoT data file not found: {file_path}")
        st.info(f"ℹ️ IoT data file '{os.path.basename(file_path)}' not found. Environmental monitoring may be limited.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df = _clean_column_names(df)
        logger.info(f"Successfully loaded {len(df)} IoT records.")
        if 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else: logger.error("IoT data missing 'timestamp' column."); return pd.DataFrame()
        numeric_iot_cols = ['avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour']
        for col in numeric_iot_cols:
            if col in df.columns: df[col] = _convert_to_numeric(df[col])
            else: df[col] = np.nan # Ensure numeric columns exist if missing from CSV
        string_iot_cols = ['clinic_id', 'room_name', 'zone_id']
        for col in string_iot_cols:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip() if col in df.columns else "Unknown"
        logger.info("IoT data cleaning complete.")
        return df
    except Exception as e:
        logger.error(f"Error loading IoT data from {file_path}: {e}", exc_info=True)
        st.warning(f"Could not load/process IoT data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, hash_funcs={gpd.GeoDataFrame: hash_geodataframe}, show_spinner="Loading zone geographic and attribute data...")
def load_zone_data(attributes_path: str = None, geometries_path: str = None) -> Optional[gpd.GeoDataFrame]:
    attributes_path = attributes_path or app_config.ZONE_ATTRIBUTES_CSV
    geometries_path = geometries_path or app_config.ZONE_GEOMETRIES_GEOJSON
    logger.info(f"Loading zone attributes from {attributes_path} and geometries from {geometries_path}")
    if not os.path.exists(attributes_path) or not os.path.exists(geometries_path):
        err_msg_list = []
        if not os.path.exists(attributes_path): err_msg_list.append(f"Zone attributes file '{os.path.basename(attributes_path)}' not found.")
        if not os.path.exists(geometries_path): err_msg_list.append(f"Zone geometries file '{os.path.basename(geometries_path)}' not found.")
        full_err_msg = " ".join(err_msg_list)
        logger.error(full_err_msg); st.error(f"🚨 **Critical GIS Data Error:** {full_err_msg}"); return None
    try:
        zone_attributes_df = pd.read_csv(attributes_path); zone_attributes_df = _clean_column_names(zone_attributes_df)
        zone_geometries_gdf = gpd.read_file(geometries_path); zone_geometries_gdf = _clean_column_names(zone_geometries_gdf)
        if 'zone_id' not in zone_attributes_df.columns or 'zone_id' not in zone_geometries_gdf.columns:
            logger.error("Missing 'zone_id' in zone attributes or geometries."); st.error("🚨 Key 'zone_id' missing in GIS files."); return None
        zone_attributes_df['zone_id'] = zone_attributes_df['zone_id'].astype(str).str.strip()
        zone_geometries_gdf['zone_id'] = zone_geometries_gdf['zone_id'].astype(str).str.strip()
        if 'zone_display_name' in zone_attributes_df.columns: zone_attributes_df.rename(columns={'zone_display_name': 'name'}, inplace=True)
        elif 'name' not in zone_attributes_df.columns and 'zone_id' in zone_attributes_df.columns : zone_attributes_df['name'] = zone_attributes_df['zone_id']
        
        merged_gdf = zone_geometries_gdf.merge(zone_attributes_df, on="zone_id", how="left", suffixes=('_geom', ''))
        for col_attr in zone_attributes_df.columns:
            if f"{col_attr}_geom" in merged_gdf.columns and col_attr in merged_gdf.columns and col_attr != 'zone_id':
                 merged_gdf[col_attr] = merged_gdf[col_attr].fillna(merged_gdf[f"{col_attr}_geom"])
                 merged_gdf.drop(columns=[f"{col_attr}_geom"], inplace=True, errors='ignore')
        if 'geometry_geom' in merged_gdf.columns and 'geometry' not in merged_gdf.columns : merged_gdf.rename(columns={'geometry_geom':'geometry'}, inplace=True)
        if 'geometry' in merged_gdf.columns and merged_gdf.geometry.name != 'geometry': merged_gdf = merged_gdf.set_geometry('geometry', inplace=False) # Ensure 'geometry' is the active geometry column
        
        if merged_gdf.crs is None: merged_gdf = merged_gdf.set_crs(app_config.DEFAULT_CRS, allow_override=True)
        elif merged_gdf.crs.to_string().upper() != app_config.DEFAULT_CRS.upper(): merged_gdf = merged_gdf.to_crs(app_config.DEFAULT_CRS)
        
        required_zone_cols = ['zone_id', 'name', 'population', 'geometry', 'num_clinics', 'socio_economic_index', 'avg_travel_time_clinic_min']
        for rz_col in required_zone_cols:
            if rz_col not in merged_gdf.columns:
                logger.warning(f"Required zone column '{rz_col}' missing after merge. Adding default.")
                if rz_col == 'population' or rz_col == 'num_clinics': merged_gdf[rz_col] = 0.0
                elif rz_col == 'socio_economic_index': merged_gdf[rz_col] = 0.5
                elif rz_col == 'avg_travel_time_clinic_min': merged_gdf[rz_col] = 30.0 # Default travel time
                elif rz_col == 'name': merged_gdf[rz_col] = "Zone " + merged_gdf['zone_id'].astype(str) if 'zone_id' in merged_gdf else "Unknown Zone"
                elif rz_col != 'geometry': merged_gdf[rz_col] = "Unknown" # String cols default
        for num_col in ['population', 'socio_economic_index', 'num_clinics', 'avg_travel_time_clinic_min']:
            if num_col in merged_gdf.columns: merged_gdf[num_col] = _convert_to_numeric(merged_gdf[num_col], 0 if num_col in ['population', 'num_clinics'] else (0.5 if num_col=='socio_economic_index' else 30.0))
        logger.info(f"Successfully loaded and merged zone data: {len(merged_gdf)} zones.")
        return merged_gdf
    except Exception as e:
        logger.error(f"Error loading/merging zone data: {e}", exc_info=True); st.error(f"Error with zone GIS data: {e}"); return None

# --- Data Enrichment and Aggregation Functions ---
def enrich_zone_geodata_with_health_aggregates(
    zone_gdf: gpd.GeoDataFrame, health_df: pd.DataFrame, iot_df: Optional[pd.DataFrame] = None
) -> gpd.GeoDataFrame:
    if zone_gdf is None or zone_gdf.empty or 'zone_id' not in zone_gdf.columns:
        return zone_gdf if zone_gdf is not None else gpd.GeoDataFrame(columns=['zone_id', 'name', 'geometry', 'population'], crs=app_config.DEFAULT_CRS)
    enriched = zone_gdf.copy()
    if 'population' not in enriched.columns: enriched['population'] = 0.0
    enriched['population'] = _convert_to_numeric(enriched['population'], 0.0)

    agg_cols_to_initialize = ['total_population_health_data', 'avg_risk_score', 'total_patient_encounters', 'active_tb_cases', 'active_malaria_cases', 'hiv_positive_cases', 'pneumonia_cases', 'total_referrals_made', 'successful_referrals', 'avg_test_turnaround_critical', 'perc_critical_tests_tat_met', 'prevalence_per_1000', 'total_active_key_infections', 'avg_daily_steps_zone', 'zone_avg_co2', 'facility_coverage_score']
    for col in agg_cols_to_initialize: enriched[col] = 0.0

    if health_df is not None and not health_df.empty and 'zone_id' in health_df.columns:
        health_df['zone_id'] = health_df['zone_id'].astype(str).str.strip()
        health_df_for_agg = health_df.copy()
        
        enriched = _robust_merge_agg(enriched, health_df_for_agg.groupby('zone_id')['patient_id'].nunique().reset_index(), 'total_population_health_data')
        enriched = _robust_merge_agg(enriched, health_df_for_agg.groupby('zone_id')['ai_risk_score'].mean().reset_index(), 'avg_risk_score')
        enriched = _robust_merge_agg(enriched, health_df_for_agg.groupby('zone_id')['encounter_id'].nunique().reset_index(), 'total_patient_encounters')
        enriched = _robust_merge_agg(enriched, health_df_for_agg[health_df_for_agg['condition'].str.contains("TB", case=False, na=False)].groupby('zone_id')['patient_id'].nunique().reset_index(), 'active_tb_cases')
        enriched = _robust_merge_agg(enriched, health_df_for_agg[health_df_for_agg['condition'].str.contains("Malaria", case=False, na=False)].groupby('zone_id')['patient_id'].nunique().reset_index(), 'active_malaria_cases')
        enriched = _robust_merge_agg(enriched, health_df_for_agg[health_df_for_agg['condition'].str.contains("HIV-Positive", case=False, na=False)].groupby('zone_id')['patient_id'].nunique().reset_index(), 'hiv_positive_cases')
        enriched = _robust_merge_agg(enriched, health_df_for_agg[health_df_for_agg['condition'].str.contains("Pneumonia", case=False, na=False)].groupby('zone_id')['patient_id'].nunique().reset_index(), 'pneumonia_cases')
        
        key_conditions_burden = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia']
        total_key_inf_agg = health_df_for_agg[health_df_for_agg['condition'].isin(key_conditions_burden)].groupby('zone_id')['patient_id'].nunique().reset_index()
        enriched = _robust_merge_agg(enriched, total_key_inf_agg, 'total_active_key_infections')

        if 'referral_status' in health_df_for_agg.columns:
            enriched = _robust_merge_agg(enriched, health_df_for_agg[health_df_for_agg['referral_status'].notna() & (~health_df_for_agg['referral_status'].isin(['N/A', 'Unknown']))].groupby('zone_id')['encounter_id'].nunique().reset_index(), 'total_referrals_made')
            if 'referral_outcome' in health_df_for_agg.columns:
                suc_out = ['Completed', 'Service Provided', 'Attended Consult', 'Attended Followup', 'Attended']
                enriched = _robust_merge_agg(enriched, health_df_for_agg[health_df_for_agg['referral_outcome'].isin(suc_out)].groupby('zone_id')['encounter_id'].nunique().reset_index(), 'successful_referrals')
        
        crit_test_keys = [k for k, v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get("critical", False)]
        if crit_test_keys: # Ensure using actual test_type keys from CSV, not display names for filtering
            tat_df = health_df_for_agg[(health_df_for_agg['test_type'].isin(crit_test_keys)) & (health_df_for_agg['test_turnaround_days'].notna()) & (~health_df_for_agg['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown','Indeterminate']))].copy()
            if not tat_df.empty:
                enriched = _robust_merge_agg(enriched, tat_df.groupby('zone_id')['test_turnaround_days'].mean().reset_index(), 'avg_test_turnaround_critical')
                def _check_tat_met_enrich(row_tat_enrich):
                    cfg = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(row_tat_enrich['test_type']) # Match row['test_type'] against config keys
                    return row_tat_enrich['test_turnaround_days'] <= (cfg['target_tat_days'] if cfg and 'target_tat_days' in cfg else app_config.TARGET_TEST_TURNAROUND_DAYS)
                tat_df['tat_met_flag'] = tat_df.apply(_check_tat_met_enrich, axis=1)
                perc_met_agg = tat_df.groupby('zone_id')['tat_met_flag'].mean().reset_index()
                # Correctly rename the aggregated column before robust_merge_agg expects a generic name
                perc_met_agg.rename(columns={'tat_met_flag': 'value_col_for_merge'}, inplace=True) 
                enriched = _robust_merge_agg(enriched, perc_met_agg, 'perc_critical_tests_tat_met')
                # Convert back to percentage if it's not already done by _robust_merge_agg logic handling
                enriched['perc_critical_tests_tat_met'] = enriched['perc_critical_tests_tat_met'] * 100

        if 'avg_daily_steps' in health_df_for_agg.columns:
            enriched = _robust_merge_agg(enriched, health_df_for_agg.groupby('zone_id')['avg_daily_steps'].mean().reset_index(), 'avg_daily_steps_zone')

    if iot_df is not None and not iot_df.empty and 'zone_id' in iot_df.columns and 'avg_co2_ppm' in iot_df.columns:
        iot_df['zone_id'] = iot_df['zone_id'].astype(str).str.strip()
        enriched = _robust_merge_agg(enriched, iot_df.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(), 'zone_avg_co2')
    
    if 'total_active_key_infections' in enriched.columns and 'population' in enriched.columns:
         enriched['prevalence_per_1000'] = enriched.apply(lambda r: (r['total_active_key_infections']/r['population'])*1000 if pd.notna(r['population']) and r['population']>0 and pd.notna(r['total_active_key_infections']) else 0.0, axis=1).fillna(0.0)
    if 'num_clinics' in enriched.columns and 'population' in enriched.columns: # Assuming 'num_clinics' is loaded into enriched GDF
        enriched['facility_coverage_score'] = enriched.apply(lambda r: (r['num_clinics']/r['population'])*10000*5 if pd.notna(r['population']) and r['population']>0 and pd.notna(r['num_clinics']) else 0.0, axis=1).fillna(0.0).clip(0,100)
    elif 'facility_coverage_score' not in enriched.columns: enriched['facility_coverage_score'] = 0.0
    
    for col in agg_cols_to_initialize:
        if col in enriched.columns:
            if not pd.api.types.is_numeric_dtype(enriched[col]): enriched[col] = pd.to_numeric(enriched[col], errors='coerce')
            enriched[col].fillna(0.0, inplace=True)
        else: enriched[col] = 0.0
    logger.info("Zone GeoDataFrame enrichment complete.")
    return enriched

# --- KPI Calculation Functions ---
def get_overall_kpis(health_df: pd.DataFrame, date_filter_start: Optional[str] = None, date_filter_end: Optional[str] = None) -> Dict[str, Any]:
    kpis = { "total_patients": 0, "avg_patient_risk": 0.0, "active_tb_cases_current": 0, "malaria_rdt_positive_rate_period": 0.0, "hiv_rapid_positive_rate_period": 0.0, "key_supply_stockout_alerts": 0 }
    if health_df is None or health_df.empty: return kpis
    df = health_df.copy()
    if 'encounter_date' not in df.columns or df['encounter_date'].isnull().all(): return kpis
    df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
    df.dropna(subset=['encounter_date'], inplace=True) # Crucial after coercion

    if date_filter_start: df = df[df['encounter_date'] >= pd.to_datetime(date_filter_start, errors='coerce')]
    if date_filter_end: df = df[df['encounter_date'] <= pd.to_datetime(date_filter_end, errors='coerce')]
    df.dropna(subset=['encounter_date'], inplace=True)
    if df.empty: return kpis
    
    kpis["total_patients"] = df['patient_id'].nunique()
    kpis["avg_patient_risk"] = df['ai_risk_score'].mean() if 'ai_risk_score' in df and df['ai_risk_score'].notna().any() else 0.0
    kpis["active_tb_cases_current"] = df[df['condition'].str.contains("TB", case=False, na=False)]['patient_id'].nunique() if 'condition' in df.columns else 0
    
    mal_rdt_key = next((k for k,v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get('display_name', '').lower() == 'malaria rdt'), "RDT-Malaria") # Match on display_name, fallback key
    malaria_rdt_df = df[(df['test_type'] == app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_rdt_key,{}).get("display_name")) & (~df['test_result'].isin(["Pending", "Rejected Sample", "Unknown"]))]
    if not malaria_rdt_df.empty: kpis["malaria_rdt_positive_rate_period"] = (malaria_rdt_df[malaria_rdt_df['test_result'] == 'Positive'].shape[0] / len(malaria_rdt_df)) * 100 if len(malaria_rdt_df) > 0 else 0.0

    hiv_rapid_key = next((k for k,v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get('display_name', '').lower() == 'hiv rapid test'), "HIV-Rapid")
    hiv_rapid_df = df[(df['test_type'] == app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(hiv_rapid_key,{}).get("display_name")) & (~df['test_result'].isin(["Pending", "Rejected Sample", "Unknown"]))]
    if not hiv_rapid_df.empty: kpis["hiv_rapid_positive_rate_period"] = (hiv_rapid_df[hiv_rapid_df['test_result'] == 'Positive'].shape[0] / len(hiv_rapid_df)) * 100 if len(hiv_rapid_df) > 0 else 0.0
    
    if 'item' in df.columns and 'item_stock_agg_zone' in df.columns and 'consumption_rate_per_day' in df.columns:
        supply_df = df.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last')
        supply_df['days_of_supply'] = supply_df['item_stock_agg_zone'] / (supply_df['consumption_rate_per_day'].replace(0, np.nan)) # Avoid DivByZero
        supply_df.dropna(subset=['days_of_supply'], inplace=True)
        kpis['key_supply_stockout_alerts'] = supply_df[supply_df['days_of_supply'] < app_config.CRITICAL_SUPPLY_DAYS]['item'].nunique()
    return kpis

def get_chw_summary(health_df_daily: pd.DataFrame) -> Dict[str, Any]:
    summary = { "visits_today": 0, "tb_contacts_to_trace_today": 0, "sti_symptomatic_referrals_today": 0, "avg_patient_risk_visited_today": np.nan, "high_risk_followups_today": 0, "patients_low_spo2_visited_today": 0, "patients_fever_visited_today": 0, "avg_patient_steps_visited_today": np.nan, "patients_fall_detected_today": 0 }
    if health_df_daily is None or health_df_daily.empty: return summary
    chw_enc_df = health_df_daily.copy()
    is_chw_encounter = False
    if 'chw_visit' in chw_enc_df.columns and chw_enc_df['chw_visit'].dtype in [np.int64, np.float64] and chw_enc_df['chw_visit'].sum() > 0 :
        chw_enc_df = chw_enc_df[chw_enc_df['chw_visit'] == 1]; is_chw_encounter = True
    elif 'encounter_type' in chw_enc_df.columns and chw_enc_df['encounter_type'].str.contains("CHW", case=False, na=False).any():
        chw_enc_df = chw_enc_df[chw_enc_df['encounter_type'].str.contains("CHW", case=False, na=False)]; is_chw_encounter = True
    if not is_chw_encounter and not chw_enc_df.empty : # if no explicit chw flag, assume all in health_df_daily is chw relevant if this func is called for chw dash
         pass # use all of chw_enc_df
    elif chw_enc_df.empty: return summary

    summary["visits_today"] = chw_enc_df['patient_id'].nunique()
    if all(c in chw_enc_df.columns for c in ['condition', 'referral_reason', 'referral_status']):
        summary["tb_contacts_to_trace_today"] = chw_enc_df[(chw_enc_df['condition'] == 'TB') & (chw_enc_df['referral_reason'].str.contains("Contact Tracing|Investigation", case=False, na=False)) & (chw_enc_df['referral_status'] == 'Pending')]['patient_id'].nunique()
        summary["sti_symptomatic_referrals_today"] = chw_enc_df[(chw_enc_df['condition'].str.contains("STI", case=False, na=False)) & (chw_enc_df.get('patient_reported_symptoms', pd.Series(dtype=str)) != "Unknown") & (chw_enc_df['referral_status'] == 'Pending')]['patient_id'].nunique()
    if 'ai_risk_score' in chw_enc_df.columns and chw_enc_df['ai_risk_score'].notna().any():
        summary["avg_patient_risk_visited_today"] = chw_enc_df['ai_risk_score'].mean()
        summary["high_risk_followups_today"] = chw_enc_df[chw_enc_df['ai_risk_score'] >= app_config.RISK_THRESHOLDS.get('high', 75)]['patient_id'].nunique()
    if 'min_spo2_pct' in chw_enc_df.columns: summary["patients_low_spo2_visited_today"] = chw_enc_df[chw_enc_df['min_spo2_pct'] < app_config.SPO2_LOW_THRESHOLD_PCT]['patient_id'].nunique()
    temp_col = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in chw_enc_df.columns and chw_enc_df['vital_signs_temperature_celsius'].notna().any() else ('max_skin_temp_celsius' if 'max_skin_temp_celsius' in chw_enc_df.columns else None)
    if temp_col and temp_col in chw_enc_df.columns: summary["patients_fever_visited_today"] = chw_enc_df[chw_enc_df[temp_col] >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C]['patient_id'].nunique()
    if 'avg_daily_steps' in chw_enc_df.columns and chw_enc_df['avg_daily_steps'].notna().any(): summary["avg_patient_steps_visited_today"] = chw_enc_df['avg_daily_steps'].mean()
    if 'fall_detected_today' in chw_enc_df.columns: summary["patients_fall_detected_today"] = chw_enc_df[chw_enc_df['fall_detected_today'] > 0]['patient_id'].nunique()
    return summary

def get_patient_alerts_for_chw(health_df_daily: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_THRESHOLDS['chw_alert_moderate'], risk_threshold_high: int = app_config.RISK_THRESHOLDS['chw_alert_high']) -> pd.DataFrame:
    if health_df_daily is None or health_df_daily.empty: return pd.DataFrame()
    alerts = []; df_alerts = health_df_daily.copy()
    alert_cols_chw = ['patient_id', 'ai_risk_score', 'ai_followup_priority_score', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius', 'condition', 'referral_status', 'fall_detected_today', 'encounter_date'] # Ensure all used cols
    for col in alert_cols_chw:
        if col not in df_alerts.columns: df_alerts[col] = np.nan if pd.api.types.is_numeric_dtype(df_alerts.get(col)) else ("Unknown" if col != 'encounter_date' else pd.NaT)

    if df_alerts['ai_followup_priority_score'].notna().any():
        for _, r in df_alerts[df_alerts['ai_followup_priority_score'] >= 80].iterrows(): alerts.append({**r.to_dict(), 'alert_reason': "High AI Follow-up Priority", 'priority_score': r['ai_followup_priority_score']})
    if df_alerts['min_spo2_pct'].notna().any():
        for _, r in df_alerts[df_alerts['min_spo2_pct'] < app_config.SPO2_CRITICAL_THRESHOLD_PCT].iterrows(): alerts.append({**r.to_dict(), 'alert_reason': f"Critical SpO2 ({r['min_spo2_pct']}%)", 'priority_score': 90 + (app_config.SPO2_CRITICAL_THRESHOLD_PCT - r['min_spo2_pct'])})
    temp_c = 'vital_signs_temperature_celsius' if df_alerts['vital_signs_temperature_celsius'].notna().any() else 'max_skin_temp_celsius'
    if df_alerts[temp_c].notna().any():
        for _, r in df_alerts[df_alerts[temp_c] >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C + 1.0].iterrows(): alerts.append({**r.to_dict(), 'alert_reason': f"High Fever ({r[temp_c]}°C)", 'priority_score': 85 + (r[temp_c] - (app_config.SKIN_TEMP_FEVER_THRESHOLD_C + 1.0))})
    if df_alerts['fall_detected_today'].notna().any():
        for _, r in df_alerts[df_alerts['fall_detected_today'] > 0].iterrows(): alerts.append({**r.to_dict(), 'alert_reason': "Fall Detected", 'priority_score': 88})
    if df_alerts['ai_risk_score'].notna().any():
        for _, r in df_alerts[df_alerts['ai_risk_score'] >= risk_threshold_high].iterrows(): alerts.append({**r.to_dict(), 'alert_reason': "High General AI Risk", 'priority_score': r['ai_risk_score']})
    if 'condition' in df_alerts.columns and 'referral_status' in df_alerts.columns : # ensure columns exist
        for _, r in df_alerts[(df_alerts['condition'].isin(app_config.KEY_CONDITIONS_FOR_TRENDS[:4])) & (df_alerts['referral_status'] == 'Pending')].iterrows(): alerts.append({**r.to_dict(), 'alert_reason': f"Pending Referral for {r['condition']}", 'priority_score': 70})
    if df_alerts['ai_risk_score'].notna().any():
        for _, r in df_alerts[(df_alerts['ai_risk_score'] >= risk_threshold_moderate) & (df_alerts['ai_risk_score'] < risk_threshold_high)].iterrows(): alerts.append({**r.to_dict(), 'alert_reason': "Moderate AI Risk", 'priority_score': r['ai_risk_score']})
            
    if not alerts: return pd.DataFrame()
    alert_df = pd.DataFrame(alerts)
    if 'encounter_date' in alert_df.columns:
        alert_df['encounter_date_obj_for_dedup'] = pd.to_datetime(alert_df['encounter_date']).dt.date
        alert_df.drop_duplicates(subset=['patient_id', 'alert_reason', 'encounter_date_obj_for_dedup'], inplace=True)
        alert_df.drop(columns=['encounter_date_obj_for_dedup'], inplace=True, errors='ignore')
    else: # Fallback if encounter_date was somehow missing despite checks
         alert_df.drop_duplicates(subset=['patient_id', 'alert_reason'], inplace=True)

    alert_df['priority_score'] = alert_df['priority_score'].fillna(0).astype(int)
    sort_col = 'encounter_date' if 'encounter_date' in alert_df.columns and alert_df['encounter_date'].notna().all() else 'priority_score' # Avoid sorting by NaT
    return alert_df.sort_values(by=['priority_score', sort_col], ascending=[False, False])

def get_clinic_summary(health_df_period: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = { "overall_avg_test_turnaround": 0.0, "overall_perc_met_tat": 0.0, "total_pending_critical_tests": 0, "sample_rejection_rate": 0.0, "key_drug_stockouts": 0, "test_summary_details": {} }
    if health_df_period is None or health_df_period.empty: return summary
    df = health_df_period.copy()
    test_cols_req = ['test_type', 'test_result', 'test_turnaround_days', 'sample_status', 'encounter_date']
    for col in test_cols_req:
        if col not in df.columns: df[col] = np.nan if col=='test_turnaround_days' else ("Unknown" if col!='encounter_date' else pd.NaT)
    df['test_turnaround_days'] = _convert_to_numeric(df['test_turnaround_days']) # Ensure numeric

    conclusive_df = df[~df['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown', 'N/A', 'nan','Indeterminate']) & df['test_turnaround_days'].notna()].copy() # Use copy for .loc
    all_proc_samples_df = df[~df['sample_status'].isin(['Pending', 'Unknown', 'N/A', 'nan'])].copy()
    if not conclusive_df.empty: summary["overall_avg_test_turnaround"] = conclusive_df['test_turnaround_days'].mean()
    
    crit_test_cfgs = {k: v for k,v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get("critical")}
    crit_test_disp_names = [v['display_name'] for v in crit_test_cfgs.values()]
    crit_conclusive = conclusive_df[conclusive_df['test_type'].isin(crit_test_disp_names)].copy()
    if not crit_conclusive.empty:
        def _check_tat_met_overall_clinic(r): # Distinct name
            # Map display name back to original key to get specific TAT target
            orig_key = next((k for k,cfg_val in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if cfg_val.get('display_name') == r['test_type']), None)
            target_tat = app_config.KEY_TEST_TYPES_FOR_ANALYSIS[orig_key]['target_tat_days'] if orig_key and 'target_tat_days' in app_config.KEY_TEST_TYPES_FOR_ANALYSIS[orig_key] else app_config.TARGET_TEST_TURNAROUND_DAYS
            return r['test_turnaround_days'] <= target_tat
        crit_conclusive.loc[:, 'tat_met'] = crit_conclusive.apply(_check_tat_met_overall_clinic, axis=1)
        summary["overall_perc_met_tat"] = (crit_conclusive['tat_met'].mean() * 100) if not crit_conclusive.empty else 0.0
    summary["total_pending_critical_tests"] = df[(df['test_type'].isin(crit_test_disp_names)) & (df['test_result'] == 'Pending')]['patient_id'].nunique()
    if not all_proc_samples_df.empty: summary["sample_rejection_rate"] = (all_proc_samples_df[all_proc_samples_df['sample_status'] == 'Rejected'].shape[0] / len(all_proc_samples_df)) * 100 if len(all_proc_samples_df) > 0 else 0.0
    
    test_summary_details = {}
    for orig_key, cfg_props in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items():
        disp_name = cfg_props.get("display_name", orig_key)
        actual_test_keys = cfg_props.get("types_in_group", [orig_key]) # This should be a list of keys, not display names
        if isinstance(actual_test_keys, str): actual_test_keys = [actual_test_keys]

        # Filter df using actual_test_keys which should match the 'test_type' column in df
        grp_df = df[df['test_type'].isin(actual_test_keys)]
        if grp_df.empty: continue; stats = {}
        grp_concl = grp_df[~grp_df['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown', 'N/A', 'nan','Indeterminate']) & grp_df['test_turnaround_days'].notna()].copy()
        if not grp_concl.empty:
            stats["positive_rate"] = (grp_concl[grp_concl['test_result'] == 'Positive'].shape[0] / len(grp_concl)) * 100 if len(grp_concl) > 0 else 0.0
            stats["avg_tat_days"] = grp_concl['test_turnaround_days'].mean()
            tgt_tat = cfg_props.get("target_tat_days", app_config.TARGET_TEST_TURNAROUND_DAYS)
            grp_concl.loc[:, 'tat_met_specific'] = grp_concl['test_turnaround_days'] <= tgt_tat
            stats["perc_met_tat_target"] = grp_concl['tat_met_specific'].mean() * 100 if not grp_concl.empty else 0.0
        else: stats = {"positive_rate":0.0, "avg_tat_days":np.nan, "perc_met_tat_target":0.0}
        stats["pending_count"] = grp_df[grp_df['test_result'] == 'Pending']['patient_id'].nunique()
        stats["rejected_count"] = grp_df[grp_df['sample_status'] == 'Rejected']['patient_id'].nunique()
        stats["total_conducted_conclusive"] = len(grp_concl); test_summary_details[disp_name] = stats
    summary["test_summary_details"] = test_summary_details
    
    if 'item' in df.columns and 'item_stock_agg_zone' in df.columns and 'consumption_rate_per_day' in df.columns and app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        key_drugs_df = df[df['item'].str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False)]
        if not key_drugs_df.empty:
            latest_key_supply = key_drugs_df.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last')
            latest_key_supply['days_of_supply_calc'] = latest_key_supply['item_stock_agg_zone'] / (latest_key_supply['consumption_rate_per_day'].replace(0, np.nan))
            summary['key_drug_stockouts'] = latest_key_supply[latest_key_supply['days_of_supply_calc'] < app_config.CRITICAL_SUPPLY_DAYS]['item'].nunique()
    return summary

def get_clinic_environmental_summary(iot_df_period: pd.DataFrame) -> Dict[str, Any]:
    summary = { "avg_co2_overall": np.nan, "rooms_co2_alert_latest": 0, "avg_pm25_overall": np.nan, "rooms_pm25_alert_latest": 0, "avg_occupancy_overall": np.nan, "high_occupancy_alert_latest": False, "avg_noise_overall": np.nan, "rooms_noise_alert_latest": 0 }
    if iot_df_period is None or iot_df_period.empty: return summary
    if 'timestamp' not in iot_df_period.columns or not pd.api.types.is_datetime64_any_dtype(iot_df_period['timestamp']): return summary
    num_cols = ['avg_co2_ppm', 'avg_pm25', 'waiting_room_occupancy', 'avg_noise_db']; df_iot = iot_df_period.copy()
    for col in num_cols: df_iot[col] = _convert_to_numeric(df_iot[col], np.nan) if col in df_iot.columns else np.nan
    if df_iot['avg_co2_ppm'].notna().any(): summary["avg_co2_overall"] = df_iot['avg_co2_ppm'].mean()
    if df_iot['avg_pm25'].notna().any(): summary["avg_pm25_overall"] = df_iot['avg_pm25'].mean()
    if df_iot['waiting_room_occupancy'].notna().any(): summary["avg_occupancy_overall"] = df_iot['waiting_room_occupancy'].mean()
    if df_iot['avg_noise_db'].notna().any(): summary["avg_noise_overall"] = df_iot['avg_noise_db'].mean()
    key_cols_room = ['clinic_id', 'room_name']
    if all(c in df_iot.columns for c in key_cols_room):
        latest_reads = df_iot.sort_values('timestamp').drop_duplicates(subset=key_cols_room, keep='last')
        if not latest_reads.empty:
            if 'avg_co2_ppm' in latest_reads: summary["rooms_co2_alert_latest"] = latest_reads[latest_reads['avg_co2_ppm'] > app_config.CO2_LEVEL_ALERT_PPM].shape[0]
            if 'avg_pm25' in latest_reads: summary["rooms_pm25_alert_latest"] = latest_reads[latest_reads['avg_pm25'] > app_config.PM25_ALERT_UGM3].shape[0]
            if 'waiting_room_occupancy' in latest_reads: summary["high_occupancy_alert_latest"] = (latest_reads['waiting_room_occupancy'] > app_config.TARGET_WAITING_ROOM_OCCUPANCY).any()
            if 'avg_noise_db' in latest_reads: summary["rooms_noise_alert_latest"] = latest_reads[latest_reads['avg_noise_db'] > app_config.NOISE_LEVEL_ALERT_DB].shape[0]
    return summary

def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_THRESHOLDS['moderate']) -> pd.DataFrame:
    if health_df_period is None or health_df_period.empty: return pd.DataFrame()
    alerts_data = []; df_alerts = health_df_period.copy()
    alert_cols = ['patient_id', 'encounter_date', 'condition', 'ai_risk_score', 'ai_followup_priority_score', 'test_type', 'test_result', 'hiv_viral_load_copies_ml', 'sample_status', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius', 'referral_status', 'referral_reason']
    for col in alert_cols: # Ensure all columns exist
        if col not in df_alerts.columns: df_alerts[col] = np.nan if col in ['ai_risk_score', 'ai_followup_priority_score', 'hiv_viral_load_copies_ml', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius'] else (pd.NaT if col == 'encounter_date' else "Unknown")

    if df_alerts['ai_risk_score'].notna().any() or df_alerts['ai_followup_priority_score'].notna().any():
        for _, r in df_alerts[(df_alerts['ai_risk_score'] >= risk_threshold_moderate) | (df_alerts['ai_followup_priority_score'] >= 75)].iterrows(): alerts_data.append({**r.to_dict(), 'alert_reason': f"High AI Risk/Prio (Risk:{r.get('ai_risk_score',0):.0f}, Prio:{r.get('ai_followup_priority_score',0):.0f})", 'priority_score': max(r.get('ai_risk_score',0), r.get('ai_followup_priority_score',0))})
    
    crit_test_disp_names_clinic = [v['display_name'] for k,v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get("critical")]
    crit_pos = df_alerts[(df_alerts['test_type'].isin(crit_test_disp_names_clinic)) & (df_alerts['test_result'] == 'Positive')]
    for _, r in crit_pos.iterrows(): alerts_data.append({**r.to_dict(), 'alert_reason': f"Critical Positive: {r['test_type']}", 'priority_score': 85})
    
    if 'hiv_viral_load_copies_ml' in df_alerts and df_alerts['hiv_viral_load_copies_ml'].notna().any():
        for _, r in df_alerts[df_alerts['hiv_viral_load_copies_ml'] > 1000].iterrows(): alerts_data.append({**r.to_dict(), 'alert_reason': f"High HIV Viral Load ({r['hiv_viral_load_copies_ml']:.0f})", 'priority_score': 90})
    if 'min_spo2_pct' in df_alerts and df_alerts['min_spo2_pct'].notna().any():
        for _, r in df_alerts[df_alerts['min_spo2_pct'] < app_config.SPO2_CRITICAL_THRESHOLD_PCT].iterrows(): alerts_data.append({**r.to_dict(), 'alert_reason': f"Critically Low SpO2 ({r['min_spo2_pct']:.0f}%)", 'priority_score': 92})
    
    date_col_overdue = 'sample_collection_date' if 'sample_collection_date' in df_alerts.columns and df_alerts['sample_collection_date'].notna().any() else 'encounter_date'
    if date_col_overdue in df_alerts.columns and df_alerts[date_col_overdue].notna().any():
        pending_df = df_alerts[(df_alerts['test_result'] == 'Pending') & (df_alerts['test_type'].isin(crit_test_disp_names_clinic)) & (df_alerts[date_col_overdue].notna())].copy()
        if not pending_df.empty:
            pending_df.loc[:, 'days_pending'] = (pd.Timestamp('today').normalize() - pd.to_datetime(pending_df[date_col_overdue], errors='coerce')).dt.days
            def get_overdue_thresh_clinic(ttype_disp): # Renamed inner function
                k = next((key for key,v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v['display_name']==ttype_disp), None)
                return (app_config.KEY_TEST_TYPES_FOR_ANALYSIS[k]['target_tat_days'] if k and 'target_tat_days' in app_config.KEY_TEST_TYPES_FOR_ANALYSIS[k] else app_config.OVERDUE_PENDING_TEST_DAYS) + 2
            pending_df.loc[:, 'overdue_thresh_days'] = pending_df['test_type'].apply(get_overdue_thresh_clinic)
            overdue_crit = pending_df[pending_df['days_pending'] > pending_df['overdue_thresh_days']]
            for _, r in overdue_crit.iterrows(): alerts_data.append({**r.to_dict(), 'alert_reason': f"Overdue Pending: {r['test_type']} ({r.get('days_pending',0):.0f}d)", 'priority_score': 75 + min(r.get('days_pending',0) - r.get('overdue_thresh_days',0), 10)})
            
    if not alerts_data: return pd.DataFrame(columns=alert_cols + ['alert_reason', 'priority_score']) # Return DF with expected cols if empty
    alerts_df = pd.DataFrame(alerts_data)
    alerts_df['alert_reason'] = alerts_df['alert_reason'].astype(str).fillna("Unknown Alert")
    alerts_df['encounter_date'] = pd.to_datetime(alerts_df['encounter_date'], errors='coerce')
    alerts_df.dropna(subset=['patient_id', 'encounter_date'], inplace=True) # Must have these for grouping
    if alerts_df.empty: return pd.DataFrame(columns=alerts_df.columns) # Return with original columns if all dropped
    alerts_df['encounter_date_obj'] = alerts_df['encounter_date'].dt.date

    def aggregate_alerts_clinic(group): # Distinct name
        first_row_dict = group.iloc[0].to_dict()
        alert_reasons_str = [str(r) for r in group['alert_reason'].unique() if pd.notna(r)]
        aggregated_reason = "; ".join(alert_reasons_str) if alert_reasons_str else "General Alert"
        max_priority = group['priority_score'].max()
        # Return a dictionary that will become a row
        output_dict = {col: first_row_dict.get(col) for col in group.columns} # copy all columns from first row
        output_dict['alert_reason'] = aggregated_reason
        output_dict['priority_score'] = max_priority
        return pd.Series(output_dict)

    final_alerts_df = alerts_df.groupby(['patient_id', 'encounter_date_obj'], as_index=False).apply(aggregate_alerts_clinic, include_groups=False).reset_index(drop=True) # Use include_groups=False and reset_index
    
    if 'priority_score' in final_alerts_df.columns: final_alerts_df['priority_score'] = _convert_to_numeric(final_alerts_df['priority_score'],0).astype(int)
    else: final_alerts_df['priority_score'] = 0
    
    sort_date_col = 'encounter_date' if 'encounter_date' in final_alerts_df.columns and final_alerts_df['encounter_date'].notna().all() else 'encounter_date_obj'
    if sort_date_col not in final_alerts_df.columns and 'encounter_date_obj' in final_alerts_df.columns : sort_date_col = 'encounter_date_obj'
    elif sort_date_col not in final_alerts_df.columns : # if neither is there, can't sort by date
        return final_alerts_df.sort_values(by=['priority_score'], ascending=[False])
        
    return final_alerts_df.sort_values(by=['priority_score', sort_date_col], ascending=[False, False])

def get_district_summary_kpis(enriched_zone_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    kpis: Dict[str, Any] = { "total_population_district": 0, "avg_population_risk": np.nan, "zones_high_risk_count": 0, "overall_facility_coverage": np.nan, "district_tb_burden_total": 0, "district_malaria_burden_total": 0, "key_infection_prevalence_district_per_1000": np.nan, "population_weighted_avg_steps": np.nan, "avg_clinic_co2_district":np.nan }
    if enriched_zone_gdf is None or enriched_zone_gdf.empty: return kpis
    gdf = enriched_zone_gdf.copy()
    num_cols_dist = ['population', 'avg_risk_score', 'active_tb_cases', 'active_malaria_cases', 'total_active_key_infections', 'facility_coverage_score', 'avg_daily_steps_zone', 'zone_avg_co2']
    for col in num_cols_dist: gdf[col] = _convert_to_numeric(gdf[col], 0.0) if col in gdf.columns else 0.0
    kpis["total_population_district"] = gdf['population'].sum()
    if kpis["total_population_district"] > 0 and kpis["total_population_district"] is not np.nan: # Check not NaN
        gdf['pop_x_risk'] = gdf['population'] * gdf['avg_risk_score']
        kpis["avg_population_risk"] = gdf['pop_x_risk'].sum() / kpis["total_population_district"] if kpis["total_population_district"] > 0 else np.nan
        gdf['pop_x_facility_coverage'] = gdf['population'] * gdf['facility_coverage_score']
        kpis["overall_facility_coverage"] = gdf['pop_x_facility_coverage'].sum() / kpis["total_population_district"] if kpis["total_population_district"] > 0 else np.nan
        gdf['pop_x_steps'] = gdf['population'] * gdf['avg_daily_steps_zone']
        kpis["population_weighted_avg_steps"] = gdf['pop_x_steps'].sum() / kpis["total_population_district"] if kpis["total_population_district"] > 0 else np.nan
        kpis["key_infection_prevalence_district_per_1000"] = (gdf['total_active_key_infections'].sum() / kpis["total_population_district"]) * 1000 if kpis["total_population_district"] > 0 else np.nan
    else: # Fallbacks if total_population_district is 0 or NaN
        kpis["avg_population_risk"] = gdf['avg_risk_score'].mean() if not gdf.empty and gdf['avg_risk_score'].notna().any() else np.nan
        kpis["overall_facility_coverage"] = gdf['facility_coverage_score'].mean() if not gdf.empty and gdf['facility_coverage_score'].notna().any() else np.nan
        kpis["population_weighted_avg_steps"] = gdf['avg_daily_steps_zone'].mean() if not gdf.empty and gdf['avg_daily_steps_zone'].notna().any() else np.nan
    kpis["zones_high_risk_count"] = gdf[gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk']].shape[0] if 'avg_risk_score' in gdf.columns else 0
    kpis["district_tb_burden_total"] = int(gdf['active_tb_cases'].sum())
    kpis["district_malaria_burden_total"] = int(gdf['active_malaria_cases'].sum())
    kpis["avg_clinic_co2_district"] = gdf['zone_avg_co2'].mean() if 'zone_avg_co2' in gdf and gdf['zone_avg_co2'].notna().any() else np.nan
    return kpis

def get_trend_data(df: pd.DataFrame, value_col: str, date_col: str = 'encounter_date', period: str = 'D', agg_func: str = 'mean', filter_col: Optional[str] = None, filter_val: Optional[Any] = None) -> pd.Series:
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns: return pd.Series(dtype='float64')
    trend_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(trend_df[date_col]): trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors='coerce')
    trend_df.dropna(subset=[date_col], inplace=True) # Drop rows where date conversion failed
    if value_col not in trend_df.columns: return pd.Series(dtype='float64') # value_col might be dropped if all NaN before this check
    trend_df.dropna(subset=[value_col], inplace=True) # Drop if target value is also NaN
    if trend_df.empty: return pd.Series(dtype='float64')
    if filter_col and filter_col in trend_df.columns and filter_val is not None:
        trend_df = trend_df[trend_df[filter_col] == filter_val]
        if trend_df.empty: return pd.Series(dtype='float64')
    trend_df.set_index(date_col, inplace=True)
    if agg_func in ['mean', 'sum', 'median'] and not pd.api.types.is_numeric_dtype(trend_df[value_col]):
        trend_df[value_col] = _convert_to_numeric(trend_df[value_col], np.nan); trend_df.dropna(subset=[value_col], inplace=True)
        if trend_df.empty: return pd.Series(dtype='float64')
    try:
        if agg_func == 'nunique': trend_series = trend_df.groupby(pd.Grouper(freq=period))[value_col].nunique()
        elif agg_func == 'sum': trend_series = trend_df.groupby(pd.Grouper(freq=period))[value_col].sum()
        elif agg_func == 'median': trend_series = trend_df.groupby(pd.Grouper(freq=period))[value_col].median()
        else: trend_series = trend_df.groupby(pd.Grouper(freq=period))[value_col].mean()
    except Exception as e: logger.error(f"Trend error for {value_col} (agg {agg_func}): {e}", exc_info=True); return pd.Series(dtype='float64')
    return trend_series

def get_supply_forecast_data(health_df: pd.DataFrame, forecast_days_out: int = 30, item_filter_list: Optional[List[str]] = None) -> pd.DataFrame:
    default_cols = ['item','date','current_stock','consumption_rate','forecast_stock','forecast_days','estimated_stockout_date','lower_ci','upper_ci','initial_days_supply']
    if health_df is None or health_df.empty or not all(c in health_df.columns for c in ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']):
        return pd.DataFrame(columns=default_cols)
    supply_status_df = health_df.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
    if item_filter_list: supply_status_df = supply_status_df[supply_status_df['item'].isin(item_filter_list)]
    if supply_status_df.empty: return pd.DataFrame(columns=default_cols)
    
    forecasts = []
    for _, row in supply_status_df.iterrows():
        item, cur_stock, cons_rate, last_date = row['item'], row['item_stock_agg_zone'], row['consumption_rate_per_day'], pd.to_datetime(row['encounter_date'])
        if pd.isna(cur_stock) or pd.isna(cons_rate) or pd.isna(last_date) or cur_stock < 0: continue
        
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days_out, freq='D')
        est_stockout_date = pd.NaT
        days_rem_start = (cur_stock / cons_rate) if cons_rate > 0.001 else np.inf # Use a small epsilon for cons_rate
        
        # Calculate stockout date based on initial stock and constant consumption first
        if cons_rate > 0.001:
            days_to_stockout_from_start = cur_stock / cons_rate
            est_stockout_date = last_date + pd.to_timedelta(days_to_stockout_from_start, unit='D')
        
        for i, fc_date in enumerate(forecast_dates):
            days_out = i + 1; current_forecast_stock = cur_stock - (cons_rate * days_out)
            days_of_supply_fc = (current_forecast_stock / cons_rate) if cons_rate > 0.001 else (np.inf if current_forecast_stock > 0 else 0)
            cons_std_factor = 0.15; lower_cons = cons_rate * (1+cons_std_factor); upper_cons = max(0.01, cons_rate * (1-cons_std_factor)) # ensure upper_cons > 0 if cons_rate >0
            lower_ci_stock = cur_stock - (lower_cons*days_out); upper_ci_stock = cur_stock - (upper_cons*days_out)
            lower_ci_days = (lower_ci_stock/lower_cons) if lower_cons > 0.001 else (np.inf if lower_ci_stock > 0 else 0)
            upper_ci_days = (upper_ci_stock/upper_cons) if upper_cons > 0.001 else (np.inf if upper_ci_stock > 0 else 0)
            
            forecasts.append({'item': item, 'date': fc_date, 'current_stock': cur_stock, 'consumption_rate': cons_rate, 
                              'forecast_stock': max(0,current_forecast_stock), 
                              'forecast_days': max(0,days_of_supply_fc), 
                              'estimated_stockout_date': est_stockout_date, # Use pre-calculated stockout date
                              'lower_ci': max(0,lower_ci_days), 'upper_ci': max(0,upper_ci_days), 
                              'initial_days_supply': days_rem_start})
    if not forecasts: return pd.DataFrame(columns=default_cols)
    return pd.DataFrame(forecasts)
