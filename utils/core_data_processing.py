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
        geom_col_name = gdf.geometry.name if hasattr(gdf, 'geometry') and hasattr(gdf.geometry, 'name') else 'geometry'
        
        if geom_col_name in gdf.columns and hasattr(gdf[geom_col_name], 'is_empty') and not gdf[geom_col_name].is_empty.all():
            non_geom_cols = gdf.drop(columns=[geom_col_name], errors='ignore').columns.tolist()
            geom_hash = pd.util.hash_array(gdf[geom_col_name].to_wkt().values).sum() if hasattr(gdf[geom_col_name], 'to_wkt') else 0
        else:
            non_geom_cols = gdf.columns.tolist()
            geom_hash = 0
            
        df_hashable = gdf[non_geom_cols].copy()
        for col in df_hashable.select_dtypes(include=[np.datetime64, 'datetime64[ns]', 'datetime64[ns, UTC]']).columns: # Pandas <2.0 compatibility
             df_hashable[col] = df_hashable[col].astype(np.int64) # Convert datetimes to int for hashing

        df_hash = pd.util.hash_pandas_object(df_hashable, index=True).sum()
        return f"{df_hash}-{geom_hash}"
    except Exception as e:
        logger.warning(f"Could not hash GeoDataFrame: {e}"); return None

def _robust_merge_agg(
    left_df: pd.DataFrame, right_df: pd.DataFrame, target_col_name: str,
    on_col: str = 'zone_id', default_fill_value: Any = 0.0
) -> pd.DataFrame:
    """Robustly merges an aggregated right_df into left_df."""
    if target_col_name not in left_df.columns:
        left_df[target_col_name] = default_fill_value

    if right_df.empty or on_col not in right_df.columns:
        left_df[target_col_name] = left_df.get(target_col_name, default_fill_value)
        left_df[target_col_name].fillna(default_fill_value, inplace=True)
        return left_df

    value_col_in_right_df_candidates = [col for col in right_df.columns if col != on_col]
    if not value_col_in_right_df_candidates:
        logger.debug(f"_robust_merge_agg: Right DF for target '{target_col_name}' has no value column besides '{on_col}'.")
        left_df[target_col_name] = left_df.get(target_col_name, default_fill_value)
        left_df[target_col_name].fillna(default_fill_value, inplace=True)
        return left_df
    
    value_col_in_right_df = value_col_in_right_df_candidates[0]
    temp_agg_col_name = f"__{target_col_name}_temp_agg_{np.random.randint(1000,9999)}__"
    
    right_df_renamed = right_df[[on_col, value_col_in_right_df]].copy()
    right_df_renamed.rename(columns={value_col_in_right_df: temp_agg_col_name}, inplace=True)
    
    left_df_index_name = left_df.index.name
    left_df_reset = left_df.reset_index(drop=left_df_index_name is None) # drop only if index is default RangeIndex

    merged_df = left_df_reset.merge(right_df_renamed, on=on_col, how='left')
    
    if temp_agg_col_name in merged_df.columns:
        # If merged_df[target_col_name] did not exist it was created with default_fill_value by the function start
        merged_df[target_col_name] = np.where(
            merged_df[temp_agg_col_name].notna(),
            merged_df[temp_agg_col_name],
            merged_df.get(target_col_name, default_fill_value) # Use .get() for safety if column was somehow dropped
        )
        merged_df.drop(columns=[temp_agg_col_name], inplace=True, errors='ignore')
    
    merged_df[target_col_name].fillna(default_fill_value, inplace=True)

    if left_df_index_name and left_df_index_name in merged_df.columns: # If original index was named and became a column
        merged_df.set_index(left_df_index_name, inplace=True)
    elif left_df_index_name is None and 'index' in merged_df.columns and 'index' not in left_df.columns: # if default 'index' col was added
        merged_df.set_index('index', inplace=True)


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
            else: df[col] = pd.NaT # Add missing date columns as NaT
        numeric_cols = ['test_turnaround_days', 'quantity_dispensed', 'item_stock_agg_zone', 'consumption_rate_per_day', 'ai_risk_score', 'ai_followup_priority_score', 'vital_signs_bp_systolic', 'vital_signs_bp_diastolic', 'vital_signs_temperature_celsius', 'min_spo2_pct', 'max_skin_temp_celsius', 'avg_spo2', 'avg_daily_steps', 'resting_heart_rate', 'avg_hrv', 'avg_sleep_duration_hrs', 'sleep_score_pct', 'stress_level_score', 'fall_detected_today', 'age', 'chw_visit', 'tb_contact_traced', 'patient_latitude', 'patient_longitude']
        for col in numeric_cols:
            if col in df.columns: df[col] = _convert_to_numeric(df[col])
            else: df[col] = np.nan
        if 'hiv_viral_load_copies_ml' in df.columns: df['hiv_viral_load_copies_ml'] = pd.to_numeric(df['hiv_viral_load_copies_ml'], errors='coerce')
        elif 'hiv_viral_load_copies_ml' not in df.columns : df['hiv_viral_load_copies_ml'] = np.nan

        string_like_cols = ['encounter_id', 'patient_id', 'encounter_type', 'condition', 'diagnosis_code_icd10', 'test_type', 'test_result', 'item', 'zone_id', 'clinic_id', 'physician_id', 'notes', 'patient_reported_symptoms', 'gender', 'screening_hpv_status', 'key_chronic_conditions_summary', 'medication_adherence_self_report', 'referral_status', 'referral_reason', 'referred_to_facility_id', 'referral_outcome', 'sample_status', 'rejection_reason']
        for col in string_like_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str).str.strip()
                df.loc[df[col].isin(['', 'nan', 'None', 'N/A', '#N/A', 'np.nan']), col] = "Unknown"
            else: df[col] = "Unknown"
        required_cols = ['patient_id', 'encounter_date', 'condition', 'test_type', 'test_result', 'zone_id', 'ai_risk_score']
        for r_col in required_cols: # Ensure these absolutely essential columns exist
            if r_col not in df.columns:
                logger.warning(f"Core required column '{r_col}' not found in health records. Adding as empty/default.")
                if r_col == 'encounter_date': df[r_col] = pd.NaT
                elif r_col == 'ai_risk_score' : df[r_col] = np.nan
                else: df[r_col] = "Unknown"
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
            else: df[col] = np.nan
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
        active_geom_col = merged_gdf.geometry.name if hasattr(merged_gdf, 'geometry') else 'geometry'
        if active_geom_col != 'geometry' and 'geometry' in merged_gdf.columns: merged_gdf = merged_gdf.set_geometry('geometry', inplace=False)
        elif 'geometry' not in merged_gdf.columns and active_geom_col in merged_gdf.columns: merged_gdf = merged_gdf.set_geometry(active_geom_col, inplace=False)


        if merged_gdf.crs is None: merged_gdf = merged_gdf.set_crs(app_config.DEFAULT_CRS, allow_override=True)
        elif merged_gdf.crs.to_string().upper() != app_config.DEFAULT_CRS.upper(): merged_gdf = merged_gdf.to_crs(app_config.DEFAULT_CRS)
        
        required_zone_cols = ['zone_id', 'name', 'population', 'geometry', 'num_clinics', 'socio_economic_index', 'avg_travel_time_clinic_min']
        for rz_col in required_zone_cols:
            if rz_col not in merged_gdf.columns:
                logger.warning(f"Required zone column '{rz_col}' missing after merge. Adding default.")
                if rz_col == 'population' or rz_col == 'num_clinics': merged_gdf[rz_col] = 0.0
                elif rz_col == 'socio_economic_index': merged_gdf[rz_col] = 0.5
                elif rz_col == 'avg_travel_time_clinic_min': merged_gdf[rz_col] = 30.0
                elif rz_col == 'name': merged_gdf[rz_col] = "Zone " + merged_gdf['zone_id'].astype(str) if 'zone_id' in merged_gdf and not merged_gdf['zone_id'].empty else "Unknown Zone"
                elif rz_col != 'geometry': merged_gdf[rz_col] = "Unknown"
        for num_col in ['population', 'socio_economic_index', 'num_clinics', 'avg_travel_time_clinic_min']:
            if num_col in merged_gdf.columns: merged_gdf[num_col] = _convert_to_numeric(merged_gdf[num_col], 0 if num_col in ['population', 'num_clinics'] else (0.5 if num_col=='socio_economic_index' else 30.0))
        logger.info(f"Successfully loaded and merged zone data: {len(merged_gdf)} zones.")
        return merged_gdf
    except Exception as e:
        logger.error(f"Error loading/merging zone data: {e}", exc_info=True); st.error(f"Error with zone GIS data: {e}"); return None

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
        
        crit_test_keys_enrich = [k for k, v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get("critical", False) and k in health_df_for_agg['test_type'].unique()] # ensure test type exists in data
        if crit_test_keys_enrich:
            tat_df_enrich = health_df_for_agg[(health_df_for_agg['test_type'].isin(crit_test_keys_enrich)) & (health_df_for_agg['test_turnaround_days'].notna()) & (~health_df_for_agg['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown','Indeterminate']))].copy()
            if not tat_df_enrich.empty:
                enriched = _robust_merge_agg(enriched, tat_df_enrich.groupby('zone_id')['test_turnaround_days'].mean().reset_index(), 'avg_test_turnaround_critical')
                def _check_tat_met_enrich_local(row_tat_enrich_local):
                    cfg_local = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(row_tat_enrich_local['test_type']) # Ensure this matches keys
                    return row_tat_enrich_local['test_turnaround_days'] <= (cfg_local['target_tat_days'] if cfg_local and 'target_tat_days' in cfg_local else app_config.TARGET_TEST_TURNAROUND_DAYS)
                tat_df_enrich.loc[:,'tat_met_flag'] = tat_df_enrich.apply(_check_tat_met_enrich_local, axis=1)
                perc_met_agg_enrich = tat_df_enrich.groupby('zone_id')['tat_met_flag'].mean().reset_index()
                perc_met_agg_enrich.rename(columns={'tat_met_flag': 'value_col_for_merge'}, inplace=True) # Standardize name for helper
                enriched = _robust_merge_agg(enriched, perc_met_agg_enrich, 'perc_critical_tests_tat_met')
                if 'perc_critical_tests_tat_met' in enriched.columns:
                    enriched['perc_critical_tests_tat_met'] = enriched['perc_critical_tests_tat_met'] * 100
        
        if 'avg_daily_steps' in health_df_for_agg.columns:
            enriched = _robust_merge_agg(enriched, health_df_for_agg.groupby('zone_id')['avg_daily_steps'].mean().reset_index(), 'avg_daily_steps_zone')

    if iot_df is not None and not iot_df.empty and 'zone_id' in iot_df.columns and 'avg_co2_ppm' in iot_df.columns:
        iot_df['zone_id'] = iot_df['zone_id'].astype(str).str.strip()
        enriched = _robust_merge_agg(enriched, iot_df.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(), 'zone_avg_co2')
    
    if 'total_active_key_infections' in enriched.columns and 'population' in enriched.columns:
         enriched['prevalence_per_1000'] = enriched.apply(lambda r: (r['total_active_key_infections']/r['population'])*1000 if pd.notna(r['population']) and r['population']>0 and pd.notna(r['total_active_key_infections']) else 0.0, axis=1).fillna(0.0)
    if 'num_clinics' in enriched.columns and 'population' in enriched.columns:
        enriched['facility_coverage_score'] = enriched.apply(lambda r: (r['num_clinics']/r['population'])*10000*5 if pd.notna(r['population']) and r['population']>0 and pd.notna(r['num_clinics']) else 0.0, axis=1).fillna(0.0).clip(0,100)
    elif 'facility_coverage_score' not in enriched.columns: enriched['facility_coverage_score'] = 0.0
    
    for col in agg_cols_to_initialize:
        if col in enriched.columns:
            if not pd.api.types.is_numeric_dtype(enriched[col]): enriched[col] = pd.to_numeric(enriched[col], errors='coerce')
            enriched[col].fillna(0.0, inplace=True)
        else: enriched[col] = 0.0 # Ensure all initialized agg cols are present
    logger.info("Zone GeoDataFrame enrichment complete.")
    return enriched

def get_overall_kpis(health_df: pd.DataFrame, date_filter_start: Optional[str] = None, date_filter_end: Optional[str] = None) -> Dict[str, Any]:
    kpis = { "total_patients": 0, "avg_patient_risk": np.nan, "active_tb_cases_current": 0, "malaria_rdt_positive_rate_period": 0.0, "hiv_rapid_positive_rate_period": 0.0, "key_supply_stockout_alerts": 0 }
    if health_df is None or health_df.empty: return kpis
    df = health_df.copy()
    if 'encounter_date' not in df.columns or df['encounter_date'].isnull().all(): return kpis
    df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce')
    df.dropna(subset=['encounter_date'], inplace=True)
    if date_filter_start: df = df[df['encounter_date'] >= pd.to_datetime(date_filter_start, errors='coerce')]
    if date_filter_end: df = df[df['encounter_date'] <= pd.to_datetime(date_filter_end, errors='coerce')]
    df.dropna(subset=['encounter_date'], inplace=True)
    if df.empty: return kpis
    
    kpis["total_patients"] = df['patient_id'].nunique()
    kpis["avg_patient_risk"] = df['ai_risk_score'].mean() if 'ai_risk_score' in df and df['ai_risk_score'].notna().any() else np.nan
    kpis["active_tb_cases_current"] = df[df['condition'].str.contains("TB", case=False, na=False)]['patient_id'].nunique() if 'condition' in df.columns else 0
    
    mal_rdt_key = "RDT-Malaria" # Assuming 'test_type' column uses original keys
    malaria_rdt_df = df[(df['test_type'] == mal_rdt_key) & (~df['test_result'].isin(["Pending", "Rejected Sample", "Unknown"]))]
    if not malaria_rdt_df.empty: kpis["malaria_rdt_positive_rate_period"] = (malaria_rdt_df[malaria_rdt_df['test_result'] == 'Positive'].shape[0] / len(malaria_rdt_df)) * 100 if len(malaria_rdt_df) > 0 else 0.0

    hiv_rapid_key = "HIV-Rapid"
    hiv_rapid_df = df[(df['test_type'] == hiv_rapid_key) & (~df['test_result'].isin(["Pending", "Rejected Sample", "Unknown"]))]
    if not hiv_rapid_df.empty: kpis["hiv_rapid_positive_rate_period"] = (hiv_rapid_df[hiv_rapid_df['test_result'] == 'Positive'].shape[0] / len(hiv_rapid_df)) * 100 if len(hiv_rapid_df) > 0 else 0.0
    
    if 'item' in df.columns and 'item_stock_agg_zone' in df.columns and 'consumption_rate_per_day' in df.columns:
        supply_df = df.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last')
        supply_df['days_of_supply'] = supply_df['item_stock_agg_zone'] / (supply_df['consumption_rate_per_day'].replace(0, np.nan))
        supply_df.dropna(subset=['days_of_supply'], inplace=True)
        kpis['key_supply_stockout_alerts'] = supply_df[supply_df['days_of_supply'] < app_config.CRITICAL_SUPPLY_DAYS]['item'].nunique()
    return kpis

def get_chw_summary(health_df_daily: pd.DataFrame) -> Dict[str, Any]:
    summary = { "visits_today": 0, "tb_contacts_to_trace_today": 0, "sti_symptomatic_referrals_today": 0, "avg_patient_risk_visited_today": np.nan, "high_risk_followups_today": 0, "patients_low_spo2_visited_today": 0, "patients_fever_visited_today": 0, "avg_patient_steps_visited_today": np.nan, "patients_fall_detected_today": 0 }
    if health_df_daily is None or health_df_daily.empty: return summary
    chw_enc_df = health_df_daily.copy()
    is_chw_encounter_identified = False
    if 'chw_visit' in chw_enc_df.columns and chw_enc_df['chw_visit'].dtype in [np.int64, np.float64, int, float] and chw_enc_df['chw_visit'].sum(skipna=True) > 0 :
        chw_enc_df = chw_enc_df[chw_enc_df['chw_visit'] == 1]; is_chw_encounter_identified = True
    elif 'encounter_type' in chw_enc_df.columns and chw_enc_df['encounter_type'].str.contains("CHW", case=False, na=False).any():
        chw_enc_df = chw_enc_df[chw_enc_df['encounter_type'].str.contains("CHW", case=False, na=False)]; is_chw_encounter_identified = True
    if not is_chw_encounter_identified and not chw_enc_df.empty : pass
    elif chw_enc_df.empty: return summary

    summary["visits_today"] = chw_enc_df['patient_id'].nunique()
    if all(c in chw_enc_df.columns for c in ['condition', 'referral_reason', 'referral_status']):
        summary["tb_contacts_to_trace_today"] = chw_enc_df[(chw_enc_df['condition'] == 'TB') & (chw_enc_df['referral_reason'].str.contains("Contact Tracing|Investigation", case=False, na=False)) & (chw_enc_df['referral_status'] == 'Pending')]['patient_id'].nunique()
        summary["sti_symptomatic_referrals_today"] = chw_enc_df[(chw_enc_df['condition'].str.contains("STI", case=False, na=False)) & (chw_enc_df.get('patient_reported_symptoms', pd.Series(dtype=str)) != "Unknown") & (chw_enc_df['referral_status'] == 'Pending')]['patient_id'].nunique()
    if 'ai_risk_score' in chw_enc_df.columns and chw_enc_df['ai_risk_score'].notna().any():
        summary["avg_patient_risk_visited_today"] = chw_enc_df['ai_risk_score'].mean()
        summary["high_risk_followups_today"] = chw_enc_df[chw_enc_df['ai_risk_score'] >= app_config.RISK_THRESHOLDS.get('high', 75)]['patient_id'].nunique()
    if 'min_spo2_pct' in chw_enc_df.columns: summary["patients_low_spo2_visited_today"] = chw_enc_df[chw_enc_df['min_spo2_pct'] < app_config.SPO2_LOW_THRESHOLD_PCT]['patient_id'].nunique()
    temp_col_chw = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in chw_enc_df.columns and chw_enc_df['vital_signs_temperature_celsius'].notna().any() else ('max_skin_temp_celsius' if 'max_skin_temp_celsius' in chw_enc_df.columns else None)
    if temp_col_chw and temp_col_chw in chw_enc_df.columns and chw_enc_df[temp_col_chw].notna().any(): summary["patients_fever_visited_today"] = chw_enc_df[chw_enc_df[temp_col_chw] >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C]['patient_id'].nunique()
    if 'avg_daily_steps' in chw_enc_df.columns and chw_enc_df['avg_daily_steps'].notna().any(): summary["avg_patient_steps_visited_today"] = chw_enc_df['avg_daily_steps'].mean()
    if 'fall_detected_today' in chw_enc_df.columns and chw_enc_df['fall_detected_today'].notna().any(): summary["patients_fall_detected_today"] = chw_enc_df[chw_enc_df['fall_detected_today'] > 0]['patient_id'].nunique()
    return summary

def get_patient_alerts_for_chw(health_df_daily: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_THRESHOLDS['chw_alert_moderate'], risk_threshold_high: int = app_config.RISK_THRESHOLDS['chw_alert_high']) -> pd.DataFrame:
    if health_df_daily is None or health_df_daily.empty: return pd.DataFrame()
    alerts = []; df_alerts = health_df_daily.copy()
    alert_cols_chw = ['patient_id', 'ai_risk_score', 'ai_followup_priority_score', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius', 'condition', 'referral_status', 'fall_detected_today', 'encounter_date']
    for col in alert_cols_chw:
        if col not in df_alerts.columns: df_alerts[col] = np.nan if col in ['ai_risk_score','ai_followup_priority_score','min_spo2_pct','vital_signs_temperature_celsius','max_skin_temp_celsius','fall_detected_today'] else ("Unknown" if col in ['condition','referral_status','patient_id'] else pd.NaT)
    
    if df_alerts['ai_followup_priority_score'].notna().any():
        for _, r in df_alerts[df_alerts['ai_followup_priority_score'] >= 80].iterrows(): alerts.append({**r.to_dict(), 'alert_reason': "High AI Follow-up Priority", 'priority_score': r['ai_followup_priority_score']})
    if df_alerts['min_spo2_pct'].notna().any():
        for _, r in df_alerts[df_alerts['min_spo2_pct'] < app_config.SPO2_CRITICAL_THRESHOLD_PCT].iterrows(): alerts.append({**r.to_dict(), 'alert_reason': f"Critical SpO2 ({r['min_spo2_pct']}%)", 'priority_score': 90 + (app_config.SPO2_CRITICAL_THRESHOLD_PCT - r['min_spo2_pct'])})
    temp_c = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in df_alerts.columns and df_alerts['vital_signs_temperature_celsius'].notna().any() else ('max_skin_temp_celsius' if 'max_skin_temp_celsius' in df_alerts.columns and df_alerts['max_skin_temp_celsius'].notna().any() else None)
    if temp_c and df_alerts[temp_c].notna().any():
        for _, r in df_alerts[df_alerts[temp_c] >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C + 1.0].iterrows(): alerts.append({**r.to_dict(), 'alert_reason': f"High Fever ({r[temp_c]}°C)", 'priority_score': 85 + (r[temp_c] - (app_config.SKIN_TEMP_FEVER_THRESHOLD_C + 1.0))})
    if 'fall_detected_today' in df_alerts.columns and df_alerts['fall_detected_today'].notna().any():
        for _, r in df_alerts[df_alerts['fall_detected_today'] > 0].iterrows(): alerts.append({**r.to_dict(), 'alert_reason': "Fall Detected", 'priority_score': 88})
    if 'ai_risk_score' in df_alerts.columns and df_alerts['ai_risk_score'].notna().any():
        for _, r in df_alerts[df_alerts['ai_risk_score'] >= risk_threshold_high].iterrows(): alerts.append({**r.to_dict(), 'alert_reason': "High General AI Risk", 'priority_score': r['ai_risk_score']})
    if 'condition' in df_alerts.columns and 'referral_status' in df_alerts.columns :
        for _, r in df_alerts[(df_alerts['condition'].isin(app_config.KEY_CONDITIONS_FOR_TRENDS[:4])) & (df_alerts['referral_status'] == 'Pending')].iterrows(): alerts.append({**r.to_dict(), 'alert_reason': f"Pending Referral for {r['condition']}", 'priority_score': 70})
    if 'ai_risk_score' in df_alerts.columns and df_alerts['ai_risk_score'].notna().any():
        for _, r in df_alerts[(df_alerts['ai_risk_score'] >= risk_threshold_moderate) & (df_alerts['ai_risk_score'] < risk_threshold_high)].iterrows(): alerts.append({**r.to_dict(), 'alert_reason': "Moderate AI Risk", 'priority_score': r['ai_risk_score']})
            
    if not alerts: return pd.DataFrame(columns=alert_cols_chw + ['alert_reason', 'priority_score'])
    alert_df = pd.DataFrame(alerts)
    alert_df['encounter_date'] = pd.to_datetime(alert_df['encounter_date'], errors='coerce')
    if 'encounter_date' in alert_df.columns and alert_df['encounter_date'].notna().any():
        alert_df['encounter_date_obj_for_dedup'] = alert_df['encounter_date'].dt.date
        alert_df.drop_duplicates(subset=['patient_id', 'alert_reason', 'encounter_date_obj_for_dedup'], inplace=True, keep='first')
        alert_df.drop(columns=['encounter_date_obj_for_dedup'], inplace=True, errors='ignore')
    else: alert_df.drop_duplicates(subset=['patient_id', 'alert_reason'], inplace=True, keep='first')
    alert_df['priority_score'] = alert_df['priority_score'].fillna(0).astype(int)
    sort_cols_chw_alert = ['priority_score']
    if 'encounter_date' in alert_df.columns and alert_df['encounter_date'].notna().any() : sort_cols_chw_alert.append('encounter_date')
    return alert_df.sort_values(by=sort_cols_chw_alert, ascending=[False, False] if len(sort_cols_chw_alert)>1 else [False])

def get_clinic_summary(health_df_period: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = { "overall_avg_test_turnaround": np.nan, "overall_perc_met_tat": 0.0, "total_pending_critical_tests": 0, "sample_rejection_rate": 0.0, "key_drug_stockouts": 0, "test_summary_details": {} }
    if health_df_period is None or health_df_period.empty: return summary
    df = health_df_period.copy()
    test_cols_req = ['test_type', 'test_result', 'test_turnaround_days', 'sample_status', 'encounter_date'] # Added encounter_date
    for col in test_cols_req: # Ensure these columns exist
        if col not in df.columns: df[col] = np.nan if col == 'test_turnaround_days' else ("Unknown" if col != 'encounter_date' else pd.NaT)
    if 'test_turnaround_days' in df.columns: df['test_turnaround_days'] = _convert_to_numeric(df['test_turnaround_days'], np.nan) # Ensure numeric

    conclusive_df = df[~df['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown', 'N/A', 'nan','Indeterminate']) & df['test_turnaround_days'].notna()].copy()
    all_proc_samples_df = df[~df['sample_status'].isin(['Pending', 'Unknown', 'N/A', 'nan'])].copy() # for rejection rate
    
    if not conclusive_df.empty and conclusive_df['test_turnaround_days'].notna().any(): summary["overall_avg_test_turnaround"] = conclusive_df['test_turnaround_days'].mean()
    
    crit_test_cfgs_clinic = {k: v for k,v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get("critical")}
    # Use the actual test_type values (keys) from the data that are configured as critical
    critical_test_keys_in_data_clinic = [k for k in crit_test_cfgs_clinic.keys() if k in df['test_type'].unique()] 
    
    critical_conclusive_df = conclusive_df[conclusive_df['test_type'].isin(critical_test_keys_in_data_clinic)].copy()
    if not critical_conclusive_df.empty:
        def _check_tat_met_overall_clinic_v2(r_crit): # Distinct name
            test_key_crit = r_crit['test_type'] # This is an original key
            test_config_crit = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(test_key_crit)
            target_tat_crit = test_config_crit['target_tat_days'] if test_config_crit and 'target_tat_days' in test_config_crit else app_config.TARGET_TEST_TURNAROUND_DAYS
            return r_crit['test_turnaround_days'] <= target_tat_crit
        critical_conclusive_df.loc[:, 'tat_met'] = critical_conclusive_df.apply(_check_tat_met_overall_clinic_v2, axis=1)
        if not critical_conclusive_df['tat_met'].empty : summary["overall_perc_met_tat"] = critical_conclusive_df['tat_met'].mean() * 100
    
    summary["total_pending_critical_tests"] = df[(df['test_type'].isin(critical_test_keys_in_data_clinic)) & (df['test_result'] == 'Pending')]['patient_id'].nunique()
    if not all_proc_samples_df.empty and len(all_proc_samples_df) > 0: summary["sample_rejection_rate"] = (all_proc_samples_df[all_proc_samples_df['sample_status'] == 'Rejected'].shape[0] / len(all_proc_samples_df)) * 100
    
    test_summary_details = {}
    for original_key_from_config, cfg_props in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items():
        display_name_for_ui = cfg_props.get("display_name", original_key_from_config)
        actual_test_keys_in_data_group = cfg_props.get("types_in_group", [original_key_from_config])
        if isinstance(actual_test_keys_in_data_group, str): actual_test_keys_in_data_group = [actual_test_keys_in_data_group]

        # THIS IS THE CORRECTED ASSIGNMENT:
        grp_df = df[df['test_type'].isin(actual_test_keys_in_data_group)]
        
        stats = {"positive_rate": 0.0, "avg_tat_days": np.nan, "perc_met_tat_target": 0.0, "pending_count": 0, "rejected_count": 0, "total_conducted_conclusive": 0}
        if grp_df.empty: test_summary_details[display_name_for_ui] = stats; continue

        grp_concl = grp_df[~grp_df['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown', 'N/A', 'nan', 'Indeterminate']) & grp_df['test_turnaround_days'].notna()].copy()
        stats["total_conducted_conclusive"] = len(grp_concl)

        if not grp_concl.empty:
            positive_cases_in_concl = grp_concl[grp_concl['test_result'] == 'Positive'].shape[0]
            stats["positive_rate"] = (positive_cases_in_concl / len(grp_concl)) * 100 if len(grp_concl) > 0 else 0.0 # Guarded division
            if grp_concl['test_turnaround_days'].notna().any(): stats["avg_tat_days"] = grp_concl['test_turnaround_days'].mean()
            target_tat_specific = cfg_props.get("target_tat_days", app_config.TARGET_TEST_TURNAROUND_DAYS)
            grp_concl.loc[:, 'tat_met_specific'] = grp_concl['test_turnaround_days'] <= target_tat_specific
            if not grp_concl['tat_met_specific'].empty: stats["perc_met_tat_target"] = grp_concl['tat_met_specific'].mean() * 100
        
        if 'test_result' in grp_df.columns: stats["pending_count"] = grp_df[grp_df['test_result'] == 'Pending']['patient_id'].nunique()
        if 'sample_status' in grp_df.columns: stats["rejected_count"] = grp_df[grp_df['sample_status'] == 'Rejected']['patient_id'].nunique()
        test_summary_details[display_name_for_ui] = stats
    summary["test_summary_details"] = test_summary_details
    
    if 'item' in df.columns and 'item_stock_agg_zone' in df.columns and 'consumption_rate_per_day' in df.columns and app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        key_drugs_df = df[df['item'].str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False)]
        if not key_drugs_df.empty:
            key_drugs_df['encounter_date'] = pd.to_datetime(key_drugs_df['encounter_date'], errors='coerce')
            key_drugs_df.dropna(subset=['encounter_date'], inplace=True)
            if not key_drugs_df.empty:
                latest_key_supply = key_drugs_df.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last')
                latest_key_supply['days_of_supply_calc'] = latest_key_supply['item_stock_agg_zone'] / (latest_key_supply['consumption_rate_per_day'].replace(0, np.nan))
                summary['key_drug_stockouts'] = latest_key_supply[latest_key_supply['days_of_supply_calc'] < app_config.CRITICAL_SUPPLY_DAYS]['item'].nunique()
    return summary

def get_clinic_environmental_summary(iot_df_period: pd.DataFrame) -> Dict[str, Any]:
    summary = { "avg_co2_overall": np.nan, "rooms_co2_alert_latest": 0, "avg_pm25_overall": np.nan, "rooms_pm25_alert_latest": 0, "avg_occupancy_overall": np.nan, "high_occupancy_alert_latest": False, "avg_noise_overall": np.nan, "rooms_noise_alert_latest": 0 }
    if iot_df_period is None or iot_df_period.empty: return summary
    if 'timestamp' not in iot_df_period.columns or not pd.api.types.is_datetime64_any_dtype(iot_df_period['timestamp']): return summary
    num_cols_iot = ['avg_co2_ppm', 'avg_pm25', 'waiting_room_occupancy', 'avg_noise_db']; df_iot = iot_df_period.copy()
    for col in num_cols_iot: df_iot[col] = _convert_to_numeric(df_iot[col], np.nan) if col in df_iot.columns else np.nan
    if df_iot['avg_co2_ppm'].notna().any(): summary["avg_co2_overall"] = df_iot['avg_co2_ppm'].mean()
    if df_iot['avg_pm25'].notna().any(): summary["avg_pm25_overall"] = df_iot['avg_pm25'].mean()
    if df_iot['waiting_room_occupancy'].notna().any(): summary["avg_occupancy_overall"] = df_iot['waiting_room_occupancy'].mean()
    if df_iot['avg_noise_db'].notna().any(): summary["avg_noise_overall"] = df_iot['avg_noise_db'].mean()
    key_cols_room_env = ['clinic_id', 'room_name']
    if all(c in df_iot.columns for c in key_cols_room_env):
        latest_reads = df_iot.sort_values('timestamp').drop_duplicates(subset=key_cols_room_env, keep='last')
        if not latest_reads.empty:
            if 'avg_co2_ppm' in latest_reads and latest_reads['avg_co2_ppm'].notna().any(): summary["rooms_co2_alert_latest"] = latest_reads[latest_reads['avg_co2_ppm'] > app_config.CO2_LEVEL_ALERT_PPM].shape[0]
            if 'avg_pm25' in latest_reads and latest_reads['avg_pm25'].notna().any(): summary["rooms_pm25_alert_latest"] = latest_reads[latest_reads['avg_pm25'] > app_config.PM25_ALERT_UGM3].shape[0]
            if 'waiting_room_occupancy' in latest_reads and latest_reads['waiting_room_occupancy'].notna().any(): summary["high_occupancy_alert_latest"] = (latest_reads['waiting_room_occupancy'] > app_config.TARGET_WAITING_ROOM_OCCUPANCY).any()
            if 'avg_noise_db' in latest_reads and latest_reads['avg_noise_db'].notna().any(): summary["rooms_noise_alert_latest"] = latest_reads[latest_reads['avg_noise_db'] > app_config.NOISE_LEVEL_ALERT_DB].shape[0]
    return summary

def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_THRESHOLDS['moderate']) -> pd.DataFrame:
    if health_df_period is None or health_df_period.empty: return pd.DataFrame()
    alerts_data = []; df_alerts = health_df_period.copy()
    alert_cols_clinic = ['patient_id', 'encounter_date', 'condition', 'ai_risk_score', 'ai_followup_priority_score', 'test_type', 'test_result', 'hiv_viral_load_copies_ml', 'sample_status', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius', 'referral_status', 'referral_reason']
    for col in alert_cols_clinic:
        if col not in df_alerts.columns: df_alerts[col] = np.nan if col in ['ai_risk_score', 'ai_followup_priority_score', 'hiv_viral_load_copies_ml', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius'] else (pd.NaT if col == 'encounter_date' else "Unknown")

    if df_alerts.get('ai_risk_score', pd.Series(dtype=float)).notna().any() or df_alerts.get('ai_followup_priority_score', pd.Series(dtype=float)).notna().any():
        for _, r in df_alerts[(df_alerts.get('ai_risk_score', pd.Series(0.0)) >= risk_threshold_moderate) | (df_alerts.get('ai_followup_priority_score', pd.Series(0.0)) >= 75)].iterrows(): alerts_data.append({**r.to_dict(), 'alert_reason': f"High AI Risk/Prio (Risk:{r.get('ai_risk_score',0):.0f}, Prio:{r.get('ai_followup_priority_score',0):.0f})", 'priority_score': max(r.get('ai_risk_score',0), r.get('ai_followup_priority_score',0))})
    
    # Use actual test_type keys from data (which should map to KEY_TEST_TYPES_FOR_ANALYSIS keys)
    crit_test_keys_alert_clinic = [k for k,v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get("critical") and k in df_alerts['test_type'].unique()]
    for _, r in df_alerts[(df_alerts['test_type'].isin(crit_test_keys_alert_clinic)) & (df_alerts['test_result'] == 'Positive')].iterrows(): alerts_data.append({**r.to_dict(), 'alert_reason': f"Critical Positive: {r['test_type']}", 'priority_score': 85})
    
    if 'hiv_viral_load_copies_ml' in df_alerts and df_alerts['hiv_viral_load_copies_ml'].notna().any():
        for _, r in df_alerts[df_alerts['hiv_viral_load_copies_ml'] > 1000].iterrows(): alerts_data.append({**r.to_dict(), 'alert_reason': f"High HIV Viral Load ({r['hiv_viral_load_copies_ml']:.0f})", 'priority_score': 90})
    if 'min_spo2_pct' in df_alerts and df_alerts['min_spo2_pct'].notna().any():
        for _, r in df_alerts[df_alerts['min_spo2_pct'] < app_config.SPO2_CRITICAL_THRESHOLD_PCT].iterrows(): alerts_data.append({**r.to_dict(), 'alert_reason': f"Critically Low SpO2 ({r['min_spo2_pct']:.0f}%)", 'priority_score': 92})
    
    date_col_for_overdue_calc_clinic = 'sample_collection_date' if 'sample_collection_date' in df_alerts.columns and df_alerts['sample_collection_date'].notna().any() else 'encounter_date'
    if date_col_for_overdue_calc_clinic in df_alerts.columns and df_alerts[date_col_for_overdue_calc_clinic].notna().any():
        pending_df_alerts_clinic = df_alerts[(df_alerts['test_result'] == 'Pending') & (df_alerts['test_type'].isin(crit_test_keys_alert_clinic)) & (df_alerts[date_col_for_overdue_calc_clinic].notna())].copy()
        if not pending_df_alerts_clinic.empty:
            pending_df_alerts_clinic.loc[:, 'days_pending'] = (pd.Timestamp('today').normalize() - pd.to_datetime(pending_df_alerts_clinic[date_col_for_overdue_calc_clinic], errors='coerce')).dt.days
            def get_overdue_thresh_alert_clinic(test_type_key_from_data_clinic):
                test_cfg_clinic = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(test_type_key_from_data_clinic)
                buffer_days = 2
                return (test_cfg_clinic['target_tat_days'] if test_cfg_clinic and 'target_tat_days' in test_cfg_clinic else app_config.OVERDUE_PENDING_TEST_DAYS) + buffer_days
            pending_df_alerts_clinic.loc[:, 'overdue_thresh_days'] = pending_df_alerts_clinic['test_type'].apply(get_overdue_thresh_alert_clinic)
            overdue_crit_alerts_clinic = pending_df_alerts_clinic[pending_df_alerts_clinic['days_pending'].notna() & (pending_df_alerts_clinic['days_pending'] > pending_df_alerts_clinic['overdue_thresh_days'])]
            for _, r in overdue_crit_alerts_clinic.iterrows(): alerts_data.append({**r.to_dict(), 'alert_reason': f"Overdue Pending: {r['test_type']} ({r.get('days_pending',0):.0f}d)", 'priority_score': 75 + min(r.get('days_pending',0) - r.get('overdue_thresh_days',0), 10)})
            
    if not alerts_data: return pd.DataFrame(columns=df_alerts.columns.tolist() + ['alert_reason', 'priority_score'])
    alerts_df_agg_clinic = pd.DataFrame(alerts_data)
    alerts_df_agg_clinic['alert_reason'] = alerts_df_agg_clinic['alert_reason'].astype(str).fillna("Unknown Alert")
    alerts_df_agg_clinic['encounter_date'] = pd.to_datetime(alerts_df_agg_clinic['encounter_date'], errors='coerce')
    alerts_df_agg_clinic.dropna(subset=['patient_id', 'encounter_date'], inplace=True)
    if alerts_df_agg_clinic.empty: return pd.DataFrame(columns=alerts_df_agg_clinic.columns)
    alerts_df_agg_clinic['encounter_date_obj'] = alerts_df_agg_clinic['encounter_date'].dt.date

    def aggregate_alerts_clinic_final_v2(group): # Refined aggregator
        first_row = group.iloc[0].copy() # Start with a copy of the first row of the group
        unique_reasons = sorted(list(pd.Series([str(r) for r in group['alert_reason'] if pd.notna(r)]).unique()))
        first_row['alert_reason'] = "; ".join(unique_reasons) if unique_reasons else "General Alert"
        first_row['priority_score'] = group['priority_score'].max() if pd.notna(group['priority_score']).any() else 0
        # Keep other columns from the first row if they aren't group keys
        for col in ['patient_id', 'encounter_date_obj']:
            if col in first_row:
                del first_row[col] # Remove grouping keys to avoid them being duplicated by apply output structure
        return first_row

    if not (alerts_df_agg_clinic['patient_id'].notna().all() and alerts_df_agg_clinic['encounter_date_obj'].notna().all()):
         logger.warning("NaNs found in grouping keys for patient alerts (clinic). Filtering them out.")
         alerts_df_agg_clinic.dropna(subset=['patient_id', 'encounter_date_obj'], inplace=True)
    if alerts_df_agg_clinic.empty: return pd.DataFrame(columns=alert_cols_clinic + ['alert_reason', 'priority_score'])
    
    final_alerts_df = alerts_df_agg_clinic.groupby(['patient_id', 'encounter_date_obj'], as_index=False).apply(aggregate_alerts_clinic_final_v2)
    
    if 'priority_score' in final_alerts_df.columns: final_alerts_df['priority_score'] = _convert_to_numeric(final_alerts_df['priority_score'],0).astype(int)
    else: final_alerts_df['priority_score'] = 0
    
    sort_date_col = 'encounter_date' if 'encounter_date' in final_alerts_df.columns and final_alerts_df['encounter_date'].notna().all() else 'encounter_date_obj'
    if sort_date_col not in final_alerts_df.columns : sort_date_col = None
        
    sort_cols = ['priority_score']
    sort_ascending = [False]
    if sort_date_col:
        final_alerts_df[sort_date_col] = pd.to_datetime(final_alerts_df[sort_date_col], errors='coerce') # Ensure datetime
        sort_cols.append(sort_date_col)
        sort_ascending.append(False)
    return final_alerts_df.sort_values(by=sort_cols, ascending=sort_ascending)

def get_district_summary_kpis(enriched_zone_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    kpis: Dict[str, Any] = { "total_population_district": 0, "avg_population_risk": np.nan, "zones_high_risk_count": 0, "overall_facility_coverage": np.nan, "district_tb_burden_total": 0, "district_malaria_burden_total": 0, "key_infection_prevalence_district_per_1000": np.nan, "population_weighted_avg_steps": np.nan, "avg_clinic_co2_district":np.nan }
    if enriched_zone_gdf is None or enriched_zone_gdf.empty: return kpis
    gdf = enriched_zone_gdf.copy()
    num_cols_dist_kpi = ['population', 'avg_risk_score', 'active_tb_cases', 'active_malaria_cases', 'total_active_key_infections', 'facility_coverage_score', 'avg_daily_steps_zone', 'zone_avg_co2']
    for col in num_cols_dist_kpi: gdf[col] = _convert_to_numeric(gdf[col], 0.0) if col in gdf.columns else 0.0
    kpis["total_population_district"] = gdf['population'].sum() if 'population' in gdf.columns else 0
    
    if kpis["total_population_district"] > 0 and pd.notna(kpis["total_population_district"]):
        if 'avg_risk_score' in gdf.columns and gdf['avg_risk_score'].notna().any() and 'population' in gdf.columns and gdf['population'].notna().any(): kpis["avg_population_risk"] = np.average(gdf['avg_risk_score'].dropna(), weights=gdf.loc[gdf['avg_risk_score'].notna(), 'population'])
        if 'facility_coverage_score' in gdf.columns and gdf['facility_coverage_score'].notna().any() and 'population' in gdf.columns and gdf['population'].notna().any(): kpis["overall_facility_coverage"] = np.average(gdf['facility_coverage_score'].dropna(), weights=gdf.loc[gdf['facility_coverage_score'].notna(), 'population'])
        if 'avg_daily_steps_zone' in gdf.columns and gdf['avg_daily_steps_zone'].notna().any() and 'population' in gdf.columns and gdf['population'].notna().any(): kpis["population_weighted_avg_steps"] = np.average(gdf['avg_daily_steps_zone'].dropna(), weights=gdf.loc[gdf['avg_daily_steps_zone'].notna(), 'population'])
        if 'total_active_key_infections' in gdf.columns: kpis["key_infection_prevalence_district_per_1000"] = (gdf['total_active_key_infections'].sum() / kpis["total_population_district"]) * 1000
    else:
        kpis["avg_population_risk"] = gdf['avg_risk_score'].mean() if not gdf.empty and 'avg_risk_score' in gdf.columns and gdf['avg_risk_score'].notna().any() else np.nan
        kpis["overall_facility_coverage"] = gdf['facility_coverage_score'].mean() if not gdf.empty and 'facility_coverage_score' in gdf.columns and gdf['facility_coverage_score'].notna().any() else np.nan
        kpis["population_weighted_avg_steps"] = gdf['avg_daily_steps_zone'].mean() if not gdf.empty and 'avg_daily_steps_zone' in gdf.columns and gdf['avg_daily_steps_zone'].notna().any() else np.nan
        kpis["key_infection_prevalence_district_per_1000"] = 0.0 # Or np.nan
    kpis["zones_high_risk_count"] = gdf[gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk']].shape[0] if 'avg_risk_score' in gdf.columns else 0
    kpis["district_tb_burden_total"] = int(gdf['active_tb_cases'].sum()) if 'active_tb_cases' in gdf.columns else 0
    kpis["district_malaria_burden_total"] = int(gdf['active_malaria_cases'].sum()) if 'active_malaria_cases' in gdf.columns else 0
    kpis["avg_clinic_co2_district"] = gdf['zone_avg_co2'].mean() if 'zone_avg_co2' in gdf and gdf['zone_avg_co2'].notna().any() else np.nan
    return kpis

def get_trend_data(df: pd.DataFrame, value_col: str, date_col: str = 'encounter_date', period: str = 'D', agg_func: str = 'mean', filter_col: Optional[str] = None, filter_val: Optional[Any] = None) -> pd.Series:
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns: return pd.Series(dtype='float64')
    trend_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(trend_df[date_col]): trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors='coerce')
    trend_df.dropna(subset=[date_col], inplace=True)
    if value_col not in trend_df.columns : return pd.Series(dtype='float64') # Ensure value_col exists after potential drops
    if agg_func != 'nunique': trend_df.dropna(subset=[value_col], inplace=True)
    if trend_df.empty: return pd.Series(dtype='float64')
    if filter_col and filter_col in trend_df.columns and filter_val is not None:
        trend_df = trend_df[trend_df[filter_col] == filter_val]
        if trend_df.empty: return pd.Series(dtype='float64')
    trend_df.set_index(date_col, inplace=True)
    if agg_func in ['mean', 'sum', 'median'] and not pd.api.types.is_numeric_dtype(trend_df[value_col]):
        trend_df[value_col] = _convert_to_numeric(trend_df[value_col], np.nan); trend_df.dropna(subset=[value_col], inplace=True)
        if trend_df.empty: return pd.Series(dtype='float64')
    try:
        resampled = trend_df.groupby(pd.Grouper(freq=period))
        if agg_func == 'nunique': trend_series = resampled[value_col].nunique()
        elif agg_func == 'sum': trend_series = resampled[value_col].sum()
        elif agg_func == 'median': trend_series = resampled[value_col].median()
        else: trend_series = resampled[value_col].mean()
    except Exception as e: logger.error(f"Trend error for {value_col} (agg {agg_func}): {e}", exc_info=True); return pd.Series(dtype='float64')
    return trend_series

def get_supply_forecast_data(health_df: pd.DataFrame, forecast_days_out: int = 30, item_filter_list: Optional[List[str]] = None) -> pd.DataFrame:
    default_cols = ['item','date','current_stock','consumption_rate','forecast_stock','forecast_days','estimated_stockout_date','lower_ci','upper_ci','initial_days_supply']
    if health_df is None or health_df.empty or not all(c in health_df.columns for c in ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']):
        return pd.DataFrame(columns=default_cols)
    
    health_df_copy = health_df.copy() # Work on a copy
    health_df_copy['encounter_date'] = pd.to_datetime(health_df_copy['encounter_date'], errors='coerce')
    health_df_copy.dropna(subset=['encounter_date'], inplace=True)
    if health_df_copy.empty: return pd.DataFrame(columns=default_cols)
        
    supply_status_df = health_df_copy.loc[health_df_copy.groupby('item')['encounter_date'].idxmax()]
    
    if item_filter_list: supply_status_df = supply_status_df[supply_status_df['item'].isin(item_filter_list)]
    if supply_status_df.empty: return pd.DataFrame(columns=default_cols)
    
    forecasts = []
    for _, row in supply_status_df.iterrows():
        item, cur_stock, cons_rate, last_date = row['item'], row['item_stock_agg_zone'], row['consumption_rate_per_day'], row['encounter_date'] # last_date is already datetime
        if pd.isna(cur_stock) or pd.isna(cons_rate) or pd.isna(last_date) or cur_stock < 0: continue
        
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days_out, freq='D')
        est_stockout_date = pd.NaT
        days_rem_start = (cur_stock / cons_rate) if cons_rate > 0.001 else (np.inf if cur_stock > 0 else 0)
        
        if cons_rate > 0.001:
            days_to_stockout_from_start = cur_stock / cons_rate
            est_stockout_date = last_date + pd.to_timedelta(days_to_stockout_from_start, unit='D')
        
        for i, fc_date in enumerate(forecast_dates):
            days_out = i + 1; current_forecast_stock = cur_stock - (cons_rate * days_out)
            days_of_supply_fc = (current_forecast_stock / cons_rate) if cons_rate > 0.001 else (np.inf if current_forecast_stock > 0 else 0)
            cons_std_factor = 0.15; lower_cons = cons_rate * (1+cons_std_factor); upper_cons = max(0.01, cons_rate * (1-cons_std_factor))
            lower_ci_stock = cur_stock - (lower_cons*days_out); upper_ci_stock = cur_stock - (upper_cons*days_out)
            lower_ci_days = (lower_ci_stock/lower_cons) if lower_cons > 0.001 else (np.inf if lower_ci_stock > 0 else 0)
            upper_ci_days = (upper_ci_stock/upper_cons) if upper_cons > 0.001 else (np.inf if upper_ci_stock > 0 else 0)
            
            forecasts.append({'item': item, 'date': fc_date, 'current_stock': cur_stock, 'consumption_rate': cons_rate, 
                              'forecast_stock': max(0,current_forecast_stock), 'forecast_days': max(0,days_of_supply_fc), 
                              'estimated_stockout_date': est_stockout_date, 
                              'lower_ci': max(0,lower_ci_days), 'upper_ci': max(0,upper_ci_days), 
                              'initial_days_supply': days_rem_start})
    if not forecasts: return pd.DataFrame(columns=default_cols)
    return pd.DataFrame(forecasts)
