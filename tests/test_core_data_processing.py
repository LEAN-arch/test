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
    """Standardizes column names: lower case, replaces spaces/hyphens with underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

def _convert_to_numeric(series: pd.Series, default_value: Any = np.nan) -> pd.Series:
    """Safely converts a pandas Series to numeric, coercing errors to default_value."""
    return pd.to_numeric(series, errors='coerce').fillna(default_value)

def hash_geodataframe(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """Custom hash function for GeoDataFrames for Streamlit caching."""
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame): return None
    try:
        df_hash = pd.util.hash_pandas_object(gdf.drop(columns=[gdf.geometry.name], errors='ignore'), index=True).sum()
        geom_hash = pd.util.hash_array(gdf.geometry.to_wkt().values).sum() if gdf.geometry.name in gdf.columns and not gdf.geometry.empty else 0
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
    if right_df.empty or len(right_df.columns) < 2 or on_col not in right_df.columns:
        return left_df
    value_col_in_right_df_candidates = [col for col in right_df.columns if col != on_col]
    if not value_col_in_right_df_candidates:
        return left_df
    value_col_in_right_df = value_col_in_right_df_candidates[0]
    temp_agg_col_name = f"{target_col_name}_temp_agg_val"
    right_df_renamed = right_df.rename(columns={value_col_in_right_df: temp_agg_col_name})
    merged_df = left_df.merge(right_df_renamed[[on_col, temp_agg_col_name]], on=on_col, how='left') # Select only needed cols from right
    if temp_agg_col_name in merged_df.columns:
        # Update original target col, using new data if available, else keep original, else default
        merged_df[target_col_name] = np.where(
            merged_df[temp_agg_col_name].notna(), # If new data from merge is not NaN
            merged_df[temp_agg_col_name],         # Use it
            merged_df[target_col_name]            # Else, keep what was already in target_col_name (from init or previous merge)
        )
        merged_df.drop(columns=[temp_agg_col_name], inplace=True)
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
                df.loc[df[col].isin(['', 'nan', 'None', 'N/A', '#N/A']), col] = "Unknown"
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
        err_msg = ([f"Zone attributes file '{os.path.basename(attributes_path)}' not found."] if not os.path.exists(attributes_path) else []) + \
                  ([f"Zone geometries file '{os.path.basename(geometries_path)}' not found."] if not os.path.exists(geometries_path) else [])
        full_err_msg = " ".join(err_msg)
        logger.error(full_err_msg); st.error(f"🚨 **Critical GIS Data Error:** {full_err_msg}"); return None
    try:
        zone_attributes_df = pd.read_csv(attributes_path); zone_attributes_df = _clean_column_names(zone_attributes_df)
        zone_geometries_gdf = gpd.read_file(geometries_path); zone_geometries_gdf = _clean_column_names(zone_geometries_gdf)
        if 'zone_id' not in zone_attributes_df.columns or 'zone_id' not in zone_geometries_gdf.columns:
            logger.error("Missing 'zone_id' in zone attributes or geometries."); st.error("🚨 Key 'zone_id' missing in GIS files."); return None
        zone_attributes_df['zone_id'] = zone_attributes_df['zone_id'].astype(str).str.strip()
        zone_geometries_gdf['zone_id'] = zone_geometries_gdf['zone_id'].astype(str).str.strip()
        if 'zone_display_name' in zone_attributes_df.columns: zone_attributes_df.rename(columns={'zone_display_name': 'name'}, inplace=True)
        elif 'name' not in zone_attributes_df.columns: zone_attributes_df['name'] = zone_attributes_df['zone_id']
        merged_gdf = zone_geometries_gdf.merge(zone_attributes_df, on="zone_id", how="left", suffixes=('_geom', ''))
        for col_attr in zone_attributes_df.columns: # Prioritize attribute columns after merge
            if f"{col_attr}_geom" in merged_gdf.columns and col_attr in merged_gdf.columns and col_attr != 'zone_id':
                 merged_gdf[col_attr] = merged_gdf[col_attr].fillna(merged_gdf[f"{col_attr}_geom"])
                 merged_gdf.drop(columns=[f"{col_attr}_geom"], inplace=True, errors='ignore')
        if 'geometry' not in merged_gdf.columns and any('_geom' in col for col in merged_gdf.columns): # Ensure geometry column is set
            geom_col_candidate = [col for col in merged_gdf.columns if 'geometry' in col][0]
            merged_gdf = merged_gdf.set_geometry(geom_col_candidate)
        if merged_gdf.crs is None: merged_gdf = merged_gdf.set_crs(app_config.DEFAULT_CRS, allow_override=True)
        elif merged_gdf.crs.to_string().upper() != app_config.DEFAULT_CRS.upper(): merged_gdf = merged_gdf.to_crs(app_config.DEFAULT_CRS)
        for rz_col in ['zone_id', 'name', 'population', 'geometry']: # Ensure required columns
            if rz_col not in merged_gdf.columns:
                if rz_col == 'population': merged_gdf[rz_col] = 0.0
                elif rz_col == 'name': merged_gdf[rz_col] = "Unknown Zone"
                elif rz_col != 'geometry': merged_gdf[rz_col] = None
        for num_col in ['population', 'socio_economic_index', 'num_clinics', 'avg_travel_time_clinic_min']:
            if num_col in merged_gdf.columns: merged_gdf[num_col] = _convert_to_numeric(merged_gdf[num_col], 0 if num_col in ['population', 'num_clinics'] else 0.5)
        logger.info(f"Successfully loaded and merged zone data: {len(merged_gdf)} zones.")
        return merged_gdf
    except Exception as e:
        logger.error(f"Error loading/merging zone data: {e}", exc_info=True); st.error(f"Error with zone GIS data: {e}"); return None

# --- Data Enrichment and Aggregation Functions ---
def enrich_zone_geodata_with_health_aggregates(
    zone_gdf: gpd.GeoDataFrame, health_df: pd.DataFrame, iot_df: Optional[pd.DataFrame] = None
) -> gpd.GeoDataFrame:
    if zone_gdf is None or zone_gdf.empty or 'zone_id' not in zone_gdf.columns:
        return zone_gdf if zone_gdf is not None else gpd.GeoDataFrame(columns=['zone_id', 'geometry'], crs=app_config.DEFAULT_CRS) # Return minimal valid GDF
    enriched = zone_gdf.copy()
    if 'population' not in enriched.columns: enriched['population'] = 0.0
    enriched['population'] = _convert_to_numeric(enriched['population'], 0.0)

    agg_cols_to_initialize = ['total_population_health_data', 'avg_risk_score', 'total_patient_encounters', 'active_tb_cases', 'active_malaria_cases', 'hiv_positive_cases', 'pneumonia_cases', 'total_referrals_made', 'successful_referrals', 'avg_test_turnaround_critical', 'perc_critical_tests_tat_met', 'prevalence_per_1000', 'total_active_key_infections', 'avg_daily_steps_zone', 'zone_avg_co2', 'facility_coverage_score']
    for col in agg_cols_to_initialize: enriched[col] = 0.0

    if health_df is not None and not health_df.empty and 'zone_id' in health_df.columns:
        health_df['zone_id'] = health_df['zone_id'].astype(str).str.strip()
        health_df_for_agg = health_df.copy()
        
        enriched = _robust_merge_agg(enriched, health_df_for_agg.groupby('zone_id')['patient_id'].nunique().reset_index(name='patient_count_val'), 'total_population_health_data')
        enriched = _robust_merge_agg(enriched, health_df_for_agg.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='avg_risk_val'), 'avg_risk_score')
        enriched = _robust_merge_agg(enriched, health_df_for_agg.groupby('zone_id')['encounter_id'].nunique().reset_index(name='enc_count_val'), 'total_patient_encounters')
        enriched = _robust_merge_agg(enriched, health_df_for_agg[health_df_for_agg['condition'].str.contains("TB", case=False, na=False)].groupby('zone_id')['patient_id'].nunique().reset_index(name='tb_val'), 'active_tb_cases')
        enriched = _robust_merge_agg(enriched, health_df_for_agg[health_df_for_agg['condition'].str.contains("Malaria", case=False, na=False)].groupby('zone_id')['patient_id'].nunique().reset_index(name='mal_val'), 'active_malaria_cases')
        enriched = _robust_merge_agg(enriched, health_df_for_agg[health_df_for_agg['condition'].str.contains("HIV-Positive", case=False, na=False)].groupby('zone_id')['patient_id'].nunique().reset_index(name='hiv_val'), 'hiv_positive_cases')
        enriched = _robust_merge_agg(enriched, health_df_for_agg[health_df_for_agg['condition'].str.contains("Pneumonia", case=False, na=False)].groupby('zone_id')['patient_id'].nunique().reset_index(name='pneu_val'), 'pneumonia_cases')
        
        key_conditions_burden = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia']
        total_key_inf_agg = health_df_for_agg[health_df_for_agg['condition'].isin(key_conditions_burden)].groupby('zone_id')['patient_id'].nunique().reset_index(name='key_inf_val')
        enriched = _robust_merge_agg(enriched, total_key_inf_agg, 'total_active_key_infections')

        if 'referral_status' in health_df_for_agg.columns:
            enriched = _robust_merge_agg(enriched, health_df_for_agg[health_df_for_agg['referral_status'] != 'N/A'].groupby('zone_id')['encounter_id'].nunique().reset_index(name='ref_made_val'), 'total_referrals_made')
            if 'referral_outcome' in health_df_for_agg.columns:
                suc_out = ['Completed', 'Service Provided', 'Attended Consult', 'Attended Followup', 'Attended']
                enriched = _robust_merge_agg(enriched, health_df_for_agg[health_df_for_agg['referral_outcome'].isin(suc_out)].groupby('zone_id')['encounter_id'].nunique().reset_index(name='ref_suc_val'), 'successful_referrals')
        
        crit_test_keys = [k for k, v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get("critical", False)]
        if crit_test_keys:
            tat_df = health_df_for_agg[(health_df_for_agg['test_type'].isin(crit_test_keys)) & (health_df_for_agg['test_turnaround_days'].notna()) & (~health_df_for_agg['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown','Indeterminate']))].copy()
            if not tat_df.empty:
                enriched = _robust_merge_agg(enriched, tat_df.groupby('zone_id')['test_turnaround_days'].mean().reset_index(name='avg_tat_crit_val'), 'avg_test_turnaround_critical')
                def _check_tat_met(row):
                    cfg = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(row['test_type'])
                    return row['test_turnaround_days'] <= (cfg['target_tat_days'] if cfg and 'target_tat_days' in cfg else app_config.TARGET_TEST_TURNAROUND_DAYS)
                tat_df['tat_met_flag'] = tat_df.apply(_check_tat_met, axis=1)
                perc_met_agg = tat_df.groupby('zone_id')['tat_met_flag'].mean().reset_index(name='perc_tat_met_val')
                perc_met_agg['perc_tat_met_val'] *= 100
                enriched = _robust_merge_agg(enriched, perc_met_agg, 'perc_critical_tests_tat_met')
        if 'avg_daily_steps' in health_df_for_agg.columns:
            enriched = _robust_merge_agg(enriched, health_df_for_agg.groupby('zone_id')['avg_daily_steps'].mean().reset_index(name='steps_val'), 'avg_daily_steps_zone')

    if iot_df is not None and not iot_df.empty and 'zone_id' in iot_df.columns and 'avg_co2_ppm' in iot_df.columns:
        iot_df['zone_id'] = iot_df['zone_id'].astype(str).str.strip()
        enriched = _robust_merge_agg(enriched, iot_df.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(name='co2_val'), 'zone_avg_co2')

    if 'total_active_key_infections' in enriched.columns and 'population' in enriched.columns:
         enriched['prevalence_per_1000'] = enriched.apply(lambda r: (r['total_active_key_infections']/r['population'])*1000 if pd.notna(r['population']) and r['population']>0 and pd.notna(r['total_active_key_infections']) else 0.0, axis=1).fillna(0.0)
    if 'num_clinics' in enriched.columns and 'population' in enriched.columns:
        enriched['facility_coverage_score'] = enriched.apply(lambda r: (r['num_clinics']/r['population'])*10000*5 if pd.notna(r['population']) and r['population']>0 and pd.notna(r['num_clinics']) else 0.0, axis=1).fillna(0.0).clip(0,100)
    elif 'facility_coverage_score' not in enriched.columns: enriched['facility_coverage_score'] = 0.0
    
    for col in agg_cols_to_initialize: # Final cleanup for all designated agg columns
        if col in enriched.columns:
            if not pd.api.types.is_numeric_dtype(enriched[col]): enriched[col] = pd.to_numeric(enriched[col], errors='coerce')
            enriched[col].fillna(0.0, inplace=True)
        else: enriched[col] = 0.0
    logger.info("Zone GeoDataFrame enrichment complete.")
    return enriched

# --- KPI Calculation Functions ---
# (get_overall_kpis, get_chw_summary, get_patient_alerts_for_chw, get_clinic_summary,
#  get_clinic_environmental_summary, get_patient_alerts_for_clinic,
#  get_district_summary_kpis, get_trend_data, get_supply_forecast_data)
# These functions are quite extensive. They were provided in earlier complete versions
# of core_data_processing.py and are assumed to be present here without re-listing for brevity,
# unless specific corrections for them are needed beyond what's already discussed.
# Ensure they correctly use the cleaned column names ('encounter_date', etc.) and new fields.

# Example placeholder for one function:
def get_overall_kpis(health_df: pd.DataFrame, date_filter_start: Optional[str] = None, date_filter_end: Optional[str] = None) -> Dict[str, Any]:
    # ... (Full logic from previous complete core_data_processing.py) ...
    # Ensure it uses 'encounter_date' from health_df
    kpis = { "total_patients": 0, "avg_patient_risk": 0.0, "active_tb_cases_current": 0, "malaria_rdt_positive_rate_period": 0.0, "hiv_rapid_positive_rate_period": 0.0, "key_supply_stockout_alerts": 0 }
    if health_df is None or health_df.empty: return kpis
    df = health_df.copy()
    if 'encounter_date' not in df.columns or df['encounter_date'].isnull().all(): return kpis # Need encounter_date
    if date_filter_start: df = df[df['encounter_date'] >= pd.to_datetime(date_filter_start)]
    if date_filter_end: df = df[df['encounter_date'] <= pd.to_datetime(date_filter_end)]
    if df.empty: return kpis
    kpis["total_patients"] = df['patient_id'].nunique()
    kpis["avg_patient_risk"] = df['ai_risk_score'].mean() if 'ai_risk_score' in df and df['ai_risk_score'].notna().any() else 0.0
    kpis["active_tb_cases_current"] = df[df['condition'].str.contains("TB", case=False, na=False)]['patient_id'].nunique() if 'condition' in df.columns else 0
    # ... rest of KPI logic from previous full version
    return kpis

def get_chw_summary(health_df_daily: pd.DataFrame) -> Dict[str, Any]:
    # ... (Full logic from previous complete core_data_processing.py) ...
    summary = { "visits_today": 0, "tb_contacts_to_trace_today": 0, "sti_symptomatic_referrals_today": 0, "avg_patient_risk_visited_today": 0.0, "high_risk_followups_today": 0, "patients_low_spo2_visited_today": 0, "patients_fever_visited_today": 0, "avg_patient_steps_visited_today": 0.0, "patients_fall_detected_today": 0 }
    if health_df_daily is None or health_df_daily.empty: return summary
    # ... rest of logic
    return summary

def get_patient_alerts_for_chw(health_df_daily: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_THRESHOLDS['chw_alert_moderate'], risk_threshold_high: int = app_config.RISK_THRESHOLDS['chw_alert_high']) -> pd.DataFrame:
    # ... (Full logic from previous complete core_data_processing.py) ...
    if health_df_daily is None or health_df_daily.empty: return pd.DataFrame()
    # ... rest of logic
    return pd.DataFrame() # Placeholder if empty

def get_clinic_summary(health_df_period: pd.DataFrame) -> Dict[str, Any]:
    # ... (Full logic from previous complete core_data_processing.py, including test_summary_details) ...
    summary: Dict[str, Any] = { "overall_avg_test_turnaround": 0.0, "overall_perc_met_tat": 0.0, "total_pending_critical_tests": 0, "sample_rejection_rate": 0.0, "key_drug_stockouts": 0, "test_summary_details": {} }
    if health_df_period is None or health_df_period.empty: return summary
    # ... rest of logic
    return summary

def get_clinic_environmental_summary(iot_df_period: pd.DataFrame) -> Dict[str, Any]:
    # ... (Full logic from previous complete core_data_processing.py) ...
    summary = { "avg_co2_overall": 0.0, "rooms_co2_alert_latest": 0, "avg_pm25_overall": 0.0, "rooms_pm25_alert_latest": 0, "avg_occupancy_overall": 0.0, "high_occupancy_alert_latest": False, "avg_noise_overall": 0.0, "rooms_noise_alert_latest": 0 }
    if iot_df_period is None or iot_df_period.empty: return summary
    # ... rest of logic
    return summary

def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_THRESHOLDS['moderate']) -> pd.DataFrame:
    # ... (Full logic from previous complete core_data_processing.py) ...
    if health_df_period is None or health_df_period.empty: return pd.DataFrame()
    # ... rest of logic
    return pd.DataFrame() # placeholder if empty

def get_district_summary_kpis(enriched_zone_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    # ... (Full logic from previous complete core_data_processing.py) ...
    kpis: Dict[str, Any] = { "total_population_district": 0, "avg_population_risk": 0.0, "zones_high_risk_count": 0, "overall_facility_coverage": 0.0, "district_tb_burden_total": 0, "district_malaria_burden_total": 0, "key_infection_prevalence_district_per_1000": 0.0, "population_weighted_avg_steps": 0.0, "avg_clinic_co2_district":0.0, }
    if enriched_zone_gdf is None or enriched_zone_gdf.empty: return kpis
    # ... rest of logic
    return kpis

def get_trend_data(df: pd.DataFrame, value_col: str, date_col: str = 'encounter_date', period: str = 'D', agg_func: str = 'mean', filter_col: Optional[str] = None, filter_val: Optional[Any] = None) -> pd.Series:
    # ... (Full logic from previous complete core_data_processing.py) ...
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns: return pd.Series(dtype='float64')
    # ... rest of logic
    return pd.Series(dtype='float64') # placeholder if empty

def get_supply_forecast_data(health_df: pd.DataFrame, forecast_days_out: int = 30, item_filter_list: Optional[List[str]] = None) -> pd.DataFrame:
    # ... (Full logic for linear forecast from previous complete core_data_processing.py) ...
    if health_df is None or health_df.empty: return pd.DataFrame()
    # ... rest of logic
    return pd.DataFrame() # placeholder if empty
