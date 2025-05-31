# test/utils/core_data_processing.py
import streamlit as st # Using Streamlit for caching and potential error messaging in a Streamlit context
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import logging
from config import app_config # app_config object is imported
from typing import List, Dict, Any, Optional, Tuple

# --- Logging Setup ---
# (Assuming logging is configured in app_home.py or elsewhere if this is run as part of a larger app)
# If running this module standalone for testing, you might need to init logging here.
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names: lower case, replaces spaces with underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

def _convert_to_numeric(series: pd.Series, default_value: Any = np.nan) -> pd.Series:
    """Safely converts a pandas Series to numeric, coercing errors to default_value."""
    return pd.to_numeric(series, errors='coerce').fillna(default_value)

def hash_geodataframe(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """
    Custom hash function for GeoDataFrames for Streamlit caching.
    Computes a hash based on the DataFrame's values and the WKT representation of geometries.
    """
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame):
        return None
    try:
        # Hash non-geometry data
        df_hash = pd.util.hash_pandas_object(gdf.drop(columns=[gdf.geometry.name], errors='ignore'), index=True).sum()
        # Hash geometry data (WKT representation)
        geom_hash = pd.util.hash_array(gdf.geometry.to_wkt().values).sum() if gdf.geometry.name in gdf.columns and not gdf.geometry.empty else 0
        return f"{df_hash}-{geom_hash}"
    except Exception as e:
        logger.warning(f"Could not hash GeoDataFrame: {e}")
        return None # Fallback if hashing fails

# --- Data Loading and Basic Cleaning Functions ---
@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading health records...")
def load_health_records(file_path: str = None) -> pd.DataFrame:
    """
    Loads, cleans, and preprocesses health records from the specified CSV file.
    Handles the expanded schema.
    """
    file_path = file_path or app_config.HEALTH_RECORDS_CSV
    logger.info(f"Attempting to load health records from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Health records file not found: {file_path}")
        st.error(f"🚨 **Critical Data Error:** Health records file '{os.path.basename(file_path)}' not found. Please check configuration and file location.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path, low_memory=False)
        df = _clean_column_names(df)
        logger.info(f"Successfully loaded {len(df)} records from {file_path}.")

        # Date Conversions (add more as needed from expanded schema)
        date_cols = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date',
                     'referral_date', 'referral_outcome_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Numeric Conversions (add more from expanded schema)
        numeric_cols = [
            'test_turnaround_days', 'quantity_dispensed', 'item_stock_agg_zone',
            'consumption_rate_per_day', 'ai_risk_score', 'ai_followup_priority_score',
            'vital_signs_bp_systolic', 'vital_signs_bp_diastolic', 'vital_signs_temperature_celsius',
            'min_spo2_pct', 'max_skin_temp_celsius', 'avg_spo2', 'avg_daily_steps',
            'resting_heart_rate', 'avg_hrv', 'avg_sleep_duration_hrs', 'sleep_score_pct',
            'stress_level_score', 'fall_detected_today', 'age', 'chw_visit', 'tb_contact_traced',
            'patient_latitude', 'patient_longitude'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = _convert_to_numeric(df[col])

        # Specific cleaning for hiv_viral_load_copies_ml
        if 'hiv_viral_load_copies_ml' in df.columns:
            # Handle cases like "<50" or "Target Not Detected" by coercing to a numeric value or NaN
            # For simplicity here, coercing non-numeric to NaN, specific clinical rules might apply.
            df['hiv_viral_load_copies_ml'] = pd.to_numeric(df['hiv_viral_load_copies_ml'], errors='coerce')


        # String/Object Column Cleaning (fill NaNs with "Unknown")
        string_like_cols = [
            'encounter_id', 'patient_id', 'encounter_type', 'condition', 'diagnosis_code_icd10',
            'test_type', 'test_result', 'item', 'zone_id', 'clinic_id', 'physician_id',
            'notes', 'patient_reported_symptoms', 'gender', 'screening_hpv_status',
            'key_chronic_conditions_summary', 'medication_adherence_self_report',
            'referral_status', 'referral_reason', 'referred_to_facility_id',
            'referral_outcome', 'sample_status', 'rejection_reason'
        ]
        for col in string_like_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str).str.strip()
                df.loc[df[col].isin(['', 'nan', 'None', 'N/A', '#N/A']), col] = "Unknown" # More robust unknown
            else:
                df[col] = "Unknown" # Add column if missing and fill

        # Ensure key columns exist even if empty in CSV
        required_cols = ['patient_id', 'encounter_date', 'condition', 'test_type', 'test_result', 'zone_id', 'ai_risk_score']
        for r_col in required_cols:
            if r_col not in df.columns:
                logger.warning(f"Required column '{r_col}' not found in health records. Adding empty series.")
                if 'date' in r_col: df[r_col] = pd.Series(dtype='datetime64[ns]')
                elif 'score' in r_col: df[r_col] = pd.Series(dtype='float64')
                else: df[r_col] = pd.Series(dtype='object')
        
        logger.info("Health records cleaning and type conversion complete.")
        return df

    except Exception as e:
        logger.error(f"Error loading or processing health records from {file_path}: {e}", exc_info=True)
        st.error(f"Failed to load or process health records: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading IoT environmental data...")
def load_iot_clinic_environment_data(file_path: str = None) -> pd.DataFrame:
    """Loads, cleans, and preprocesses IoT clinic environment data."""
    file_path = file_path or app_config.IOT_CLINIC_ENVIRONMENT_CSV
    logger.info(f"Attempting to load IoT data from: {file_path}")
    if not os.path.exists(file_path):
        logger.warning(f"IoT data file not found: {file_path}")
        # Not critical enough to stop app with st.error usually, dashboards can handle empty IoT
        st.info(f"ℹ️ IoT data file '{os.path.basename(file_path)}' not found. Environmental monitoring will be unavailable.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df = _clean_column_names(df) # standardizes names
        logger.info(f"Successfully loaded {len(df)} IoT records from {file_path}.")

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            logger.error("IoT data missing 'timestamp' column.")
            return pd.DataFrame() # Essential column

        numeric_iot_cols = [
            'avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index',
            'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
            'waiting_room_occupancy', 'patient_throughput_per_hour',
            'sanitizer_dispenses_per_hour'
        ]
        for col in numeric_iot_cols:
            if col in df.columns:
                df[col] = _convert_to_numeric(df[col])

        string_iot_cols = ['clinic_id', 'room_name', 'zone_id']
        for col in string_iot_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            else: # If base IoT data doesn't have zone_id but it's needed
                df[col] = "Unknown"
        
        logger.info("IoT data cleaning and type conversion complete.")
        return df
    except Exception as e:
        logger.error(f"Error loading IoT data from {file_path}: {e}", exc_info=True)
        st.warning(f"Could not load or process IoT data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, hash_funcs={gpd.GeoDataFrame: hash_geodataframe}, show_spinner="Loading zone geographic and attribute data...")
def load_zone_data(attributes_path: str = None, geometries_path: str = None) -> Optional[gpd.GeoDataFrame]:
    """Loads zone attributes and geometries, merges them, and sets CRS."""
    attributes_path = attributes_path or app_config.ZONE_ATTRIBUTES_CSV
    geometries_path = geometries_path or app_config.ZONE_GEOMETRIES_GEOJSON
    logger.info(f"Loading zone attributes from {attributes_path} and geometries from {geometries_path}")

    if not os.path.exists(attributes_path) or not os.path.exists(geometries_path):
        err_msg = []
        if not os.path.exists(attributes_path): err_msg.append(f"Zone attributes file '{os.path.basename(attributes_path)}' not found.")
        if not os.path.exists(geometries_path): err_msg.append(f"Zone geometries file '{os.path.basename(geometries_path)}' not found.")
        full_err_msg = " ".join(err_msg)
        logger.error(full_err_msg)
        st.error(f"🚨 **Critical GIS Data Error:** {full_err_msg} Check configuration and file locations.")
        return None # Cannot proceed without zone data for district dashboard
    try:
        zone_attributes_df = pd.read_csv(attributes_path)
        zone_attributes_df = _clean_column_names(zone_attributes_df)
        
        zone_geometries_gdf = gpd.read_file(geometries_path)
        zone_geometries_gdf = _clean_column_names(zone_geometries_gdf)

        if 'zone_id' not in zone_attributes_df.columns or 'zone_id' not in zone_geometries_gdf.columns:
            logger.error("Missing 'zone_id' in zone attributes or geometries. Cannot merge.")
            st.error("🚨 Key 'zone_id' missing in GIS data files. Merge failed.")
            return None

        # Standardize 'zone_id' before merge just in case
        zone_attributes_df['zone_id'] = zone_attributes_df['zone_id'].astype(str).str.strip()
        zone_geometries_gdf['zone_id'] = zone_geometries_gdf['zone_id'].astype(str).str.strip()
        
        # Use zone_display_name as primary 'name', fall back if not present
        if 'zone_display_name' in zone_attributes_df.columns:
            zone_attributes_df.rename(columns={'zone_display_name': 'name'}, inplace=True)
        elif 'name' not in zone_attributes_df.columns and 'zone_id' in zone_attributes_df.columns:
            zone_attributes_df['name'] = zone_attributes_df['zone_id'] # Fallback name
        
        # Perform the merge
        merged_gdf = zone_geometries_gdf.merge(zone_attributes_df, on="zone_id", how="left", suffixes=('_geom', ''))
        
        # Prioritize columns from attributes if suffixes were created
        for col_attr in zone_attributes_df.columns:
            if col_attr + "_geom" in merged_gdf.columns and col_attr in merged_gdf.columns and col_attr != 'zone_id':
                 merged_gdf[col_attr] = merged_gdf[col_attr].fillna(merged_gdf[col_attr + "_geom"]) # Prioritize non-geom suffixed column if geom also brought it
                 merged_gdf.drop(columns=[col_attr + "_geom"], inplace=True, errors='ignore')
            elif col_attr + "" in merged_gdf.columns and col_attr != 'zone_id': # Attribute's column
                pass # It's fine
        
        # Ensure geometry column is set and CRS
        if 'geometry' not in merged_gdf.columns and any('_geom' in col for col in merged_gdf.columns):
            geom_col_candidate = [col for col in merged_gdf.columns if 'geometry' in col][0]
            merged_gdf = merged_gdf.set_geometry(geom_col_candidate)

        if merged_gdf.crs is None:
            merged_gdf = merged_gdf.set_crs(app_config.DEFAULT_CRS, allow_override=True)
            logger.info(f"Set CRS to {app_config.DEFAULT_CRS} for merged zone data.")
        elif merged_gdf.crs.to_string().upper() != app_config.DEFAULT_CRS.upper():
            merged_gdf = merged_gdf.to_crs(app_config.DEFAULT_CRS)
            logger.info(f"Re-projected zone data to {app_config.DEFAULT_CRS}.")
            
        required_zone_cols = ['zone_id', 'name', 'population', 'geometry']
        for rz_col in required_zone_cols:
            if rz_col not in merged_gdf.columns:
                logger.warning(f"Required zone column '{rz_col}' is missing. GIS analyses might fail.")
                if rz_col == 'population': merged_gdf[rz_col] = 0
                elif rz_col == 'name': merged_gdf[rz_col] = "Unknown Zone"
                elif rz_col == 'geometry' and isinstance(merged_gdf, gpd.GeoDataFrame): pass # gpd takes care of it mostly
                else: merged_gdf[rz_col] = None

        if 'population' in merged_gdf.columns: merged_gdf['population'] = _convert_to_numeric(merged_gdf['population'], default_value=0)
        if 'socio_economic_index' in merged_gdf.columns: merged_gdf['socio_economic_index'] = _convert_to_numeric(merged_gdf['socio_economic_index'], default_value=0.5)
        if 'num_clinics' in merged_gdf.columns: merged_gdf['num_clinics'] = _convert_to_numeric(merged_gdf['num_clinics'], default_value=0)

        logger.info(f"Successfully loaded and merged zone data: {len(merged_gdf)} zones.")
        return merged_gdf

    except Exception as e:
        logger.error(f"Error loading or merging zone data: {e}", exc_info=True)
        st.error(f"Error with zone GIS data: {e}")
        return None


# --- Data Enrichment and Aggregation Functions ---
def enrich_zone_geodata_with_health_aggregates(
    zone_gdf: gpd.GeoDataFrame,
    health_df: pd.DataFrame,
    iot_df: Optional[pd.DataFrame] = None
) -> gpd.GeoDataFrame:
    """
    Enriches the zone GeoDataFrame with aggregated health and IoT metrics.
    This is a key function for the District Dashboard.
    """
    if zone_gdf is None or zone_gdf.empty or 'zone_id' not in zone_gdf.columns:
        logger.warning("enrich_zone_geodata: Base zone_gdf is invalid or missing 'zone_id'. Returning as is or empty.")
        return zone_gdf if zone_gdf is not None else gpd.GeoDataFrame()
    
    enriched = zone_gdf.copy()
    if 'population' not in enriched.columns: enriched['population'] = 0
    enriched['population'] = _convert_to_numeric(enriched['population'], 0)

    # Default columns for aggregation
    agg_cols_to_initialize = [
        'total_population_health_data', 'avg_risk_score', 'total_patient_encounters',
        'active_tb_cases', 'active_malaria_cases', 'hiv_positive_cases',
        'pneumonia_cases', 'total_referrals_made', 'successful_referrals',
        'avg_test_turnaround_critical', 'perc_critical_tests_tat_met',
        'prevalence_per_1000', 'total_active_key_infections',
        'avg_daily_steps_zone', 'zone_avg_co2', 'facility_coverage_score' # New coverage score
    ]
    for col in agg_cols_to_initialize:
        enriched[col] = 0.0 # Initialize with float for safe division later

    if health_df is not None and not health_df.empty and 'zone_id' in health_df.columns:
        health_df['zone_id'] = health_df['zone_id'].astype(str).str.strip()
        health_df_for_agg = health_df.copy()
        
        # Population with health data (using unique patients)
        pop_with_data = health_df_for_agg.groupby('zone_id')['patient_id'].nunique().reset_index(name='total_population_health_data')
        enriched = enriched.merge(pop_with_data, on='zone_id', how='left')
        enriched['total_population_health_data'].fillna(0, inplace=True)

        # Average AI Risk Score per zone
        avg_risk = health_df_for_agg.groupby('zone_id')['ai_risk_score'].mean().reset_index(name='avg_risk_score')
        enriched = enriched.merge(avg_risk, on='zone_id', how='left')

        # Total patient encounters
        encounters = health_df_for_agg.groupby('zone_id')['encounter_id'].nunique().reset_index(name='total_patient_encounters')
        enriched = enriched.merge(encounters, on='zone_id', how='left')

        # Specific disease counts (active cases, assumes current data represents active)
        # These are examples; definitions of "active" can be complex
        # Count unique patients per condition in each zone
        active_tb = health_df_for_agg[health_df_for_agg['condition'].str.contains("TB", case=False, na=False)].groupby('zone_id')['patient_id'].nunique().reset_index(name='active_tb_cases')
        enriched = enriched.merge(active_tb, on='zone_id', how='left')

        active_malaria = health_df_for_agg[health_df_for_agg['condition'].str.contains("Malaria", case=False, na=False)].groupby('zone_id')['patient_id'].nunique().reset_index(name='active_malaria_cases')
        enriched = enriched.merge(active_malaria, on='zone_id', how='left')
        
        hiv_positive = health_df_for_agg[health_df_for_agg['condition'].str.contains("HIV-Positive", case=False, na=False)].groupby('zone_id')['patient_id'].nunique().reset_index(name='hiv_positive_cases')
        enriched = enriched.merge(hiv_positive, on='zone_id', how='left')

        pneumonia_cases = health_df_for_agg[health_df_for_agg['condition'].str.contains("Pneumonia", case=False, na=False)].groupby('zone_id')['patient_id'].nunique().reset_index(name='pneumonia_cases')
        enriched = enriched.merge(pneumonia_cases, on='zone_id', how='left')
        
        # Key infections combined
        key_conditions_for_burden = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia'] #Subset from app_config.KEY_CONDITIONS_FOR_TRENDS perhaps
        key_infections_df = health_df_for_agg[health_df_for_agg['condition'].isin(key_conditions_for_burden)]
        total_key_inf = key_infections_df.groupby('zone_id')['patient_id'].nunique().reset_index(name='total_active_key_infections')
        enriched = enriched.merge(total_key_inf, on='zone_id', how='left')
        enriched['prevalence_per_1000'] = enriched.apply(lambda row: (row['total_active_key_infections'] / row['population']) * 1000 if row['population'] > 0 else 0, axis=1)


        # Referrals
        if 'referral_status' in health_df_for_agg.columns:
            referrals_made = health_df_for_agg[health_df_for_agg['referral_status'] != 'N/A'].groupby('zone_id')['encounter_id'].nunique().reset_index(name='total_referrals_made')
            enriched = enriched.merge(referrals_made, on='zone_id', how='left')
            
            if 'referral_outcome' in health_df_for_agg.columns:
                 # Example: define successful outcomes. This could be complex.
                successful_outcomes = ['Completed', 'Service Provided', 'Attended Consult', 'Attended Followup', 'Attended']
                successful_refs = health_df_for_agg[health_df_for_agg['referral_outcome'].isin(successful_outcomes)].groupby('zone_id')['encounter_id'].nunique().reset_index(name='successful_referrals')
                enriched = enriched.merge(successful_refs, on='zone_id', how='left')

        # Test Turnaround Time (TAT) - focusing on critical tests meeting target
        # Requires understanding which tests are critical from app_config
        critical_tests_cfg = {k: v for k,v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get("critical", False)}
        critical_test_names = list(critical_tests_cfg.keys())
        
        if not critical_test_names: logger.warning("No critical tests defined in app_config for TAT enrichment.")
        else:
            tat_df = health_df_for_agg[
                (health_df_for_agg['test_type'].isin(critical_test_names)) &
                (health_df_for_agg['test_turnaround_days'].notna()) &
                (~health_df_for_agg['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown']))
            ].copy()

            if not tat_df.empty:
                # Average TAT for critical tests
                avg_tat_critical = tat_df.groupby('zone_id')['test_turnaround_days'].mean().reset_index(name='avg_test_turnaround_critical')
                enriched = enriched.merge(avg_tat_critical, on='zone_id', how='left')
                
                # Percentage of critical tests meeting their specific TAT target
                def check_tat_met(row):
                    test_config = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(row['test_type'])
                    if test_config and 'target_tat_days' in test_config:
                        return row['test_turnaround_days'] <= test_config['target_tat_days']
                    return row['test_turnaround_days'] <= app_config.TARGET_TEST_TURNAROUND_DAYS # Fallback
                
                tat_df['tat_met'] = tat_df.apply(check_tat_met, axis=1)
                perc_tat_met_critical = tat_df.groupby('zone_id')['tat_met'].mean().reset_index(name='perc_critical_tests_tat_met')
                perc_tat_met_critical['perc_critical_tests_tat_met'] *= 100 # convert to percentage
                enriched = enriched.merge(perc_tat_met_critical, on='zone_id', how='left')


        # Wellness metrics like average daily steps
        if 'avg_daily_steps' in health_df_for_agg.columns:
            avg_steps = health_df_for_agg.groupby('zone_id')['avg_daily_steps'].mean().reset_index(name='avg_daily_steps_zone')
            enriched = enriched.merge(avg_steps, on='zone_id', how='left')

    # IoT Data Aggregation (e.g., average CO2 per zone)
    if iot_df is not None and not iot_df.empty and 'zone_id' in iot_df.columns and 'avg_co2_ppm' in iot_df.columns:
        iot_df['zone_id'] = iot_df['zone_id'].astype(str).str.strip()
        zone_co2 = iot_df.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(name='zone_avg_co2')
        enriched = enriched.merge(zone_co2, on='zone_id', how='left')
    
    # Placeholder for Facility Coverage Score calculation
    # This would require more complex logic, potentially relating num_clinics, population, travel time, etc.
    # Example: (num_clinics_per_10k_pop_score * 0.4) + (travel_time_score * 0.3) + (socio_economic_score * 0.3)
    # For now, a simplified version or default:
    if 'num_clinics' in enriched.columns:
        enriched['facility_coverage_score'] = enriched.apply(
            lambda row: (row['num_clinics'] / row['population']) * 10000 * 5 if row['population'] > 0 else 0, axis=1
        ) # Very basic: #clinics per 10k pop * 5 (max score 100 if 2 clinics per 1k)
        enriched['facility_coverage_score'] = enriched['facility_coverage_score'].clip(0, 100)


    # Fill NaNs for all newly created aggregate columns with 0
    for col in agg_cols_to_initialize: # Iterate through the master list of columns that were merged
        if col in enriched.columns:
            enriched[col].fillna(0, inplace=True)
        else: # If a merge didn't happen for some reason (e.g. source data for that metric was empty)
            enriched[col] = 0.0
            
    logger.info("Zone GeoDataFrame enrichment complete.")
    return enriched


# --- KPI Calculation Functions (for different dashboards) ---

def get_overall_kpis(health_df: pd.DataFrame, date_filter_start: Optional[str] = None, date_filter_end: Optional[str] = None) -> Dict[str, Any]:
    """Calculates overall KPIs for the home page or general overview."""
    kpis = {
        "total_patients": 0, "avg_patient_risk": 0.0, "active_tb_cases_current": 0,
        "malaria_rdt_positive_rate_period": 0.0, "hiv_rapid_positive_rate_period": 0.0,
        "key_supply_stockout_alerts": 0
    }
    if health_df is None or health_df.empty: return kpis

    df = health_df.copy()
    if date_filter_start: df = df[df['encounter_date'] >= pd.to_datetime(date_filter_start)]
    if date_filter_end: df = df[df['encounter_date'] <= pd.to_datetime(date_filter_end)]
    if df.empty: return kpis

    kpis["total_patients"] = df['patient_id'].nunique()
    kpis["avg_patient_risk"] = df['ai_risk_score'].mean() if 'ai_risk_score' in df and df['ai_risk_score'].notna().any() else 0.0
    
    # Active TB cases (unique patients with TB condition in period)
    tb_df = df[df['condition'].str.contains("TB", case=False, na=False)]
    kpis["active_tb_cases_current"] = tb_df['patient_id'].nunique()

    # Malaria RDT positivity
    malaria_rdt_df = df[(df['test_type'] == app_config.KEY_TEST_TYPES_FOR_ANALYSIS['RDT-Malaria']['display_name']) & (~df['test_result'].isin(["Pending", "Rejected Sample", "Unknown"]))]
    if not malaria_rdt_df.empty:
        pos_malaria_rdt = malaria_rdt_df[malaria_rdt_df['test_result'] == 'Positive'].shape[0]
        kpis["malaria_rdt_positive_rate_period"] = (pos_malaria_rdt / len(malaria_rdt_df)) * 100 if len(malaria_rdt_df) > 0 else 0.0

    # HIV Rapid Test Positivity
    hiv_rapid_df = df[(df['test_type'] == app_config.KEY_TEST_TYPES_FOR_ANALYSIS['HIV-Rapid']['display_name']) & (~df['test_result'].isin(["Pending", "Rejected Sample", "Unknown"]))]
    if not hiv_rapid_df.empty:
        pos_hiv_rapid = hiv_rapid_df[hiv_rapid_df['test_result'] == 'Positive'].shape[0]
        kpis["hiv_rapid_positive_rate_period"] = (pos_hiv_rapid / len(hiv_rapid_df)) * 100 if len(hiv_rapid_df) > 0 else 0.0
    
    # Stockouts (Simplified - sum of items low on stock_agg_zone if it's per item)
    # A true stockout calculation would need per-item, per-location daily stock. This is illustrative.
    if 'item' in df.columns and 'item_stock_agg_zone' in df.columns and 'consumption_rate_per_day' in df.columns:
        supply_df = df.drop_duplicates(subset=['item', 'zone_id'], keep='last') # Latest stock status per item per zone
        supply_df['days_of_supply'] = supply_df['item_stock_agg_zone'] / (supply_df['consumption_rate_per_day'].replace(0, np.nan))
        kpis['key_supply_stockout_alerts'] = supply_df[supply_df['days_of_supply'] < app_config.CRITICAL_SUPPLY_DAYS]['item'].nunique()
    
    return kpis


def get_chw_summary(health_df_daily: pd.DataFrame) -> Dict[str, Any]:
    """Calculates summary KPIs for a CHW for a specific day. Assumes health_df_daily is pre-filtered."""
    summary = {
        "visits_today": 0, "tb_contacts_to_trace_today": 0,
        "sti_symptomatic_referrals_today": 0, "avg_patient_risk_visited_today": 0.0,
        "high_risk_followups_today": 0, "patients_low_spo2_visited_today": 0,
        "patients_fever_visited_today": 0, "avg_patient_steps_visited_today": 0.0,
        "patients_fall_detected_today": 0
    }
    if health_df_daily is None or health_df_daily.empty: return summary
    
    # Filter for actual CHW visits/encounters (encounter_type or chw_visit flag)
    chw_encounters_df = health_df_daily.copy() # Assuming already filtered if this is a daily view post-CHW selection
    if 'chw_visit' in chw_encounters_df.columns and chw_encounters_df['chw_visit'].sum() > 0 : # Use explicit chw_visit if present
        chw_encounters_df = chw_encounters_df[chw_encounters_df['chw_visit'] == 1]
    elif 'encounter_type' in chw_encounters_df.columns and chw_encounters_df['encounter_type'].str.contains("CHW", case=False, na=False).any():
        chw_encounters_df = chw_encounters_df[chw_encounters_df['encounter_type'].str.contains("CHW", case=False, na=False)]
    # If neither above, assume all records are CHW relevant if this function is called specifically for CHW dash.
    if chw_encounters_df.empty: return summary

    summary["visits_today"] = chw_encounters_df['patient_id'].nunique() # Unique patients visited
    
    # Example: TB contacts traced/to trace
    # This logic would need specific fields like 'is_tb_contact', 'contact_tracing_status'
    # Placeholder: count referrals for TB investigation
    if 'condition' in chw_encounters_df.columns and 'referral_reason' in chw_encounters_df.columns:
        summary["tb_contacts_to_trace_today"] = chw_encounters_df[
            (chw_encounters_df['condition'] == 'TB') & 
            (chw_encounters_df['referral_reason'].str.contains("Contact Tracing|Investigation", case=False, na=False)) &
            (chw_encounters_df['referral_status'] == 'Pending')
        ]['patient_id'].nunique()

        summary["sti_symptomatic_referrals_today"] = chw_encounters_df[
             (chw_encounters_df['condition'].str.contains("STI", case=False, na=False)) & 
             (chw_encounters_df['patient_reported_symptoms'] != "Unknown") & # Assuming symptoms mean referral needed
             (chw_encounters_df['referral_status'] == 'Pending')
        ]['patient_id'].nunique()

    if 'ai_risk_score' in chw_encounters_df.columns and chw_encounters_df['ai_risk_score'].notna().any():
        summary["avg_patient_risk_visited_today"] = chw_encounters_df['ai_risk_score'].mean()
        summary["high_risk_followups_today"] = chw_encounters_df[
            chw_encounters_df['ai_risk_score'] >= app_config.RISK_THRESHOLDS.get('high', 75)
        ]['patient_id'].nunique()

    if 'min_spo2_pct' in chw_encounters_df.columns:
        summary["patients_low_spo2_visited_today"] = chw_encounters_df[chw_encounters_df['min_spo2_pct'] < app_config.SPO2_LOW_THRESHOLD_PCT]['patient_id'].nunique()
    
    # Use vital_signs_temperature_celsius if available, else max_skin_temp_celsius
    temp_col_chw = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in chw_encounters_df.columns and chw_encounters_df['vital_signs_temperature_celsius'].notna().any() else 'max_skin_temp_celsius'
    if temp_col_chw in chw_encounters_df.columns:
        summary["patients_fever_visited_today"] = chw_encounters_df[chw_encounters_df[temp_col_chw] >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C]['patient_id'].nunique()

    if 'avg_daily_steps' in chw_encounters_df.columns and chw_encounters_df['avg_daily_steps'].notna().any():
        summary["avg_patient_steps_visited_today"] = chw_encounters_df['avg_daily_steps'].mean()
    
    if 'fall_detected_today' in chw_encounters_df.columns:
        summary["patients_fall_detected_today"] = chw_encounters_df[chw_encounters_df['fall_detected_today'] > 0]['patient_id'].nunique()
        
    return summary

def get_patient_alerts_for_chw(health_df_daily: pd.DataFrame,
                               risk_threshold_moderate: int = app_config.RISK_THRESHOLDS['chw_alert_moderate'],
                               risk_threshold_high: int = app_config.RISK_THRESHOLDS['chw_alert_high']) -> pd.DataFrame:
    """Generates a DataFrame of prioritized patient alerts for CHWs for a specific day."""
    if health_df_daily is None or health_df_daily.empty: return pd.DataFrame()
    
    alerts = []
    # Ensure working with a copy and that essential columns exist
    df_for_alerts = health_df_daily.copy()
    for col in ['patient_id', 'ai_risk_score', 'ai_followup_priority_score', 'min_spo2_pct', 'max_skin_temp_celsius', 'condition', 'referral_status', 'fall_detected_today']:
        if col not in df_for_alerts.columns: df_for_alerts[col] = np.nan # or "Unknown" for strings

    # Priority 1: High AI Follow-up Priority Score (if available and used)
    high_ai_priority = df_for_alerts[df_for_alerts['ai_followup_priority_score'] >= 80] # Example threshold
    for _, row in high_ai_priority.iterrows():
        alerts.append({**row.to_dict(), 'alert_reason': "High AI Follow-up Priority", 'priority_score': row['ai_followup_priority_score']})

    # Priority 2: Critical Vitals (Low SpO2, High Fever, Falls)
    critical_spo2 = df_for_alerts[df_for_alerts['min_spo2_pct'] < app_config.SPO2_CRITICAL_THRESHOLD_PCT]
    for _, row in critical_spo2.iterrows():
        alerts.append({**row.to_dict(), 'alert_reason': f"Critical SpO2 ({row['min_spo2_pct']}%)", 'priority_score': 90 + (app_config.SPO2_CRITICAL_THRESHOLD_PCT - row['min_spo2_pct'])})

    fever_col_chw_alert = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in df_for_alerts.columns and df_for_alerts['vital_signs_temperature_celsius'].notna().any() else 'max_skin_temp_celsius'
    high_fever = df_for_alerts[df_for_alerts[fever_col_chw_alert] >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C + 1.0] # e.g., >39C
    for _, row in high_fever.iterrows():
        alerts.append({**row.to_dict(), 'alert_reason': f"High Fever ({row[fever_col_chw_alert]}°C)", 'priority_score': 85 + (row[fever_col_chw_alert] - (app_config.SKIN_TEMP_FEVER_THRESHOLD_C + 1.0))})

    falls = df_for_alerts[df_for_alerts['fall_detected_today'] > 0]
    for _, row in falls.iterrows():
        alerts.append({**row.to_dict(), 'alert_reason': "Fall Detected", 'priority_score': 88})
        
    # Priority 3: High AI Risk Score (if not covered by ai_followup_priority_score)
    high_risk_general = df_for_alerts[df_for_alerts['ai_risk_score'] >= risk_threshold_high]
    for _, row in high_risk_general.iterrows():
        alerts.append({**row.to_dict(), 'alert_reason': "High General AI Risk", 'priority_score': row['ai_risk_score']})

    # Priority 4: Pending Referrals for Critical Conditions
    critical_cond_pending_referral = df_for_alerts[
        (df_for_alerts['condition'].isin(app_config.KEY_CONDITIONS_FOR_TRENDS[:4])) & # Example: TB, Malaria, HIV, Pneumonia
        (df_for_alerts['referral_status'] == 'Pending')
    ]
    for _, row in critical_cond_pending_referral.iterrows():
        alerts.append({**row.to_dict(), 'alert_reason': f"Pending Referral for {row['condition']}", 'priority_score': 70})

    # Priority 5: Moderate AI Risk Score
    moderate_risk = df_for_alerts[
        (df_for_alerts['ai_risk_score'] >= risk_threshold_moderate) &
        (df_for_alerts['ai_risk_score'] < risk_threshold_high)
    ]
    for _, row in moderate_risk.iterrows():
        alerts.append({**row.to_dict(), 'alert_reason': "Moderate AI Risk", 'priority_score': row['ai_risk_score']})
        
    if not alerts: return pd.DataFrame()
    alert_df = pd.DataFrame(alerts).drop_duplicates(subset=['patient_id', 'alert_reason']) # Avoid duplicate alerts for same patient-reason on same day
    alert_df['priority_score'] = alert_df['priority_score'].fillna(0).astype(int)
    return alert_df.sort_values(by='priority_score', ascending=False)


def get_clinic_summary(health_df_period: pd.DataFrame) -> Dict[str, Any]:
    """Calculates summary KPIs for a clinic over a defined period."""
    summary: Dict[str, Any] = {
        "overall_avg_test_turnaround": 0.0, "overall_perc_met_tat": 0.0,
        "total_pending_critical_tests": 0, "sample_rejection_rate": 0.0,
        "key_drug_stockouts": 0, "test_summary_details": {} # New: detailed per-test-group stats
    }
    if health_df_period is None or health_df_period.empty: return summary
    
    df = health_df_period.copy()
    # Ensure required columns exist for calculations
    test_cols_req = ['test_type', 'test_result', 'test_turnaround_days', 'sample_status', 'rejection_reason']
    for col in test_cols_req:
        if col not in df.columns: 
            logger.warning(f"Clinic Summary: Missing essential test column '{col}'. Test analytics will be affected.")
            if 'rate' in col or 'turnaround' in col: df[col] = np.nan
            else: df[col] = "Unknown"
            
    # Filter out tests that are not conclusive for TAT and positivity calculations
    conclusive_tests_df = df[~df['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown', 'N/A', 'nan', 'Indeterminate']) & df['test_turnaround_days'].notna()]
    all_processed_samples_df = df[~df['sample_status'].isin(['Pending', 'Unknown', 'N/A', 'nan'])] # For rejection rate denominator

    # --- Overall Test Performance ---
    if not conclusive_tests_df.empty:
        summary["overall_avg_test_turnaround"] = conclusive_tests_df['test_turnaround_days'].mean()
    
    # Critical tests TAT met
    critical_tests_config = {name: props for name, props in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if props.get("critical", False)}
    critical_test_display_names = [props['display_name'] for props in critical_tests_config.values()]
    
    critical_conclusive_df = conclusive_tests_df[conclusive_tests_df['test_type'].isin(critical_test_display_names)].copy()
    if not critical_conclusive_df.empty:
        def check_tat_met_overall(row):
            original_test_key = next((key for key, cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if cfg['display_name'] == row['test_type']), None)
            if original_test_key and 'target_tat_days' in app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_test_key]:
                return row['test_turnaround_days'] <= app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_test_key]['target_tat_days']
            return row['test_turnaround_days'] <= app_config.TARGET_TEST_TURNAROUND_DAYS # Global default if specific not found
        
        critical_conclusive_df['tat_met'] = critical_conclusive_df.apply(check_tat_met_overall, axis=1)
        summary["overall_perc_met_tat"] = (critical_conclusive_df['tat_met'].mean() * 100) if not critical_conclusive_df.empty else 0.0

    # Pending Critical Tests
    pending_critical_df = df[(df['test_type'].isin(critical_test_display_names)) & (df['test_result'] == 'Pending')]
    summary["total_pending_critical_tests"] = pending_critical_df['patient_id'].nunique() # Count unique patients with pending critical tests

    # Sample Rejection Rate
    if not all_processed_samples_df.empty and 'sample_status' in all_processed_samples_df.columns:
        rejected_count = all_processed_samples_df[all_processed_samples_df['sample_status'] == 'Rejected'].shape[0]
        summary["sample_rejection_rate"] = (rejected_count / len(all_processed_samples_df)) * 100 if len(all_processed_samples_df) > 0 else 0.0

    # --- Detailed Per-Test-Group Summary ---
    test_summary_details = {}
    for original_key, config_props in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items():
        display_name = config_props.get("display_name", original_key)
        # Actual test types this display_name might group (could be itself or a list from a new 'types_in_group' key in config)
        actual_test_types_in_group = config_props.get("types_in_group", [original_key]) 
        if isinstance(actual_test_types_in_group, str) : actual_test_types_in_group = [actual_test_types_in_group]


        group_df = df[df['test_type'].isin(actual_test_types_in_group)] # Use actual_test_types from our mapping if it's a group
        if group_df.empty: continue

        group_stats = {}
        group_conclusive = group_df[~group_df['test_result'].isin(['Pending', 'Rejected Sample', 'Unknown', 'N/A', 'nan','Indeterminate']) & group_df['test_turnaround_days'].notna()]
        
        if not group_conclusive.empty:
            group_stats["positive_rate"] = (group_conclusive[group_conclusive['test_result'] == 'Positive'].shape[0] / len(group_conclusive)) * 100 if len(group_conclusive) > 0 else 0.0
            group_stats["avg_tat_days"] = group_conclusive['test_turnaround_days'].mean()
            
            target_tat_specific = config_props.get("target_tat_days", app_config.TARGET_TEST_TURNAROUND_DAYS)
            group_conclusive_copy = group_conclusive.copy() # Avoid SettingWithCopyWarning
            group_conclusive_copy.loc[:, 'tat_met_specific'] = group_conclusive_copy['test_turnaround_days'] <= target_tat_specific
            group_stats["perc_met_tat_target"] = group_conclusive_copy['tat_met_specific'].mean() * 100

        group_stats["pending_count"] = group_df[group_df['test_result'] == 'Pending']['patient_id'].nunique()
        group_stats["rejected_count"] = group_df[group_df['sample_status'] == 'Rejected']['patient_id'].nunique() # Unique patients with rejected samples for this test type
        group_stats["total_conducted_conclusive"] = len(group_conclusive)
        
        test_summary_details[display_name] = group_stats
    summary["test_summary_details"] = test_summary_details
    
    # Key Drug Stockouts (using similar logic to get_overall_kpis for consistency, applied to current period's df)
    if 'item' in df.columns and 'item_stock_agg_zone' in df.columns and 'consumption_rate_per_day' in df.columns and app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        key_drugs_df = df[df['item'].str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False)]
        if not key_drugs_df.empty:
            latest_key_supply_df = key_drugs_df.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last') # latest status in period
            latest_key_supply_df['days_of_supply_calc'] = latest_key_supply_df['item_stock_agg_zone'] / (latest_key_supply_df['consumption_rate_per_day'].replace(0, np.nan))
            summary['key_drug_stockouts'] = latest_key_supply_df[latest_key_supply_df['days_of_supply_calc'] < app_config.CRITICAL_SUPPLY_DAYS]['item'].nunique()
            
    return summary


def get_clinic_environmental_summary(iot_df_period: pd.DataFrame) -> Dict[str, Any]:
    """Calculates summary environmental KPIs for a clinic over a period."""
    summary = {
        "avg_co2_overall": 0.0, "rooms_co2_alert_latest": 0,
        "avg_pm25_overall": 0.0, "rooms_pm25_alert_latest": 0,
        "avg_occupancy_overall": 0.0, "high_occupancy_alert_latest": False,
        "avg_noise_overall": 0.0, "rooms_noise_alert_latest": 0
    }
    if iot_df_period is None or iot_df_period.empty: return summary
    
    # Ensure 'timestamp' and other key numeric columns exist and are numeric
    if 'timestamp' not in iot_df_period.columns or not pd.api.types.is_datetime64_any_dtype(iot_df_period['timestamp']):
        logger.warning("Clinic Env Summary: Timestamp invalid or missing. Cannot compute time-sensitive alerts.")
        # Return early or with only averages if time based alerting is key
        return summary 
    
    num_cols = ['avg_co2_ppm', 'avg_pm25', 'waiting_room_occupancy', 'avg_noise_db']
    for col in num_cols: 
        if col not in iot_df_period.columns: iot_df_period[col] = np.nan
        else: iot_df_period[col] = _convert_to_numeric(iot_df_period[col], np.nan)
        
    if iot_df_period['avg_co2_ppm'].notna().any(): summary["avg_co2_overall"] = iot_df_period['avg_co2_ppm'].mean()
    if iot_df_period['avg_pm25'].notna().any(): summary["avg_pm25_overall"] = iot_df_period['avg_pm25'].mean()
    if iot_df_period['waiting_room_occupancy'].notna().any(): summary["avg_occupancy_overall"] = iot_df_period['waiting_room_occupancy'].mean()
    if iot_df_period['avg_noise_db'].notna().any(): summary["avg_noise_overall"] = iot_df_period['avg_noise_db'].mean()

    # Latest readings per room for alerts (assuming data has 'room_name' and 'clinic_id')
    key_cols_room = ['clinic_id', 'room_name']
    if all(c in iot_df_period.columns for c in key_cols_room):
        latest_readings_per_room = iot_df_period.sort_values('timestamp').drop_duplicates(subset=key_cols_room, keep='last')
        if not latest_readings_per_room.empty:
            if 'avg_co2_ppm' in latest_readings_per_room:
                 summary["rooms_co2_alert_latest"] = latest_readings_per_room[latest_readings_per_room['avg_co2_ppm'] > app_config.CO2_LEVEL_ALERT_PPM].shape[0]
            if 'avg_pm25' in latest_readings_per_room:
                summary["rooms_pm25_alert_latest"] = latest_readings_per_room[latest_readings_per_room['avg_pm25'] > app_config.PM25_ALERT_UGM3].shape[0]
            if 'waiting_room_occupancy' in latest_readings_per_room:
                summary["high_occupancy_alert_latest"] = (latest_readings_per_room['waiting_room_occupancy'] > app_config.TARGET_WAITING_ROOM_OCCUPANCY).any()
            if 'avg_noise_db' in latest_readings_per_room:
                summary["rooms_noise_alert_latest"] = latest_readings_per_room[latest_readings_per_room['avg_noise_db'] > app_config.NOISE_LEVEL_ALERT_DB].shape[0]
    return summary


def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_THRESHOLDS['moderate']) -> pd.DataFrame:
    """Identifies high-risk patients for clinical review based on various factors."""
    if health_df_period is None or health_df_period.empty: return pd.DataFrame()
    alerts_data = []
    df_for_alerts = health_df_period.copy()
    
    # Ensure necessary columns exist
    clinic_alert_cols = ['patient_id', 'encounter_date', 'condition', 'ai_risk_score', 
                         'test_type', 'test_result', 'hiv_viral_load_copies_ml', 
                         'sample_status', 'rejection_reason', 'min_spo2_pct',
                         'ai_followup_priority_score'] # using this too
    for col in clinic_alert_cols:
        if col not in df_for_alerts.columns:
            df_for_alerts[col] = np.nan if 'score' in col or 'load' in col or 'spo2' in col else "Unknown"

    # 1. High AI Risk Score or High AI Followup Priority Score
    high_risk_ai_df = df_for_alerts[(df_for_alerts['ai_risk_score'] >= risk_threshold_moderate) | (df_for_alerts['ai_followup_priority_score'] >= 75)] # example threshold for followup
    for _, row in high_risk_ai_df.iterrows():
        reason = "High AI Risk"
        if pd.notna(row['ai_followup_priority_score']) and row['ai_followup_priority_score'] >= 75:
            reason = f"High AI Follow-up Priority ({row['ai_followup_priority_score']:.0f})"
        elif pd.notna(row['ai_risk_score']):
             reason = f"High AI Risk ({row['ai_risk_score']:.0f})"
        alerts_data.append({**row.to_dict(), 'alert_reason': reason, 'priority_score': max(row.get('ai_risk_score',0), row.get('ai_followup_priority_score',0))})

    # 2. Critical Positive Test Results for Key Diseases
    critical_positive_results_df = df_for_alerts[
        (df_for_alerts['test_type'].isin([cfg['display_name'] for name, cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if cfg.get('critical')])) &
        (df_for_alerts['test_result'] == 'Positive')
    ]
    for _, row in critical_positive_results_df.iterrows():
        alerts_data.append({**row.to_dict(), 'alert_reason': f"Critical Positive: {row['test_type']}", 'priority_score': 85})
    
    # 3. High HIV Viral Load
    if 'hiv_viral_load_copies_ml' in df_for_alerts.columns and df_for_alerts['hiv_viral_load_copies_ml'].notna().any():
        high_vl_df = df_for_alerts[df_for_alerts['hiv_viral_load_copies_ml'] > 1000] # Example: WHO definition of virologic failure
        for _, row in high_vl_df.iterrows():
             alerts_data.append({**row.to_dict(), 'alert_reason': f"High HIV Viral Load ({row['hiv_viral_load_copies_ml']:.0f})", 'priority_score': 90})

    # 4. Critical Low SpO2 (if recorded during encounter, implies clinical observation not just CHW wearable)
    critical_spo2_clinic = df_for_alerts[df_for_alerts['min_spo2_pct'] < app_config.SPO2_CRITICAL_THRESHOLD_PCT] # e.g., < 90%
    for _, row in critical_spo2_clinic.iterrows():
         alerts_data.append({**row.to_dict(), 'alert_reason': f"Critically Low SpO2 ({row['min_spo2_pct']:.0f}%)", 'priority_score': 92})

    # 5. Overdue Pending Critical Tests
    # Consider 'sample_registered_lab_date' or 'sample_collection_date'
    date_col_for_overdue = 'sample_registered_lab_date' if 'sample_registered_lab_date' in df_for_alerts.columns and df_for_alerts['sample_registered_lab_date'].notna().any() else 'sample_collection_date'
    if date_col_for_overdue in df_for_alerts.columns:
        pending_df = df_for_alerts[
            (df_for_alerts['test_result'] == 'Pending') &
            (df_for_alerts['test_type'].isin([cfg['display_name'] for name, cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if cfg.get('critical')])) &
            (df_for_alerts[date_col_for_overdue].notna())
        ].copy()
        if not pending_df.empty:
            pending_df['days_pending'] = (pd.Timestamp('today').normalize() - pending_df[date_col_for_overdue]).dt.days
            # Get target TAT for each test type to define 'overdue' relative to its target
            def get_target_tat_for_alert(test_type_display_name):
                original_key = next((k for k,v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == test_type_display_name), None)
                if original_key: return app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_key].get('target_tat_days', app_config.OVERDUE_PENDING_TEST_DAYS)
                return app_config.OVERDUE_PENDING_TEST_DAYS # Default if not found
            
            pending_df['overdue_threshold_days'] = pending_df['test_type'].apply(get_target_tat_for_alert)
            overdue_critical_tests_df = pending_df[pending_df['days_pending'] > pending_df['overdue_threshold_days']]

            for _, row in overdue_critical_tests_df.iterrows():
                alerts_data.append({**row.to_dict(), 'alert_reason': f"Overdue Pending Test: {row['test_type']} ({row['days_pending']:.0f} days)", 'priority_score': 75 + min(row['days_pending'] - row['overdue_threshold_days'], 10) }) # Add a bit more priority for longer overdue

    if not alerts_data: return pd.DataFrame()
    alerts_df = pd.DataFrame(alerts_data)
    # Aggregate multiple alerts for the same patient on the same day
    # Group by patient_id and encounter_date, then combine reasons and take max priority
    def aggregate_alerts(group):
        first_row = group.iloc[0].copy()
        first_row['alert_reason'] = "; ".join(group['alert_reason'].unique())
        first_row['priority_score'] = group['priority_score'].max()
        return first_row
    
    # Need 'encounter_date' to be a date object if not already for grouping
    alerts_df['encounter_date_obj'] = pd.to_datetime(alerts_df['encounter_date']).dt.date
    final_alerts_df = alerts_df.groupby(['patient_id', 'encounter_date_obj'], as_index=False).apply(aggregate_alerts, include_groups=False)
    # Reset index might be needed if apply changes it weirdly without include_groups=False for new pandas
    final_alerts_df.reset_index(drop=True, inplace=True)
    final_alerts_df['priority_score'] = _convert_to_numeric(final_alerts_df['priority_score'],0).astype(int)
    
    return final_alerts_df.sort_values(by=['priority_score', 'encounter_date'], ascending=[False, False])


def get_district_summary_kpis(enriched_zone_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Aggregates KPIs from the enriched_zone_gdf for a district-level view."""
    kpis: Dict[str, Any] = {
        "total_population_district": 0, "avg_population_risk": 0.0,
        "zones_high_risk_count": 0, "overall_facility_coverage": 0.0,
        "district_tb_burden_total": 0, "district_malaria_burden_total": 0,
        "key_infection_prevalence_district_per_1000": 0.0,
        "population_weighted_avg_steps": 0.0, "avg_clinic_co2_district":0.0,
    }
    if enriched_zone_gdf is None or enriched_zone_gdf.empty: return kpis

    gdf = enriched_zone_gdf.copy()
    # Ensure numeric types for calculation
    for col in ['population', 'avg_risk_score', 'active_tb_cases', 'active_malaria_cases', 'total_active_key_infections', 'facility_coverage_score', 'avg_daily_steps_zone', 'zone_avg_co2']:
        if col in gdf.columns: gdf[col] = _convert_to_numeric(gdf[col], 0)
        else: gdf[col] = 0.0 # Add column if missing and fill with 0

    kpis["total_population_district"] = gdf['population'].sum()
    
    if kpis["total_population_district"] > 0:
        # Population-weighted average risk
        gdf['pop_x_risk'] = gdf['population'] * gdf['avg_risk_score']
        kpis["avg_population_risk"] = gdf['pop_x_risk'].sum() / kpis["total_population_district"]
        
        # Population-weighted facility coverage
        gdf['pop_x_facility_coverage'] = gdf['population'] * gdf['facility_coverage_score']
        kpis["overall_facility_coverage"] = gdf['pop_x_facility_coverage'].sum() / kpis["total_population_district"]

        # Population-weighted average steps
        gdf['pop_x_steps'] = gdf['population'] * gdf['avg_daily_steps_zone']
        kpis["population_weighted_avg_steps"] = gdf['pop_x_steps'].sum() / kpis["total_population_district"]

        # Overall prevalence
        total_key_infections_district = gdf['total_active_key_infections'].sum()
        kpis["key_infection_prevalence_district_per_1000"] = (total_key_infections_district / kpis["total_population_district"]) * 1000 if kpis["total_population_district"] > 0 else 0.0

    else: # if total population is 0, avoid division by zero
        kpis["avg_population_risk"] = gdf['avg_risk_score'].mean() if not gdf.empty else 0.0
        kpis["overall_facility_coverage"] = gdf['facility_coverage_score'].mean() if not gdf.empty else 0.0
        kpis["population_weighted_avg_steps"] = gdf['avg_daily_steps_zone'].mean() if not gdf.empty else 0.0
        kpis["key_infection_prevalence_district_per_1000"] = 0.0


    kpis["zones_high_risk_count"] = gdf[gdf['avg_risk_score'] >= app_config.RISK_THRESHOLDS['district_zone_high_risk']].shape[0]
    kpis["district_tb_burden_total"] = int(gdf['active_tb_cases'].sum())
    kpis["district_malaria_burden_total"] = int(gdf['active_malaria_cases'].sum())
    kpis["avg_clinic_co2_district"] = gdf['zone_avg_co2'].mean() if 'zone_avg_co2' in gdf and gdf['zone_avg_co2'].notna().any() else 0.0
    
    return kpis


def get_trend_data(df: pd.DataFrame, value_col: str, date_col: str = 'encounter_date', period: str = 'D',
                   agg_func: str = 'mean', filter_col: Optional[str] = None,
                   filter_val: Optional[Any] = None) -> pd.Series:
    """Generic function to aggregate data for trend analysis."""
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.Series(dtype='float64')
    
    trend_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(trend_df[date_col]):
        trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors='coerce')
    
    trend_df.dropna(subset=[date_col, value_col], inplace=True)
    if trend_df.empty: return pd.Series(dtype='float64')

    if filter_col and filter_col in trend_df.columns and filter_val is not None:
        trend_df = trend_df[trend_df[filter_col] == filter_val]
        if trend_df.empty: return pd.Series(dtype='float64')
            
    trend_df.set_index(date_col, inplace=True)
    
    # Ensure value_col is numeric for mean/sum, allow nunique for others
    if agg_func in ['mean', 'sum', 'median'] and not pd.api.types.is_numeric_dtype(trend_df[value_col]):
        trend_df[value_col] = _convert_to_numeric(trend_df[value_col], np.nan)
        trend_df.dropna(subset=[value_col], inplace=True)
        if trend_df.empty: return pd.Series(dtype='float64')

    try:
        if agg_func == 'nunique':
            trend_series = trend_df.groupby(pd.Grouper(freq=period))[value_col].nunique()
        elif agg_func == 'sum':
            trend_series = trend_df.groupby(pd.Grouper(freq=period))[value_col].sum()
        elif agg_func == 'median':
            trend_series = trend_df.groupby(pd.Grouper(freq=period))[value_col].median()
        else: # default to mean
            trend_series = trend_df.groupby(pd.Grouper(freq=period))[value_col].mean()
    except Exception as e:
        logger.error(f"Error generating trend data for {value_col} with agg {agg_func}: {e}", exc_info=True)
        return pd.Series(dtype='float64')
        
    return trend_series


def get_supply_forecast_data(health_df: pd.DataFrame, forecast_days_out: int = 30, 
                             item_filter_list: Optional[List[str]] = None) -> pd.DataFrame:
    """Forecasts supply levels based on consumption rates."""
    forecasts = []
    if health_df is None or health_df.empty or not all(c in health_df.columns for c in ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']):
        logger.warning("Supply forecast: Missing essential columns in health_df.")
        return pd.DataFrame()

    # Use latest available stock and consumption rate for each item (and zone if applicable)
    # Assuming item_stock_agg_zone is the latest snapshot
    supply_status_df = health_df.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last') # Simplified to item for now

    if item_filter_list:
        supply_status_df = supply_status_df[supply_status_df['item'].isin(item_filter_list)]
    
    if supply_status_df.empty:
        logger.info("Supply forecast: No items to forecast after filtering or data is empty.")
        return pd.DataFrame()

    for _, row in supply_status_df.iterrows():
        item_name = row['item']
        current_stock = row['item_stock_agg_zone']
        consumption_rate = row['consumption_rate_per_day']
        last_record_date = row['encounter_date']

        if pd.isna(current_stock) or pd.isna(consumption_rate) or current_stock < 0:
            logger.debug(f"Skipping forecast for {item_name}: missing stock ({current_stock}) or rate ({consumption_rate}).")
            continue

        # Generate forecast dates
        forecast_dates = pd.date_range(start=last_record_date + pd.Timedelta(days=1), periods=forecast_days_out, freq='D')
        
        current_forecast_stock = current_stock
        estimated_stockout_date = None
        days_remaining_at_forecast_start = (current_stock / consumption_rate) if consumption_rate > 0 else np.inf
        
        for i, forecast_date in enumerate(forecast_dates):
            days_out = i + 1
            # Linear consumption model
            current_forecast_stock = current_stock - (consumption_rate * days_out)
            
            # Calculate days of supply from the forecast_date perspective
            days_of_supply_forecast = (current_forecast_stock / consumption_rate) if consumption_rate > 0 else (np.inf if current_forecast_stock > 0 else 0)

            # Confidence Interval (simplified placeholder - true CI needs a model)
            # Lower CI could assume higher consumption, Upper CI lower consumption
            consumption_rate_std_dev_factor = 0.15 # Assume 15% std dev on consumption rate for CI
            lower_consumption_rate = consumption_rate * (1 + consumption_rate_std_dev_factor)
            upper_consumption_rate = max(0.1, consumption_rate * (1 - consumption_rate_std_dev_factor)) # ensure not zero or negative

            lower_ci_stock = current_stock - (lower_consumption_rate * days_out)
            upper_ci_stock = current_stock - (upper_consumption_rate * days_out)
            
            lower_ci_days_supply = (lower_ci_stock / lower_consumption_rate) if lower_consumption_rate > 0 else (np.inf if lower_ci_stock > 0 else 0)
            upper_ci_days_supply = (upper_ci_stock / upper_consumption_rate) if upper_consumption_rate > 0 else (np.inf if upper_ci_stock > 0 else 0)

            if current_forecast_stock <= 0 and estimated_stockout_date is None:
                # Refined stockout_date calculation: number of days from *start* until stock reaches 0
                if consumption_rate > 0:
                    days_to_stockout_from_start = current_stock / consumption_rate 
                    estimated_stockout_date = last_record_date + pd.to_timedelta(days_to_stockout_from_start, unit='D')

            forecasts.append({
                'item': item_name,
                'date': forecast_date, # This is the date being forecasted for
                'current_stock': current_stock, # Stock at the beginning of the forecast period
                'consumption_rate': consumption_rate,
                'forecast_stock': max(0, current_forecast_stock), # Stock cannot be negative
                'forecast_days': max(0, days_of_supply_forecast),
                'estimated_stockout_date': estimated_stockout_date, # This will be same for all rows of an item if calculated
                'lower_ci': max(0, lower_ci_days_supply),
                'upper_ci': max(0, upper_ci_days_supply),
                'initial_days_supply': days_remaining_at_forecast_start
            })
            
        # If never stocked out within forecast period but consumption is positive
        if estimated_stockout_date is None and consumption_rate > 0:
            days_to_stockout_from_start = current_stock / consumption_rate
            estimated_stockout_date_final = last_record_date + pd.to_timedelta(days_to_stockout_from_start, unit='D')
            for entry in forecasts:
                if entry['item'] == item_name and entry['estimated_stockout_date'] is None:
                     entry['estimated_stockout_date'] = estimated_stockout_date_final
        elif estimated_stockout_date is None and consumption_rate <= 0 and current_stock > 0: # No consumption, positive stock
             for entry in forecasts:
                if entry['item'] == item_name and entry['estimated_stockout_date'] is None:
                    entry['estimated_stockout_date'] = pd.NaT # Indefinite supply


    if not forecasts: return pd.DataFrame()
    forecast_df = pd.DataFrame(forecasts)
    forecast_df['estimated_stockout_date'] = pd.to_datetime(forecast_df['estimated_stockout_date'], errors='coerce')
    logger.info(f"Supply forecast generated for {forecast_df['item'].nunique()} items.")
    return forecast_df
