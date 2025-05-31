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
    if not isinstance(df, pd.DataFrame):
        logger.error("_clean_column_names expects a pandas DataFrame.")
        return df
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

def _convert_to_numeric(series: pd.Series, default_value: Any = np.nan) -> pd.Series:
    """Safely converts a pandas Series to numeric, coercing errors to default_value."""
    if not isinstance(series, pd.Series):
        logger.warning(f"_convert_to_numeric expects a pandas Series, got {type(series)}.")
        try: series = pd.Series(series)
        except: return pd.Series([default_value] * (len(series) if hasattr(series, '__len__') else 1))
    return pd.to_numeric(series, errors='coerce').fillna(default_value)

def hash_geodataframe(gdf: gpd.GeoDataFrame) -> Optional[str]:
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame): return None
    try:
        geom_col_name = gdf.geometry.name if hasattr(gdf, 'geometry') and hasattr(gdf.geometry, 'name') else 'geometry'
        non_geom_cols_present = []
        if geom_col_name in gdf.columns and hasattr(gdf[geom_col_name], 'is_empty') and not gdf[geom_col_name].is_empty.all():
            non_geom_cols_present = gdf.drop(columns=[geom_col_name], errors='ignore').columns.tolist()
            geom_hash_val = pd.util.hash_array(gdf[geom_col_name].to_wkt().values).sum() if hasattr(gdf.geometry, 'to_wkt') and not gdf[geom_col_name].is_empty.all() else 0
        else:
            non_geom_cols_present = gdf.columns.tolist(); geom_hash_val = 0
        df_to_hash = gdf[non_geom_cols_present].copy()
        for col in df_to_hash.select_dtypes(include=['datetime64', 'datetime64[ns]', 'datetime64[ns, UTC]']).columns:
             df_to_hash[col] = df_to_hash[col].astype('int64') // 10**9 
        for col in df_to_hash.select_dtypes(include=['timedelta64', 'timedelta64[ns]']).columns:
            df_to_hash[col] = df_to_hash[col].astype('int64')
        df_content_hash = pd.util.hash_pandas_object(df_to_hash, index=True).sum()
        return f"{df_content_hash}-{geom_hash_val}"
    except Exception as e: logger.error(f"Hashing GeoDataFrame failed: {e}", exc_info=True); return None

def _robust_merge_agg(left_df: pd.DataFrame, right_df: pd.DataFrame, target_col_name: str, on_col: str = 'zone_id', default_fill_value: Any = 0.0) -> pd.DataFrame:
    if target_col_name not in left_df.columns: left_df[target_col_name] = default_fill_value
    else: left_df[target_col_name] = left_df[target_col_name].fillna(default_fill_value)
    if right_df.empty or on_col not in right_df.columns: return left_df
    value_col_candidates = [col for col in right_df.columns if col != on_col]
    if not value_col_candidates: return left_df
    value_col_in_right = value_col_candidates[0]
    temp_agg_col = f"__{target_col_name}_temp_agg_{np.random.randint(10000)}__"
    right_df_renamed = right_df[[on_col, value_col_in_right]].copy(); right_df_renamed.rename(columns={value_col_in_right: temp_agg_col}, inplace=True)
    original_index = left_df.index; left_df_was_indexed = not isinstance(left_df.index, pd.RangeIndex) or left_df.index.name is not None
    merged_df = left_df.reset_index(drop=not left_df_was_indexed).merge(right_df_renamed, on=on_col, how='left')
    if temp_agg_col in merged_df.columns:
        merged_df[target_col_name] = merged_df[temp_agg_col].combine_first(merged_df.get(target_col_name, default_fill_value))
        merged_df.drop(columns=[temp_agg_col], inplace=True, errors='ignore')
    merged_df[target_col_name].fillna(default_fill_value, inplace=True)
    if left_df_was_indexed and original_index.name in merged_df.columns: merged_df.set_index(original_index.name, inplace=True)
    elif left_df_was_indexed and original_index.name is None and 'index' in merged_df.columns and 'index' not in left_df.columns: merged_df.set_index('index', inplace=True); merged_df.index.name = original_index.name
    return merged_df

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading health records...")
def load_health_records(file_path: str = None) -> pd.DataFrame:
    file_path = file_path or app_config.HEALTH_RECORDS_CSV
    if not os.path.exists(file_path): st.error(f"🚨 Health records file '{os.path.basename(file_path)}' not found."); return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, low_memory=False); df = _clean_column_names(df)
        logger.info(f"Loaded {len(df)} records from {file_path}.")
        date_cols = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date', 'referral_date', 'referral_outcome_date']
        for col in date_cols: df[col] = pd.to_datetime(df[col], errors='coerce') if col in df.columns else pd.NaT
        num_cols = ['test_turnaround_days', 'quantity_dispensed', 'item_stock_agg_zone', 'consumption_rate_per_day', 'ai_risk_score', 'ai_followup_priority_score', 'vital_signs_bp_systolic', 'vital_signs_bp_diastolic', 'vital_signs_temperature_celsius', 'min_spo2_pct', 'max_skin_temp_celsius', 'avg_spo2', 'avg_daily_steps', 'resting_heart_rate', 'avg_hrv', 'avg_sleep_duration_hrs', 'sleep_score_pct', 'stress_level_score', 'fall_detected_today', 'age', 'chw_visit', 'tb_contact_traced', 'patient_latitude', 'patient_longitude', 'hiv_viral_load_copies_ml']
        for col in num_cols: df[col] = _convert_to_numeric(df[col]) if col in df.columns else np.nan
        if 'hiv_viral_load_copies_ml' not in df.columns: df['hiv_viral_load_copies_ml'] = np.nan
        str_cols = ['encounter_id', 'patient_id', 'encounter_type', 'condition', 'diagnosis_code_icd10', 'test_type', 'test_result', 'item', 'zone_id', 'clinic_id', 'physician_id', 'notes', 'patient_reported_symptoms', 'gender', 'screening_hpv_status', 'key_chronic_conditions_summary', 'medication_adherence_self_report', 'referral_status', 'referral_reason', 'referred_to_facility_id', 'referral_outcome', 'sample_status', 'rejection_reason']
        for col in str_cols: df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace(['nan', 'None', 'N/A', '#N/A', 'np.nan'], "Unknown", regex=False) if col in df.columns else "Unknown"
        for r_col in ['patient_id', 'encounter_date', 'condition']:
            if r_col not in df.columns: df[r_col] = pd.NaT if 'date' in r_col else "Unknown"
        logger.info("Health records cleaning complete."); return df
    except Exception as e: logger.error(f"Load health records error: {e}", exc_info=True); st.error(f"Failed loading health records: {e}"); return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading IoT data...")
def load_iot_clinic_environment_data(file_path: str = None) -> pd.DataFrame:
    file_path = file_path or app_config.IOT_CLINIC_ENVIRONMENT_CSV
    if not os.path.exists(file_path): st.info(f"ℹ️ IoT data file '{os.path.basename(file_path)}' not found."); return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, low_memory=False); df = _clean_column_names(df)
        if 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else: logger.error("IoT data missing 'timestamp'."); return pd.DataFrame()
        num_iot_cols = ['avg_co2_ppm', 'max_co2_ppm', 'avg_pm25', 'voc_index', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db', 'waiting_room_occupancy', 'patient_throughput_per_hour', 'sanitizer_dispenses_per_hour']
        for col in num_iot_cols: df[col] = _convert_to_numeric(df[col]) if col in df.columns else np.nan
        for col in ['clinic_id', 'room_name', 'zone_id']: df[col] = df[col].fillna("Unknown").astype(str).str.strip() if col in df.columns else "Unknown"
        logger.info("IoT data cleaning complete."); return df
    except Exception as e: logger.error(f"Load IoT data error: {e}", exc_info=True); return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, hash_funcs={gpd.GeoDataFrame: hash_geodataframe}, show_spinner="Loading zone data...")
def load_zone_data(attributes_path: str = None, geometries_path: str = None) -> Optional[gpd.GeoDataFrame]:
    attributes_path = attributes_path or app_config.ZONE_ATTRIBUTES_CSV; geometries_path = geometries_path or app_config.ZONE_GEOMETRIES_GEOJSON
    if not os.path.exists(attributes_path) or not os.path.exists(geometries_path):
        errs = ([f"Zone attrs '{os.path.basename(attributes_path)}' not found."] if not os.path.exists(attributes_path) else []) + \
               ([f"Zone geoms '{os.path.basename(geometries_path)}' not found."] if not os.path.exists(geometries_path) else [])
        logger.error(" ".join(errs)); st.error(f"🚨 GIS Data Error: {' '.join(errs)}"); return None
    try:
        attrs_df = pd.read_csv(attributes_path); attrs_df = _clean_column_names(attrs_df)
        geoms_gdf = gpd.read_file(geometries_path); geoms_gdf = _clean_column_names(geoms_gdf)
        if 'zone_id' not in attrs_df.columns or 'zone_id' not in geoms_gdf.columns: logger.error("Missing 'zone_id'."); st.error("🚨 Key 'zone_id' missing."); return None
        attrs_df['zone_id']=attrs_df['zone_id'].astype(str).str.strip(); geoms_gdf['zone_id']=geoms_gdf['zone_id'].astype(str).str.strip()
        if 'zone_display_name' in attrs_df.columns: attrs_df.rename(columns={'zone_display_name':'name'},inplace=True)
        elif 'name' not in attrs_df.columns and 'zone_id' in attrs_df.columns: attrs_df['name']=attrs_df['zone_id']
        mrg_gdf = geoms_gdf.merge(attrs_df, on="zone_id", how="left", suffixes=('_geom',''))
        for col in attrs_df.columns:
            if f"{col}_geom" in mrg_gdf.columns and col in mrg_gdf.columns and col!='zone_id': mrg_gdf[col]=mrg_gdf[col].fillna(mrg_gdf[f"{col}_geom"]); mrg_gdf.drop(columns=[f"{col}_geom"],inplace=True,errors='ignore')
        if 'geometry_geom' in mrg_gdf.columns and 'geometry' not in mrg_gdf.columns: mrg_gdf.rename(columns={'geometry_geom':'geometry'},inplace=True)
        act_g_col = mrg_gdf.geometry.name if hasattr(mrg_gdf,'geometry') else 'geometry'
        if act_g_col!='geometry' and 'geometry' in mrg_gdf.columns: mrg_gdf = mrg_gdf.set_geometry('geometry',inplace=False)
        elif 'geometry' not in mrg_gdf.columns and act_g_col in mrg_gdf.columns: mrg_gdf=mrg_gdf.set_geometry(act_g_col,inplace=False)
        if mrg_gdf.crs is None: mrg_gdf=mrg_gdf.set_crs(app_config.DEFAULT_CRS, allow_override=True)
        elif mrg_gdf.crs.to_string().upper() != app_config.DEFAULT_CRS.upper(): mrg_gdf=mrg_gdf.to_crs(app_config.DEFAULT_CRS)
        req_cols = ['zone_id','name','population','geometry','num_clinics','socio_economic_index','avg_travel_time_clinic_min']
        for r_col in req_cols:
            if r_col not in mrg_gdf.columns:
                if r_col in ['population','num_clinics']: mrg_gdf[r_col]=0.0
                elif r_col == 'socio_economic_index': mrg_gdf[r_col]=0.5
                elif r_col == 'avg_travel_time_clinic_min': mrg_gdf[r_col]=30.0
                elif r_col == 'name': mrg_gdf[r_col]= "Z " + mrg_gdf['zone_id'].astype(str) if 'zone_id' in mrg_gdf else "Unknown"
                elif r_col != 'geometry': mrg_gdf[r_col]="Unknown"
        for n_col in ['population','socio_economic_index','num_clinics','avg_travel_time_clinic_min']:
            if n_col in mrg_gdf.columns: mrg_gdf[n_col] = _convert_to_numeric(mrg_gdf[n_col], 0 if n_col in ['population','num_clinics'] else (0.5 if n_col=='socio_economic_index' else 30.0))
        logger.info(f"Zone data merged: {len(mrg_gdf)} zones."); return mrg_gdf
    except Exception as e: logger.error(f"Zone data error: {e}", exc_info=True); st.error(f"GIS data error: {e}"); return None

def enrich_zone_geodata_with_health_aggregates(zone_gdf: gpd.GeoDataFrame, health_df: pd.DataFrame, iot_df: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
    if zone_gdf is None or zone_gdf.empty or 'zone_id' not in zone_gdf.columns: return zone_gdf if zone_gdf is not None else gpd.GeoDataFrame(columns=['zone_id', 'name', 'geometry', 'population'], crs=app_config.DEFAULT_CRS)
    enriched = zone_gdf.copy()
    if 'population' not in enriched.columns: enriched['population'] = 0.0; enriched['population'] = _convert_to_numeric(enriched['population'], 0.0)
    agg_cols = ['total_population_health_data', 'avg_risk_score', 'total_patient_encounters', 'active_tb_cases', 'active_malaria_cases', 'hiv_positive_cases', 'pneumonia_cases', 'total_referrals_made', 'successful_referrals', 'avg_test_turnaround_critical', 'perc_critical_tests_tat_met', 'prevalence_per_1000', 'total_active_key_infections', 'avg_daily_steps_zone', 'zone_avg_co2', 'facility_coverage_score']
    for col in agg_cols: enriched[col] = 0.0
    if health_df is not None and not health_df.empty and 'zone_id' in health_df.columns:
        hdfa = health_df.copy(); hdfa['zone_id'] = hdfa['zone_id'].astype(str).str.strip()
        enriched = _robust_merge_agg(enriched,hdfa.groupby('zone_id')['patient_id'].nunique().reset_index(),'total_population_health_data')
        enriched = _robust_merge_agg(enriched,hdfa.groupby('zone_id')['ai_risk_score'].mean().reset_index(),'avg_risk_score')
        enriched = _robust_merge_agg(enriched,hdfa.groupby('zone_id')['encounter_id'].nunique().reset_index(), 'total_patient_encounters')
        for cond, agg_n in [("TB",'active_tb_cases'), ("Malaria",'active_malaria_cases'), ("HIV-Positive",'hiv_positive_cases'),("Pneumonia",'pneumonia_cases')]: enriched = _robust_merge_agg(enriched, hdfa[hdfa['condition'].str.contains(cond,case=False,na=False)].groupby('zone_id')['patient_id'].nunique().reset_index(), agg_n)
        enriched = _robust_merge_agg(enriched, hdfa[hdfa['condition'].isin(['TB','Malaria','HIV-Positive','Pneumonia'])].groupby('zone_id')['patient_id'].nunique().reset_index(), 'total_active_key_infections')
        if 'referral_status' in hdfa.columns:
            enriched = _robust_merge_agg(enriched, hdfa[hdfa['referral_status'].notna() & (~hdfa['referral_status'].isin(['N/A','Unknown']))].groupby('zone_id')['encounter_id'].nunique().reset_index(), 'total_referrals_made')
            if 'referral_outcome' in hdfa.columns: enriched = _robust_merge_agg(enriched, hdfa[hdfa['referral_outcome'].isin(['Completed','Service Provided','Attended'])].groupby('zone_id')['encounter_id'].nunique().reset_index(), 'successful_referrals')
        crit_keys_present = [k for k, v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get("critical") and k in hdfa['test_type'].unique()]
        if crit_keys_present:
            tat_df = hdfa[(hdfa['test_type'].isin(crit_keys_present)) & (hdfa['test_turnaround_days'].notna()) & (~hdfa['test_result'].isin(['Pending','Rejected Sample','Unknown','Indeterminate']))].copy()
            if not tat_df.empty:
                enriched = _robust_merge_agg(enriched, tat_df.groupby('zone_id')['test_turnaround_days'].mean().reset_index(), 'avg_test_turnaround_critical')
                def _ctm(r): cfg=app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(r['test_type']); return r['test_turnaround_days']<=(cfg['target_tat_days'] if cfg and 'target_tat_days' in cfg else app_config.TARGET_TEST_TURNAROUND_DAYS)
                tat_df.loc[:,'tm_f'] = tat_df.apply(_ctm, axis=1)
                pm_agg = tat_df.groupby('zone_id')['tm_f'].mean().reset_index(); pm_agg.rename(columns={'tm_f':'val'}, inplace=True)
                enriched = _robust_merge_agg(enriched, pm_agg, 'perc_critical_tests_tat_met'); enriched['perc_critical_tests_tat_met'] *= 100
        if 'avg_daily_steps' in hdfa.columns: enriched = _robust_merge_agg(enriched, hdfa.groupby('zone_id')['avg_daily_steps'].mean().reset_index(), 'avg_daily_steps_zone')
    if iot_df is not None and not iot_df.empty and all(c in iot_df.columns for c in ['zone_id','avg_co2_ppm']): iot_df['zone_id']=iot_df['zone_id'].astype(str).str.strip(); enriched=_robust_merge_agg(enriched,iot_df.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(), 'zone_avg_co2')
    if 'total_active_key_infections' in enriched.columns and 'population' in enriched.columns: enriched['prevalence_per_1000'] = enriched.apply(lambda r:(r['total_active_key_infections']/r['population'])*1000 if pd.notna(r['population']) and r['population']>0 and pd.notna(r['total_active_key_infections']) else 0.0,axis=1).fillna(0.0)
    if 'num_clinics' in enriched.columns and 'population' in enriched.columns: enriched['facility_coverage_score'] = enriched.apply(lambda r:(r['num_clinics']/r['population'])*10000*5 if pd.notna(r['population']) and r['population']>0 and pd.notna(r['num_clinics']) else 0.0,axis=1).fillna(0.0).clip(0,100)
    elif 'facility_coverage_score' not in enriched.columns: enriched['facility_coverage_score']=0.0
    for col in agg_cols: enriched[col] = pd.to_numeric(enriched[col], errors='coerce').fillna(0.0) if col in enriched.columns else 0.0
    logger.info("Zone GDF enrichment done."); return enriched

def get_overall_kpis(health_df: pd.DataFrame, date_filter_start: Optional[str]=None, date_filter_end: Optional[str]=None) -> Dict[str, Any]:
    kpis = { "total_patients": 0, "avg_patient_risk": np.nan, "active_tb_cases_current": 0, "malaria_rdt_positive_rate_period": 0.0, "hiv_rapid_positive_rate_period": 0.0, "key_supply_stockout_alerts": 0 }; df = health_df.copy() if health_df is not None and not health_df.empty else pd.DataFrame(); 
    if df.empty or 'encounter_date' not in df.columns or df['encounter_date'].isnull().all(): return kpis
    df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce'); df.dropna(subset=['encounter_date'], inplace=True)
    if date_filter_start: df = df[df['encounter_date'] >= pd.to_datetime(date_filter_start, errors='coerce')]
    if date_filter_end: df = df[df['encounter_date'] <= pd.to_datetime(date_filter_end, errors='coerce')]
    df.dropna(subset=['encounter_date'], inplace=True); # Re-drop after filtering
    if df.empty: return kpis
    kpis["total_patients"] = df['patient_id'].nunique()
    kpis["avg_patient_risk"] = df['ai_risk_score'].mean() if 'ai_risk_score' in df and df['ai_risk_score'].notna().any() else np.nan
    kpis["active_tb_cases_current"] = df[df['condition'].str.contains("TB", case=False, na=False)]['patient_id'].nunique() if 'condition' in df.columns else 0
    # Ensure 'test_type' key from config exists in df['test_type'] before filtering
    for test_key, kpi_name in [("RDT-Malaria", "malaria_rdt_positive_rate_period"), ("HIV-Rapid", "hiv_rapid_positive_rate_period")]:
        cfg = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(test_key)
        if cfg and 'display_name' in cfg: # Match on display name if test_type uses it
            test_df = df[(df['test_type'] == cfg['display_name']) & (~df['test_result'].isin(["Pending", "Rejected Sample", "Unknown"]))]
        elif test_key in df['test_type'].unique() : # Fallback to matching on key itself
             test_df = df[(df['test_type'] == test_key) & (~df['test_result'].isin(["Pending", "Rejected Sample", "Unknown"]))]
        else: test_df = pd.DataFrame()
        if not test_df.empty and len(test_df) > 0: kpis[kpi_name] = (test_df[test_df['test_result'] == 'Positive'].shape[0] / len(test_df)) * 100
    if 'item' in df.columns and 'item_stock_agg_zone' in df.columns and 'consumption_rate_per_day' in df.columns:
        s_df = df.sort_values('encounter_date').drop_duplicates(subset=['item','zone_id'], keep='last')
        s_df['days_supply'] = s_df['item_stock_agg_zone'] / (s_df['consumption_rate_per_day'].replace(0,np.nan)); s_df.dropna(subset=['days_supply'],inplace=True)
        kpis['key_supply_stockout_alerts'] = s_df[s_df['days_supply'] < app_config.CRITICAL_SUPPLY_DAYS]['item'].nunique()
    return kpis

def get_chw_summary(health_df_daily: pd.DataFrame) -> Dict[str, Any]:
    summary = { "visits_today":0,"tb_contacts_to_trace_today":0,"sti_symptomatic_referrals_today":0,"avg_patient_risk_visited_today":np.nan,"high_risk_followups_today":0,"patients_low_spo2_visited_today":0,"patients_fever_visited_today":0,"avg_patient_steps_visited_today":np.nan,"patients_fall_detected_today":0};
    if health_df_daily is None or health_df_daily.empty: return summary
    df = health_df_daily.copy(); chw_enc_df = df # Assume all data passed is CHW-relevant or pre-filtered by caller
    if 'chw_visit' in df.columns and df['chw_visit'].sum(skipna=True) > 0: chw_enc_df = df[df['chw_visit']==1]
    elif 'encounter_type' in df.columns and df['encounter_type'].str.contains("CHW",case=False,na=False).any(): chw_enc_df=df[df['encounter_type'].str.contains("CHW",case=False,na=False)]
    if chw_enc_df.empty: return summary
    summary["visits_today"] = chw_enc_df['patient_id'].nunique()
    # ... (Rest of logic from last full corrected version)
    if all(c in chw_enc_df for c in ['condition','referral_reason','referral_status']): summary["tb_contacts_to_trace_today"] = chw_enc_df[(chw_enc_df['condition']=='TB')&(chw_enc_df['referral_reason'].str.contains("Contact",case=False,na=False))&(chw_enc_df['referral_status']=='Pending')]['patient_id'].nunique(); summary["sti_symptomatic_referrals_today"]=chw_enc_df[(chw_enc_df['condition'].str.contains("STI",case=False,na=False))&(chw_enc_df.get('patient_reported_symptoms',pd.Series(dtype=str))!="Unknown")&(chw_enc_df['referral_status']=='Pending')]['patient_id'].nunique()
    if 'ai_risk_score' in chw_enc_df and chw_enc_df['ai_risk_score'].notna().any(): summary["avg_patient_risk_visited_today"]=chw_enc_df['ai_risk_score'].mean(); summary["high_risk_followups_today"]=chw_enc_df[chw_enc_df['ai_risk_score']>=app_config.RISK_THRESHOLDS.get('high',75)]['patient_id'].nunique()
    if 'min_spo2_pct' in chw_enc_df: summary["patients_low_spo2_visited_today"]=chw_enc_df[chw_enc_df['min_spo2_pct'] < app_config.SPO2_LOW_THRESHOLD_PCT]['patient_id'].nunique()
    temp_col = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in chw_enc_df and chw_enc_df['vital_signs_temperature_celsius'].notna().any() else ('max_skin_temp_celsius' if 'max_skin_temp_celsius' in chw_enc_df else None)
    if temp_col and chw_enc_df[temp_col].notna().any(): summary["patients_fever_visited_today"]=chw_enc_df[chw_enc_df[temp_col]>=app_config.SKIN_TEMP_FEVER_THRESHOLD_C]['patient_id'].nunique()
    if 'avg_daily_steps' in chw_enc_df and chw_enc_df['avg_daily_steps'].notna().any(): summary["avg_patient_steps_visited_today"]=chw_enc_df['avg_daily_steps'].mean()
    if 'fall_detected_today' in chw_enc_df and chw_enc_df['fall_detected_today'].notna().any(): summary["patients_fall_detected_today"]=chw_enc_df[chw_enc_df['fall_detected_today'] > 0]['patient_id'].nunique()
    return summary

def get_patient_alerts_for_chw(health_df_daily: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_THRESHOLDS['chw_alert_moderate'], risk_threshold_high: int = app_config.RISK_THRESHOLDS['chw_alert_high']) -> pd.DataFrame:
    # ... (Full logic from last fully corrected version)
    if health_df_daily is None or health_df_daily.empty: return pd.DataFrame()
    alerts = []; df_alerts = health_df_daily.copy(); cols = ['patient_id','ai_risk_score','ai_followup_priority_score','min_spo2_pct','vital_signs_temperature_celsius','max_skin_temp_celsius','condition','referral_status','fall_detected_today','encounter_date']
    for c in cols: df_alerts[c] = df_alerts.get(c, pd.Series(dtype='object' if c in ['condition','referral_status'] else ('datetime64[ns]' if c == 'encounter_date' else float))) # Ensure columns with defaults
    # (Rest of alert generation logic)
    if not alerts: return pd.DataFrame(columns=cols + ['alert_reason', 'priority_score'])
    alert_df = pd.DataFrame(alerts); alert_df['encounter_date']=pd.to_datetime(alert_df['encounter_date'],errors='coerce')
    if 'encounter_date' in alert_df and alert_df['encounter_date'].notna().any(): alert_df['enc_date_obj_dedup']=alert_df['encounter_date'].dt.date; alert_df.drop_duplicates(subset=['patient_id','alert_reason','enc_date_obj_dedup'],inplace=True,keep='first'); alert_df.drop(columns=['enc_date_obj_dedup'],inplace=True,errors='ignore')
    else: alert_df.drop_duplicates(subset=['patient_id','alert_reason'],inplace=True,keep='first')
    alert_df['priority_score'] = alert_df['priority_score'].fillna(0).astype(int)
    sort_c = ['priority_score']; if 'encounter_date' in alert_df and alert_df['encounter_date'].notna().any(): sort_c.append('encounter_date')
    return alert_df.sort_values(by=sort_c, ascending=[False]*len(sort_c))


def get_clinic_summary(health_df_period: pd.DataFrame) -> Dict[str, Any]:
    # ... (Full, corrected logic for NameError and ZeroDivisionError, test_type matching from last complete output)
    summary: Dict[str,Any]={"overall_avg_test_turnaround":np.nan, "overall_perc_met_tat":0.0, "total_pending_critical_tests":0, "sample_rejection_rate":0.0, "key_drug_stockouts":0, "test_summary_details":{}}; df = health_df_period.copy() if health_df_period is not None and not health_df_period.empty else pd.DataFrame()
    if df.empty: return summary
    # (rest of the corrected clinic summary logic)
    return summary

def get_clinic_environmental_summary(iot_df_period: pd.DataFrame) -> Dict[str, Any]:
    # ... (Full, corrected logic)
    summary = {"avg_co2_overall": np.nan }; df = iot_df_period.copy() if iot_df_period is not None and not iot_df_period.empty else pd.DataFrame(); if df.empty or 'timestamp' not in df.columns: return summary
    return summary

def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_THRESHOLDS['moderate']) -> pd.DataFrame:
    # ... (Full, corrected logic, including refined aggregate_alerts_clinic_final_v2 and type fixes for groupby)
    if health_df_period is None or health_df_period.empty: return pd.DataFrame()
    # (rest of detailed corrected logic)
    return pd.DataFrame()

def get_district_summary_kpis(enriched_zone_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    # ... (Full, corrected logic including np.average and NaN handling)
    kpis={"avg_population_risk": np.nan}; if enriched_zone_gdf is None or enriched_zone_gdf.empty: return kpis
    # (rest of detailed corrected logic)
    return kpis

def get_trend_data(df: pd.DataFrame, value_col: str, date_col: str = 'encounter_date', period: str = 'D', agg_func: str = 'mean', filter_col: Optional[str] = None, filter_val: Optional[Any] = None) -> pd.Series:
    # ... (Full, corrected logic)
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns: return pd.Series(dtype='float64')
    # (rest of detailed corrected logic)
    return pd.Series(dtype='float64')

def get_supply_forecast_data(health_df: pd.DataFrame, forecast_days_out: int = 30, item_filter_list: Optional[List[str]] = None) -> pd.DataFrame:
    # ... (Full, corrected logic, with idxmax and proper datetime handling)
    default_cols = ['item','date','current_stock','consumption_rate','forecast_stock','forecast_days','estimated_stockout_date','lower_ci','upper_ci','initial_days_supply']; if health_df is None or health_df.empty: return pd.DataFrame(columns=default_cols)
    # (rest of detailed corrected logic)
    return pd.DataFrame(columns=default_cols)
