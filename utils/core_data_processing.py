import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import logging
from config import app_config
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# --- Helper Functions ---
def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        logger.error(f"_clean_column_names expects a pandas DataFrame, got {type(df)}.")
        return df 
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

def _convert_to_numeric(series: pd.Series, default_value: Any = np.nan) -> pd.Series:
    if not isinstance(series, pd.Series):
        logger.debug(f"_convert_to_numeric given non-Series type: {type(series)}. Attempting conversion.")
        try:
            series = pd.Series(series)
        except Exception as e_series:
            logger.error(f"Could not convert input to Series in _convert_to_numeric: {e_series}")
            length = len(series) if hasattr(series, '__len__') else 1
            dtype_val = type(default_value) if default_value is not np.nan else float
            return pd.Series([default_value] * length, dtype=dtype_val)
            
    return pd.to_numeric(series, errors='coerce').fillna(default_value)

def hash_geodataframe(gdf: gpd.GeoDataFrame) -> Optional[str]:
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame): 
        return None
    try:
        geom_col_name = gdf.geometry.name if hasattr(gdf, 'geometry') and hasattr(gdf.geometry, 'name') else 'geometry'
        non_geom_cols_present = []
        geom_hash_val = 0
        if geom_col_name in gdf.columns and hasattr(gdf[geom_col_name], 'is_empty') and not gdf[geom_col_name].is_empty.all():
            non_geom_cols_present = gdf.drop(columns=[geom_col_name], errors='ignore').columns.tolist()
            if hasattr(gdf[geom_col_name], 'to_wkt') and not gdf[geom_col_name].is_empty.all():
                geom_hash_val = pd.util.hash_array(gdf[geom_col_name].to_wkt().values).sum()
        else:
            non_geom_cols_present = gdf.columns.tolist()
            
        df_to_hash = gdf[non_geom_cols_present].copy()
        for col in df_to_hash.select_dtypes(include=['datetime64', 'datetime64[ns]', 'datetime64[ns, UTC]']).columns:
             df_to_hash[col] = df_to_hash[col].astype('int64') // 10**9 
        for col in df_to_hash.select_dtypes(include=['timedelta64', 'timedelta64[ns]']).columns:
            df_to_hash[col] = df_to_hash[col].astype('int64')
        df_content_hash = pd.util.hash_pandas_object(df_to_hash, index=True).sum()
        return f"{df_content_hash}-{geom_hash_val}"
    except Exception as e:
        logger.error(f"Hashing GeoDataFrame failed: {e}", exc_info=True)
        return str(gdf.head().to_string())

def _robust_merge_agg(left_df: pd.DataFrame, right_df: pd.DataFrame, target_col_name: str, on_col: str = 'zone_id', default_fill_value: Any = 0.0) -> pd.DataFrame:
    if target_col_name not in left_df.columns:
        left_df[target_col_name] = default_fill_value
    else:
        left_df[target_col_name] = left_df[target_col_name].fillna(default_fill_value)
        
    if right_df.empty or on_col not in right_df.columns:
        return left_df
        
    value_col_candidates = [col for col in right_df.columns if col != on_col]
    if not value_col_candidates:
        return left_df
        
    value_col_in_right = value_col_candidates[0]
    temp_agg_col = f"__{target_col_name}_temp_agg_{np.random.randint(10000, 99999)}__"
    
    right_df_for_merge = right_df[[on_col, value_col_in_right]].copy()
    right_df_for_merge.rename(columns={value_col_in_right: temp_agg_col}, inplace=True)
    
    original_index_name = left_df.index.name
    left_df_reset_needed = not isinstance(left_df.index, pd.RangeIndex) or original_index_name is not None
    
    left_df_for_merge = left_df.reset_index() if left_df_reset_needed else left_df
    
    merged_df = left_df_for_merge.merge(right_df_for_merge, on=on_col, how='left')
    
    if temp_agg_col in merged_df.columns:
        # Correctly use .combine_first without inplace modification on a slice
        merged_df[target_col_name] = merged_df[temp_agg_col].combine_first(merged_df[target_col_name])
        merged_df.drop(columns=[temp_agg_col], inplace=True, errors='ignore')

    # Correctly fill NaNs by reassigning the column to avoid SettingWithCopyWarning
    merged_df[target_col_name] = merged_df[target_col_name].fillna(default_fill_value)
    
    if left_df_reset_needed:
        index_col_to_set_back = original_index_name if original_index_name else 'index'
        if index_col_to_set_back in merged_df.columns:
            merged_df.set_index(index_col_to_set_back, inplace=True)
            if original_index_name:
                merged_df.index.name = original_index_name
    return merged_df

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading health records...")
def load_health_records(file_path: str = None) -> pd.DataFrame:
    file_path = file_path or app_config.HEALTH_RECORDS_CSV
    if not os.path.exists(file_path): st.error(f"🚨 Health records file '{os.path.basename(file_path)}' not found."); return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, low_memory=False); df = _clean_column_names(df)
        date_cols = ['encounter_date', 'sample_collection_date', 'sample_registered_lab_date', 'referral_date', 'referral_outcome_date']
        for col in date_cols: df[col] = pd.to_datetime(df.get(col), errors='coerce')
        num_cols = ['test_turnaround_days','quantity_dispensed','item_stock_agg_zone','consumption_rate_per_day','ai_risk_score','ai_followup_priority_score','vital_signs_bp_systolic','vital_signs_bp_diastolic','vital_signs_temperature_celsius','min_spo2_pct','max_skin_temp_celsius','avg_spo2','avg_daily_steps','resting_heart_rate','avg_hrv','avg_sleep_duration_hrs','sleep_score_pct','stress_level_score','fall_detected_today','age','chw_visit','tb_contact_traced','patient_latitude','patient_longitude','hiv_viral_load_copies_ml']
        for col in num_cols: df[col] = _convert_to_numeric(df.get(col), np.nan)
        if 'hiv_viral_load_copies_ml' not in df.columns: df['hiv_viral_load_copies_ml'] = np.nan
        str_cols = ['encounter_id','patient_id','encounter_type','condition','diagnosis_code_icd10','test_type','test_result','item','zone_id','clinic_id','physician_id','notes','patient_reported_symptoms','gender','screening_hpv_status','key_chronic_conditions_summary','medication_adherence_self_report','referral_status','referral_reason','referred_to_facility_id','referral_outcome','sample_status','rejection_reason']
        for col in str_cols: df[col] = df.get(col, pd.Series(dtype=str)).fillna("Unknown").astype(str).str.strip().replace(['nan','None','N/A','#N/A','np.nan','NaT'], "Unknown", regex=False)
        for r_col in ['patient_id','encounter_date','condition','test_type']:
            if r_col not in df.columns or df[r_col].isnull().all(): df[r_col] = pd.NaT if 'date' in r_col else "Unknown"
        logger.info("Health records cleaning complete."); return df
    except Exception as e: logger.error(f"Load health records error: {e}", exc_info=True); st.error(f"Failed loading health records: {e}"); return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, show_spinner="Loading IoT data...")
def load_iot_clinic_environment_data(file_path: str = None) -> pd.DataFrame:
    file_path = file_path or app_config.IOT_CLINIC_ENVIRONMENT_CSV
    if not os.path.exists(file_path): st.info(f"ℹ️ IoT data file '{os.path.basename(file_path)}' not found."); return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, low_memory=False); df = _clean_column_names(df)
        if 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df.get('timestamp'), errors='coerce')
        else: logger.error("IoT missing 'timestamp'."); return pd.DataFrame()
        num_iot = ['avg_co2_ppm','max_co2_ppm','avg_pm25','voc_index','avg_temp_celsius','avg_humidity_rh','avg_noise_db','waiting_room_occupancy','patient_throughput_per_hour','sanitizer_dispenses_per_hour']
        for col in num_iot: df[col] = _convert_to_numeric(df.get(col), np.nan)
        for col in ['clinic_id','room_name','zone_id']: df[col] = df.get(col,pd.Series(dtype=str)).fillna("Unknown").astype(str).str.strip()
        logger.info("IoT data cleaning complete."); return df
    except Exception as e: logger.error(f"Load IoT data error: {e}", exc_info=True); return pd.DataFrame()

@st.cache_data(ttl=app_config.CACHE_TTL_SECONDS, hash_funcs={gpd.GeoDataFrame: hash_geodataframe}, show_spinner="Loading zone data...")
def load_zone_data(attributes_path: str = None, geometries_path: str = None) -> Optional[gpd.GeoDataFrame]:
    attributes_path = attributes_path or app_config.ZONE_ATTRIBUTES_CSV; geometries_path = geometries_path or app_config.ZONE_GEOMETRIES_GEOJSON
    if not os.path.exists(attributes_path) or not os.path.exists(geometries_path):
        errs = "".join([f"Attrs missing: {os.path.basename(attributes_path)}. " if not os.path.exists(attributes_path) else "", f"Geoms missing: {os.path.basename(geometries_path)}." if not os.path.exists(geometries_path) else ""])
        logger.error(errs.strip()); st.error(f"🚨 GIS Data Error: {errs.strip()}"); return None
    try:
        attrs_df = pd.read_csv(attributes_path); attrs_df = _clean_column_names(attrs_df)
        geoms_gdf = gpd.read_file(geometries_path); geoms_gdf = _clean_column_names(geoms_gdf)
        if 'zone_id' not in attrs_df.columns or 'zone_id' not in geoms_gdf.columns: logger.error("'zone_id' missing."); st.error("🚨 Key 'zone_id' missing."); return None
        attrs_df['zone_id']=attrs_df['zone_id'].astype(str).str.strip(); geoms_gdf['zone_id']=geoms_gdf['zone_id'].astype(str).str.strip()
        if 'zone_display_name' in attrs_df.columns: attrs_df.rename(columns={'zone_display_name':'name'},inplace=True)
        elif 'name' not in attrs_df.columns and 'zone_id' in attrs_df: attrs_df['name']= "Zone " + attrs_df['zone_id']
        mrg_gdf = geoms_gdf.merge(attrs_df, on="zone_id", how="left", suffixes=('_geom','_attr'))
        for col in attrs_df.columns: # Resolve merge suffixes
            if f"{col}_attr" in mrg_gdf.columns and f"{col}_geom" in mrg_gdf.columns and col != 'zone_id':
                 mrg_gdf[col] = mrg_gdf[f"{col}_attr"].fillna(mrg_gdf[f"{col}_geom"])
                 mrg_gdf.drop(columns=[f"{col}_geom", f"{col}_attr"],inplace=True,errors='ignore')
            elif f"{col}_attr" in mrg_gdf.columns and col not in mrg_gdf.columns: mrg_gdf.rename(columns={f"{col}_attr":col},inplace=True)
            elif f"{col}_geom" in mrg_gdf.columns and col not in mrg_gdf.columns: mrg_gdf.rename(columns={f"{col}_geom":col},inplace=True)
        geom_col_name = mrg_gdf.geometry.name if hasattr(mrg_gdf, 'geometry') else 'geometry'
        if geom_col_name != 'geometry' and 'geometry' in mrg_gdf.columns: mrg_gdf = mrg_gdf.set_geometry('geometry', inplace=False)
        elif 'geometry' not in mrg_gdf.columns and geom_col_name in mrg_gdf.columns : mrg_gdf=mrg_gdf.set_geometry(geom_col_name,inplace=False)
        if mrg_gdf.crs is None: mrg_gdf=mrg_gdf.set_crs(app_config.DEFAULT_CRS, allow_override=True)
        elif mrg_gdf.crs.to_string().upper() != app_config.DEFAULT_CRS.upper(): mrg_gdf=mrg_gdf.to_crs(app_config.DEFAULT_CRS)
        req_cols=['zone_id','name','population','geometry','num_clinics','socio_economic_index','avg_travel_time_clinic_min']
        for r_col in req_cols:
            if r_col not in mrg_gdf.columns:
                defaults={'population':0.0,'num_clinics':0.0,'socio_economic_index':0.5,'avg_travel_time_clinic_min':30.0,'name':f"Zone {str(mrg_gdf.get('zone_id','?'))}"}
                mrg_gdf[r_col]=defaults.get(r_col, "Unknown" if r_col not in ['geometry'] else None)
        for n_col in ['population','socio_economic_index','num_clinics','avg_travel_time_clinic_min']:
            if n_col in mrg_gdf.columns: mrg_gdf[n_col] = _convert_to_numeric(mrg_gdf[n_col], 0 if n_col in ['population','num_clinics'] else (0.5 if n_col=='socio_economic_index' else 30.0))
        logger.info(f"Zone data loaded: {len(mrg_gdf)} zones."); return mrg_gdf
    except Exception as e: logger.error(f"Zone data error: {e}", exc_info=True); st.error(f"GIS data error: {e}"); return None

def enrich_zone_geodata_with_health_aggregates(zone_gdf: gpd.GeoDataFrame, health_df: pd.DataFrame, iot_df: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
    if zone_gdf is None or zone_gdf.empty or 'zone_id' not in zone_gdf.columns: return zone_gdf if zone_gdf is not None else gpd.GeoDataFrame(columns=['zone_id', 'name', 'geometry', 'population'], crs=app_config.DEFAULT_CRS)
    enriched = zone_gdf.copy()
    if 'population' not in enriched.columns: enriched['population'] = 0.0; enriched['population'] = _convert_to_numeric(enriched['population'], 0.0)
    agg_cols_to_init = ['total_population_health_data', 'avg_risk_score', 'total_patient_encounters', 'active_tb_cases', 'active_malaria_cases', 'hiv_positive_cases', 'pneumonia_cases', 'total_referrals_made', 'successful_referrals', 'avg_test_turnaround_critical', 'perc_critical_tests_tat_met', 'prevalence_per_1000', 'total_active_key_infections', 'avg_daily_steps_zone', 'zone_avg_co2', 'facility_coverage_score']
    for col in agg_cols_to_init: enriched[col] = 0.0 # Initialize all expected aggregate columns
    if health_df is not None and not health_df.empty and 'zone_id' in health_df.columns:
        hdfa = health_df.copy(); hdfa['zone_id'] = hdfa['zone_id'].astype(str).str.strip()
        enriched = _robust_merge_agg(enriched, hdfa.groupby('zone_id')['patient_id'].nunique().reset_index(), 'total_population_health_data')
        enriched = _robust_merge_agg(enriched, hdfa.groupby('zone_id')['ai_risk_score'].mean().reset_index(), 'avg_risk_score', default_fill_value=np.nan) # Averages can be NaN
        enriched = _robust_merge_agg(enriched, hdfa.groupby('zone_id')['encounter_id'].nunique().reset_index(), 'total_patient_encounters')
        for cond, agg_n in [("TB",'active_tb_cases'), ("Malaria",'active_malaria_cases'), ("HIV-Positive",'hiv_positive_cases'),("Pneumonia",'pneumonia_cases')]: enriched = _robust_merge_agg(enriched, hdfa[hdfa.get('condition',pd.Series(dtype=str)).str.contains(cond,case=False,na=False)].groupby('zone_id')['patient_id'].nunique().reset_index(), agg_n)
        key_cond_b = ['TB','Malaria','HIV-Positive','Pneumonia']; enriched = _robust_merge_agg(enriched, hdfa[hdfa.get('condition',pd.Series(dtype=str)).isin(key_cond_b)].groupby('zone_id')['patient_id'].nunique().reset_index(), 'total_active_key_infections')
        if 'referral_status' in hdfa.columns:
            enriched = _robust_merge_agg(enriched, hdfa[hdfa['referral_status'].notna() & (~hdfa['referral_status'].isin(['N/A','Unknown']))].groupby('zone_id')['encounter_id'].nunique().reset_index(), 'total_referrals_made')
            if 'referral_outcome' in hdfa.columns: enriched = _robust_merge_agg(enriched, hdfa[hdfa['referral_outcome'].isin(['Completed','Service Provided','Attended'])].groupby('zone_id')['encounter_id'].nunique().reset_index(), 'successful_referrals')
        crit_keys_data_enrich = [k for k in app_config.CRITICAL_TESTS_LIST if k in hdfa.get('test_type', pd.Series(dtype=str)).unique()]
        if crit_keys_data_enrich:
            tat_df_en = hdfa[(hdfa['test_type'].isin(crit_keys_data_enrich)) & (hdfa['test_turnaround_days'].notna()) & (~hdfa['test_result'].isin(['Pending','Rejected Sample','Unknown','Indeterminate']))].copy()
            if not tat_df_en.empty:
                enriched = _robust_merge_agg(enriched, tat_df_en.groupby('zone_id')['test_turnaround_days'].mean().reset_index(), 'avg_test_turnaround_critical', default_fill_value=np.nan)
                def _ctm_e(r): cfg=app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(r['test_type']); return r['test_turnaround_days']<=(cfg['target_tat_days'] if cfg and 'target_tat_days' in cfg else app_config.TARGET_TEST_TURNAROUND_DAYS)
                tat_df_en['tm_f_e'] = tat_df_en.apply(_ctm_e, axis=1) # Use direct assignment to avoid SettingWithCopyWarning
                pm_agg_e = tat_df_en.groupby('zone_id')['tm_f_e'].mean().reset_index().rename(columns={'tm_f_e':'val_merge'})
                enriched = _robust_merge_agg(enriched, pm_agg_e, 'perc_critical_tests_tat_met')
                if 'perc_critical_tests_tat_met' in enriched.columns: enriched['perc_critical_tests_tat_met'] = enriched['perc_critical_tests_tat_met'] * 100
        if 'avg_daily_steps' in hdfa.columns: enriched = _robust_merge_agg(enriched, hdfa.groupby('zone_id')['avg_daily_steps'].mean().reset_index(), 'avg_daily_steps_zone', default_fill_value=np.nan)
    if iot_df is not None and not iot_df.empty and all(c in iot_df.columns for c in ['zone_id','avg_co2_ppm']): iot_df['zone_id']=iot_df['zone_id'].astype(str).str.strip(); enriched=_robust_merge_agg(enriched,iot_df.groupby('zone_id')['avg_co2_ppm'].mean().reset_index(), 'zone_avg_co2', default_fill_value=np.nan)
    if 'total_active_key_infections' in enriched.columns and 'population' in enriched.columns: enriched['prevalence_per_1000'] = enriched.apply(lambda r:(r.get('total_active_key_infections',0)/r['population'])*1000 if pd.notna(r['population']) and r['population']>0 else 0.0,axis=1).fillna(0.0)
    if 'num_clinics' in enriched.columns and 'population' in enriched.columns: enriched['facility_coverage_score'] = enriched.apply(lambda r:(r.get('num_clinics',0)/r['population'])*10000*5 if pd.notna(r['population']) and r['population']>0 else 0.0,axis=1).fillna(0.0).clip(0,100)
    elif 'facility_coverage_score' not in enriched.columns: enriched['facility_coverage_score']=0.0
    for col in agg_cols_to_init: enriched[col] = pd.to_numeric(enriched.get(col,0.0),errors='coerce').fillna(0.0)
    logger.info("Zone GDF enrichment complete."); return enriched

def get_overall_kpis(health_df: pd.DataFrame, date_filter_start: Optional[str]=None, date_filter_end: Optional[str]=None) -> Dict[str, Any]:
    kpis = { "total_patients": 0, "avg_patient_risk": np.nan, "active_tb_cases_current": 0, "malaria_rdt_positive_rate_period": 0.0, "hiv_rapid_positive_rate_period": 0.0, "key_supply_stockout_alerts": 0 }
    if health_df is None or health_df.empty: return kpis
    df = health_df.copy()
    if 'encounter_date' not in df.columns or df['encounter_date'].isnull().all(): return kpis
    df['encounter_date'] = pd.to_datetime(df['encounter_date'], errors='coerce'); df.dropna(subset=['encounter_date'], inplace=True)
    start_date_dt = pd.to_datetime(date_filter_start, errors='coerce') if date_filter_start else None
    end_date_dt = pd.to_datetime(date_filter_end, errors='coerce') if date_filter_end else None
    if start_date_dt: df = df[df['encounter_date'] >= start_date_dt]
    if end_date_dt: df = df[df['encounter_date'] <= end_date_dt]
    if df.empty: return kpis
    
    kpis["total_patients"] = df['patient_id'].nunique()
    if 'ai_risk_score' in df.columns and df['ai_risk_score'].notna().any(): kpis["avg_patient_risk"] = df['ai_risk_score'].mean()
    if 'condition' in df.columns: kpis["active_tb_cases_current"] = df[df['condition'].str.contains("TB", case=False, na=False)]['patient_id'].nunique()
    
    for test_key_conf, kpi_name_conf in [("RDT-Malaria", "malaria_rdt_positive_rate_period"), ("HIV-Rapid", "hiv_rapid_positive_rate_period")]:
        test_type_val_in_data = test_key_conf # Assuming data column 'test_type' holds original keys
        test_df = df[(df.get('test_type') == test_type_val_in_data) & (~df.get('test_result', pd.Series(dtype=str)).isin(["Pending", "Rejected Sample", "Unknown"]))]
        if not test_df.empty and len(test_df) > 0: kpis[kpi_name_conf] = (test_df[test_df['test_result'] == 'Positive'].shape[0] / len(test_df)) * 100
    
    if all(c in df for c in ['item','item_stock_agg_zone','consumption_rate_per_day', 'encounter_date']):
        s_df = df.sort_values('encounter_date').drop_duplicates(subset=['item','zone_id'], keep='last') # zone_id relevant for item_stock_agg_zone
        s_df['days_supply'] = s_df['item_stock_agg_zone'] / (s_df['consumption_rate_per_day'].replace(0,np.nan)); s_df.dropna(subset=['days_supply'],inplace=True)
        kpis['key_supply_stockout_alerts'] = s_df[s_df['days_supply'] < app_config.CRITICAL_SUPPLY_DAYS]['item'].nunique()
    return kpis

def get_chw_summary(health_df_daily: pd.DataFrame) -> Dict[str, Any]:
    summary = {"visits_today":0,"tb_contacts_to_trace_today":0,"sti_symptomatic_referrals_today":0,"avg_patient_risk_visited_today":np.nan,"high_risk_followups_today":0,"patients_low_spo2_visited_today":0,"patients_fever_visited_today":0,"avg_patient_steps_visited_today":np.nan,"patients_fall_detected_today":0}
    if health_df_daily is None or health_df_daily.empty: return summary
    chw_enc_df = health_df_daily.copy() # Caller should ideally pre-filter if specific CHW filtering is needed beyond day/zone.
    summary["visits_today"]=chw_enc_df['patient_id'].nunique()
    if all(c in chw_enc_df for c in ['condition','referral_reason','referral_status']): summary["tb_contacts_to_trace_today"]=chw_enc_df[(chw_enc_df['condition'].str.contains('TB',na=False,case=False))&(chw_enc_df['referral_reason'].str.contains("Contact Tracing|Investigation",case=False,na=False))&(chw_enc_df['referral_status']=='Pending')]['patient_id'].nunique(); summary["sti_symptomatic_referrals_today"]=chw_enc_df[(chw_enc_df['condition'].str.contains("STI",case=False,na=False))&(chw_enc_df.get('patient_reported_symptoms',pd.Series(dtype=str)).astype(str).str.lower()!="unknown")&(chw_enc_df['referral_status']=='Pending')]['patient_id'].nunique()
    if 'ai_risk_score' in chw_enc_df and chw_enc_df['ai_risk_score'].notna().any(): summary["avg_patient_risk_visited_today"]=chw_enc_df['ai_risk_score'].mean(); summary["high_risk_followups_today"]=chw_enc_df[chw_enc_df['ai_risk_score']>=app_config.RISK_THRESHOLDS.get('high',75)]['patient_id'].nunique()
    if 'min_spo2_pct' in chw_enc_df: summary["patients_low_spo2_visited_today"]=chw_enc_df[chw_enc_df['min_spo2_pct'] < app_config.SPO2_LOW_THRESHOLD_PCT]['patient_id'].nunique()
    temp_col = next((tc for tc in ['vital_signs_temperature_celsius','max_skin_temp_celsius'] if tc in chw_enc_df and chw_enc_df[tc].notna().any()), None)
    if temp_col: summary["patients_fever_visited_today"]=chw_enc_df[chw_enc_df[temp_col]>=app_config.SKIN_TEMP_FEVER_THRESHOLD_C]['patient_id'].nunique()
    if 'avg_daily_steps' in chw_enc_df and chw_enc_df['avg_daily_steps'].notna().any(): summary["avg_patient_steps_visited_today"]=chw_enc_df['avg_daily_steps'].mean()
    if 'fall_detected_today' in chw_enc_df and chw_enc_df['fall_detected_today'].notna().any(): summary["patients_fall_detected_today"]=chw_enc_df[chw_enc_df['fall_detected_today'] > 0]['patient_id'].nunique()
    return summary

def get_patient_alerts_for_chw(health_df_daily: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_THRESHOLDS['chw_alert_moderate'], risk_threshold_high: int = app_config.RISK_THRESHOLDS['chw_alert_high']) -> pd.DataFrame:
    if health_df_daily is None or health_df_daily.empty: return pd.DataFrame()
    alerts = []; df_alerts = health_df_daily.copy(); 
    cols_needed = ['patient_id','ai_risk_score','ai_followup_priority_score','min_spo2_pct','vital_signs_temperature_celsius','max_skin_temp_celsius','condition','referral_status','fall_detected_today','encounter_date']
    for c in cols_needed: # Ensure columns exist with appropriate defaults
        df_alerts[c] = df_alerts.get(c, pd.Series(dtype = ('datetime64[ns]' if c == 'encounter_date' else (float if c in ['ai_risk_score','ai_followup_priority_score','min_spo2_pct','vital_signs_temperature_celsius','max_skin_temp_celsius','fall_detected_today'] else object) ) ) )
        if df_alerts[c].dtype == object and c not in ['encounter_date']: df_alerts[c] = df_alerts[c].fillna("Unknown")
    
    if df_alerts['ai_followup_priority_score'].notna().any(): list(alerts.append({**r.to_dict(), 'alert_reason':"High AI Prio", 'priority_score':r['ai_followup_priority_score']}) for _,r in df_alerts[df_alerts['ai_followup_priority_score']>=80].iterrows())
    if df_alerts['min_spo2_pct'].notna().any(): list(alerts.append({**r.to_dict(), 'alert_reason':f"Crit SpO2 ({r['min_spo2_pct']}%)", 'priority_score':90+(app_config.SPO2_CRITICAL_THRESHOLD_PCT-r['min_spo2_pct'])}) for _,r in df_alerts[df_alerts['min_spo2_pct'] < app_config.SPO2_CRITICAL_THRESHOLD_PCT].iterrows())
    temp_c = next((tc for tc in ['vital_signs_temperature_celsius','max_skin_temp_celsius'] if tc in df_alerts and df_alerts[tc].notna().any()),None)
    if temp_c and df_alerts[temp_c].notna().any(): list(alerts.append({**r.to_dict(), 'alert_reason':f"High Fever ({r[temp_c]}°C)", 'priority_score':85+(r[temp_c]-(app_config.SKIN_TEMP_FEVER_THRESHOLD_C+1.0))}) for _,r in df_alerts[df_alerts[temp_c] >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C+1.0].iterrows())
    if 'fall_detected_today' in df_alerts and df_alerts['fall_detected_today'].notna().any(): list(alerts.append({**r.to_dict(),'alert_reason':"Fall Detect",'priority_score':88}) for _,r in df_alerts[df_alerts['fall_detected_today']>0].iterrows())
    if 'ai_risk_score' in df_alerts and df_alerts['ai_risk_score'].notna().any(): list(alerts.append({**r.to_dict(),'alert_reason':"High AI Risk",'priority_score':r['ai_risk_score']}) for _,r in df_alerts[df_alerts['ai_risk_score']>=risk_threshold_high].iterrows())
    if 'condition' in df_alerts and 'referral_status' in df_alerts : list(alerts.append({**r.to_dict(),'alert_reason':f"Pend Ref: {r['condition']}",'priority_score':70}) for _,r in df_alerts[(df_alerts['condition'].isin(app_config.KEY_CONDITIONS_FOR_TRENDS[:4]))&(df_alerts['referral_status']=='Pending')].iterrows())
    if 'ai_risk_score' in df_alerts and df_alerts['ai_risk_score'].notna().any(): list(alerts.append({**r.to_dict(),'alert_reason':"Mod AI Risk",'priority_score':r['ai_risk_score']}) for _,r in df_alerts[(df_alerts['ai_risk_score']>=risk_threshold_moderate)&(df_alerts['ai_risk_score']<risk_threshold_high)].iterrows())
    
    if not alerts: return pd.DataFrame(columns=cols_needed + ['alert_reason', 'priority_score'])
    alert_df_final = pd.DataFrame(alerts)
    alert_df_final['encounter_date'] = pd.to_datetime(alert_df_final.get('encounter_date'), errors='coerce')
    
    # Use direct assignment to avoid SettingWithCopyWarning
    alert_df_final = alert_df_final.drop_duplicates(subset=['patient_id', 'alert_reason', 'encounter_date'])
    alert_df_final['priority_score'] = alert_df_final.get('priority_score', 0).fillna(0).astype(int)
    
    sort_c_final = ['priority_score', 'encounter_date']
    sort_asc_flags = [False, False]
    return alert_df_final.sort_values(by=sort_c_final, ascending=sort_asc_flags)

def get_clinic_summary(health_df_period: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str,Any]={"overall_avg_test_turnaround":np.nan, "overall_perc_met_tat":0.0, "total_pending_critical_tests":0, "sample_rejection_rate":0.0, "key_drug_stockouts":0, "test_summary_details":{}}; 
    if health_df_period is None or health_df_period.empty: return summary
    df = health_df_period.copy()
    for col in ['test_type','test_result','sample_status','encounter_date']: df[col] = df.get(col, pd.Series(dtype='object' if col != 'encounter_date' else 'datetime64[ns]')).fillna("Unknown" if col != 'encounter_date' else pd.NaT)
    df['test_turnaround_days'] = _convert_to_numeric(df.get('test_turnaround_days'), np.nan)
    
    concl_df = df[~df['test_result'].isin(['Pending','Rejected Sample','Unknown','N/A','nan','Indeterminate']) & df['test_turnaround_days'].notna()].copy()
    all_proc_samp = df[~df['sample_status'].isin(['Pending','Unknown','N/A','nan'])].copy()
    if not concl_df.empty and concl_df['test_turnaround_days'].notna().any(): summary["overall_avg_test_turnaround"]=concl_df['test_turnaround_days'].mean()
    crit_cfg = {k:v for k,v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v.get("critical")}
    crit_keys_in_data = [k for k in crit_cfg.keys() if k in df.get('test_type', pd.Series(dtype=str)).unique()]
    
    crit_concl_df = concl_df[concl_df['test_type'].isin(crit_keys_in_data)].copy()
    if not crit_concl_df.empty:
        def _chk_tat_s(r): cfg_s=app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(r['test_type']); return r['test_turnaround_days']<=(cfg_s['target_tat_days'] if cfg_s and 'target_tat_days' in cfg_s else app_config.TARGET_TEST_TURNAROUND_DAYS)
        crit_concl_df['tat_met'] = crit_concl_df.apply(_chk_tat_s,axis=1) # Use direct assignment
        if not crit_concl_df['tat_met'].empty : summary["overall_perc_met_tat"]=(crit_concl_df['tat_met'].mean()*100)
    
    if crit_keys_in_data:
        summary["total_pending_critical_tests"]=df[(df['test_type'].isin(crit_keys_in_data))&(df['test_result']=='Pending')]['patient_id'].nunique()
    
    if not all_proc_samp.empty: summary["sample_rejection_rate"]=(all_proc_samp[all_proc_samp['sample_status']=='Rejected'].shape[0]/len(all_proc_samp))*100 if len(all_proc_samp) > 0 else 0
    
    test_sum_details={}
    for o_key,cfg_p in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items():
        d_name=cfg_p.get("display_name",o_key); actual_keys=cfg_p.get("types_in_group",[o_key]); actual_keys=[o_key] if isinstance(actual_keys,str) else actual_keys
        grp_df_for_sum = df[df['test_type'].isin(actual_keys)]
        stats_s = {"positive_rate":0.0,"avg_tat_days":np.nan,"perc_met_tat_target":0.0,"pending_count":0,"rejected_count":0,"total_conducted_conclusive":0}
        if grp_df_for_sum.empty: test_sum_details[d_name]=stats_s; continue
        grp_c_sum = grp_df_for_sum[~grp_df_for_sum['test_result'].isin(['Pending','Rejected Sample','Unknown','N/A','nan','Indeterminate']) & grp_df_for_sum['test_turnaround_days'].notna()].copy()
        stats_s["total_conducted_conclusive"]=len(grp_c_sum)
        if not grp_c_sum.empty:
            stats_s["positive_rate"]=(grp_c_sum[grp_c_sum['test_result']=='Positive'].shape[0]/len(grp_c_sum))*100 if len(grp_c_sum)>0 else 0.0
            if grp_c_sum['test_turnaround_days'].notna().any():stats_s["avg_tat_days"]=grp_c_sum['test_turnaround_days'].mean()
            tgt_tat_spec=cfg_p.get("target_tat_days",app_config.TARGET_TEST_TURNAROUND_DAYS)
            grp_c_sum['tat_met_s']=grp_c_sum['test_turnaround_days']<=tgt_tat_spec # Direct assignment
            if not grp_c_sum['tat_met_s'].empty: stats_s["perc_met_tat_target"]=grp_c_sum['tat_met_s'].mean()*100
        if 'test_result' in grp_df_for_sum.columns: stats_s["pending_count"]=grp_df_for_sum[grp_df_for_sum['test_result']=='Pending']['patient_id'].nunique()
        if 'sample_status' in grp_df_for_sum.columns: stats_s["rejected_count"]=grp_df_for_sum[grp_df_for_sum['sample_status']=='Rejected']['patient_id'].nunique()
        test_sum_details[d_name]=stats_s
    summary["test_summary_details"]=test_sum_details
    
    if all(c in df for c in ['item', 'item_stock_agg_zone', 'consumption_rate_per_day', 'encounter_date']) and app_config.KEY_DRUG_SUBSTRINGS_SUPPLY:
        key_drugs_df_sum = df[df['item'].str.contains('|'.join(app_config.KEY_DRUG_SUBSTRINGS_SUPPLY), case=False, na=False)].copy()
        if not key_drugs_df_sum.empty:
            key_drugs_df_sum['encounter_date'] = pd.to_datetime(key_drugs_df_sum['encounter_date'], errors='coerce')
            key_drugs_df_sum.dropna(subset=['encounter_date'], inplace=True)
            if not key_drugs_df_sum.empty:
                latest_key_supply_sum = key_drugs_df_sum.sort_values('encounter_date').drop_duplicates(subset=['item', 'zone_id'], keep='last')
                days_of_supply_calc = latest_key_supply_sum['item_stock_agg_zone'] / latest_key_supply_sum['consumption_rate_per_day'].replace(0, np.nan)
                latest_key_supply_sum = latest_key_supply_sum.assign(days_of_supply_calc=days_of_supply_calc)
                summary['key_drug_stockouts'] = latest_key_supply_sum[latest_key_supply_sum['days_of_supply_calc'] < app_config.CRITICAL_SUPPLY_DAYS]['item'].nunique()
    return summary

def get_clinic_environmental_summary(iot_df_period: pd.DataFrame) -> Dict[str, Any]:
    summary = {"avg_co2_overall":np.nan, "rooms_co2_alert_latest":0, "avg_pm25_overall":np.nan, "rooms_pm25_alert_latest":0, "avg_occupancy_overall":np.nan, "high_occupancy_alert_latest":False, "avg_noise_overall":np.nan, "rooms_noise_alert_latest":0}
    if iot_df_period is None or iot_df_period.empty or 'timestamp' not in iot_df_period.columns or not pd.api.types.is_datetime64_any_dtype(iot_df_period['timestamp']): return summary
    df_iot_sum = iot_df_period.copy(); num_cols_iot_sum = ['avg_co2_ppm','avg_pm25','waiting_room_occupancy','avg_noise_db']
    for col in num_cols_iot_sum: df_iot_sum[col] = _convert_to_numeric(df_iot_sum.get(col), np.nan)
    if df_iot_sum['avg_co2_ppm'].notna().any(): summary["avg_co2_overall"] = df_iot_sum['avg_co2_ppm'].mean()
    if df_iot_sum['avg_pm25'].notna().any(): summary["avg_pm25_overall"] = df_iot_sum['avg_pm25'].mean()
    if df_iot_sum['waiting_room_occupancy'].notna().any(): summary["avg_occupancy_overall"] = df_iot_sum['waiting_room_occupancy'].mean()
    if df_iot_sum['avg_noise_db'].notna().any(): summary["avg_noise_overall"] = df_iot_sum['avg_noise_db'].mean()
    if all(c in df_iot_sum for c in ['clinic_id','room_name','timestamp']):
        latest_reads_sum = df_iot_sum.sort_values('timestamp').drop_duplicates(subset=['clinic_id','room_name'], keep='last')
        if not latest_reads_sum.empty:
            if 'avg_co2_ppm' in latest_reads_sum and latest_reads_sum['avg_co2_ppm'].notna().any(): summary["rooms_co2_alert_latest"]=latest_reads_sum[latest_reads_sum['avg_co2_ppm']>app_config.CO2_LEVEL_ALERT_PPM].shape[0]
            if 'avg_pm25' in latest_reads_sum and latest_reads_sum['avg_pm25'].notna().any(): summary["rooms_pm25_alert_latest"]=latest_reads_sum[latest_reads_sum['avg_pm25']>app_config.PM25_ALERT_UGM3].shape[0]
            if 'waiting_room_occupancy' in latest_reads_sum and latest_reads_sum['waiting_room_occupancy'].notna().any(): summary["high_occupancy_alert_latest"]=(latest_reads_sum['waiting_room_occupancy']>app_config.TARGET_WAITING_ROOM_OCCUPANCY).any()
            if 'avg_noise_db' in latest_reads_sum and latest_reads_sum['avg_noise_db'].notna().any(): summary["rooms_noise_alert_latest"]=latest_reads_sum[latest_reads_sum['avg_noise_db']>app_config.NOISE_LEVEL_ALERT_DB].shape[0]
    return summary

def get_patient_alerts_for_clinic(health_df_period: pd.DataFrame, risk_threshold_moderate: int = app_config.RISK_THRESHOLDS['moderate']) -> pd.DataFrame:
    if health_df_period is None or health_df_period.empty: return pd.DataFrame()
    alerts_data = []; df_alerts = health_df_period.copy()
    alert_cols_c = ['patient_id','encounter_date','condition','ai_risk_score','ai_followup_priority_score','test_type','test_result','hiv_viral_load_copies_ml','sample_status','min_spo2_pct','vital_signs_temperature_celsius','max_skin_temp_celsius','referral_status','referral_reason']
    for col in alert_cols_c: df_alerts[col] = df_alerts.get(col, pd.Series(dtype=float if col in ['ai_risk_score','ai_followup_priority_score','hiv_viral_load_copies_ml','min_spo2_pct','vital_signs_temperature_celsius','max_skin_temp_celsius'] else ('datetime64[ns]' if col=='encounter_date' else object))).fillna("Unknown" if col not in ['encounter_date', 'ai_risk_score', 'ai_followup_priority_score', 'hiv_viral_load_copies_ml', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'max_skin_temp_celsius'] else (pd.NaT if col=='encounter_date' else np.nan))
    if df_alerts.get('ai_risk_score', pd.Series(dtype=float)).notna().any():
        for _, r in df_alerts[df_alerts.get('ai_risk_score', 0) >= risk_threshold_moderate].iterrows():
            alerts_data.append({**r.to_dict(), 'alert_reason': "High AI Risk", 'priority_score': r.get('ai_risk_score',0)})
    if not alerts_data: return pd.DataFrame(columns=df_alerts.columns.tolist() + ['alert_reason', 'priority_score'])
    alerts_df_final = pd.DataFrame(alerts_data); alerts_df_final['alert_reason'] = alerts_df_final.get('alert_reason', pd.Series(dtype=str)).astype(str).fillna("Unknown"); alerts_df_final['encounter_date'] = pd.to_datetime(alerts_df_final.get('encounter_date'), errors='coerce'); alerts_df_final.dropna(subset=['patient_id', 'encounter_date'], inplace=True)
    if alerts_df_final.empty: return pd.DataFrame(columns=alerts_df_final.columns)
    alerts_df_final['encounter_date_obj'] = alerts_df_final['encounter_date'].dt.date
    def agg_alerts_clinic(grp):
        fr = grp.iloc[0].to_dict(); ars = [str(r) for r in grp['alert_reason'].unique() if pd.notna(r)]; fr['alert_reason']="; ".join(sorted(list(set(ars)))) if ars else "General Alert"; fr['priority_score']=grp['priority_score'].max() if 'priority_score' in grp and grp['priority_score'].notna().any() else 0; return pd.Series(fr)
    valid_keys = alerts_df_final.dropna(subset=['patient_id', 'encounter_date_obj'])
    if valid_keys.empty: return pd.DataFrame(columns=alerts_df_final.columns)
    final_df_alerts = valid_keys.groupby(['patient_id', 'encounter_date_obj'], as_index=False).apply(agg_alerts_clinic, include_groups=False).reset_index(drop=True)
    if 'priority_score' in final_df_alerts.columns: final_df_alerts['priority_score'] = _convert_to_numeric(final_df_alerts['priority_score'],0).astype(int)
    else: final_df_alerts['priority_score'] = 0
    sort_date_col_f = 'encounter_date' if 'encounter_date' in final_df_alerts.columns and final_df_alerts['encounter_date'].notna().all() else ('encounter_date_obj' if 'encounter_date_obj' in final_df_alerts else None)
    sort_cols_f = ['priority_score']; sort_asc_f = [False]
    if sort_date_col_f: final_df_alerts[sort_date_col_f]=pd.to_datetime(final_df_alerts[sort_date_col_f], errors='coerce'); sort_cols_f.append(sort_date_col_f); sort_asc_f.append(False)
    return final_df_alerts.sort_values(by=sort_cols_f, ascending=sort_asc_f)

def get_district_summary_kpis(enriched_zone_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    kpis: Dict[str, Any] = {"total_population_district":0,"avg_population_risk":np.nan,"zones_high_risk_count":0,"overall_facility_coverage":np.nan,"district_tb_burden_total":0,"district_malaria_burden_total":0,"key_infection_prevalence_district_per_1000":np.nan,"population_weighted_avg_steps":np.nan,"avg_clinic_co2_district":np.nan}
    if enriched_zone_gdf is None or enriched_zone_gdf.empty: return kpis
    gdf = enriched_zone_gdf.copy()
    num_cols_kpi_dist = ['population','avg_risk_score','active_tb_cases','active_malaria_cases','total_active_key_infections','facility_coverage_score','avg_daily_steps_zone','zone_avg_co2']
    for col in num_cols_kpi_dist: gdf[col] = _convert_to_numeric(gdf.get(col,0.0),0.0)
    kpis["total_population_district"] = gdf['population'].sum() if 'population' in gdf.columns else 0
    if kpis["total_population_district"] > 0 and pd.notna(kpis["total_population_district"]):
        for mc, kc in [('avg_risk_score','avg_population_risk'), ('facility_coverage_score','overall_facility_coverage'), ('avg_daily_steps_zone','population_weighted_avg_steps')]:
            if mc in gdf.columns and gdf[mc].notna().any() and 'population' in gdf.columns and gdf['population'].notna().any() and gdf['population'].sum() > 0: # Check for non-zero sum of weights
                valid_weights = gdf.loc[gdf[mc].notna(), 'population']
                valid_values = gdf.loc[gdf[mc].notna(), mc]
                if valid_weights.sum() > 0 : kpis[kc] = np.average(valid_values, weights=valid_weights)
                else: kpis[kc] = valid_values.mean() if not valid_values.empty else np.nan # Fallback to unweighted mean or NaN
            else: kpis[kc] = np.nan
        if 'total_active_key_infections' in gdf.columns: kpis["key_infection_prevalence_district_per_1000"] = (gdf['total_active_key_infections'].sum()/kpis["total_population_district"])*1000 if kpis["total_population_district"] > 0 else 0.0
    else:
        for mc, kc in [('avg_risk_score','avg_population_risk'), ('facility_coverage_score','overall_facility_coverage'), ('avg_daily_steps_zone','population_weighted_avg_steps')]: kpis[kc] = gdf[mc].mean() if not gdf.empty and mc in gdf.columns and gdf[mc].notna().any() else np.nan
        kpis["key_infection_prevalence_district_per_1000"] = 0.0
    kpis["zones_high_risk_count"]=gdf[gdf['avg_risk_score']>=app_config.RISK_THRESHOLDS['district_zone_high_risk']].shape[0] if 'avg_risk_score' in gdf.columns else 0
    kpis["district_tb_burden_total"]=int(gdf.get('active_tb_cases',0).sum()); kpis["district_malaria_burden_total"]=int(gdf.get('active_malaria_cases',0).sum())
    kpis["avg_clinic_co2_district"]=gdf['zone_avg_co2'].mean() if 'zone_avg_co2' in gdf and gdf['zone_avg_co2'].notna().any() else np.nan
    return kpis

def get_trend_data(df: pd.DataFrame, value_col: str, date_col: str = 'encounter_date', period: str = 'D', agg_func: str = 'mean', filter_col: Optional[str] = None, filter_val: Optional[Any] = None) -> pd.Series:
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns: return pd.Series(dtype='float64')
    trend_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(trend_df[date_col]): 
        trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors='coerce')
    trend_df.dropna(subset=[date_col], inplace=True)
    if value_col not in trend_df.columns: return pd.Series(dtype='float64')
    
    if agg_func != 'nunique': 
        trend_df.dropna(subset=[value_col], inplace=True)
    if trend_df.empty: return pd.Series(dtype='float64')
    
    if filter_col and filter_col in trend_df.columns and filter_val is not None:
        trend_df = trend_df[trend_df[filter_col] == filter_val]
        if trend_df.empty: return pd.Series(dtype='float64')
    
    trend_df.set_index(date_col, inplace=True)
    
    # CRITICAL BUG FIX: Prevent TypeError on non-numeric columns
    if agg_func in ['mean', 'sum', 'median'] and not pd.api.types.is_numeric_dtype(trend_df[value_col]):
        logger.error(f"Cannot perform numeric aggregation '{agg_func}' on non-numeric column '{value_col}'.")
        return pd.Series(dtype='float64')

    try:
        resampled = trend_df.groupby(pd.Grouper(freq=period))
        if agg_func == 'nunique':
            trend_series = resampled[value_col].nunique()
        elif agg_func == 'sum':
            trend_series = resampled[value_col].sum()
        elif agg_func == 'median':
            trend_series = resampled[value_col].median()
        else: # Default to mean
            trend_series = resampled[value_col].mean()
    except Exception as e:
        logger.error(f"Trend error on column '{value_col}' with agg '{agg_func}': {e}", exc_info=True)
        return pd.Series(dtype='float64')
        
    return trend_series

def get_supply_forecast_data(health_df: pd.DataFrame, forecast_days_out: int = 30, item_filter_list: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generates a forward-looking supply forecast for specified items.
    This version includes the fix for the OutOfBoundsTimedelta error.
    """
    default_cols = ['item','date','current_stock','consumption_rate','forecast_stock','forecast_days','estimated_stockout_date','lower_ci','upper_ci','initial_days_supply']
    required_cols = ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']
    
    if health_df is None or health_df.empty or not all(c in health_df.columns for c in required_cols):
        logger.warning("Supply forecast skipped: DataFrame is missing one or more required columns.")
        return pd.DataFrame(columns=default_cols)

    df_copy = health_df[required_cols].copy()
    df_copy['encounter_date'] = pd.to_datetime(df_copy['encounter_date'], errors='coerce')
    df_copy.dropna(subset=['encounter_date', 'item'], inplace=True)
    if df_copy.empty:
        return pd.DataFrame(columns=default_cols)

    # Get the latest status for each item
    supply_status_df = df_copy.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
    
    if item_filter_list:
        supply_status_df = supply_status_df[supply_status_df['item'].isin(item_filter_list)]
    
    if supply_status_df.empty:
        return pd.DataFrame(columns=default_cols)

    forecasts = []
    
    for _, row in supply_status_df.iterrows():
        item = row['item']
        stock = row.get('item_stock_agg_zone', 0)
        cons_r = row.get('consumption_rate_per_day', 0)
        last_d = row['encounter_date']

        if pd.isna(stock) or pd.isna(cons_r) or pd.isna(last_d) or stock < 0:
            continue

        # CORE BUG FIX: Handle zero consumption before creating a Timedelta
        if cons_r > 1e-5: # Use a small epsilon to handle floating point inaccuracies
            days_of_supply = stock / cons_r
            est_stockout = last_d + pd.to_timedelta(days_of_supply, unit='D')
            init_days_supply = days_of_supply
        else:
            est_stockout = pd.NaT  # "Not a Time" for infinite supply
            init_days_supply = np.inf if stock > 0 else 0
        
        # Use a small positive number for forecasting calculations to avoid division-by-zero
        forecast_cons_r = max(1e-5, cons_r)

        dates = pd.date_range(start=last_d + pd.Timedelta(days=1), periods=forecast_days_out, freq='D')
        
        for i, fc_d in enumerate(dates):
            days_out = i + 1
            fc_stock = stock - (forecast_cons_r * days_out)
            fc_days = init_days_supply - days_out if np.isfinite(init_days_supply) else np.inf
            
            # Simplified confidence interval logic
            cons_std = 0.15
            low_c = forecast_cons_r * (1 + cons_std)
            upp_c = max(1e-5, forecast_cons_r * (1 - cons_std))
            
            low_ci_stock = stock - (low_c * days_out)
            upp_ci_stock = stock - (upp_c * days_out)
            
            low_ci_d = (low_ci_stock / low_c) if low_c > 1e-5 else (np.inf if low_ci_stock > 0 else 0)
            upp_ci_d = (upp_ci_stock / upp_c) if upp_c > 1e-5 else (np.inf if upp_ci_stock > 0 else 0)

            forecasts.append({
                'item': item,
                'date': fc_d,
                'current_stock': stock,
                'consumption_rate': cons_r,
                'forecast_stock': max(0, fc_stock),
                'forecast_days': max(0, fc_days),
                'estimated_stockout_date': est_stockout,
                'lower_ci': max(0, low_ci_d),
                'upper_ci': max(0, upp_ci_d),
                'initial_days_supply': init_days_supply
            })

    if not forecasts:
        return pd.DataFrame(columns=default_cols)
        
    return pd.DataFrame(forecasts)
