# tests/test_core_data_processing.py
import pytest
import pandas as pd
import geopandas as gpd
import os
import numpy as np
from unittest.mock import patch, MagicMock
from config import app_config

from utils.core_data_processing import (
    load_health_records, load_zone_data, load_iot_clinic_environment_data,
    enrich_zone_geodata_with_health_aggregates, get_overall_kpis,
    get_chw_summary, get_patient_alerts_for_chw, get_clinic_summary,
    get_clinic_environmental_summary, get_patient_alerts_for_clinic,
    get_trend_data, get_supply_forecast_data, get_district_summary_kpis
)

# === Test Data Loading Functions ===
@patch('utils.core_data_processing.pd.read_csv')
@patch('utils.core_data_processing.os.path.exists')
def test_load_health_records_success(mock_os_exists, mock_pd_read_csv, sample_health_records_df_main, mocker):
    mock_os_exists.return_value = True
    mock_pd_read_csv.return_value = sample_health_records_df_main.copy()
    mock_st_error = mocker.patch('utils.core_data_processing.st.error')
    df = load_health_records()
    mock_pd_read_csv.assert_called_once_with(app_config.HEALTH_RECORDS_CSV, low_memory=False)
    assert not df.empty
    assert "ai_risk_score" in df.columns and pd.api.types.is_numeric_dtype(df['ai_risk_score'])
    assert "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df['date'])
    assert "condition" in df.columns and (df['condition'].dtype == 'object' or pd.api.types.is_string_dtype(df['condition']))
    # Check for some expected cleaning, e.g., 'Unknown' for empty strings or N/A
    if 'notes' in df.columns: assert not df['notes'].isin(['', 'nan', 'None']).any() # Should be 'Unknown'
    mock_st_error.assert_not_called()

@patch('utils.core_data_processing.os.path.exists')
def test_load_health_records_file_not_found(mock_os_exists, mocker):
    mock_os_exists.return_value = False
    mock_st_error = mocker.patch('utils.core_data_processing.st.error')
    df = load_health_records()
    assert df.empty
    mock_st_error.assert_called_once() # Checks that st.error was called
    assert f"Data file '{os.path.basename(app_config.HEALTH_RECORDS_CSV)}' not found" in mock_st_error.call_args[0][0]


@patch('utils.core_data_processing.gpd.read_file')
@patch('utils.core_data_processing.pd.read_csv')
@patch('utils.core_data_processing.os.path.exists')
def test_load_zone_data_success(mock_os_exists, mock_pd_read_csv, mock_gpd_read_file, sample_zone_attributes_df_main, sample_zone_geometries_gdf_main, mocker):
    mock_os_exists.return_value = True
    mock_pd_read_csv.return_value = sample_zone_attributes_df_main.copy()
    # Ensure mock_gpd_read_file returns a GDF with a CRS
    gdf_with_crs = sample_zone_geometries_gdf_main.copy()
    if gdf_with_crs.crs is None: gdf_with_crs = gdf_with_crs.set_crs(app_config.DEFAULT_CRS)
    mock_gpd_read_file.return_value = gdf_with_crs

    mock_st_error = mocker.patch('utils.core_data_processing.st.error')
    merged_gdf = load_zone_data()
    assert isinstance(merged_gdf, gpd.GeoDataFrame)
    assert not merged_gdf.empty
    assert 'name' in merged_gdf.columns and 'population' in merged_gdf.columns
    assert 'geometry' in merged_gdf.columns
    assert merged_gdf.crs is not None and merged_gdf.crs.to_string().upper() == app_config.DEFAULT_CRS.upper()
    mock_st_error.assert_not_called()


@patch('utils.core_data_processing.pd.read_csv')
@patch('utils.core_data_processing.os.path.exists')
def test_load_iot_data_success(mock_os_exists, mock_pd_read_csv, sample_iot_clinic_df_main, mocker):
    mock_os_exists.return_value = True
    mock_pd_read_csv.return_value = sample_iot_clinic_df_main.copy()
    mock_st_error = mocker.patch('utils.core_data_processing.st.error')
    df = load_iot_clinic_environment_data()
    assert not df.empty
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
    assert 'zone_id' in df.columns and not df['zone_id'].str.contains('Unknown', na=False).all() # Check if zone_id derived
    assert pd.api.types.is_numeric_dtype(df['avg_co2_ppm'])
    mock_st_error.assert_not_called()

# === Test Data Enrichment ===
def test_enrich_zone_geodata_with_health_aggregates(sample_enriched_gdf_main): # Uses the comprehensive fixture
    # The sample_enriched_gdf_main fixture now calls the actual enrichment function
    enriched_gdf = sample_enriched_gdf_main
    assert isinstance(enriched_gdf, gpd.GeoDataFrame)
    assert not enriched_gdf.empty
    # Check for presence of key aggregated columns
    expected_cols = ['avg_risk_score', 'active_tb_cases', 'prevalence_per_1000', 'facility_coverage_score', 'zone_avg_co2', 'avg_daily_steps_zone']
    for col in expected_cols: assert col in enriched_gdf.columns and pd.api.types.is_numeric_dtype(enriched_gdf[col])
    
    # Specific checks (these would need exact calculations based on sample_health_records & sample_iot to be precise)
    assert enriched_gdf['avg_risk_score'].mean() > 0 # Assuming some risk exists
    assert enriched_gdf['zone_avg_co2'].mean() > 0   # Assuming some CO2 readings exist for some zones

def test_enrich_zone_geodata_empty_health_iot_df(sample_zone_geometries_gdf_main, sample_zone_attributes_df_main, empty_health_df_with_schema, empty_iot_df_with_schema):
    base_gdf = sample_zone_geometries_gdf_main.merge(sample_zone_attributes_df_main, on="zone_id", how="left")
    enriched = enrich_zone_geodata_with_health_aggregates(base_gdf, empty_health_df_with_schema, empty_iot_df_with_schema)
    assert isinstance(enriched, gpd.GeoDataFrame)
    assert not enriched.empty # Should return base_gdf with default values for agg cols
    assert 'avg_risk_score' in enriched.columns and enriched['avg_risk_score'].fillna(0).eq(0).all()
    assert 'zone_avg_co2' in enriched.columns and enriched['zone_avg_co2'].fillna(0).eq(0).all()

# === Test KPI Calculation Functions ===
def test_get_overall_kpis(sample_health_records_df_main):
    latest_date = sample_health_records_df_main['date'].max()
    kpis = get_overall_kpis(sample_health_records_df_main, date_filter_end=latest_date)
    assert kpis['total_patients'] > 0
    assert kpis['avg_patient_risk'] >= 0
    assert kpis['active_tb_cases_current'] >= 0 # Could be 0
    assert kpis['malaria_rdt_positive_rate_period'] >= 0.0

def test_get_chw_summary_single_day(sample_health_records_df_main):
    day_df = sample_health_records_df_main[sample_health_records_df_main['date'] == pd.to_datetime('2023-10-03')]
    summary = get_chw_summary(day_df)
    assert summary['visits_today'] >= 0
    assert 'avg_patient_risk_visited_today' in summary
    assert summary['patients_low_spo2_visited_today'] >= 0

def test_get_patient_alerts_for_chw_single_day(sample_health_records_df_main):
    day_df = sample_health_records_df_main[sample_health_records_df_main['date'] == pd.to_datetime('2023-10-03')]
    alerts = get_patient_alerts_for_chw(day_df)
    assert isinstance(alerts, pd.DataFrame)
    if not alerts.empty:
        assert 'alert_reason' in alerts.columns and 'priority_score' in alerts.columns

def test_get_clinic_summary_period(sample_health_records_df_main):
    period_df = sample_health_records_df_main[sample_health_records_df_main['date'] <= pd.to_datetime('2023-10-07')]
    summary = get_clinic_summary(period_df)
    assert 'tb_sputum_positivity' in summary and summary['tb_sputum_positivity'] >= 0.0
    assert 'key_drug_stockouts' in summary and summary['key_drug_stockouts'] >= 0

def test_get_clinic_environmental_summary_period(sample_iot_clinic_df_main):
    period_iot_df = sample_iot_clinic_df_main[sample_iot_clinic_df_main['timestamp'] <= pd.to_datetime('2023-10-02T23:59:59Z')]
    summary = get_clinic_environmental_summary(period_iot_df)
    assert 'avg_co2_overall' in summary
    assert summary['rooms_pm25_alert_latest'] >= 0

def test_get_patient_alerts_for_clinic_period(sample_health_records_df_main):
    period_df = sample_health_records_df_main[sample_health_records_df_main['date'] <= pd.to_datetime('2023-10-07')]
    alerts = get_patient_alerts_for_clinic(period_df)
    assert isinstance(alerts, pd.DataFrame)
    if not alerts.empty:
        assert 'alert_reason' in alerts.columns and 'priority_score' in alerts.columns

def test_get_trend_data(sample_health_records_df_main):
    trend = get_trend_data(sample_health_records_df_main, 'ai_risk_score', period='W', agg_func='mean')
    assert isinstance(trend, pd.Series)
    if not trend.empty: assert pd.api.types.is_datetime64_any_dtype(trend.index)

def test_get_supply_forecast_data(sample_health_records_df_main):
    forecast = get_supply_forecast_data(sample_health_records_df_main, forecast_days_out=7)
    assert isinstance(forecast, pd.DataFrame)
    if not forecast.empty:
        assert 'forecast_days' in forecast.columns and 'estimated_stockout_date' in forecast.columns

def test_get_district_summary_kpis(sample_enriched_gdf_main):
    if sample_enriched_gdf_main.empty: pytest.skip("Enriched GDF fixture is empty.")
    kpis = get_district_summary_kpis(sample_enriched_gdf_main)
    assert kpis['avg_population_risk'] >= 0
    assert kpis['zones_high_risk_count'] >= 0
    assert 'key_infection_prevalence_district_per_1000' in kpis

# Test cases with empty inputs
def test_empty_inputs_for_kpi_functions(empty_health_df_with_schema, empty_iot_df_with_schema, empty_gdf_with_schema):
    assert get_overall_kpis(empty_health_df_with_schema)['total_patients'] == 0
    assert get_chw_summary(empty_health_df_with_schema)['visits_today'] == 0
    assert get_patient_alerts_for_chw(empty_health_df_with_schema).empty
    assert get_clinic_summary(empty_health_df_with_schema)['tb_sputum_positivity'] == 0.0
    assert get_clinic_environmental_summary(empty_iot_df_with_schema)['avg_co2_overall'] == 0.0
    assert get_patient_alerts_for_clinic(empty_health_df_with_schema).empty
    assert get_district_summary_kpis(empty_gdf_with_schema)['avg_population_risk'] == 0.0
    assert get_trend_data(empty_health_df_with_schema, 'ai_risk_score').empty
    assert get_supply_forecast_data(empty_health_df_with_schema).empty
