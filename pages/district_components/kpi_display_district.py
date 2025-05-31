# test/pages/district_components/kpi_display_district.py
import streamlit as st
import pandas as pd # Not strictly needed here, but good practice if kpi calcs were complex
from config import app_config
from utils.ui_visualization_helpers import render_kpi_card

def render_district_kpis(district_overall_kpis, district_gdf_main_enriched): # Added GDF for total zones if needed
    st.subheader("District-Wide Key Performance Indicators (Latest Aggregated Zonal Data)")
    
    if district_overall_kpis is None or not district_overall_kpis:
        st.warning("District-Wide KPIs cannot be displayed: Calculation returned no data.")
        return

    kpi_cols_row1_dist = st.columns(4)
    with kpi_cols_row1_dist[0]:
        avg_pop_risk_val = district_overall_kpis.get('avg_population_risk', 0.0)
        render_kpi_card("Avg. Population AI Risk", f"{avg_pop_risk_val:.1f}", "🎯",
                        status="High" if avg_pop_risk_val >= app_config.RISK_THRESHOLDS['high'] else ("Moderate" if avg_pop_risk_val >= app_config.RISK_THRESHOLDS['moderate'] else "Low"),
                        help_text="Population-weighted average AI risk score across all zones.")
    with kpi_cols_row1_dist[1]:
        facility_coverage_val = district_overall_kpis.get('overall_facility_coverage', 0.0)
        render_kpi_card("Facility Coverage Score", f"{facility_coverage_val:.1f}%", "🏥",
                        status="Good High" if facility_coverage_val >= 80 else ("Moderate" if facility_coverage_val >= app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD else "Bad Low"),
                        help_text="Population-weighted score reflecting access and capacity of health facilities.")
    with kpi_cols_row1_dist[2]:
        high_risk_zones_num = district_overall_kpis.get('zones_high_risk_count', 0)
        total_zones_val = len(district_gdf_main_enriched) if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty else 1
        perc_high_risk_zones = (high_risk_zones_num / total_zones_val) * 100 if total_zones_val > 0 else 0.0
        render_kpi_card("High AI Risk Zones", f"{high_risk_zones_num} ({perc_high_risk_zones:.0f}%)", "⚠️",
                        status="High" if perc_high_risk_zones > 25 else ("Moderate" if high_risk_zones_num > 0 else "Low"),
                        help_text=f"Number (and %) of zones with average AI risk score ≥ {app_config.RISK_THRESHOLDS['district_zone_high_risk']}.")
    with kpi_cols_row1_dist[3]:
        district_prevalence_val = district_overall_kpis.get('key_infection_prevalence_district_per_1000', 0.0)
        render_kpi_card("Overall Key Inf. Prevalence", f"{district_prevalence_val:.1f} /1k Pop", "📈",
                        status="High" if district_prevalence_val > 50 else ("Moderate" if district_prevalence_val > 20 else "Low"),
                        help_text="Combined prevalence of key infectious diseases per 1,000 population.")

    st.markdown("##### Key Disease Burdens & District Wellness / Environment")
    kpi_cols_row2_dist = st.columns(4)
    with kpi_cols_row2_dist[0]:
        tb_total_burden = district_overall_kpis.get('district_tb_burden_total', 0)
        render_kpi_card("Total Active TB Cases", str(tb_total_burden), "🫁",
                        status="High" if tb_total_burden > (len(district_gdf_main_enriched) * app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty else 20) else ("Moderate" if tb_total_burden > 0 else "Low"),
                        help_text="Total active TB cases identified across the district.")
    with kpi_cols_row2_dist[1]:
        malaria_total_burden = district_overall_kpis.get('district_malaria_burden_total',0)
        render_kpi_card("Total Active Malaria Cases", str(malaria_total_burden), "🦟",
                        status="High" if malaria_total_burden > (len(district_gdf_main_enriched) * app_config.INTERVENTION_MALARIA_BURDEN_HIGH_THRESHOLD if district_gdf_main_enriched is not None and not district_gdf_main_enriched.empty else 50) else ("Moderate" if malaria_total_burden > 0 else "Low"),
                        help_text="Total active Malaria cases identified across the district.")
    with kpi_cols_row2_dist[2]:
        avg_steps_district = district_overall_kpis.get('population_weighted_avg_steps', 0.0)
        render_kpi_card("Avg. Patient Steps (Pop. Weighted)", f"{avg_steps_district:,.0f}", "👣",
                        status="Good High" if avg_steps_district >= app_config.TARGET_DAILY_STEPS else ("Moderate" if avg_steps_district >= app_config.TARGET_DAILY_STEPS * 0.7 else "Bad Low"),
                        help_text=f"Population-weighted average daily steps. Target: {app_config.TARGET_DAILY_STEPS:,.0f} steps.")
    with kpi_cols_row2_dist[3]:
        avg_co2_district_val = district_overall_kpis.get('avg_clinic_co2_district',0.0)
        render_kpi_card("Avg. Clinic CO2 (District)", f"{avg_co2_district_val:.0f} ppm", "💨",
                        status="High" if avg_co2_district_val > app_config.CO2_LEVEL_ALERT_PPM else ("Moderate" if avg_co2_district_val > app_config.CO2_LEVEL_IDEAL_PPM else "Low"),
                        help_text="District average of zonal mean CO2 levels in clinics.")
