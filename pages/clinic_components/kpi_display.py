# test/pages/clinic_components/kpi_display.py
import streamlit as st
from config import app_config
from utils.ui_visualization_helpers import render_kpi_card
import numpy as np # For np.nan if default value is NaN

def render_main_clinic_kpis(clinic_service_kpis, date_range_display_str):
    st.subheader(f"Overall Clinic Performance Summary {date_range_display_str}")
    kpi_cols_main_clinic = st.columns(4)
    with kpi_cols_main_clinic[0]:
        overall_tat_val = clinic_service_kpis.get('overall_avg_test_turnaround', np.nan) # Default to NaN for averages
        render_kpi_card("Overall Avg. TAT", f"{overall_tat_val:.1f}d" if pd.notna(overall_tat_val) else "N/A", "⏱️",
                        status="High" if pd.notna(overall_tat_val) and overall_tat_val > (app_config.TARGET_TEST_TURNAROUND_DAYS + 1) else ("Moderate" if pd.notna(overall_tat_val) and overall_tat_val > app_config.TARGET_TEST_TURNAROUND_DAYS else ("Low" if pd.notna(overall_tat_val) else "Neutral")),
                        help_text=f"Average TAT for all conclusive tests. Target: ≤{app_config.TARGET_TEST_TURNAROUND_DAYS} days.")
    with kpi_cols_main_clinic[1]:
        perc_met_tat_val = clinic_service_kpis.get('overall_perc_met_tat', 0.0)
        render_kpi_card("% Critical Tests TAT Met", f"{perc_met_tat_val:.1f}%", "🎯",
                        status="Good High" if perc_met_tat_val >= app_config.TARGET_OVERALL_TESTS_MEETING_TAT_PCT else "Bad Low",
                        help_text=f"Critical tests meeting TAT. Target: ≥{app_config.TARGET_OVERALL_TESTS_MEETING_TAT_PCT}%.")
    with kpi_cols_main_clinic[2]:
        pending_crit_val = clinic_service_kpis.get('total_pending_critical_tests', 0)
        render_kpi_card("Pending Critical Tests", str(pending_crit_val), "⏳",
                        status="High" if pending_crit_val > 10 else ("Moderate" if pending_crit_val > 0 else "Low"), 
                        help_text="Unique patients with critical tests still awaiting results.")
    with kpi_cols_main_clinic[3]:
        rejection_rate_val = clinic_service_kpis.get('sample_rejection_rate', 0.0)
        render_kpi_card("Sample Rejection Rate", f"{rejection_rate_val:.1f}%", "🚫",
                        status="High" if rejection_rate_val > app_config.TARGET_SAMPLE_REJECTION_RATE_PCT else "Low",
                        help_text=f"Overall sample rejection rate. Target: <{app_config.TARGET_SAMPLE_REJECTION_RATE_PCT}%.")

def render_disease_specific_kpis(clinic_service_kpis):
    st.markdown("##### Disease-Specific Test Positivity Rates (Selected Period)")
    test_details_for_kpis = clinic_service_kpis.get("test_summary_details", {})
    if not test_details_for_kpis:
        st.caption("Detailed test statistics are not available.")
        return
        
    kpi_cols_disease_pos = st.columns(4)

    tb_gx_key = next((k for k, v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if "genexpert" in v.get("display_name", "").lower()), "Sputum-GeneXpert")
    tb_gx_display_name = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(tb_gx_key, {}).get("display_name", "TB GeneXpert")
    with kpi_cols_disease_pos[0]:
        tb_pos_rate = test_details_for_kpis.get(tb_gx_display_name, {}).get("positive_rate", 0.0)
        render_kpi_card(f"{tb_gx_display_name} Pos.", f"{tb_pos_rate:.1f}%", "🫁", status="High" if tb_pos_rate > 10 else ("Moderate" if tb_pos_rate > 5 else "Low"))

    mal_rdt_key = next((k for k, v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if "rdt-malaria" in k.lower()), "RDT-Malaria")
    mal_rdt_display_name = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(mal_rdt_key, {}).get("display_name", "Malaria RDT")
    with kpi_cols_disease_pos[1]:
        mal_pos_rate = test_details_for_kpis.get(mal_rdt_display_name, {}).get("positive_rate", 0.0)
        render_kpi_card(f"{mal_rdt_display_name} Pos.", f"{mal_pos_rate:.1f}%", "🦟", status="High" if mal_pos_rate > app_config.TARGET_MALARIA_POSITIVITY_RATE else "Low")

    hiv_rapid_key = next((k for k, v in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if "hiv-rapid" in k.lower()), "HIV-Rapid")
    hiv_rapid_display_name = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(hiv_rapid_key, {}).get("display_name", "HIV Rapid Test")
    with kpi_cols_disease_pos[2]:
        hiv_pos_rate = test_details_for_kpis.get(hiv_rapid_display_name, {}).get("positive_rate", 0.0)
        render_kpi_card(f"{hiv_rapid_display_name} Pos.", f"{hiv_pos_rate:.1f}%", "🩸", status="Moderate" if hiv_pos_rate > 2 else "Low")

    with kpi_cols_disease_pos[3]:
        drug_stockouts_val = clinic_service_kpis.get('key_drug_stockouts', 0)
        render_kpi_card("Key Drug Stockouts", str(drug_stockouts_val), "💊", status="High" if drug_stockouts_val > 0 else "Low", help_text=f"Key drugs with <{app_config.CRITICAL_SUPPLY_DAYS} days supply.")
