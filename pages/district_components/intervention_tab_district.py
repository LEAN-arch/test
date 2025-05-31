# test/pages/district_components/intervention_tab_district.py
import streamlit as st
import pandas as pd
from config import app_config
import logging

logger = logging.getLogger(__name__)

def render_intervention_planning_tab(district_gdf_main_enriched):
    st.header("🎯 Intervention Planning Insights")

    if district_gdf_main_enriched is None or district_gdf_main_enriched.empty or \
       'geometry' not in district_gdf_main_enriched.columns: # Basic check
        st.info("Intervention planning insights require successfully loaded and enriched geographic zone data.")
        return

    st.markdown("Identify zones for targeted interventions based on customizable criteria related to health risks, disease burdens, resource accessibility, and environmental factors from the latest aggregated data.")
    
    criteria_lambdas_intervention_dist = {
        f"High Avg. AI Risk (Score ≥ {app_config.RISK_THRESHOLDS['district_zone_high_risk']})":
            lambda df_interv: df_interv.get('avg_risk_score', pd.Series(dtype=float)) >= app_config.RISK_THRESHOLDS['district_zone_high_risk'],
        f"Low Facility Coverage (< {app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD}%)":
            lambda df_interv: df_interv.get('facility_coverage_score', pd.Series(dtype=float)) < app_config.INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD,
        f"High Key Inf. Prevalence (Top {100 - app_config.INTERVENTION_PREVALENCE_HIGH_PERCENTILE*100:.0f}%)":
            lambda df_interv: df_interv.get('prevalence_per_1000', pd.Series(dtype=float)) >= df_interv.get('prevalence_per_1000', pd.Series(dtype=float)).quantile(app_config.INTERVENTION_PREVALENCE_HIGH_PERCENTILE) if 'prevalence_per_1000' in df_interv and df_interv['prevalence_per_1000'].notna().any() and len(df_interv['prevalence_per_1000'].dropna()) > 1 else pd.Series([False]*len(df_interv), index=df_interv.index),
        f"High TB Burden (Abs. Cases > {app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD} per zone)":
            lambda df_interv: df_interv.get('active_tb_cases', pd.Series(dtype=float)) > app_config.INTERVENTION_TB_BURDEN_HIGH_THRESHOLD,
        f"High Avg. Clinic CO2 (> {app_config.CO2_LEVEL_IDEAL_PPM}ppm in zone)":
            lambda df_interv: df_interv.get('zone_avg_co2', pd.Series(dtype=float)) > app_config.CO2_LEVEL_IDEAL_PPM
    }
    
    available_criteria_for_intervention_dist = {}
    base_cols_for_test = ['avg_risk_score', 'facility_coverage_score', 'prevalence_per_1000', 'active_tb_cases', 'zone_avg_co2']
    for name_crit_int, func_crit_int in criteria_lambdas_intervention_dist.items():
        try:
            # Create a minimal test DataFrame ensuring expected columns exist, even if GDF is empty.
            # This prevents lambdas from failing on .get() if column doesn't exist during the test call.
            if not district_gdf_main_enriched.empty:
                test_apply_df_interv = district_gdf_main_enriched.head(1) # Use actual GDF if possible
            else: # If GDF is empty, create a dummy for lambda applicability check
                test_apply_df_interv = pd.DataFrame(columns=base_cols_for_test, data=[[0]*len(base_cols_for_test)])

            # Ensure the dummy df has all columns the lambda *might* try to access with .get()
            for expected_col in base_cols_for_test:
                if expected_col not in test_apply_df_interv.columns:
                    test_apply_df_interv[expected_col] = 0.0 
            
            func_crit_int(test_apply_df_interv) # Test if lambda function can be called
            available_criteria_for_intervention_dist[name_crit_int] = func_crit_int
        except Exception as e_crit_test_apply:
            logger.debug(f"Intervention criterion '{name_crit_int}' not available for selection. Error: {e_crit_test_apply}")

    if not available_criteria_for_intervention_dist:
        st.warning("No intervention criteria can be applied. Relevant data columns may be missing or encounter errors.")
        return

    default_selection_interv = list(available_criteria_for_intervention_dist.keys())[0:min(2, len(available_criteria_for_intervention_dist))]
    selected_criteria_names_interv = st.multiselect(
        "Select Criteria to Identify Priority Zones (Zones meeting ANY selected criteria will be shown):",
        options=list(available_criteria_for_intervention_dist.keys()),
        default=default_selection_interv,
        key="district_intervention_criteria_multiselect_comp_v1" # Component-specific key
    )
    
    if not selected_criteria_names_interv:
        st.info("Please select at least one criterion above to identify priority zones.")
        return

    final_intervention_mask_dist = pd.Series([False] * len(district_gdf_main_enriched), index=district_gdf_main_enriched.index)
    for crit_name_selected_interv in selected_criteria_names_interv:
        crit_func_selected_interv = available_criteria_for_intervention_dist[crit_name_selected_interv]
        try:
            current_crit_mask_interv = crit_func_selected_interv(district_gdf_main_enriched)
            if isinstance(current_crit_mask_interv, pd.Series) and current_crit_mask_interv.dtype == 'bool':
                final_intervention_mask_dist = final_intervention_mask_dist | current_crit_mask_interv.fillna(False)
            else:
                logger.warning(f"Intervention criterion '{crit_name_selected_interv}' did not produce a valid boolean Series. Type: {type(current_crit_mask_interv)}")
        except Exception as e_crit_apply_interv_actual:
            logger.error(f"Error applying actual intervention criterion '{crit_name_selected_interv}': {e_crit_apply_interv_actual}", exc_info=True)
            st.warning(f"Could not apply criterion: {crit_name_selected_interv}. Check logs for details.")

    priority_zones_df_for_interv = district_gdf_main_enriched[final_intervention_mask_dist].copy()
    
    if not priority_zones_df_for_interv.empty:
        st.markdown(f"###### Identified **{len(priority_zones_df_for_interv)}** Zone(s) Meeting Selected Intervention Criteria:")
        cols_interv_table = ['name', 'population', 'avg_risk_score', 'total_active_key_infections', 'prevalence_per_1000', 'facility_coverage_score', 'zone_avg_co2', 'active_tb_cases', 'active_malaria_cases']
        actual_cols_interv_table = [col for col in cols_interv_table if col in priority_zones_df_for_interv.columns]
        
        sort_by = []; sort_asc = []
        if 'avg_risk_score' in actual_cols_interv_table: sort_by.append('avg_risk_score'); sort_asc.append(False)
        if 'prevalence_per_1000' in actual_cols_interv_table: sort_by.append('prevalence_per_1000'); sort_asc.append(False)
        if 'facility_coverage_score' in actual_cols_interv_table: sort_by.append('facility_coverage_score'); sort_asc.append(True)
        
        interv_df_sorted = priority_zones_df_for_interv.sort_values(by=sort_by, ascending=sort_asc) if sort_by else priority_zones_df_for_interv
        
        # Prepare for st.dataframe (ensure serializable)
        df_to_show_interv = interv_df_sorted[actual_cols_interv_table].copy()
        for col in df_to_show_interv.select_dtypes(include=['object']).columns:
            df_to_show_interv[col] = df_to_show_interv[col].fillna('N/A').astype(str)
        for col in df_to_show_interv.select_dtypes(include=['number']).columns:
             df_to_show_interv[col] = df_to_show_interv[col].fillna(0) # Fill numeric NaNs with 0 for display consistency
            
        st.dataframe(
            df_to_show_interv, use_container_width=True, hide_index=True, height=min(400, len(df_to_show_interv)*38 + 58),
            column_config={ "name": st.column_config.TextColumn("Zone Name"), "population": st.column_config.NumberColumn("Population", format="%,.0f"), "avg_risk_score": st.column_config.ProgressColumn("Avg. AI Risk", format="%.1f", min_value=0, max_value=100), "total_active_key_infections": st.column_config.NumberColumn("Key Inf.", format="%.0f"), "prevalence_per_1000": st.column_config.NumberColumn("Prevalence/1k", format="%.1f"), "facility_coverage_score": st.column_config.NumberColumn("Facility Cov. (%)", format="%.1f%%"), "zone_avg_co2": st.column_config.NumberColumn("Avg Clinic CO2", format="%.0f ppm"), "active_tb_cases": st.column_config.NumberColumn("TB Cases", format="%.0f"), "active_malaria_cases": st.column_config.NumberColumn("Malaria Cases", format="%.0f")}
        )
    else:
        st.success("✅ No zones currently meet the selected high-priority criteria based on the available aggregated data.")
