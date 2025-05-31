# test/pages/clinic_components/testing_insights_tab.py
import streamlit as st
import pandas as pd
import numpy as np 
from config import app_config
from utils.core_data_processing import get_trend_data 
from utils.ui_visualization_helpers import plot_donut_chart, plot_annotated_line_chart, plot_bar_chart, render_kpi_card
import logging

logger = logging.getLogger(__name__)

def render_testing_insights(filtered_health_df_clinic, clinic_service_kpis):
    st.subheader("🔬 In-depth Laboratory Testing Performance & Trends")
    
    if filtered_health_df_clinic.empty:
        st.info("No health data available for the selected period to display detailed testing insights.")
        return

    detailed_test_stats_tab = clinic_service_kpis.get("test_summary_details", {})
    
    if not detailed_test_stats_tab:
         st.warning("No detailed test summary statistics could be generated. This might be due to missing test data or configuration issues in `app_config.KEY_TEST_TYPES_FOR_ANALYSIS`.")
         return

    # --- Test Group Selection ---
    # Populate options only with test groups that have some data (conclusive, pending, or rejected)
    active_test_groups = sorted([
        disp_name for disp_name, stats in detailed_test_stats_tab.items() 
        if stats.get('total_conducted_conclusive',0) > 0 or stats.get('pending_count',0) > 0 or stats.get('rejected_count',0) > 0
    ])
    
    critical_test_exists_in_config = any(props.get("critical", False) for props in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.values())
    
    test_group_options_tab = []
    if critical_test_exists_in_config : test_group_options_tab.append("All Critical Tests Summary")
    test_group_options_tab.extend(active_test_groups)

    if not test_group_options_tab:
         st.info("No test groups with activity or critical tests defined for detailed analysis in this period.")
         return
         
    selected_test_group_display = st.selectbox(
        "Focus on Test Group/Type:", options=test_group_options_tab,
        key="clinic_test_group_select_tab_component_v2", # Incremented key
        help="Select a test group for detailed metrics and trends, or view a summary for all critical tests."
    )
    st.markdown("---")

    # --- Display based on selection ---
    if selected_test_group_display == "All Critical Tests Summary":
        st.markdown("###### **Performance Metrics for All Critical Tests (Period Average)**")
        crit_test_table_data = []
        for group_disp_name, stats in detailed_test_stats_tab.items():
            original_group_key = next((k for k, v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == group_disp_name), None)
            if original_group_key and app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(original_group_key, {}).get("critical"):
                crit_test_table_data.append({
                    "Test Group": group_disp_name, 
                    "Positivity (%)": stats.get("positive_rate", 0.0),
                    "Avg. TAT (Days)": stats.get("avg_tat_days", np.nan), 
                    "% Met TAT Target": stats.get("perc_met_tat_target", 0.0),
                    "Pending Count": stats.get("pending_count", 0), 
                    "Rejected Count": stats.get("rejected_count", 0),
                    "Total Conclusive": stats.get("total_conducted_conclusive", 0)
                })
        if crit_test_table_data:
            st.dataframe(pd.DataFrame(crit_test_table_data), use_container_width=True, hide_index=True,
                         column_config={
                             "Positivity (%)": st.column_config.NumberColumn(format="%.1f%%"),
                             "Avg. TAT (Days)": st.column_config.NumberColumn(format="%.1f d"), # Added 'd'
                             "% Met TAT Target": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)
                         })
        else: st.caption("No data for critical tests found in this period or no critical tests configured.")

    elif selected_test_group_display in detailed_test_stats_tab:
        stats_selected_group = detailed_test_stats_tab[selected_test_group_display]
        st.markdown(f"###### **Detailed Metrics for: {selected_test_group_display}**")
        
        kpi_cols_test_detail_tab = st.columns(5)
        with kpi_cols_test_detail_tab[0]: render_kpi_card("Positivity Rate", f"{stats_selected_group.get('positive_rate',0.0):.1f}%", "➕")
        avg_tat_disp = stats_selected_group.get('avg_tat_days', np.nan)
        with kpi_cols_test_detail_tab[1]: render_kpi_card("Avg. TAT", f"{avg_tat_disp:.1f}d" if pd.notna(avg_tat_disp) else "N/A", "⏱️")
        with kpi_cols_test_detail_tab[2]: render_kpi_card("% Met TAT Target", f"{stats_selected_group.get('perc_met_tat_target',0.0):.1f}%", "🎯")
        with kpi_cols_test_detail_tab[3]: render_kpi_card("Pending Tests", str(stats_selected_group.get('pending_count',0)), "⏳")
        with kpi_cols_test_detail_tab[4]: render_kpi_card("Rejected Samples", str(stats_selected_group.get('rejected_count',0)), "🚫")

        plot_cols_test_detail_tab = st.columns(2)
        original_key_for_selected = next((k for k,v_cfg in app_config.KEY_TEST_TYPES_FOR_ANALYSIS.items() if v_cfg.get("display_name") == selected_test_group_display), None)
        
        if original_key_for_selected and original_key_for_selected in app_config.KEY_TEST_TYPES_FOR_ANALYSIS: # Check key exists
            cfg_selected_test = app_config.KEY_TEST_TYPES_FOR_ANALYSIS[original_key_for_selected]
            # Test types in data should ideally be the original keys for robust matching
            actual_test_keys_for_plot = cfg_selected_test.get("types_in_group", [original_key_for_selected])
            if isinstance(actual_test_keys_for_plot, str): actual_test_keys_for_plot = [actual_test_keys_for_plot]
            target_tat_for_plot = cfg_selected_test.get("target_tat_days", app_config.TARGET_TEST_TURNAROUND_DAYS)

            with plot_cols_test_detail_tab[0]:
                st.markdown(f"**Daily Avg. TAT for {selected_test_group_display}**")
                df_tat_plot_src = filtered_health_df_clinic[(filtered_health_df_clinic['test_type'].isin(actual_test_keys_for_plot)) & (filtered_health_df_clinic['test_turnaround_days'].notna()) & (~filtered_health_df_clinic['test_result'].isin(['Pending','Unknown','Rejected Sample', 'Indeterminate']))].copy()
                if not df_tat_plot_src.empty:
                    tat_trend_plot = get_trend_data(df_tat_plot_src, 'test_turnaround_days', period='D', date_col='encounter_date', agg_func='mean')
                    if not tat_trend_plot.empty: st.plotly_chart(plot_annotated_line_chart(tat_trend_plot, f"Avg. TAT Trend", y_axis_title="Days", target_line=target_tat_for_plot, target_label=f"Target {target_tat_for_plot}d", height=app_config.COMPACT_PLOT_HEIGHT-20, date_format="%d %b"), use_container_width=True)
                    else: st.caption("No aggregated TAT trend data.")
                else: st.caption("No conclusive tests with TAT data for this group.")
            with plot_cols_test_detail_tab[1]:
                st.markdown(f"**Daily Test Volume for {selected_test_group_display}**")
                df_vol_plot_src = filtered_health_df_clinic[filtered_health_df_clinic['test_type'].isin(actual_test_keys_for_plot)].copy()
                if not df_vol_plot_src.empty:
                    conducted_vol = get_trend_data(df_vol_plot_src[~df_vol_plot_src['test_result'].isin(['Pending','Unknown','Rejected Sample', 'Indeterminate'])], 'patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Conclusive")
                    pending_vol = get_trend_data(df_vol_plot_src[df_vol_plot_src['test_result'] == 'Pending'], 'patient_id', date_col='encounter_date', period='D', agg_func='count').rename("Pending")
                    if not conducted_vol.empty or not pending_vol.empty:
                        vol_trend_df = pd.concat([conducted_vol, pending_vol], axis=1).fillna(0).reset_index()
                        date_col_melt = 'encounter_date' if 'encounter_date' in vol_trend_df.columns else ('date' if 'date' in vol_trend_df.columns else vol_trend_df.columns[0])
                        vol_melt_df = vol_trend_df.melt(id_vars=date_col_melt, value_vars=['Conclusive', 'Pending'], var_name='Status', value_name='Count')
                        st.plotly_chart(plot_bar_chart(vol_melt_df, x_col=date_col_melt, y_col='Count', color_col='Status', title=f"Daily Volume Trend", barmode='stack', height=app_config.COMPACT_PLOT_HEIGHT-20, y_is_count=True, text_format='d'), use_container_width=True)
                    else: st.caption("No volume data.")
                else: st.caption(f"No tests matching '{selected_test_group_display}' for volume trend.")
        else: st.warning(f"Configuration for '{selected_test_group_display}' not found for plotting trends.")
    else:
         st.info(f"No activity data found for test group: '{selected_test_group_display}' in this period.")
    
    st.markdown("---"); st.markdown("###### **Overdue Pending Tests (All test types, older than their target TAT + buffer)**")
    op_df_source_clinic = filtered_health_df_clinic.copy()
    # Use sample_collection_date as primary, then test_date (sample_registered_lab_date), then encounter_date
    if 'sample_collection_date' in op_df_source_clinic.columns and op_df_source_clinic['sample_collection_date'].notna().any():
        date_col_for_pending_calc = 'sample_collection_date'
    elif 'sample_registered_lab_date' in op_df_source_clinic.columns and op_df_source_clinic['sample_registered_lab_date'].notna().any():
        date_col_for_pending_calc = 'sample_registered_lab_date'
    else:
        date_col_for_pending_calc = 'encounter_date' # Fallback
        
    overdue_df_clinic = op_df_source_clinic[(op_df_source_clinic['test_result'] == 'Pending') & (op_df_source_clinic[date_col_for_pending_calc].notna())].copy()
    if not overdue_df_clinic.empty:
        # Ensure date_col_for_pending_calc is datetime before calculation
        overdue_df_clinic.loc[:, date_col_for_pending_calc] = pd.to_datetime(overdue_df_clinic[date_col_for_pending_calc], errors='coerce')
        overdue_df_clinic.dropna(subset=[date_col_for_pending_calc], inplace=True)
        
        if not overdue_df_clinic.empty:
            overdue_df_clinic.loc[:, 'days_pending_calc'] = (pd.Timestamp('today').normalize() - overdue_df_clinic[date_col_for_pending_calc]).dt.days
            
            def get_specific_overdue_threshold_testing_tab(test_type_key_from_data_testing_tab):
                test_config_testing_tab = app_config.KEY_TEST_TYPES_FOR_ANALYSIS.get(test_type_key_from_data_testing_tab)
                buffer_days_testing_tab = 2
                if test_config_testing_tab and 'target_tat_days' in test_config_testing_tab:
                    return test_config_testing_tab['target_tat_days'] + buffer_days_testing_tab
                return app_config.OVERDUE_PENDING_TEST_DAYS + buffer_days_testing_tab
            
            # Ensure test_type column has original keys
            overdue_df_clinic.loc[:, 'effective_overdue_days'] = overdue_df_clinic['test_type'].apply(get_specific_overdue_threshold_testing_tab)
            overdue_df_final_display_clinic = overdue_df_clinic[overdue_df_clinic['days_pending_calc'] > overdue_df_clinic['effective_overdue_days']]
            
            if not overdue_df_final_display_clinic.empty:
                st.dataframe(overdue_df_final_display_clinic[['patient_id', 'test_type', date_col_for_pending_calc, 'days_pending_calc', 'effective_overdue_days']].sort_values('days_pending_calc', ascending=False).head(10),
                             column_config={date_col_for_pending_calc:st.column_config.DateColumn("Sample/Encounter Date"),
                                            "days_pending_calc":st.column_config.NumberColumn("Days Pending",format="%d"),
                                            "effective_overdue_days":st.column_config.NumberColumn("Overdue If > (days)",format="%d")},
                             height=300, use_container_width=True)
            else: st.success(f"✅ No tests pending longer than their target TAT + buffer.")
        else: st.caption("No valid pending tests to evaluate after date cleaning.") # After date cleaning
    else: st.caption("No pending tests found to evaluate for overdue status in this period.")

    if 'sample_status' in filtered_health_df_clinic.columns and 'rejection_reason' in filtered_health_df_clinic.columns:
        st.markdown("---"); st.markdown("###### **Sample Rejection Analysis (Period)**")
        rejected_samples_df_tab_clinic = filtered_health_df_clinic[filtered_health_df_clinic['sample_status'] == 'Rejected'].copy()
        if not rejected_samples_df_tab_clinic.empty:
            rejection_reason_counts_clinic = rejected_samples_df_tab_clinic['rejection_reason'].value_counts().reset_index(); rejection_reason_counts_clinic.columns = ['Rejection Reason', 'Count']
            col_rej_donut_cl, col_rej_table_cl = st.columns([0.45, 0.55])
            with col_rej_donut_cl:
                if not rejection_reason_counts_clinic.empty: st.plotly_chart(plot_donut_chart(rejection_reason_counts_clinic, 'Rejection Reason', 'Count', "Top Sample Rejection Reasons", height=app_config.COMPACT_PLOT_HEIGHT + 20, values_are_counts=True), use_container_width=True)
                else: st.caption("No rejection reason data.")
            with col_rej_table_cl:
                st.caption("Rejected Samples List (Top 10 in Period)"); st.dataframe(rejected_samples_df_tab_clinic[['patient_id', 'test_type', 'encounter_date', 'rejection_reason']].head(10), height=280, use_container_width=True)
        else: st.info("✅ No rejected samples recorded in this period.")
