# test/pages/district_components/comparison_tab_district.py
import streamlit as st
import pandas as pd
import numpy as np # For np.inf if needed, though not directly here
from config import app_config
from utils.ui_visualization_helpers import plot_bar_chart
import logging

logger = logging.getLogger(__name__)

def render_zonal_comparison_tab(district_gdf_main_enriched):
    st.header("📊 Zonal Comparative Analysis (Based on Latest Aggregates)")
    
    if district_gdf_main_enriched is None or district_gdf_main_enriched.empty or \
       'geometry' not in district_gdf_main_enriched.columns: # Basic check
        st.info("Zonal comparison requires successfully loaded and enriched geographic zone data.")
        return

    st.markdown("Compare zones using aggregated health, resource, environmental, and socio-economic metrics from the latest available data.")
    
    # Re-define or pass map_metric_options_config_dist from main page, if it's calculated there based on GDF
    # For simplicity in component, define it here.
    # This assumes district_gdf_main_enriched already has all these columns calculated.
    map_metric_options_config = {
        "Avg. AI Risk Score": {"col": "avg_risk_score", "colorscale": "Reds_r", "format": "{:.1f}"},
        "Population Density (Pop/SqKm)": {"col": "population_density", "colorscale": "Plasma_r", "format":"{:,.1f}"}, # Assuming it's calculated
        "Prevalence per 1,000 (Key Inf.)": {"col": "prevalence_per_1000", "colorscale": "YlOrRd_r", "format": "{:.1f}"},
        "Facility Coverage Score": {"col": "facility_coverage_score", "colorscale": "Greens", "format": "{:.1f}%"},
        "Active TB Cases": {"col": "active_tb_cases", "colorscale": "Purples_r", "format": "{:.0f}"},
        "Avg. Zone CO2 (Clinics)": {"col": "zone_avg_co2", "colorscale": "Blues_r", "format": "{:.0f} ppm"},
        "Number of Clinics": {"col": "num_clinics", "colorscale": "Blues", "format":"{:.0f}"},
    }
    
    comp_table_metrics_dict = {
        name: details for name, details in map_metric_options_config.items()
        if details["col"] in district_gdf_main_enriched.columns and district_gdf_main_enriched[details["col"]].notna().any()
    }
    
    if not comp_table_metrics_dict:
        st.info("No metrics available for Zonal Comparison table/chart. Check GDF enrichment and available columns.")
        return

    # --- Zonal Statistics Table ---
    st.subheader("Zonal Statistics Table")
    # Ensure 'name' exists, else use 'zone_id'
    display_name_col = 'name' if 'name' in district_gdf_main_enriched.columns else 'zone_id'
    cols_for_comp_table_display = [display_name_col] + [d['col'] for d in comp_table_metrics_dict.values()]
    
    # Ensure all selected columns for display actually exist in the DataFrame
    actual_cols_for_table = [col for col in cols_for_comp_table_display if col in district_gdf_main_enriched.columns]
    df_for_comp_table_display = district_gdf_main_enriched[actual_cols_for_table].copy()
    df_for_comp_table_display.rename(columns={display_name_col:'Zone'}, inplace=True) # Use 'Zone' as display
    
    if 'Zone' in df_for_comp_table_display.columns: df_for_comp_table_display.set_index('Zone', inplace=True, drop=False) # Keep 'Zone' as a column for bar chart too
    
    # Filter out 'Zone' if it's now the index and also in style_formats keys
    style_formats_comp_dist = {
        details["col"]: details.get("format", "{:.1f}")
        for _, details in comp_table_metrics_dict.items()
        if details["col"] in df_for_comp_table_display.columns and details["col"] != 'Zone' # Exclude 'Zone' from formatting if it's just the index name
    }
    
    styler_obj_comp_dist = df_for_comp_table_display.drop(columns=['Zone'] if 'Zone' in df_for_comp_table_display.index.names else []).style.format(style_formats_comp_dist) # Format only value columns

    for metric_display_name, details_style_comp in comp_table_metrics_dict.items():
        col_name_to_style = details_style_comp["col"]
        if col_name_to_style in df_for_comp_table_display.columns and col_name_to_style != 'Zone': # Ensure column exists and not trying to style 'Zone' index name
            colorscale_name = details_style_comp.get("colorscale", "Blues")
            try:
                styler_obj_comp_dist = styler_obj_comp_dist.background_gradient(subset=[col_name_to_style], cmap=colorscale_name, axis=0)
            except Exception as e_style:
                 logger.warning(f"Could not apply background_gradient for {col_name_to_style} with cmap {colorscale_name}: {e_style}")
    st.dataframe(styler_obj_comp_dist, use_container_width=True, height=min(len(df_for_comp_table_display) * 40 + 60, 600))

    # --- Visual Comparison Chart ---
    st.subheader("Visual Comparison Chart")
    selected_bar_metric_name_dist_comp_viz = st.selectbox(
        "Select Metric for Bar Chart Comparison:",
        list(comp_table_metrics_dict.keys()),
        key="district_comp_barchart_component_v1" # Component-specific key
    )
    selected_bar_details_dist_comp_viz = comp_table_metrics_dict.get(selected_bar_metric_name_dist_comp_viz)
    
    if selected_bar_details_dist_comp_viz:
        bar_col_for_comp_viz = selected_bar_details_dist_comp_viz["col"]
        text_format_bar = selected_bar_details_dist_comp_viz.get("format", ",.1f")
        sort_asc_bar_viz = "_r" not in selected_bar_details_dist_comp_viz.get("colorscale", "")
        
        # Use df_for_comp_table_display which already has 'Zone' as a column
        # Make sure 'Zone' (or display_name_col) is passed as x_col
        x_col_bar = 'Zone' if 'Zone' in df_for_comp_table_display.columns else display_name_col
        
        if x_col_bar in df_for_comp_table_display.columns and bar_col_for_comp_viz in df_for_comp_table_display.columns:
            st.plotly_chart(plot_bar_chart(
                df_for_comp_table_display, x_col=x_col_bar, y_col=bar_col_for_comp_viz,
                title=f"{selected_bar_metric_name_dist_comp_viz} by Zone",
                x_axis_title="Zone Name", height=app_config.DEFAULT_PLOT_HEIGHT + 150,
                sort_values_by=bar_col_for_comp_viz, ascending=sort_asc_bar_viz,
                text_auto=True, text_format=text_format_bar
            ), use_container_width=True)
        else:
            st.warning("Could not generate comparison bar chart due to missing columns.")
