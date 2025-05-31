# test/pages/district_components/map_display_district.py
import streamlit as st
import pandas as pd
import numpy as np
import logging # Added logging
from config import app_config
from utils.ui_visualization_helpers import plot_layered_choropleth_map

logger = logging.getLogger(__name__) # Added logger

def render_district_map(district_gdf_main_enriched):
    st.subheader("🗺️ Interactive Health & Environment Map of the District")
    
    if district_gdf_main_enriched is None or district_gdf_main_enriched.empty or \
       'geometry' not in district_gdf_main_enriched.columns or \
       district_gdf_main_enriched.geometry.is_empty.all():
        st.error("🚨 District map cannot be displayed: Geographic data is unusable or unavailable.")
        return

    # Map metric options defined here, similar to how it was in the main page
    # This could also be passed from the main page if it's generated dynamically there
    map_metric_options_config = {
        "Avg. AI Risk Score": {"col": "avg_risk_score", "colorscale": "Reds_r", "format": "{:.1f}"},
        "Total Active Key Infections": {"col": "total_active_key_infections", "colorscale": "OrRd_r", "format": "{:.0f}"},
        "Prevalence per 1,000 (Key Inf.)": {"col": "prevalence_per_1000", "colorscale": "YlOrRd_r", "format": "{:.1f}"},
        "Facility Coverage Score": {"col": "facility_coverage_score", "colorscale": "Greens", "format": "{:.1f}%"},
        "Active TB Cases": {"col": "active_tb_cases", "colorscale": "Purples_r", "format": "{:.0f}"},
        "Active Malaria Cases": {"col": "active_malaria_cases", "colorscale": "Oranges_r", "format": "{:.0f}"},
        "Avg. Patient Steps (Zone)": {"col": "avg_daily_steps_zone", "colorscale": "Cividis", "format": "{:,.0f}"},
        "Avg. Zone CO2 (Clinics)": {"col": "zone_avg_co2", "colorscale": "Blues_r", "format": "{:.0f} ppm"},
        "Population": {"col": "population", "colorscale": "Viridis", "format": "{:,.0f}"},
        "Number of Clinics": {"col": "num_clinics", "colorscale": "Blues", "format":"{:.0f}"},
        "Socio-Economic Index": {"col": "socio_economic_index", "colorscale": "Tealgrn", "format": "{:.2f}"}
    }
    
    # Add population density if available (was calculated in main orchestrator or enrichment)
    if 'population_density' in district_gdf_main_enriched.columns:
         map_metric_options_config["Population Density (Pop/SqKm)"] = {"col": "population_density", "colorscale": "Plasma_r", "format": "{:,.1f}"}

    available_map_metrics_for_select = {
        disp_name: details for disp_name, details in map_metric_options_config.items()
        if details["col"] in district_gdf_main_enriched.columns and district_gdf_main_enriched[details["col"]].notna().any()
    }

    if not available_map_metrics_for_select:
        st.warning("No metrics with valid data are currently available for map visualization.")
        return

    selected_map_metric_display_name = st.selectbox(
        "Select Metric to Visualize on Map:",
        list(available_map_metrics_for_select.keys()),
        key="district_map_metric_selector_comp_v1", # Component-specific key
        help="Choose a metric for spatial visualization."
    )
    selected_map_metric_config = available_map_metrics_for_select.get(selected_map_metric_display_name)
    
    if selected_map_metric_config:
        map_val_col = selected_map_metric_config["col"]
        map_colorscale = selected_map_metric_config["colorscale"]
        
        hover_cols_base = ['name', map_val_col, 'population'] # Always try to include these
        if 'num_clinics' in district_gdf_main_enriched.columns: hover_cols_base.append('num_clinics')
        if 'facility_coverage_score' in district_gdf_main_enriched.columns: hover_cols_base.append('facility_coverage_score')
        # Ensure all hover_cols actually exist in the GDF and remove duplicates
        final_hover_cols_map = list(dict.fromkeys([col for col in hover_cols_base if col in district_gdf_main_enriched.columns]))

        map_figure = plot_layered_choropleth_map(
            gdf=district_gdf_main_enriched, value_col=map_val_col,
            title=f"District Map: {selected_map_metric_display_name}",
            id_col='zone_id', # Ensure this column name is consistent
            color_continuous_scale=map_colorscale,
            hover_cols=final_hover_cols_map,
            height=app_config.MAP_PLOT_HEIGHT,
            mapbox_style=app_config.MAPBOX_STYLE
        )
        st.plotly_chart(map_figure, use_container_width=True)
    else:
        st.info("Please select a metric from the dropdown to display on the map.")
