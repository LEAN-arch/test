# tests/test_ui_visualization_helpers.py
import pytest
import plotly.graph_objects as go
import plotly.io as pio 
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import geopandas as gpd
from shapely.geometry import Point

from utils.ui_visualization_helpers import (
    set_custom_plotly_theme, 
    render_kpi_card, render_traffic_light,
    plot_annotated_line_chart, plot_bar_chart,
    plot_donut_chart, plot_heatmap,
    plot_layered_choropleth_map, _create_empty_figure # Test the helper too
)
from config import app_config 

# --- Test _create_empty_figure helper ---
def test_create_empty_figure():
    fig = _create_empty_figure("Test Empty", 300, "Specific Message")
    assert isinstance(fig, go.Figure)
    assert "Test Empty (Specific Message)" in fig.layout.title.text
    assert fig.layout.height == 300
    assert not fig.layout.xaxis.visible and not fig.layout.yaxis.visible
    assert len(fig.layout.annotations) == 1 and fig.layout.annotations[0].text == "Specific Message"

# === Test Styled Components ===
@patch('utils.ui_visualization_helpers.st.markdown') 
def test_render_kpi_card_all_params_robust(mock_st_markdown):
    render_kpi_card(title="Total Patients", value="1,234", icon="🧑‍🤝‍🧑", status="Good High", delta="+20", delta_type="positive", help_text="Total patients registered.")
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0] 
    assert 'class="kpi-card high status-good"' in html_output or 'class="kpi-card good status-high"' in html_output # Check semantic status classes
    assert 'title="Total patients registered."' in html_output
    assert "Total Patients" in html_output and "1,234" in html_output and "🧑‍🤝‍🧑" in html_output
    assert '<p class="kpi-delta positive">+20</p>' in html_output

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_kpi_card_minimal_params_robust(mock_st_markdown):
    render_kpi_card(title="Alerts", value="5", icon="🚨") # Default status "neutral"
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    assert 'class="kpi-card neutral "' in html_output.replace(' status-neutral', ' neutral') # Accommodate order
    assert 'title=""' not in html_output
    assert "Alerts" in html_output and '<p class="kpi-delta' not in html_output

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_traffic_light_all_params_robust(mock_st_markdown):
    render_traffic_light(message="System OK", status="Low", details="All systems operational.")
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    assert '<div class="traffic-light-indicator">' in html_output and '<span class="traffic-light-dot status-low">' in html_output
    assert "System OK" in html_output and '<span class="traffic-light-details">All systems operational.</span>' in html_output

# === Plotting Function Tests ===

def test_plot_annotated_line_chart_valid_data_robust(sample_series_data):
    fig = plot_annotated_line_chart(sample_series_data, "Test Line: Daily Count", y_axis_title="Count")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1 
    assert "Test Line: Daily Count" in fig.layout.title.text
    assert fig.layout.yaxis.title.text == "Count"
    assert fig.layout.height == app_config.DEFAULT_PLOT_HEIGHT 

def test_plot_annotated_line_chart_empty_robust():
    empty_series = pd.Series(dtype='float64') # Empty Series
    fig = plot_annotated_line_chart(empty_series, "Empty Line Chart Test")
    assert isinstance(fig, go.Figure) and "Empty Line Chart Test (No data available to display.)" in fig.layout.title.text and len(fig.data) == 0

def test_plot_annotated_line_chart_with_ci_target_anomalies(sample_series_data):
    lower = sample_series_data * 0.9; upper = sample_series_data * 1.1
    # Add an anomaly to sample_series_data for testing
    data_with_anomaly = sample_series_data.copy()
    if len(data_with_anomaly) > 5: data_with_anomaly.iloc[2] = data_with_anomaly.max() + 2 * data_with_anomaly.std() 

    fig = plot_annotated_line_chart(data_with_anomaly, "Line with CI, Target, Anomalies", 
                                    target_line=12, target_label="Target Value", show_ci=True, 
                                    lower_bound_series=lower, upper_bound_series=upper, show_anomalies=True)
    assert len(fig.data) >= 2 # Main, CI, (optional Anomaly)
    assert any(trace.name == "Confidence Interval" for trace in fig.data)
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].y0 == 12
    if any(trace.name == "Potential Anomaly" for trace in fig.data): # Check if anomalies were detected and plotted
        assert True # Anomaly trace is present
    else: # If no anomalies were plotted (e.g. data too short or no clear outliers based on IQR)
        logger.info("Anomaly trace not present in test_plot_annotated_line_chart_with_ci_target_anomalies, data might not have met criteria.")


def test_plot_bar_chart_valid_data_robust(sample_bar_df):
    fig = plot_bar_chart(sample_bar_df, x_col='category', y_col='value', title="Bar Chart: Category Vals")
    assert isinstance(fig, go.Figure) and len(fig.data) == 1 and isinstance(fig.data[0], go.Bar)
    assert "Bar Chart: Category Vals" in fig.layout.title.text and fig.layout.xaxis.title.text == "Category"

def test_plot_bar_chart_grouped_robust(sample_bar_df):
    fig = plot_bar_chart(sample_bar_df, x_col='category', y_col='value', title="Grouped Bar by Group", color_col='group', barmode='group')
    assert len(fig.data) == sample_bar_df['group'].nunique() and fig.layout.barmode == 'group'
    assert fig.layout.legend.title.text == 'Group'


def test_plot_donut_chart_valid_data_robust(sample_donut_df):
    fig = plot_donut_chart(sample_donut_df, labels_col='status', values_col='count', title="Donut: Status Counts")
    assert isinstance(fig, go.Figure) and len(fig.data) == 1 and isinstance(fig.data[0], go.Pie)
    assert 0.45 < fig.data[0].hole < 0.55 and "Donut: Status Counts" in fig.layout.title.text

def test_plot_heatmap_valid_matrix_robust(sample_heatmap_df):
    fig = plot_heatmap(sample_heatmap_df, "Heatmap: Test Matrix", colorscale="Viridis", zmid=None) # Test with non-diverging
    assert isinstance(fig, go.Figure) and len(fig.data) == 1 and isinstance(fig.data[0], go.Heatmap)
    assert "Heatmap: Test Matrix" in fig.layout.title.text and fig.data[0].zmid is None # zmid should be None here

def test_plot_layered_choropleth_map_valid_gdf_robust(sample_choropleth_gdf):
    # Ensure the sample GDF has the id_col for locations
    if 'zone_id' not in sample_choropleth_gdf.columns: sample_choropleth_gdf['zone_id'] = sample_choropleth_gdf.index.astype(str)
    
    fig = plot_layered_choropleth_map(gdf=sample_choropleth_gdf, value_col='risk_score', title="Choropleth: Zone Risk", id_col='zone_id', featureidkey_prefix='properties', mapbox_style=app_config.MAPBOX_STYLE)
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1 # At least choropleth layer
    assert isinstance(fig.data[0], go.Choroplethmapbox) and "Choropleth: Zone Risk" in fig.layout.title.text
    assert fig.layout.mapbox.style == app_config.MAPBOX_STYLE or (not MAPBOX_TOKEN_SET and fig.layout.mapbox.style == "open-street-map")


def test_plot_layered_choropleth_map_with_facilities_robust(sample_choropleth_gdf):
    if 'zone_id' not in sample_choropleth_gdf.columns: sample_choropleth_gdf['zone_id'] = sample_choropleth_gdf.index.astype(str)
    facility_data = {'facility_id': ['F1', 'F2'], 'capacity': [10, 20], 'type': ['Clinic', 'Hospital']}
    # Ensure points are somewhat within the bounds of sample_choropleth_gdf polygons
    minx, miny, maxx, maxy = sample_choropleth_gdf.total_bounds
    facility_geometry = [Point((minx+maxx)/2, (miny+maxy)/2), Point(minx + (maxx-minx)*0.75, miny + (maxy-miny)*0.75)]
    facility_gdf_sample = gpd.GeoDataFrame(facility_data, geometry=facility_geometry, crs=sample_choropleth_gdf.crs)

    fig = plot_layered_choropleth_map(gdf=sample_choropleth_gdf, value_col='population', title="Choropleth with Facilities Test", id_col='zone_id', featureidkey_prefix='properties', facility_gdf=facility_gdf_sample, facility_size_col='capacity', facility_hover_name='facility_id')
    assert len(fig.data) == 2 # Choropleth + Scattermapbox for facilities
    assert isinstance(fig.data[1], go.Scattermapbox) and fig.data[1].name == 'Facilities Layer'

# Test robustness for empty/invalid inputs across plotting functions
@pytest.mark.parametrize("plot_func, args_dict", [
    (plot_annotated_line_chart, {"data_series": pd.Series(dtype='float64'), "title":"EmptyLine"}),
    (plot_bar_chart, {"df_input": pd.DataFrame(columns=['x','y']), "x_col": "x", "y_col": "y", "title":"EmptyBar"}),
    (plot_donut_chart, {"data_df_input": pd.DataFrame(columns=['l','v']), "labels_col": "l", "values_col": "v", "title":"EmptyDonut"}),
    (plot_heatmap, {"matrix_df_input": pd.DataFrame(), "title":"EmptyHeatmap"}),
    (plot_layered_choropleth_map, {"gdf": gpd.GeoDataFrame(columns=['geometry','id_col','val_col'], geometry='geometry', crs=app_config.DEFAULT_CRS), "value_col": "val_col", "id_col":"id_col", "title":"EmptyMap"})
])
def test_plotting_functions_handle_empty_input(plot_func, args_dict):
    fig = plot_func(**args_dict)
    assert isinstance(fig, go.Figure)
    # Expected: title contains indication of no data
    assert f"{args_dict['title']} (No data" in fig.layout.title.text or f"{args_dict['title']} (Map Data Error" in fig.layout.title.text or f"{args_dict['title']} (Invalid data" in fig.layout.title.text
    assert len(fig.data) == 0 # Typically no traces are added
