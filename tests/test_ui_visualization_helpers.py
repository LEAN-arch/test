# tests/test_ui_visualization_helpers.py
import pytest
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock # Added MagicMock for MAPBOX_TOKEN_SET
import geopandas as gpd
from shapely.geometry import Point

from utils.ui_visualization_helpers import (
    set_custom_plotly_theme,
    render_kpi_card, render_traffic_light,
    plot_annotated_line_chart, plot_bar_chart,
    plot_donut_chart, plot_heatmap,
    plot_layered_choropleth_map, _create_empty_figure, _get_theme_color,
    MAPBOX_TOKEN_SET # Import the flag to potentially mock it
)
from config import app_config

set_custom_plotly_theme() # Apply theme for testing outputs

# Mock MAPBOX_TOKEN_SET for tests where its value matters for map styles
@pytest.fixture(autouse=True) # Apply to all tests in this module
def mock_mapbox_token_status(mocker):
    # You can set this to True or False depending on what you want to test by default
    # Or, individual tests can re-patch it if they need a specific state.
    mocker.patch('utils.ui_visualization_helpers.MAPBOX_TOKEN_SET', True)


def test_create_empty_figure_with_message():
    fig = _create_empty_figure("Test Empty Title", 350, "Custom No Data Message")
    assert isinstance(fig, go.Figure)
    assert "Test Empty Title (Custom No Data Message)" in fig.layout.title.text
    assert fig.layout.height == 350
    assert not fig.layout.xaxis.visible and not fig.layout.yaxis.visible
    assert len(fig.layout.annotations) == 1 and fig.layout.annotations[0].text == "Custom No Data Message"

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_kpi_card_all_params_robust(mock_st_markdown):
    render_kpi_card(title="Total Patients", value="1,234", icon="🧑‍🤝‍🧑", status="Good High", delta="+20", delta_type="positive", help_text="Total patients registered.")
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    # Check for the combined class pattern (e.g., "high status-good" or "good status-high")
    assert ('class="kpi-card high status-good"' in html_output or \
            'class="kpi-card good status-high"' in html_output or \
            'class="kpi-card good high"' in html_output or # Allow for order variation if space is inconsistent
            'class="kpi-card high good"' in html_output)
    assert 'title="Total patients registered."' in html_output
    assert "Total Patients" in html_output and "1,234" in html_output and "🧑‍🤝‍🧑" in html_output
    assert '<p class="kpi-delta positive">+20</p>' in html_output

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_kpi_card_minimal_params_robust(mock_st_markdown):
    render_kpi_card(title="Alerts", value="5", icon="🚨")
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    assert 'class="kpi-card neutral' in html_output.lower()
    assert 'title=""' not in html_output # No empty title attribute
    assert "Alerts" in html_output and '<p class="kpi-delta' not in html_output

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_traffic_light_standard(mock_st_markdown):
    render_traffic_light(message="Network Status", status="Low", details="All connections stable.")
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    assert '<div class="traffic-light-indicator">' in html_output and '<span class="traffic-light-dot status-low">' in html_output
    assert "Network Status" in html_output and '<span class="traffic-light-details">All connections stable.</span>' in html_output

def test_plot_annotated_line_chart_valid_data_robust(sample_series_data): # Uses conftest fixture
    fig = plot_annotated_line_chart(sample_series_data, "Test Line: Daily Count", y_axis_title="Count")
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1
    assert "Test Line: Daily Count" in fig.layout.title.text and fig.layout.yaxis.title.text == "Count"
    assert fig.layout.height == app_config.DEFAULT_PLOT_HEIGHT

def test_plot_annotated_line_chart_empty_robust():
    empty_series = pd.Series(dtype='float64')
    fig = plot_annotated_line_chart(empty_series, "Empty Line Chart Test")
    assert isinstance(fig, go.Figure) and "Empty Line Chart Test (No data available to display.)" in fig.layout.title.text
    assert len(fig.data) == 0

def test_plot_annotated_line_chart_with_ci_target_anomalies(sample_series_data): # Uses conftest
    lower = sample_series_data * 0.9
    upper = sample_series_data * 1.1
    data_with_anomaly = sample_series_data.copy()
    # Introduce a clearer anomaly for robust testing if series length allows
    if len(data_with_anomaly) > 5 :
        anomaly_idx = data_with_anomaly.index[2]
        data_with_anomaly.loc[anomaly_idx] = data_with_anomaly.max() + 2 * data_with_anomaly.std() # Force anomaly
    
    fig = plot_annotated_line_chart(
        data_with_anomaly, "Line with CI, Target, Anomalies",
        target_line=12, target_label="Target Value",
        show_ci=True, lower_bound_series=lower, upper_bound_series=upper,
        show_anomalies=True
    )
    assert len(fig.data) >= 2 # Main line + CI fill, maybe anomalies trace
    assert any(trace.name == "Confidence Interval" for trace in fig.data if hasattr(trace, 'name'))
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].y0 == 12 # Check target line
    # Check if anomaly was actually plotted if data_with_anomaly has it
    if not data_with_anomaly[data_with_anomaly > (data_with_anomaly.quantile(0.75) + 1.5 * (data_with_anomaly.quantile(0.75) - data_with_anomaly.quantile(0.25))) ].empty or \
       not data_with_anomaly[data_with_anomaly < (data_with_anomaly.quantile(0.25) - 1.5 * (data_with_anomaly.quantile(0.75) - data_with_anomaly.quantile(0.25))) ].empty :
        assert any(trace.name == 'Potential Anomaly' for trace in fig.data if hasattr(trace, 'name'))


def test_plot_bar_chart_valid_data_robust(sample_bar_df): # Uses conftest
    fig = plot_bar_chart(sample_bar_df, x_col='category', y_col='value', title="Bar Chart: Category Vals")
    assert isinstance(fig, go.Figure) and len(fig.data) == 1 and isinstance(fig.data[0], go.Bar)
    assert "Bar Chart: Category Vals" in fig.layout.title.text
    assert fig.layout.xaxis.title.text.lower() == "category" # Ensure X axis title is set

def test_plot_bar_chart_grouped_robust(sample_bar_df): # Uses conftest
    fig = plot_bar_chart(sample_bar_df, x_col='category', y_col='value', title="Grouped Bar by Group", color_col='group', barmode='group')
    assert len(fig.data) == sample_bar_df['group'].nunique()
    assert fig.layout.barmode == 'group'
    assert fig.layout.legend.title.text.lower() == 'group' # Ensure legend title

def test_plot_donut_chart_valid_data_robust(sample_donut_df): # Uses conftest
    fig = plot_donut_chart(sample_donut_df, labels_col='status', values_col='count', title="Donut: Status Counts")
    assert isinstance(fig, go.Figure) and len(fig.data) == 1 and isinstance(fig.data[0], go.Pie)
    assert 0.4 < fig.data[0].hole < 0.6 # Check hole is set for donut
    assert "Donut: Status Counts" in fig.layout.title.text

def test_plot_heatmap_valid_matrix_robust(sample_heatmap_df): # Uses conftest
    fig = plot_heatmap(sample_heatmap_df, "Heatmap: Test Matrix", colorscale="Viridis", zmid=None) # Test with zmid=None explicitly
    assert isinstance(fig, go.Figure) and len(fig.data) == 1 and isinstance(fig.data[0], go.Heatmap)
    assert "Heatmap: Test Matrix" in fig.layout.title.text
    assert fig.data[0].zmid is None # Ensure zmid respected

def test_plot_layered_choropleth_map_valid_gdf_robust(sample_choropleth_gdf, mocker): # Uses conftest
    mocker.patch('utils.ui_visualization_helpers.MAPBOX_TOKEN_SET', True) # Ensure token is "set" for this test
    if 'zone_id' not in sample_choropleth_gdf.columns: # sample_choropleth_gdf should have zone_id
        sample_choropleth_gdf['zone_id'] = [f"Z{i}" for i in range(len(sample_choropleth_gdf))]

    fig = plot_layered_choropleth_map(
        gdf=sample_choropleth_gdf, value_col='risk_score',
        title="Choropleth: Zone Risk", id_col='zone_id',
        featureidkey_prefix='properties', # This means it expects zone_id under properties in GeoJSON structure
        mapbox_style=app_config.MAPBOX_STYLE # Use configured style
    )
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1
    assert isinstance(fig.data[0], go.Choroplethmapbox)
    assert "Choropleth: Zone Risk" in fig.layout.title.text
    
    # Determine expected map style based on mocked MAPBOX_TOKEN_SET
    expected_style = app_config.MAPBOX_STYLE # Default from config if token is mocked as True
    # If we were to test MAPBOX_TOKEN_SET = False, then expected_style would become "open-street-map"
    # if app_config.MAPBOX_STYLE was a token-requiring one.
    # This logic is handled inside plot_layered_choropleth_map itself.
    assert fig.layout.mapbox.style == expected_style

def test_plot_layered_choropleth_map_with_facilities_robust(sample_choropleth_gdf, sample_iot_clinic_df_main, mocker):
    mocker.patch('utils.ui_visualization_helpers.MAPBOX_TOKEN_SET', True)
    if 'zone_id' not in sample_choropleth_gdf.columns:
        sample_choropleth_gdf['zone_id'] = [f"Z{i}" for i in range(len(sample_choropleth_gdf))]
    
    sample_choropleth_gdf_valid = sample_choropleth_gdf[
        sample_choropleth_gdf.geometry.is_valid & ~sample_choropleth_gdf.geometry.is_empty
    ]
    if sample_choropleth_gdf_valid.empty:
        pytest.skip("Skipping facility map test: base GDF has no valid geometries.")

    # Create dummy facility points from centroids of valid zone geometries for testing
    facility_points = sample_choropleth_gdf_valid.geometry.centroid
    facility_test_gdf = gpd.GeoDataFrame({
        'facility_name': sample_choropleth_gdf_valid.get('name', pd.Series([f"Fac{i}" for i in range(len(facility_points))])),
        'capacity': np.random.randint(5, 25, size=len(facility_points))
    }, geometry=facility_points, crs=sample_choropleth_gdf.crs)

    fig = plot_layered_choropleth_map(
        gdf=sample_choropleth_gdf_valid, value_col='population', title="Choropleth with Test Facilities",
        id_col='zone_id', facility_gdf=facility_test_gdf,
        facility_size_col='capacity', facility_hover_name='facility_name'
    )
    assert len(fig.data) >= 2 # Choropleth + Scattermapbox for facilities
    assert isinstance(fig.data[0], go.Choroplethmapbox)
    assert isinstance(fig.data[1], go.Scattermapbox) and fig.data[1].name == 'Facilities Layer'

@pytest.mark.parametrize("plot_func, required_args_dict, title_arg_name", [
    (plot_annotated_line_chart, {"data_series": pd.Series(dtype='float64')}, "title"),
    (plot_bar_chart, {"df_input": pd.DataFrame(columns=['x','y']), "x_col": "x", "y_col": "y"}, "title"),
    (plot_donut_chart, {"data_df_input": pd.DataFrame(columns=['l','v']), "labels_col": "l", "values_col": "v"}, "title"),
    (plot_heatmap, {"matrix_df_input": pd.DataFrame()}, "title"), # An empty DF should trigger empty figure
    (plot_layered_choropleth_map, {
        "gdf": gpd.GeoDataFrame(columns=['geometry','id_col','val_col'], geometry='geometry', crs=app_config.DEFAULT_CRS),
        "value_col": "val_col", "id_col":"id_col"
    }, "title")
])
def test_plotting_functions_empty_input_handling(plot_func, required_args_dict, title_arg_name):
    test_title = f"Empty Data Test for {plot_func.__name__}"
    args_for_call = {**required_args_dict, title_arg_name: test_title}
    fig = plot_func(**args_for_call)
    assert isinstance(fig, go.Figure)
    # Check for variations in the "No data" message or specific error messages handled by _create_empty_figure
    assert (f"{test_title} (No data available to display.)" in fig.layout.title.text or
            f"{test_title} (Map data unavailable or configuration error.)" in fig.layout.title.text or
            f"{test_title} (Invalid or empty geometries.)" in fig.layout.title.text or
            f"{test_title} (No positive data to display.)" in fig.layout.title.text or # for donut
            f"{test_title} (All data non-numeric.)" in fig.layout.title.text or # for heatmap
            f"{test_title} (No numeric data for y-axis" in fig.layout.title.text or # for bar chart
            f"{test_title} (Missing ID col" in fig.layout.title.text) # for map
    assert len(fig.data) == 0 # Empty figures should have no data traces

def test_get_theme_color_utility():
    color1 = _get_theme_color(0); assert isinstance(color1, str) and color1.startswith("#")
    color_fallback = _get_theme_color(100, fallback_color="#FF0000") # Index out of typical theme colorway bounds
    # It might return a color from the default plotly theme if custom has fewer, or fallback
    assert color_fallback == "#FF0000" or (isinstance(color_fallback, str) and color_fallback.startswith("#"))

    # Test disease colors
    tb_color_expected = app_config.DISEASE_COLORS.get("TB", "#CCCCCC")
    tb_color_actual = _get_theme_color("TB", color_type="disease", fallback_color="#CCCCCC")
    assert tb_color_actual == tb_color_expected

    unknown_disease_color = _get_theme_color("Flu", color_type="disease", fallback_color="#DDDDDD")
    assert unknown_disease_color == "#DDDDDD" # Since "Flu" is not in app_config.DISEASE_COLORS

    # Test risk status colors
    high_risk_color_expected = app_config.RISK_STATUS_COLORS.get("High", "#BBBBBB")
    high_risk_color_actual = _get_theme_color("High", color_type="risk_status", fallback_color="#BBBBBB")
    assert high_risk_color_actual == high_risk_color_expected
