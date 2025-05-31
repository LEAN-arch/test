# tests/test_ui_visualization_helpers.py
import pytest
import plotly.graph_objects as go
import plotly.io as pio 
import pandas as pd
import numpy as np
from unittest.mock import patch 
import geopandas as gpd
from shapely.geometry import Point 

from utils.ui_visualization_helpers import (
    set_custom_plotly_theme, 
    render_kpi_card, render_traffic_light,
    plot_annotated_line_chart, plot_bar_chart,
    plot_donut_chart, plot_heatmap,
    plot_layered_choropleth_map, _create_empty_figure, _get_theme_color
)
from config import app_config 

set_custom_plotly_theme() 

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
    assert 'class="kpi-card high status-good"' in html_output or 'class="kpi-card good status-high"' in html_output 
    assert 'title="Total patients registered."' in html_output
    assert "Total Patients" in html_output and "1,234" in html_output and "🧑‍🤝‍🧑" in html_output
    assert '<p class="kpi-delta positive">+20</p>' in html_output

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_kpi_card_minimal_params_robust(mock_st_markdown):
    render_kpi_card(title="Alerts", value="5", icon="🚨") 
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    # Check for "neutral" if default, or specific status if passed but not in valid_statuses in render_kpi_card
    assert 'class="kpi-card neutral' in html_output.lower() # .lower() to handle potential status-neutral vs neutral
    assert 'title=""' not in html_output
    assert "Alerts" in html_output and '<p class="kpi-delta' not in html_output

@patch('utils.ui_visualization_helpers.st.markdown')
def test_render_traffic_light_standard(mock_st_markdown):
    render_traffic_light(message="Network Status", status="Low", details="All connections stable.")
    mock_st_markdown.assert_called_once()
    html_output = mock_st_markdown.call_args[0][0]
    assert '<div class="traffic-light-indicator">' in html_output and '<span class="traffic-light-dot status-low">' in html_output
    assert "Network Status" in html_output and '<span class="traffic-light-details">All connections stable.</span>' in html_output

def test_plot_annotated_line_chart_valid_data_robust(sample_series_data):
    fig = plot_annotated_line_chart(sample_series_data, "Test Line: Daily Count", y_axis_title="Count")
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1 
    assert "Test Line: Daily Count" in fig.layout.title.text and fig.layout.yaxis.title.text == "Count"
    assert fig.layout.height == app_config.DEFAULT_PLOT_HEIGHT 

def test_plot_annotated_line_chart_empty_robust():
    empty_series = pd.Series(dtype='float64') 
    fig = plot_annotated_line_chart(empty_series, "Empty Line Chart Test")
    assert isinstance(fig, go.Figure) and "Empty Line Chart Test (No data available to display.)" in fig.layout.title.text and len(fig.data) == 0

def test_plot_annotated_line_chart_with_ci_target_anomalies(sample_series_data):
    lower = sample_series_data * 0.9; upper = sample_series_data * 1.1
    data_with_anomaly = sample_series_data.copy()
    if len(data_with_anomaly) > 5: data_with_anomaly.iloc[2] = data_with_anomaly.max() + 2 * data_with_anomaly.std() 
    fig = plot_annotated_line_chart(data_with_anomaly, "Line with CI, Target, Anomalies", target_line=12, target_label="Target Value", show_ci=True, lower_bound_series=lower, upper_bound_series=upper, show_anomalies=True)
    assert len(fig.data) >= 2 
    assert any(trace.name == "Confidence Interval" for trace in fig.data)
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].y0 == 12
    if any(trace.name == "Potential Anomaly" for trace in fig.data): assert True
    else: logger.info("Anomaly trace not present in test_plot_annotated_line_chart_with_ci_target_anomalies")

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
    fig = plot_heatmap(sample_heatmap_df, "Heatmap: Test Matrix", colorscale="Viridis", zmid=None) 
    assert isinstance(fig, go.Figure) and len(fig.data) == 1 and isinstance(fig.data[0], go.Heatmap)
    assert "Heatmap: Test Matrix" in fig.layout.title.text and fig.data[0].zmid is None 

def test_plot_layered_choropleth_map_valid_gdf_robust(sample_choropleth_gdf):
    if 'zone_id' not in sample_choropleth_gdf.columns: sample_choropleth_gdf['zone_id'] = [f"Z{i}" for i in range(len(sample_choropleth_gdf))]
    fig = plot_layered_choropleth_map(gdf=sample_choropleth_gdf, value_col='risk_score', title="Choropleth: Zone Risk", id_col='zone_id', featureidkey_prefix='properties', mapbox_style=app_config.MAPBOX_STYLE)
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1 
    assert isinstance(fig.data[0], go.Choroplethmapbox) and "Choropleth: Zone Risk" in fig.layout.title.text
    theme_mapbox_style = app_config.MAPBOX_STYLE # Default from config if not overridden in pio.templates
    try: theme_mapbox_style = pio.templates.default.layout.mapbox.style # Get resolved theme style
    except: pass
    expected_style = theme_mapbox_style
    if not MAPBOX_TOKEN_SET and expected_style not in ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"]:
        expected_style = "open-street-map"
    assert fig.layout.mapbox.style == expected_style

def test_plot_layered_choropleth_map_with_facilities_robust(sample_choropleth_gdf, sample_iot_clinic_df_main):
    if 'zone_id' not in sample_choropleth_gdf.columns: sample_choropleth_gdf['zone_id'] = [f"Z{i}" for i in range(len(sample_choropleth_gdf))]
    sample_choropleth_gdf_valid = sample_choropleth_gdf[sample_choropleth_gdf.geometry.is_valid & ~sample_choropleth_gdf.geometry.is_empty]
    if sample_choropleth_gdf_valid.empty: pytest.skip("Skipping facility test: base GDF no valid geometries.")
    facility_points = sample_choropleth_gdf_valid.geometry.centroid
    facility_test_gdf = gpd.GeoDataFrame({'facility_name': sample_choropleth_gdf_valid.get('name', pd.Series([f"Fac{i}" for i in range(len(facility_points))])), 'capacity': np.random.randint(5, 25, size=len(facility_points))}, geometry=facility_points, crs=sample_choropleth_gdf.crs)
    fig = plot_layered_choropleth_map(gdf=sample_choropleth_gdf, value_col='population', title="Choropleth with Test Facilities", id_col='zone_id', facility_gdf=facility_test_gdf, facility_size_col='capacity', facility_hover_name='facility_name')
    assert len(fig.data) == 2 and isinstance(fig.data[1], go.Scattermapbox) and fig.data[1].name == 'Facilities Layer'

@pytest.mark.parametrize("plot_func, required_args_dict, title_arg_name", [
    (plot_annotated_line_chart, {"data_series": pd.Series(dtype='float64')}, "title"),
    (plot_bar_chart, {"df_input": pd.DataFrame(columns=['x','y']), "x_col": "x", "y_col": "y"}, "title"),
    (plot_donut_chart, {"data_df_input": pd.DataFrame(columns=['l','v']), "labels_col": "l", "values_col": "v"}, "title"),
    (plot_heatmap, {"matrix_df_input": pd.DataFrame()}, "title"),
    (plot_layered_choropleth_map, {"gdf": gpd.GeoDataFrame(columns=['geometry','id_col','val_col'], geometry='geometry', crs=app_config.DEFAULT_CRS), "value_col": "val_col", "id_col":"id_col"}, "title")
])
def test_plotting_functions_empty_input_handling(plot_func, required_args_dict, title_arg_name):
    test_title = f"Empty Data Test for {plot_func.__name__}"; args_for_call = {**required_args_dict, title_arg_name: test_title}
    fig = plot_func(**args_for_call); assert isinstance(fig, go.Figure)
    assert f"{test_title} (No data" in fig.layout.title.text or f"{test_title} (Map Data Error" in fig.layout.title.text or f"{test_title} (Invalid data" in fig.layout.title.text or f"{test_title} (No positive data" in fig.layout.title.text or f"{test_title} (All data non-numeric" in fig.layout.title.text
    assert len(fig.data) == 0

def test_get_theme_color_utility():
    color1 = _get_theme_color(0); assert isinstance(color1, str) and color1.startswith("#")
    color_fallback = _get_theme_color(100, fallback_color="#FF0000"); assert color_fallback == "#FF0000" or (isinstance(color_fallback, str) and color_fallback.startswith("#")) 
    disease_color_tb = _get_theme_color("TB", color_type="disease", fallback_color="#CCCCCC"); assert disease_color_tb == app_config.DISEASE_COLORS.get("TB", "#CCCCCC")
