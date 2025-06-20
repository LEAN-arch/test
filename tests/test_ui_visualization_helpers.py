# test/utils/ui_visualization_helpers.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging
import plotly.io as pio
from config import app_config
import html
import geopandas as gpd
import os
from typing import Optional, List, Dict, Any

# Logging Setup - Ensure logger is available
logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
                    format=app_config.LOG_FORMAT, datefmt=app_config.LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

MAPBOX_TOKEN_SET = False
try:
    MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
    if MAPBOX_ACCESS_TOKEN and MAPBOX_ACCESS_TOKEN.strip() and "YOUR_MAPBOX_ACCESS_TOKEN" not in MAPBOX_ACCESS_TOKEN and len(MAPBOX_ACCESS_TOKEN) > 20:
        px.set_mapbox_access_token(MAPBOX_ACCESS_TOKEN)
        MAPBOX_TOKEN_SET = True
        logger.info("Mapbox access token found and set for Plotly Express.")
    else:
        log_msg = "MAPBOX_ACCESS_TOKEN environment variable not found or is invalid placeholder."
        logger.warning(f"{log_msg} Map styles requiring a token may not work; defaulting to open styles.")
except Exception as e_token:
    logger.error(f"Error setting Mapbox token: {e_token}")

def _get_theme_color(index: Any = 0, fallback_color: str = "#007bff", color_type: str = "general") -> str:
    """Safely retrieves a color from various theme configurations or Plotly's default."""
    try:
        if color_type == "disease" and hasattr(app_config, 'DISEASE_COLORS') and app_config.DISEASE_COLORS:
            if isinstance(index, str) and index in app_config.DISEASE_COLORS: return app_config.DISEASE_COLORS[index]
        if color_type == "risk_status" and hasattr(app_config, 'RISK_STATUS_COLORS') and app_config.RISK_STATUS_COLORS:
            if isinstance(index, str) and index in app_config.RISK_STATUS_COLORS: return app_config.RISK_STATUS_COLORS[index]

        # Try to get colorway from the active Plotly template
        active_template_layout = pio.templates.get(pio.templates.default, {}).get('layout', {})
        colorway = active_template_layout.get('colorway', px.colors.qualitative.Plotly) # Fallback to Plotly's default

        if colorway:
            num_idx = index if isinstance(index, int) else hash(str(index)) % len(colorway) # Get a consistent index for strings
            return colorway[num_idx % len(colorway)]

    except Exception as e_color_get:
        logger.warning(f"Could not retrieve theme color for index/key '{index}', type '{color_type}': {e_color_get}. Using fallback: {fallback_color}")
    return fallback_color

def set_custom_plotly_theme():
    """Sets a custom Plotly theme ('custom_health_theme') as the default."""
    theme_font_family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji"'
    theme_primary_text_color = "#343a40"; theme_grid_color = "#e9ecef"; theme_border_color = "#ced4da"
    theme_paper_bg_color = "#f8f9fa" ; theme_plot_bg_color = "#FFFFFF"
    
    custom_theme = go.layout.Template()
    custom_theme.layout.font = dict(family=theme_font_family, size=12, color=theme_primary_text_color)
    custom_theme.layout.paper_bgcolor = theme_paper_bg_color
    custom_theme.layout.plot_bgcolor = theme_plot_bg_color
    custom_theme.layout.colorway = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#17a2b8', '#6f42c1', '#fd7e14', '#20c997', '#6610f2', '#e83e8c']
    
    axis_common = dict(gridcolor=theme_grid_color, linecolor=theme_border_color, zerolinecolor=theme_grid_color, zerolinewidth=1, title_font_size=13, tickfont_size=11, automargin=True, title_standoff=10 ) # Reduced standoff
    custom_theme.layout.xaxis = {**axis_common}
    custom_theme.layout.yaxis = {**axis_common}
    custom_theme.layout.title = dict(font=dict(size=16, weight='bold'), x=0.02, xanchor='left', y=0.97, yanchor='top', pad=dict(t=25, b=15)) # More padding for title
    custom_theme.layout.legend = dict(bgcolor='rgba(255,255,255,0.9)', bordercolor=theme_border_color, borderwidth=1, orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font_size=11, traceorder='normal')
    custom_theme.layout.margin = dict(l=60, r=20, t=80, b=60) # Adjusted margins
    
    default_mapbox_style = app_config.MAPBOX_STYLE
    if not MAPBOX_TOKEN_SET and app_config.MAPBOX_STYLE not in ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"]:
        default_mapbox_style = "open-street-map"
    custom_theme.layout.mapbox = dict(style=default_mapbox_style, center=dict(lat=app_config.MAP_DEFAULT_CENTER_LAT, lon=app_config.MAP_DEFAULT_CENTER_LON), zoom=app_config.MAP_DEFAULT_ZOOM)
    
    # Combine with plotly base theme for defaults not explicitly set
    pio.templates["custom_health_theme"] = custom_theme
    pio.templates.default = "plotly+custom_health_theme"
    logger.info("Custom Plotly theme 'custom_health_theme' combined with 'plotly' set as default.")
set_custom_plotly_theme() # Initialize theme on import

def render_kpi_card(title: str, value: str, icon: str, status: str = "neutral", delta: Optional[str] = None, delta_type: str = "neutral", help_text: Optional[str] = None, icon_is_html: bool = False):
    valid_statuses = {"high", "moderate", "low", "neutral", "good", "bad"} # good/bad are semantic prefixes
    valid_delta_types = {"positive", "negative", "neutral"}
    
    status_parts = status.lower().split()
    base_status_class = next((s for s in status_parts if s in valid_statuses and s not in ["good", "bad"]), "neutral")
    semantic_class = next((f"status-{s}" for s in status_parts if s in ["good", "bad"]), "")
    
    final_delta_type = delta_type.lower() if delta_type and delta_type.lower() in valid_delta_types else "neutral"
    delta_str = str(delta) if delta is not None and str(delta).strip() else ""
    delta_html = f'<p class="kpi-delta {final_delta_type}">{html.escape(delta_str)}</p>' if delta_str else ''
    tooltip_html = f'title="{html.escape(str(help_text))}"' if help_text and str(help_text).strip() else ''
    icon_display = str(icon) if icon_is_html else html.escape(str(icon) if icon is not None else "●")
    
    combined_status_class = f"{base_status_class} {semantic_class}".strip()
    html_content = f'<div class="kpi-card {combined_status_class}" {tooltip_html}><div class="kpi-card-header"><div class="kpi-icon">{icon_display}</div><h3 class="kpi-title">{html.escape(str(title))}</h3></div><div class="kpi-body"><p class="kpi-value">{html.escape(str(value))}</p>{delta_html}</div></div>'.replace("\n", "")
    st.markdown(html_content, unsafe_allow_html=True)

def render_traffic_light(message: str, status: str, details: str = ""):
    valid_statuses_tl = {"high", "moderate", "low", "neutral"} # Case-insensitive check
    dot_status_class = "status-" + (status.lower() if status and status.lower() in map(str.lower, valid_statuses_tl) else "neutral")
    details_html = f'<span class="traffic-light-details">{html.escape(str(details))}</span>' if details and str(details).strip() else ""
    html_content = f'<div class="traffic-light-indicator"><span class="traffic-light-dot {dot_status_class}"></span><span class="traffic-light-message">{html.escape(str(message))}</span>{details_html}</div>'.replace("\n", "")
    st.markdown(html_content, unsafe_allow_html=True)

def _create_empty_figure(title: str, height: Optional[int], message: str = "No data available to display.") -> go.Figure:
    fig = go.Figure()
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT
    fig.update_layout(title_text=f"{title} ({message})", height=final_height, xaxis={'visible': False}, yaxis={'visible': False}, annotations=[dict(text=message, xref="paper", yref="paper", showarrow=False, font=dict(size=14))])
    return fig

def plot_layered_choropleth_map(gdf: gpd.GeoDataFrame, value_col: str, title: str, id_col: str = 'zone_id', featureidkey_prefix: str = 'properties', color_continuous_scale: str = "Blues_r", hover_cols: Optional[List[str]] = None, facility_gdf: Optional[gpd.GeoDataFrame] = None, facility_size_col: Optional[str] = None, facility_hover_name: Optional[str] = None, facility_color: Optional[str] = None, height: Optional[int] = None, center_lat: Optional[float] = None, center_lon: Optional[float] = None, zoom_level: Optional[int] = None, mapbox_style: Optional[str] = None ) -> go.Figure:
    final_height = height if height is not None else app_config.MAP_PLOT_HEIGHT
    error_msg_map = "Map data unavailable or configuration error."
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty: return _create_empty_figure(title, final_height, error_msg_map)
    active_geom_col = gdf.geometry.name
    if active_geom_col not in gdf.columns or not hasattr(gdf[active_geom_col], 'is_empty') or gdf[active_geom_col].is_empty.all() or not hasattr(gdf[active_geom_col], 'is_valid') or not gdf[active_geom_col].is_valid.any():
        return _create_empty_figure(title, final_height, "Invalid or empty geometries.")
    gdf_plot = gdf.copy()
    if id_col not in gdf_plot.columns or value_col not in gdf_plot.columns: return _create_empty_figure(title, final_height, f"Missing ID col '{id_col}' or Value col '{value_col}'.")
    if not pd.api.types.is_numeric_dtype(gdf_plot[value_col]): gdf_plot[value_col] = pd.to_numeric(gdf_plot[value_col], errors='coerce')
    gdf_plot[value_col].fillna(0, inplace=True) # Fill NaNs for color scale after potential coercion
    featureidkey_path = f"{featureidkey_prefix}.{id_col}" if featureidkey_prefix and featureidkey_prefix.strip() else id_col
    gdf_plot[id_col] = gdf_plot[id_col].astype(str) # Ensure ID is string for geojson featureidkey matching
    gdf_for_geojson = gdf_plot[gdf_plot.geometry.is_valid & ~gdf_plot.geometry.is_empty].copy()
    if gdf_for_geojson.empty: return _create_empty_figure(title, final_height, "No valid geometries remaining for map after filtering.")
    
    effective_mapbox_style = mapbox_style or pio.templates.default.layout.get('mapbox',{}).get('style', app_config.MAPBOX_STYLE)
    if not MAPBOX_TOKEN_SET and effective_mapbox_style not in ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"]:
        logger.warning(f"Map ('{title}'): Style '{effective_mapbox_style}' may require token. Defaulting to 'open-street-map'.")
        effective_mapbox_style = "open-street-map"
    
    hover_name_col = "name" if "name" in gdf_plot.columns else id_col
    default_hover_data = [hover_name_col, value_col, 'population']; final_hover_data_list = hover_cols if hover_cols is not None else default_hover_data
    hover_data_for_plot = {col: True for col in final_hover_data_list if col in gdf_plot.columns and col != hover_name_col and gdf_plot[col].notna().any()}
    labels_for_plot = {col: str(col).replace('_', ' ').title() for col in [value_col] + list(hover_data_for_plot.keys())}

    try:
        fig = px.choropleth_mapbox(data_frame=gdf_for_geojson, geojson=gdf_for_geojson.geometry.__geo_interface__, locations=id_col, featureidkey=featureidkey_path, color=value_col, color_continuous_scale=color_continuous_scale, opacity=0.75, hover_name=hover_name_col, hover_data=hover_data_for_plot, labels=labels_for_plot, mapbox_style=effective_mapbox_style, center={"lat": center_lat or app_config.MAP_DEFAULT_CENTER_LAT, "lon": center_lon or app_config.MAP_DEFAULT_CENTER_LON}, zoom=zoom_level if zoom_level is not None else app_config.MAP_DEFAULT_ZOOM )
    except Exception as e_px: logger.error(f"MAP ERROR ({title}): px.choropleth_mapbox failed: {e_px}", exc_info=True); return _create_empty_figure(title, final_height, f"Map rendering error: {str(e_px)[:100]}")

    if facility_gdf is not None and not facility_gdf.empty and 'geometry' in facility_gdf.columns:
        facility_plot_gdf = facility_gdf[facility_gdf.geometry.geom_type == 'Point'].copy()
        if not facility_plot_gdf.empty:
            facility_hover_text_series = facility_plot_gdf.get(facility_hover_name, pd.Series(["Facility"] * len(facility_plot_gdf), index=facility_plot_gdf.index)) if facility_hover_name else pd.Series(["Facility"] * len(facility_plot_gdf), index=facility_plot_gdf.index)
            facility_marker_size_series = pd.Series([10] * len(facility_plot_gdf), index=facility_plot_gdf.index) # Default size
            if facility_size_col and facility_size_col in facility_plot_gdf.columns and pd.api.types.is_numeric_dtype(facility_plot_gdf[facility_size_col]):
                sizes = pd.to_numeric(facility_plot_gdf[facility_size_col], errors='coerce').fillna(0); min_s_px, max_s_px = 6, 20; min_v, max_v = sizes.min(), sizes.max()
                if max_v > min_v and max_v > 0 : facility_marker_size_series = min_s_px + ((sizes - min_v) * (max_s_px - min_s_px) / (max_v - min_v))
                elif sizes.notna().any() and sizes.max() > 0 : facility_marker_size_series = pd.Series([(min_s_px + max_s_px) / 2] * len(facility_plot_gdf), index=facility_plot_gdf.index)
            final_facility_color = facility_color if facility_color else _get_theme_color(5, fallback_color='#6F42C1')
            fig.add_trace(go.Scattermapbox(lon=facility_plot_gdf.geometry.x, lat=facility_plot_gdf.geometry.y, mode='markers', marker=go.scattermapbox.Marker(size=facility_marker_size_series, sizemin=5, color=final_facility_color, opacity=0.9, allowoverlap=True), text=facility_hover_text_series, hoverinfo='text', name='Facilities Layer'))
    fig.update_layout(title_text=title, height=final_height, margin={"r":10,"t":60,"l":10,"b":10}, legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.7)'))
    return fig

def plot_annotated_line_chart(data_series: pd.Series, title: str, y_axis_title: str = "Value", color: Optional[str] = None, target_line: Optional[float] = None, target_label: Optional[str] = None, show_ci: bool = False, lower_bound_series: Optional[pd.Series] = None, upper_bound_series: Optional[pd.Series] = None, height: Optional[int] = None, show_anomalies: bool = True, date_format: str = "%b %Y") -> go.Figure:
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT
    if not isinstance(data_series, pd.Series) or data_series.empty: return _create_empty_figure(title, final_height)
    data_series_numeric = pd.to_numeric(data_series, errors='coerce')
    if data_series_numeric.isnull().all() and not data_series.empty : return _create_empty_figure(title, final_height, "All data non-numeric.")
    elif data_series_numeric.empty and not data_series.empty : return _create_empty_figure(title, final_height, "No valid numeric data after coercion.") # If all became NaN
    
    fig = go.Figure(); line_color_val = color if color else _get_theme_color(0)
    fig.add_trace(go.Scatter(x=data_series_numeric.index, y=data_series_numeric.values, mode="lines+markers", name=y_axis_title, line=dict(color=line_color_val, width=2.5), marker=dict(size=6), customdata=data_series_numeric.values, hovertemplate=(f'<b>Date</b>: %{{x|{date_format}}}<br><b>{y_axis_title}</b>: %{{customdata:,.2f}}<extra></extra>')))
    if show_ci and lower_bound_series is not None and upper_bound_series is not None and not lower_bound_series.empty and not upper_bound_series.empty:
        common_idx_ci = data_series_numeric.index.intersection(lower_bound_series.index).intersection(upper_bound_series.index)
        if not common_idx_ci.empty:
            ls = pd.to_numeric(lower_bound_series.reindex(common_idx_ci), errors='coerce'); us = pd.to_numeric(upper_bound_series.reindex(common_idx_ci), errors='coerce'); valid_ci_mask = ls.notna() & us.notna() & (us >= ls) # ensure upper >= lower
            if valid_ci_mask.any():
                x_ci_plot = common_idx_ci[valid_ci_mask]; y_upper_plot = us[valid_ci_mask]; y_lower_plot = ls[valid_ci_mask]
                fill_color_rgba = f"rgba({','.join(str(int(c, 16)) for c in (line_color_val[1:3], line_color_val[3:5], line_color_val[5:7]))},0.15)" if line_color_val.startswith('#') and len(line_color_val) == 7 else "rgba(0,123,255,0.15)"
                fig.add_trace(go.Scatter(x=list(x_ci_plot) + list(x_ci_plot[::-1]), y=list(y_upper_plot.values) + list(y_lower_plot.values[::-1]), fill="toself", fillcolor=fill_color_rgba, line=dict(width=0), name="Confidence Interval", hoverinfo='skip'))
    if target_line is not None: fig.add_hline(y=target_line, line_dash="dot", line_color="#e74c3c", line_width=1.5, annotation_text=target_label if target_label else f"Target: {target_line:,.2f}", annotation_position="top right", annotation_font_size=10, annotation_font_color="#c0392b")
    if show_anomalies and len(data_series_numeric.dropna()) > 10 and data_series_numeric.nunique() > 1:
        q1 = data_series_numeric.quantile(0.25); q3 = data_series_numeric.quantile(0.75); iqr = q3 - q1
        if pd.notna(iqr) and iqr > 1e-9 : # Avoid issues with zero IQR
            upper_b = q3 + 1.5 * iqr; lower_b = q1 - 1.5 * iqr
            anomalies = data_series_numeric[(data_series_numeric < lower_b) | (data_series_numeric > upper_b)]
            if not anomalies.empty: fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies.values, mode='markers', marker=dict(color=_get_theme_color(6, fallback_color='#fd7e14'), size=9, symbol='x-thin-open', line=dict(width=2.5)), name='Potential Anomaly', customdata=anomalies.values, hovertemplate=(f'<b>Anomaly Date</b>: %{{x|{date_format}}}<br><b>Value</b>: %{{customdata:,.2f}}<extra></extra>')))
    final_xaxis_title = data_series_numeric.index.name if data_series_numeric.index.name and str(data_series_numeric.index.name).strip() else "Date"
    fig.update_layout(title_text=title, xaxis_title=final_xaxis_title, yaxis_title=y_axis_title, height=final_height, hovermode="x unified", legend=dict(traceorder='normal'))
    return fig

def plot_bar_chart(df_input: pd.DataFrame, x_col: str, y_col: str, title: str, color_col: Optional[str] = None, barmode: str = 'group', orientation: str = 'v', y_axis_title: Optional[str] = None, x_axis_title: Optional[str] = None, height: Optional[int] = None, text_auto: bool = True, sort_values_by: Optional[str] = None, ascending: bool = True, text_format: str = ',.0f', color_discrete_map: Optional[Dict] = None) -> go.Figure:
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT
    if df_input is None or df_input.empty or x_col not in df_input.columns or y_col not in df_input.columns: return _create_empty_figure(title, final_height)
    df = df_input.copy(); df[y_col] = pd.to_numeric(df[y_col], errors='coerce'); df.dropna(subset=[x_col, y_col], inplace=True) # Drop if y_col is NaN after coercion
    if df.empty: return _create_empty_figure(title, final_height, f"No valid numeric data for y-axis '{y_col}'.")
    final_y_title = y_axis_title if y_axis_title else y_col.replace('_', ' ').title(); final_x_title = x_axis_title if x_axis_title else x_col.replace('_', ' ').title()
    if sort_values_by and sort_values_by in df.columns:
        try: df.sort_values(by=sort_values_by, ascending=ascending, inplace=True, na_position='last', key=(lambda c_data: pd.to_numeric(c_data, errors='ignore') if pd.api.types.is_numeric_dtype(df[sort_values_by]) else c_data.astype(str))) # sort numeric as num, else as str
        except Exception as e_sort: logger.warning(f"Bar chart sort by '{sort_values_by}' failed: {e_sort}.")
    legend_title_text = color_col.replace('_',' ').title() if color_col and color_col in df.columns else None
    
    fig = px.bar(df, x=x_col, y=y_col, title=title, color=color_col, barmode=barmode, orientation=orientation, height=final_height, labels={y_col: final_y_title, x_col: final_x_title, color_col: legend_title_text if legend_title_text else ""}, text_auto=text_auto, color_discrete_map=color_discrete_map if color_discrete_map else _get_theme_color(color_type="disease")) # Example for default colors if disease related
    base_hover_bar = f'<b>{final_x_title}</b>: %{{x}}<br><b>{final_y_title}</b>: %{{y:{text_format}}}'
    hover_template_bar = base_hover_bar + (f'<br><b>{legend_title_text}</b>: %{{customdata[0]}}<extra></extra>' if color_col and color_col in df.columns and not df[[color_col]].empty else '<extra></extra>')
    
    actual_text_format = f'.{text_format.split(".")[-1]}' if '.' in text_format else text_format # e.g., ',.0f' -> '.0f'
    texttemplate_val = (f'%{{y:{actual_text_format}}}' if text_auto and orientation == 'v' else (f'%{{x:{actual_text_format}}}' if text_auto and orientation == 'h' else None))
    
    fig.update_traces(marker_line_width=0.7, marker_line_color='rgba(30,30,30,0.6)', textfont_size=10, textangle=0, textposition='auto' if orientation == 'v' else 'outside', cliponaxis=False, texttemplate=texttemplate_val, hovertemplate=hover_template_bar, customdata=df[[color_col]] if color_col and color_col in df.columns else None)
    fig.update_layout(yaxis_title=final_y_title, xaxis_title=final_x_title, uniformtext_minsize=8, uniformtext_mode='hide', legend_title_text=legend_title_text)
    if orientation == 'h' and (not sort_values_by or sort_values_by == y_col): fig.update_layout(yaxis={'categoryorder':'total ascending' if ascending else 'total descending'}) # Sort y-axis labels by value if horizontal
    return fig

def plot_donut_chart(data_df_input: pd.DataFrame, labels_col: str, values_col: str, title: str, height: Optional[int] = None, color_discrete_map: Optional[Dict] = None, pull_segments: float = 0.03, center_text: Optional[str] = None) -> go.Figure:
    final_height = height if height is not None else app_config.COMPACT_PLOT_HEIGHT + 40 # Compact by default for donuts
    if data_df_input is None or data_df_input.empty or labels_col not in data_df_input.columns or values_col not in data_df_input.columns: return _create_empty_figure(title, final_height)
    df = data_df_input.copy(); df[values_col] = pd.to_numeric(df[values_col], errors='coerce').fillna(0); df = df[df[values_col] > 0] # Only positive values
    if df.empty: return _create_empty_figure(title, final_height, "No positive data to display.")
    df.sort_values(by=values_col, ascending=False, inplace=True)
    
    plot_colors_final = None
    if color_discrete_map:
        plot_colors_final = [color_discrete_map.get(label, _get_theme_color(i)) for i, label in enumerate(df[labels_col])]
    elif hasattr(app_config,"DISEASE_COLORS") and any(str(lbl) in app_config.DISEASE_COLORS for lbl in df[labels_col]): # Check if any label matches a disease color key
        plot_colors_final = [_get_theme_color(str(lbl), color_type="disease", fallback_color=_get_theme_color(i)) for i,lbl in enumerate(df[labels_col])]
    
    fig = go.Figure(data=[go.Pie(labels=df[labels_col], values=df[values_col], hole=0.50, pull=[pull_segments if i < 3 else 0 for i in range(len(df))], textinfo='label+percent', hoverinfo='label+value+percent', insidetextorientation='radial', marker=dict(colors=plot_colors_final, line=dict(color='#ffffff', width=2)), sort=False)]) # sort=False as df already sorted
    annotations_list_donut = [dict(text=str(center_text), x=0.5, y=0.5, font_size=16, showarrow=False)] if center_text else None
    fig.update_layout(title_text=title, height=final_height, showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.15, traceorder="normal"), annotations=annotations_list_donut, margin=dict(l=20, r=100, t=60, b=20)) # Adjust right margin for legend
    return fig

def plot_heatmap(matrix_df_input: pd.DataFrame, title: str, height: Optional[int] = None, colorscale: str = "RdBu_r", zmid: Optional[float] = 0, text_auto: bool = True, text_format: str = ".2f", show_colorbar: bool = True) -> go.Figure:
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT + 80
    if not isinstance(matrix_df_input, pd.DataFrame) or matrix_df_input.empty: return _create_empty_figure(title, final_height, "Invalid data for Heatmap.")
    df_numeric = matrix_df_input.copy().apply(pd.to_numeric, errors='coerce') # Convert all to numeric
    if df_numeric.isnull().all().all(): return _create_empty_figure(title, final_height, "All data non-numeric or empty after coercion.")
    df_plot_heatmap = df_numeric.fillna(0) # Fill NaNs for plotting (could be an option)
    z_vals = df_plot_heatmap.values; text_vals_heatmap = np.around(z_vals, decimals=int(text_format[-2]) if text_format.endswith('f') and text_format[-2].isdigit() else 2) if not df_plot_heatmap.empty else None
    
    # Set zmid to None if all values are same sign or if range is very small (to prevent bad colorscale centering)
    z_flat_no_nan = z_vals[~np.isnan(z_vals)]
    zmid_final = zmid
    if len(z_flat_no_nan) > 0:
        if not (np.any(z_flat_no_nan < 0) and np.any(z_flat_no_nan > 0)): # All pos or all neg
            zmid_final = None
    else: # All NaNs or empty
        zmid_final = None
        
    fig = go.Figure(data=go.Heatmap(z=z_vals, x=df_plot_heatmap.columns.tolist(), y=df_plot_heatmap.index.tolist(), colorscale=colorscale, zmid=zmid_final, text=text_vals_heatmap if text_auto else None, texttemplate=f"%{{text:{text_format}}}" if text_auto and text_vals_heatmap is not None else "", hoverongaps=False, xgap=1.5, ygap=1.5, colorbar=dict(thickness=18, len=0.85, tickfont_size=10, title_side="right", outlinewidth=0.8, outlinecolor=_get_theme_color(0,"#ced4da")) if show_colorbar else None ))
    max_x_label_len = max((len(str(c)) for c in df_plot_heatmap.columns if c is not None), default=0)
    rotate_x = -40 if len(df_plot_heatmap.columns) > 7 or max_x_label_len > 8 else 0
    fig.update_layout(title_text=title, height=final_height, xaxis_showgrid=False, yaxis_showgrid=False, xaxis_tickangle=rotate_x, yaxis_autorange='reversed', plot_bgcolor='rgba(0,0,0,0)')
    return fig
