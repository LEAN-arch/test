# health_hub/utils/ui_visualization_helpers.py
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
        log_msg = "MAPBOX_ACCESS_TOKEN environment variable not found."
        if MAPBOX_ACCESS_TOKEN: log_msg = "MAPBOX_ACCESS_TOKEN environment variable is a placeholder or too short."
        logger.warning(f"{log_msg} Map styles requiring a token may not work; defaulting to open styles.")
except Exception as e_token: 
    logger.error(f"Error setting Mapbox token: {e_token}")


def _get_theme_color(index=0, fallback_color="#007bff", color_type="general"):
    """
    Safely retrieves a color from various theme configurations or Plotly's default.
    color_type can be 'general', 'disease', 'risk_status' to access specific palettes if defined.
    """
    try:
        if color_type == "disease" and hasattr(app_config, 'DISEASE_COLORS') and app_config.DISEASE_COLORS:
            # For disease, index might be a disease name string
            if isinstance(index, str) and index in app_config.DISEASE_COLORS:
                return app_config.DISEASE_COLORS[index]
            # Fallback to general colorway if specific disease color not found or index is numeric
            
        if color_type == "risk_status" and hasattr(app_config, 'RISK_STATUS_COLORS') and app_config.RISK_STATUS_COLORS:
            if isinstance(index, str) and index in app_config.RISK_STATUS_COLORS:
                return app_config.RISK_STATUS_COLORS[index]

        # Try custom theme's main colorway
        custom_template = pio.templates.get("custom_health_theme", None)
        if custom_template and hasattr(custom_template, 'layout') and \
           hasattr(custom_template.layout, 'colorway') and custom_template.layout.colorway:
            # Ensure index is integer for list access
            num_index = index if isinstance(index, int) else 0 
            return custom_template.layout.colorway[num_index % len(custom_template.layout.colorway)]
        
        plotly_default_template = pio.templates.get("plotly", None)
        if plotly_default_template and hasattr(plotly_default_template, 'layout') and \
           hasattr(plotly_default_template.layout, 'colorway') and plotly_default_template.layout.colorway:
            logger.debug("Using color from base 'plotly' theme colorway as fallback.")
            num_index = index if isinstance(index, int) else 0
            return plotly_default_template.layout.colorway[num_index % len(plotly_default_template.layout.colorway)]
            
    except Exception as e_color_get:
        logger.warning(f"Could not retrieve theme color for index/key '{index}', type '{color_type}': {e_color_get}. Using fallback: {fallback_color}")
    return fallback_color


def set_custom_plotly_theme():
    theme_font_family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji"'
    theme_primary_text_color = "#343a40"
    theme_grid_color = "#e9ecef"
    theme_border_color = "#ced4da"
    theme_paper_bg_color = "#f8f9fa" # Default from original CSS
    theme_plot_bg_color = "#FFFFFF"  

    custom_theme = go.layout.Template()
    custom_theme.layout.font = dict(family=theme_font_family, size=12, color=theme_primary_text_color)
    custom_theme.layout.paper_bgcolor = theme_paper_bg_color
    custom_theme.layout.plot_bgcolor = theme_plot_bg_color
    custom_theme.layout.colorway = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#17a2b8', '#6f42c1', '#fd7e14', '#20c997', '#6610f2', '#e83e8c']

    axis_common = dict(gridcolor=theme_grid_color, linecolor=theme_border_color, zerolinecolor=theme_grid_color, zerolinewidth=1, title_font_size=13, tickfont_size=11, automargin=True, title_standoff=15 )
    custom_theme.layout.xaxis = {**axis_common}
    custom_theme.layout.yaxis = {**axis_common}
    custom_theme.layout.title = dict(font=dict(size=18), x=0.02, xanchor='left', y=0.97, yanchor='top', pad=dict(t=20, b=10)) 
    custom_theme.layout.legend = dict(bgcolor='rgba(255,255,255,0.9)', bordercolor=theme_border_color, borderwidth=1, orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1, font_size=11, traceorder='normal')
    custom_theme.layout.margin = dict(l=70, r=30, t=90, b=70) 
    
    default_mapbox_style = app_config.MAPBOX_STYLE
    if not MAPBOX_TOKEN_SET and app_config.MAPBOX_STYLE not in ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"]:
        default_mapbox_style = "open-street-map" 
        logger.info(f"Plotly theme: Mapbox style '{app_config.MAPBOX_STYLE}' requires token, defaulting mapbox.style to 'open-street-map'.")
    custom_theme.layout.mapbox = dict(style=default_mapbox_style, center=dict(lat=app_config.MAP_DEFAULT_CENTER_LAT, lon=app_config.MAP_DEFAULT_CENTER_LON), zoom=app_config.MAP_DEFAULT_ZOOM)
    
    pio.templates["custom_health_theme"] = custom_theme
    pio.templates.default = "plotly+custom_health_theme" 
    logger.info("Custom Plotly theme 'custom_health_theme' set as default.")

set_custom_plotly_theme()

def render_kpi_card(title, value, icon, status="neutral", delta=None, delta_type="neutral", help_text=None, icon_is_html=False):
    valid_statuses = {"high", "moderate", "low", "neutral", "good", "bad"} 
    valid_delta_types = {"positive", "negative", "neutral"}
    final_status_class = status.lower() if status and status.lower() in valid_statuses else "neutral"
    final_delta_type = delta_type.lower() if delta_type and delta_type.lower() in valid_delta_types else "neutral"
    semantic_status_class = "" # Initialize to empty string
    # Check if status itself is 'good' or 'bad' to add as a secondary class for more specific CSS targeting
    if status and status.lower() in ["good", "bad"]: 
        semantic_status_class = "status-" + status.lower()
    
    delta_str = str(delta) if delta is not None and str(delta).strip() else ""
    delta_html = f'<p class="kpi-delta {final_delta_type}">{html.escape(delta_str)}</p>' if delta_str else ""
    tooltip_html = f'title="{html.escape(str(help_text))}"' if help_text and str(help_text).strip() else ''
    icon_display = str(icon) if icon_is_html else html.escape(str(icon) if icon is not None else "●")
    combined_status_class = f"{final_status_class} {semantic_status_class}".strip() # Make sure status-good/bad comes after main color status
    html_content = f'<div class="kpi-card {combined_status_class}" {tooltip_html}><div class="kpi-card-header"><div class="kpi-icon">{icon_display}</div><h3 class="kpi-title">{html.escape(str(title))}</h3></div><div class="kpi-body"><p class="kpi-value">{html.escape(str(value))}</p>{delta_html}</div></div>'.replace("\n", "")
    st.markdown(html_content, unsafe_allow_html=True)

def render_traffic_light(message, status, details=""):
    valid_statuses_tl = {"High", "Moderate", "Low", "Neutral"}
    dot_status_class = "status-" + (status.lower() if status and status.lower() in valid_statuses_tl else "neutral")
    details_html = f'<span class="traffic-light-details">{html.escape(str(details))}</span>' if details and str(details).strip() else ""
    html_content = f'<div class="traffic-light-indicator"><span class="traffic-light-dot {dot_status_class}"></span><span class="traffic-light-message">{html.escape(str(message))}</span>{details_html}</div>'.replace("\n", "")
    st.markdown(html_content, unsafe_allow_html=True)

def _create_empty_figure(title, height, message="No data available to display."):
    fig = go.Figure()
    fig.update_layout(title_text=f"{title} ({message})", height=height, xaxis={'visible': False}, yaxis={'visible': False}, annotations=[dict(text=message, xref="paper", yref="paper", showarrow=False, font=dict(size=14))])
    return fig

def plot_layered_choropleth_map(gdf: gpd.GeoDataFrame, value_col: str, title: str, id_col: str = 'zone_id', featureidkey_prefix: str = 'properties', color_continuous_scale: str = "Blues_r", hover_cols: list = None, facility_gdf: gpd.GeoDataFrame = None, facility_size_col: str = None, facility_hover_name: str = None, facility_color: str = None, height: int = None, center_lat: float = None, center_lon: float = None, zoom_level: int = None, mapbox_style: str = None):
    final_height = height if height is not None else app_config.MAP_PLOT_HEIGHT
    error_msg_map = "Map data unavailable or configuration error."
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty: return _create_empty_figure(title, final_height, error_msg_map)
    active_geom_col = gdf.geometry.name
    if active_geom_col not in gdf.columns or gdf[active_geom_col].is_empty.all() or not gdf[active_geom_col].is_valid.any(): return _create_empty_figure(title, final_height, "Invalid or empty geometries.")
    gdf_plot = gdf.copy()
    if id_col not in gdf_plot.columns or value_col not in gdf_plot.columns: return _create_empty_figure(title, final_height, f"Missing ID col '{id_col}' or Value col '{value_col}'.")
    if not pd.api.types.is_numeric_dtype(gdf_plot[value_col]): gdf_plot[value_col] = pd.to_numeric(gdf_plot[value_col], errors='coerce')
    if gdf_plot[value_col].isnull().all(): gdf_plot[value_col] = gdf_plot[value_col].fillna(0)
    featureidkey_path = f"{featureidkey_prefix}.{id_col}" if featureidkey_prefix and featureidkey_prefix.strip() else id_col
    gdf_plot[id_col] = gdf_plot[id_col].astype(str)
    gdf_for_geojson = gdf_plot[gdf_plot.geometry.is_valid & ~gdf_plot.geometry.is_empty].copy()
    if gdf_for_geojson.empty: return _create_empty_figure(title, final_height, "No valid geometries for map.")
    
    # Safely get mapbox_style from theme or fallback
    default_map_style = app_config.MAPBOX_STYLE
    try: default_map_style = pio.templates.default.layout.mapbox.style
    except: pass # Keep app_config.MAPBOX_STYLE if theme access fails
    current_mapbox_style = mapbox_style if mapbox_style else default_map_style
    if not MAPBOX_TOKEN_SET and current_mapbox_style not in ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"]: current_mapbox_style = "open-street-map"
    
    hover_name_col = "name" if "name" in gdf_plot.columns else id_col
    default_hover_data = [hover_name_col, value_col, 'population']; final_hover_data_list = hover_cols if hover_cols else default_hover_data
    hover_data_for_plot = {col: True for col in final_hover_data_list if col in gdf_plot.columns and col != hover_name_col}

    fig_args = {"data_frame": gdf_for_geojson, "locations": id_col, "featureidkey": featureidkey_path, "color": value_col, "color_continuous_scale": color_continuous_scale, "opacity": 0.75, "hover_name": hover_name_col, "hover_data": hover_data_for_plot, "labels": {col: col.replace('_', ' ').title() for col in [value_col] + list(hover_data_for_plot.keys())}, "mapbox_style": current_mapbox_style, "center": {"lat": center_lat or app_config.MAP_DEFAULT_CENTER_LAT, "lon": center_lon or app_config.MAP_DEFAULT_CENTER_LON}, "zoom": zoom_level if zoom_level is not None else app_config.MAP_DEFAULT_ZOOM }
    try: fig = px.choropleth_mapbox(**fig_args)
    except Exception as e_px: logger.error(f"MAP ERROR ({title}): px.choropleth_mapbox failed: {e_px}", exc_info=True); return _create_empty_figure(title, final_height, f"Map rendering error: {e_px}")

    if facility_gdf is not None and not facility_gdf.empty and 'geometry' in facility_gdf.columns:
        facility_plot_gdf = facility_gdf[facility_gdf.geometry.geom_type == 'Point'].copy()
        if not facility_plot_gdf.empty:
            facility_hover_text = facility_plot_gdf.get(facility_hover_name, "Facility") if facility_hover_name else "Facility"
            facility_marker_size = 10 
            if facility_size_col and facility_size_col in facility_plot_gdf.columns and pd.api.types.is_numeric_dtype(facility_plot_gdf[facility_size_col]):
                sizes = pd.to_numeric(facility_plot_gdf[facility_size_col], errors='coerce').fillna(0); min_s_px, max_s_px = 6, 20; min_v, max_v = sizes.min(), sizes.max()
                if max_v > min_v: facility_marker_size = min_s_px + ((sizes - min_v) * (max_s_px - min_s_px) / (max_v - min_v))
                elif sizes.notna().any(): facility_marker_size = (min_s_px + max_s_px) / 2
            final_facility_color = facility_color if facility_color else _get_theme_color(5) 
            fig.add_trace(go.Scattermapbox(lon=facility_plot_gdf.geometry.x, lat=facility_plot_gdf.geometry.y, mode='markers', marker=go.scattermapbox.Marker(size=facility_marker_size, sizemin=5, color=final_facility_color, opacity=0.9, allowoverlap=True), text=facility_hover_text, hoverinfo='text', name='Facilities Layer'))
    fig.update_layout(title_text=title, height=final_height, margin={"r":10,"t":60,"l":10,"b":10}, legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.7)'))
    return fig

def plot_annotated_line_chart(data_series: pd.Series, title: str, y_axis_title: str = "Value", color: str = None, target_line: float = None, target_label: str = None, show_ci: bool = False, lower_bound_series: pd.Series = None, upper_bound_series: pd.Series = None, height: int = None, show_anomalies: bool = True, date_format: str = "%b %Y"):
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT
    if not isinstance(data_series, pd.Series) or data_series.empty: return _create_empty_figure(title, final_height)
    fig = go.Figure(); line_color_val = color if color else _get_theme_color(0)
    fig.add_trace(go.Scatter(x=data_series.index, y=data_series.values, mode="lines+markers", name=y_axis_title, line=dict(color=line_color_val, width=2.8), marker=dict(size=7, symbol='circle'), customdata=data_series.values, hovertemplate=(f'<b>Date</b>: %{{x|{date_format}}}<br><b>{y_axis_title}</b>: %{{customdata:,.2f}}<extra></extra>')))
    if show_ci and lower_bound_series is not None and upper_bound_series is not None and not lower_bound_series.empty and not upper_bound_series.empty:
        common_idx_ci = data_series.index.intersection(lower_bound_series.index).intersection(upper_bound_series.index)
        if not common_idx_ci.empty:
            ls = pd.to_numeric(lower_bound_series.reindex(common_idx_ci), errors='coerce'); us = pd.to_numeric(upper_bound_series.reindex(common_idx_ci), errors='coerce'); valid_ci_mask = ls.notna() & us.notna()
            if valid_ci_mask.any():
                x_ci_plot = common_idx_ci[valid_ci_mask]; y_upper_plot = us[valid_ci_mask]; y_lower_plot = ls[valid_ci_mask]
                fill_color_rgba = f"rgba({','.join(str(int(c, 16)) for c in (line_color_val[1:3], line_color_val[3:5], line_color_val[5:7]))},0.18)" if line_color_val.startswith('#') and len(line_color_val) == 7 else "rgba(0,123,255,0.18)"
                fig.add_trace(go.Scatter(x=list(x_ci_plot) + list(x_ci_plot[::-1]), y=list(y_upper_plot.values) + list(y_lower_plot.values[::-1]), fill="toself", fillcolor=fill_color_rgba, line=dict(width=0), name="Confidence Interval", hoverinfo='skip'))
    if target_line is not None: fig.add_hline(y=target_line, line_dash="dot", line_color="#e74c3c", line_width=1.8, annotation_text=target_label if target_label else f"Target: {target_line:,.2f}", annotation_position="top right", annotation_font_size=11, annotation_font_color="#c0392b")
    if show_anomalies and len(data_series) > 10 and data_series.nunique() > 1: 
        q1 = data_series.quantile(0.25); q3 = data_series.quantile(0.75); iqr = q3 - q1
        if pd.notna(iqr) and iqr > 1e-7 : 
            upper_b = q3 + 1.5 * iqr; lower_b = q1 - 1.5 * iqr; anomalies = data_series[(data_series < lower_b) | (data_series > upper_b)]
            if not anomalies.empty: fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies.values, mode='markers', marker=dict(color='#fd7e14', size=11, symbol='x-thin-open', line=dict(width=2.8)), name='Potential Anomaly', customdata=anomalies.values, hovertemplate=(f'<b>Anomaly Date</b>: %{{x|{date_format}}}<br><b>Value</b>: %{{customdata:,.2f}}<extra></extra>')))
    final_xaxis_title = data_series.index.name if data_series.index.name else "Date"
    fig.update_layout(title_text=title, xaxis_title=final_xaxis_title, yaxis_title=y_axis_title, height=final_height, hovermode="x unified", legend=dict(traceorder='normal'))
    return fig

def plot_bar_chart(df_input, x_col: str, y_col: str, title: str, color_col: str = None, barmode: str = 'group', orientation: str = 'v', y_axis_title: str = None, x_axis_title: str = None, height: int = None, text_auto: bool = True, sort_values_by: str = None, ascending: bool = True, text_format: str = ',.0f', color_discrete_map: dict = None):
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT
    if df_input is None or df_input.empty or x_col not in df_input.columns or y_col not in df_input.columns: return _create_empty_figure(title, final_height)
    df = df_input.copy(); df[y_col] = pd.to_numeric(df[y_col], errors='coerce'); df.dropna(subset=[x_col, y_col], inplace=True)
    if df.empty: return _create_empty_figure(title, final_height, f"No numeric data for y-axis '{y_col}'.")
    final_y_title = y_axis_title if y_axis_title else y_col.replace('_', ' ').title(); final_x_title = x_axis_title if x_axis_title else x_col.replace('_', ' ').title()
    if sort_values_by and sort_values_by in df.columns:
        try: df.sort_values(by=sort_values_by, ascending=ascending, inplace=True, na_position='last', key=(lambda c_data: c_data.astype(str) if not pd.api.types.is_numeric_dtype(c_data) else None))
        except Exception as e_sort: logger.warning(f"Bar chart sort by '{sort_values_by}' failed: {e_sort}.")
    legend_title = color_col.replace('_',' ').title() if color_col and color_col in df.columns else None
    fig = px.bar(df, x=x_col, y=y_col, title=title, color=color_col, barmode=barmode, orientation=orientation, height=final_height, labels={y_col: final_y_title, x_col: final_x_title, color_col: legend_title if legend_title else ""}, text_auto=text_auto, color_discrete_map=color_discrete_map)
    base_hover_bar = f'<b>{final_x_title}</b>: %{{x}}<br><b>{final_y_title}</b>: %{{y:{text_format}}}'; hover_template_bar = base_hover_bar + (f'<br><b>{legend_title}</b>: %{{customdata[0]}}<extra></extra>' if color_col and color_col in df else '<extra></extra>')
    fig.update_traces(marker_line_width=0.8, marker_line_color='rgba(40,40,40,0.5)', textfont_size=11, textangle=0, textposition='auto' if orientation == 'v' else 'outside', cliponaxis=False, texttemplate=f'%{{y:{text_format}}}' if text_auto and orientation == 'v' else (f'%{{x:{text_format}}}' if text_auto and orientation == 'h' else None), hovertemplate=hover_template_bar, customdata=df[[color_col]] if color_col and color_col in df else None)
    fig.update_layout(yaxis_title=final_y_title, xaxis_title=final_x_title, uniformtext_minsize=9, uniformtext_mode='hide', legend_title_text=legend_title)
    if orientation == 'h' and not sort_values_by: fig.update_layout(yaxis={'categoryorder':'total ascending' if ascending else 'total descending'})
    return fig


def plot_donut_chart(
    data_df_input, labels_col: str, values_col: str, title: str, height: int = None,
    color_discrete_map: dict = None, pull_segments: float = 0.03, center_text: str = None
):
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT + 40
    if data_df_input is None or data_df_input.empty or labels_col not in data_df_input.columns or values_col not in data_df_input.columns:
        logger.warning(f"Donut Chart '{title}': Empty or invalid input DataFrame, or missing columns '{labels_col}'/'{values_col}'.")
        return _create_empty_figure(title, final_height)

    df = data_df_input.copy()
    df[values_col] = pd.to_numeric(df[values_col], errors='coerce').fillna(0)
    df = df[df[values_col] > 0] 
    if df.empty:
        logger.warning(f"Donut Chart '{title}': No positive values to plot after filtering.")
        return _create_empty_figure(title, final_height, "No positive data to display.")
    
    # Sort DataFrame by values_col descending so that legend and pulls are consistent.
    # go.Pie trace itself has a 'sort' attribute which defaults to True and sorts by values.
    # We pre-sort here for consistent color mapping if needed.
    df.sort_values(by=values_col, ascending=False, inplace=True)

    plot_colors_donut_list = []
    # Try to get colors from color_discrete_map first
    if color_discrete_map: 
        plot_colors_donut_list = [color_discrete_map.get(lbl, _get_theme_color(i, color_type='disease')) for i, lbl in enumerate(df[labels_col])]
    # Then try app_config.DISEASE_COLORS if available
    elif hasattr(app_config,"DISEASE_COLORS") and app_config.DISEASE_COLORS:
         plot_colors_donut_list = [app_config.DISEASE_COLORS.get(lbl, _get_theme_color(i)) for i,lbl in enumerate(df[labels_col])]
    # Fallback to general theme colors
    else: 
        plot_colors_donut_list = [_get_theme_color(i) for i in range(len(df[labels_col]))]
    
    fig = go.Figure(data=[go.Pie(
        labels=df[labels_col], 
        values=df[values_col], 
        hole=0.52, 
        pull=[pull_segments if i < 3 else 0 for i in range(len(df))], # Pull top 3 segments
        textinfo='label+percent', 
        hoverinfo='label+value+percent', 
        insidetextorientation='radial', 
        marker=dict(colors=plot_colors_donut_list, line=dict(color='#ffffff', width=2.2)),
        sort=True # Default is True, explicitly stating it for clarity. Sorts slices by values.
    )])
    
    annotations_list_donut = []
    if center_text:
        annotations_list_donut.append(
            dict(text=center_text, x=0.5, y=0.5, font_size=18, showarrow=False, 
                 font_color=_get_theme_color(0, fallback_color="#343a40") # Get a default dark text color
            )
        )

    # --- CORRECTED LEGEND ---
    fig.update_layout(
        title_text=title, 
        height=final_height, 
        showlegend=True, 
        legend=dict(
            orientation="v", 
            yanchor="middle", y=0.5, 
            xanchor="right", x=1.18, 
            traceorder="normal" # Changed from "sorted". 'normal' or 'reversed' are common for legend item order.
                                # The pie slices themselves are sorted by the `sort=True` in `go.Pie`.
        ), 
        annotations=annotations_list_donut if annotations_list_donut else None
    )
    return fig

# ... (plot_heatmap and other functions remain the same) ...

def plot_heatmap(matrix_df_input, title: str, height: int = None, colorscale: str = "RdBu_r", zmid: float = 0, text_auto: bool = True, text_format: str = ".2f", show_colorbar: bool = True):
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT + 100
    if not isinstance(matrix_df_input, pd.DataFrame) or matrix_df_input.empty: return _create_empty_figure(title, final_height, "Invalid data for Heatmap.")
    df_numeric = matrix_df_input.copy()
    for col in df_numeric.columns: df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    if df_numeric.isnull().all().all() and not matrix_df_input.empty: return _create_empty_figure(title, final_height, "All data non-numeric.")
    df_plot_heatmap = df_numeric.fillna(0) 
    z_vals = df_plot_heatmap.values; text_vals_heatmap = np.around(z_vals, decimals=int(text_format[-2]) if text_format.endswith('f') and text_format[-2].isdigit() else 2) if not df_plot_heatmap.empty else None
    zmid_final = zmid if pd.Series(z_vals.flatten()).min() < 0 and pd.Series(z_vals.flatten()).max() > 0 else None 
    fig = go.Figure(data=go.Heatmap(z=z_vals, x=df_plot_heatmap.columns.tolist(), y=df_plot_heatmap.index.tolist(), colorscale=colorscale, zmid=zmid_final, text=text_vals_heatmap if text_auto else None, texttemplate=f"%{{text:{text_format}}}" if text_auto and text_vals_heatmap is not None else "", hoverongaps=False, xgap=1.8, ygap=1.8, colorbar=dict(thickness=20, len=0.9, tickfont_size=10, title_side="right", outlinewidth=1, outlinecolor=_get_theme_color(0,"#ced4da")) if show_colorbar else None ))
    max_label_len = 0
    if not df_plot_heatmap.columns.empty: max_label_len = max(len(str(c)) for c in df_plot_heatmap.columns if c is not None)
    rotate_x = -40 if len(df_plot_heatmap.columns) > 8 or max_label_len > 10 else 0
    fig.update_layout(title_text=title, height=final_height, xaxis_showgrid=False, yaxis_showgrid=False, xaxis_tickangle=rotate_x, yaxis_autorange='reversed', plot_bgcolor='rgba(0,0,0,0)')
    return fig
