# test/utils/ui_visualization_helpers.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging
import plotly.io as pio
from config import app_config # Assuming app_config.py is in a 'config' directory accessible by Python
import html
import geopandas as gpd # Optional, only if gdf type hints are strictly enforced by linter
import os
from typing import Optional, List, Dict, Any # For type hints

# Logging Setup
# BasicConfig should ideally be in the main app script (app_home.py)
# For a utility module, just get the logger.
# If run standalone, this logger might not output unless configured elsewhere.
logger = logging.getLogger(__name__)

MAPBOX_TOKEN_SET = False
try:
    MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
    if MAPBOX_ACCESS_TOKEN and MAPBOX_ACCESS_TOKEN.strip() and "YOUR_MAPBOX_ACCESS_TOKEN" not in MAPBOX_ACCESS_TOKEN and len(MAPBOX_ACCESS_TOKEN) > 20:
        px.set_mapbox_access_token(MAPBOX_ACCESS_TOKEN)
        MAPBOX_TOKEN_SET = True
        logger.info("Mapbox access token found and set for Plotly Express.")
    else:
        # Warning if token is placeholder or missing but not an error
        log_msg = "MAPBOX_ACCESS_TOKEN environment variable not found or is an invalid placeholder."
        if not MAPBOX_ACCESS_TOKEN: log_msg = "MAPBOX_ACCESS_TOKEN environment variable not found."
        logger.warning(f"{log_msg} Map styles requiring a token may not work; defaulting to open styles if needed.")
except Exception as e_token:
    logger.error(f"Error occurred while trying to set Mapbox token: {e_token}")

# test/utils/ui_visualization_helpers.py
# ... (other imports and MAPBOX_TOKEN_SET logic) ...

def _get_theme_color(index: Any = 0, fallback_color: str = "#007bff", color_type: str = "general") -> str:
    # ... (This function is likely okay, but let's ensure its fallback behavior)
    try:
        if color_type == "disease" and hasattr(app_config, 'DISEASE_COLORS') and app_config.DISEASE_COLORS:
            if isinstance(index, str) and index in app_config.DISEASE_COLORS: return app_config.DISEASE_COLORS[index]
        if color_type == "risk_status" and hasattr(app_config, 'RISK_STATUS_COLORS') and app_config.RISK_STATUS_COLORS:
            if isinstance(index, str) and index in app_config.RISK_STATUS_COLORS: return app_config.RISK_STATUS_COLORS[index]
        
        # Use pio.templates.default which should be correctly set by set_custom_plotly_theme
        # Safely access the colorway from the default template
        default_template_name = pio.templates.default
        colorway = px.colors.qualitative.Plotly # Default fallback colorway
        if default_template_name and default_template_name in pio.templates:
            active_template_layout = pio.templates[default_template_name].layout
            if hasattr(active_template_layout, 'colorway') and active_template_layout.colorway:
                colorway = active_template_layout.colorway
        
        if colorway: # Ensure colorway is not None or empty
            num_idx = index if isinstance(index, int) else abs(hash(str(index))) % len(colorway) # abs() for hash
            return colorway[num_idx % len(colorway)]
    except Exception as e_color_get:
        logger.warning(f"Could not retrieve theme color for index/key '{index}', type '{color_type}': {e_color_get}. Using fallback: {fallback_color}")
    return fallback_color


def set_custom_plotly_theme():
    theme_font_family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji"'
    theme_primary_text_color = "#343a40"; theme_grid_color = "#e9ecef"; theme_border_color = "#ced4da"
    theme_paper_bg_color = "#f8f9fa" ; theme_plot_bg_color = "#FFFFFF"
    
    # Define the default colorway as a list
    default_colorway_list = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#17a2b8', '#6f42c1', '#fd7e14', '#20c997', '#6610f2', '#e83e8c']

    # Try to use values from app_config.DISEASE_COLORS for a more thematic colorway,
    # but ensure it's a list and has fallbacks if not enough colors.
    configured_colorway = []
    if hasattr(app_config, 'DISEASE_COLORS') and isinstance(app_config.DISEASE_COLORS, dict):
        # Take values, but ensure enough colors, fallback to default if too few
        configured_colorway = list(app_config.DISEASE_COLORS.values())
        if len(configured_colorway) < 5: # Arbitrary threshold for "enough" colors
            configured_colorway.extend(default_colorway_list[len(configured_colorway):]) # Pad with defaults
        if not configured_colorway: # If DISEASE_COLORS was empty
            configured_colorway = default_colorway_list
    else:
        configured_colorway = default_colorway_list

    layout_settings = {
        'font': dict(family=theme_font_family, size=12, color=theme_primary_text_color),
        'paper_bgcolor': theme_paper_bg_color,
        'plot_bgcolor': theme_plot_bg_color,
        'colorway': configured_colorway, # CORRECTED: Ensure this is always a list
        'xaxis': dict(gridcolor=theme_grid_color, linecolor=theme_border_color, zerolinecolor=theme_grid_color, zerolinewidth=1, title_font_size=13, tickfont_size=11, automargin=True, title_standoff=10),
        'yaxis': dict(gridcolor=theme_grid_color, linecolor=theme_border_color, zerolinecolor=theme_grid_color, zerolinewidth=1, title_font_size=13, tickfont_size=11, automargin=True, title_standoff=10),
        'title': dict(
            font=dict(family=theme_font_family, size=18, color="#1A2557"), # Removed 'weight'
            x=0.05, xanchor='left', y=0.95, yanchor='top', pad=dict(t=30, b=15, l=5) # Adjusted padding
        ),
        'legend': dict(bgcolor='rgba(255,255,255,0.85)', bordercolor=theme_border_color, borderwidth=1, orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font_size=11),
        'margin': dict(l=70, r=30, t=100, b=70) # Increased top margin for title
    }
    
    mapbox_style_to_use = app_config.MAPBOX_STYLE
    if not MAPBOX_TOKEN_SET and mapbox_style_to_use not in ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"]:
        mapbox_style_to_use = "open-street-map"
    layout_settings['mapbox'] = dict(style=mapbox_style_to_use, center=dict(lat=app_config.MAP_DEFAULT_CENTER_LAT, lon=app_config.MAP_DEFAULT_CENTER_LON), zoom=app_config.MAP_DEFAULT_ZOOM)
    
    custom_template_obj = go.layout.Template(layout=go.Layout(**layout_settings))
    pio.templates["custom_health_theme"] = custom_template_obj
    pio.templates.default = "plotly+custom_health_theme" 
    logger.info("Custom Plotly theme 'custom_health_theme' set as default.")

# Ensure it's called on import - this applies the theme globally for subsequent plot generations.
set_custom_plotly_theme()

# ... (rest of ui_visualization_helpers.py as previously corrected for plot scales, etc.) ... # Initialize theme upon module import

def render_kpi_card(title: str, value: str, icon: str, status: str = "neutral", delta: Optional[str] = None, delta_type: str = "neutral", help_text: Optional[str] = None, icon_is_html: bool = False):
    valid_statuses = {"high", "moderate", "low", "neutral", "good", "bad"}; valid_delta_types = {"positive", "negative", "neutral"}
    status_parts = status.lower().split(); base_status_class = next((s for s in status_parts if s in valid_statuses and s not in ["good", "bad"]), "neutral"); semantic_class = next((f"status-{s}" for s in status_parts if s in ["good", "bad"]), "")
    final_delta_type = delta_type.lower() if delta_type and delta_type.lower() in valid_delta_types else "neutral"; delta_str = str(delta) if delta is not None and str(delta).strip() else ""; delta_html = f'<p class="kpi-delta {final_delta_type}">{html.escape(delta_str)}</p>' if delta_str else ''
    tooltip_html = f'title="{html.escape(str(help_text))}"' if help_text and str(help_text).strip() else ''; icon_display = str(icon) if icon_is_html else html.escape(str(icon) if icon is not None else "●")
    combined_status_class = f"{base_status_class} {semantic_class}".strip()
    html_content = f'<div class="kpi-card {combined_status_class}" {tooltip_html}><div class="kpi-card-header"><div class="kpi-icon">{icon_display}</div><h3 class="kpi-title">{html.escape(str(title))}</h3></div><div class="kpi-body"><p class="kpi-value">{html.escape(str(value))}</p>{delta_html}</div></div>'.replace("\n", "")
    st.markdown(html_content, unsafe_allow_html=True)

def render_traffic_light(message: str, status: str, details: str = ""):
    valid_statuses_tl = {"high", "moderate", "low", "neutral"}; dot_status_class = "status-" + (status.lower() if status and status.lower() in map(str.lower, valid_statuses_tl) else "neutral")
    details_html = f'<span class="traffic-light-details">{html.escape(str(details))}</span>' if details and str(details).strip() else ""
    html_content = f'<div class="traffic-light-indicator"><span class="traffic-light-dot {dot_status_class}"></span><span class="traffic-light-message">{html.escape(str(message))}</span>{details_html}</div>'.replace("\n", "")
    st.markdown(html_content, unsafe_allow_html=True)

def _create_empty_figure(title: str, height: Optional[int], message: str = "No data available to display.") -> go.Figure:
    fig = go.Figure(); final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT
    fig.update_layout(title_text=f"{title} ({message})", height=final_height, xaxis={'visible': False}, yaxis={'visible': False}, annotations=[dict(text=message, xref="paper", yref="paper", showarrow=False, font=dict(size=14))]); return fig

def plot_layered_choropleth_map(gdf: gpd.GeoDataFrame, value_col: str, title: str, id_col: str = 'zone_id', featureidkey_prefix: str = 'properties', color_continuous_scale: str = "Blues_r", hover_cols: Optional[List[str]] = None, facility_gdf: Optional[gpd.GeoDataFrame] = None, facility_size_col: Optional[str] = None, facility_hover_name: Optional[str] = None, facility_color: Optional[str] = None, height: Optional[int] = None, center_lat: Optional[float] = None, center_lon: Optional[float] = None, zoom_level: Optional[int] = None, mapbox_style: Optional[str] = None ) -> go.Figure:
    final_height = height if height is not None else app_config.MAP_PLOT_HEIGHT
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty: return _create_empty_figure(title, final_height, "Map data unavailable.")
    active_geom_col = gdf.geometry.name if hasattr(gdf,'geometry') and hasattr(gdf.geometry,'name') else 'geometry'
    if active_geom_col not in gdf.columns or not hasattr(gdf[active_geom_col], 'is_empty') or gdf[active_geom_col].is_empty.all() or not hasattr(gdf[active_geom_col], 'is_valid') or not gdf[active_geom_col].is_valid.any(): return _create_empty_figure(title, final_height, "Invalid geometries.")
    gdf_plot = gdf.copy();
    if id_col not in gdf_plot.columns or value_col not in gdf_plot.columns: return _create_empty_figure(title, final_height, f"Missing ID/Value col.")
    if not pd.api.types.is_numeric_dtype(gdf_plot[value_col]): gdf_plot[value_col] = pd.to_numeric(gdf_plot[value_col], errors='coerce')
    gdf_plot[value_col].fillna(0, inplace=True); gdf_plot[id_col] = gdf_plot[id_col].astype(str)
    gdf_for_geojson = gdf_plot[gdf_plot.geometry.is_valid & ~gdf_plot.geometry.is_empty].copy()
    if gdf_for_geojson.empty: return _create_empty_figure(title, final_height, "No valid geoms for map.")
    effective_mapbox_style = mapbox_style or pio.templates.default.layout.get('mapbox',{}).get('style', app_config.MAPBOX_STYLE)
    if not MAPBOX_TOKEN_SET and effective_mapbox_style not in ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"]: logger.warning(f"Map ('{title}'): Style '{effective_mapbox_style}' may require token. Defaulting to 'open-street-map'."); effective_mapbox_style = "open-street-map"
    hover_name_col = "name" if "name" in gdf_plot.columns else id_col
    default_hover = [hover_name_col, value_col, 'population']; final_hover_list = hover_cols if hover_cols is not None else default_hover
    hover_data_plot = {col: True for col in final_hover_list if col in gdf_plot.columns and col != hover_name_col and gdf_plot[col].notna().any()}
    labels_plot = {col: str(col).replace('_', ' ').title() for col in [value_col] + list(hover_data_plot.keys())}
    try: fig = px.choropleth_mapbox(data_frame=gdf_for_geojson, geojson=gdf_for_geojson.geometry.__geo_interface__, locations=id_col, featureidkey=f"{featureidkey_prefix}.{id_col}" if featureidkey_prefix else id_col, color=value_col, color_continuous_scale=color_continuous_scale, opacity=0.75, hover_name=hover_name_col, hover_data=hover_data_plot, labels=labels_plot, mapbox_style=effective_mapbox_style, center={"lat": center_lat or app_config.MAP_DEFAULT_CENTER_LAT, "lon": center_lon or app_config.MAP_DEFAULT_CENTER_LON}, zoom=zoom_level if zoom_level is not None else app_config.MAP_DEFAULT_ZOOM )
    except Exception as e: logger.error(f"Map Error ({title}): {e}", exc_info=True); return _create_empty_figure(title, final_height, f"Map error: {str(e)[:100]}")
    if facility_gdf is not None and not facility_gdf.empty and 'geometry' in facility_gdf.columns:
        facility_plot_gdf = facility_gdf[facility_gdf.geometry.geom_type == 'Point'].copy()
        if not facility_plot_gdf.empty:
            facility_hover_text = facility_plot_gdf.get(facility_hover_name, pd.Series(["Facility"] * len(facility_plot_gdf), index=facility_plot_gdf.index)) if facility_hover_name else pd.Series(["Facility"] * len(facility_plot_gdf), index=facility_plot_gdf.index)
            f_size = pd.Series([10] * len(facility_plot_gdf), index=facility_plot_gdf.index)
            if facility_size_col and facility_size_col in facility_plot_gdf.columns and pd.api.types.is_numeric_dtype(facility_plot_gdf[facility_size_col]):
                sizes = pd.to_numeric(facility_plot_gdf[facility_size_col], errors='coerce').fillna(0); mn_s, mx_s, mn_v, mx_v = 6,20,sizes.min(),sizes.max()
                if mx_v > mn_v and mx_v > 0: f_size = mn_s + ((sizes-mn_v)*(mx_s-mn_s)/(mx_v-mn_v))
                elif sizes.notna().any() and sizes.max() > 0: f_size = pd.Series([(mn_s+mx_s)/2]*len(facility_plot_gdf),index=facility_plot_gdf.index)
            f_color = facility_color if facility_color else _get_theme_color(5, fallback_color='#6F42C1') 
            fig.add_trace(go.Scattermapbox(lon=facility_plot_gdf.geometry.x, lat=facility_plot_gdf.geometry.y, mode='markers', marker=go.scattermapbox.Marker(size=f_size, sizemin=5, color=f_color, opacity=0.9, allowoverlap=True), text=facility_hover_text, hoverinfo='text', name='Facilities Layer'))
    fig.update_layout(title_text=title, height=final_height, margin={"r":10,"t":60,"l":10,"b":10}, legend=dict(yanchor="top",y=0.98,xanchor="left",x=0.01,bgcolor='rgba(255,255,255,0.7)'))
    return fig

def plot_annotated_line_chart(data_series: pd.Series, title: str, y_axis_title: str = "Value", color: Optional[str] = None, target_line: Optional[float] = None, target_label: Optional[str] = None, show_ci: bool = False, lower_bound_series: Optional[pd.Series] = None, upper_bound_series: Optional[pd.Series] = None, height: Optional[int] = None, show_anomalies: bool = True, date_format: str = "%b %Y", y_is_count: bool = False) -> go.Figure:
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT
    if not isinstance(data_series, pd.Series) or data_series.empty: return _create_empty_figure(title, final_height)
    data_series_numeric = pd.to_numeric(data_series, errors='coerce')
    if data_series_numeric.isnull().all(): return _create_empty_figure(title, final_height, "All data non-numeric or became NaN.")
    
    fig = go.Figure(); line_color_val = color if color else _get_theme_color(0)
    y_hover_format = 'd' if y_is_count else ',.2f'; hovertemplate_str = f'<b>Date</b>: %{{x|{date_format}}}<br><b>{y_axis_title}</b>: %{{customdata:{y_hover_format}}}<extra></extra>'
    fig.add_trace(go.Scatter(x=data_series_numeric.index, y=data_series_numeric.values, mode="lines+markers", name=y_axis_title, line=dict(color=line_color_val, width=2.5), marker=dict(size=6), customdata=data_series_numeric.values, hovertemplate=hovertemplate_str))
    if show_ci and lower_bound_series is not None and upper_bound_series is not None and not lower_bound_series.empty and not upper_bound_series.empty:
        common_idx_ci = data_series_numeric.index.intersection(lower_bound_series.index).intersection(upper_bound_series.index)
        if not common_idx_ci.empty: 
            ls = pd.to_numeric(lower_bound_series.reindex(common_idx_ci),errors='coerce'); us = pd.to_numeric(upper_bound_series.reindex(common_idx_ci),errors='coerce'); 
            valid_ci_mask=ls.notna() & us.notna() & (us >= ls)
            if valid_ci_mask.any(): 
                x_ci,y_up,y_low = common_idx_ci[valid_ci_mask],us[valid_ci_mask],ls[valid_ci_mask]; 
                fill_rgba = f"rgba({','.join(str(int(c,16)) for c in (line_color_val[1:3],line_color_val[3:5],line_color_val[5:7]))},0.15)" if line_color_val.startswith('#') and len(line_color_val)==7 else "rgba(0,123,255,0.15)"
                fig.add_trace(go.Scatter(x=list(x_ci)+list(x_ci[::-1]), y=list(y_up.values)+list(y_low.values[::-1]), fill="toself", fillcolor=fill_rgba, line=dict(width=0), name="Confidence Interval", hoverinfo='skip'))
    if target_line is not None: fig.add_hline(y=target_line, line_dash="dot", line_color="#e74c3c", line_width=1.5, annotation_text=target_label if target_label else f"Target: {target_line:,.2f}", annotation_position="top right", annotation_font_size=10, annotation_font_color="#c0392b")
    if show_anomalies and len(data_series_numeric.dropna()) > 10 and data_series_numeric.nunique() > 1:
        q1=data_series_numeric.quantile(0.25); q3=data_series_numeric.quantile(0.75); iqr=q3-q1
        if pd.notna(iqr) and iqr > 1e-9: 
            upper_b=q3+1.5*iqr; lower_b=q1-1.5*iqr; anomalies=data_series_numeric[(data_series_numeric < lower_b)|(data_series_numeric > upper_b)]
            if not anomalies.empty: fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies.values, mode='markers', marker=dict(color=_get_theme_color(6,fallback_color='#fd7e14'), size=9, symbol='x-thin-open', line=dict(width=2.5)), name='Potential Anomaly', customdata=anomalies.values, hovertemplate=(f'<b>Anomaly Date</b>: %{{x|{date_format}}}<br><b>Value</b>: %{{customdata:{y_hover_format}}}<extra></extra>')))
    
    final_xaxis_title = data_series_numeric.index.name if data_series_numeric.index.name and str(data_series_numeric.index.name).strip() else "Date"
    yaxis_layout = dict(title_text=y_axis_title, rangemode='tozero' if y_is_count and data_series_numeric.min() >= 0 else 'normal')
    if y_is_count:
        yaxis_layout['tickformat'] = 'd'
        max_v = data_series_numeric.max(); min_v_check = data_series_numeric.min()
        if pd.notna(max_v) and max_v > 0:
            if max_v <= 1 and pd.notna(min_v_check) and min_v_check >=0 : yaxis_layout['dtick'] = 0.5 
            elif max_v <= 10: yaxis_layout['dtick'] = 1
            elif max_v <= 25 and (max_v - (min_v_check if pd.notna(min_v_check) else 0)) <= 25 : yaxis_layout['dtick'] = 2 if max_v > 5 else 1
            elif max_v <= 50: yaxis_layout['dtick'] = 5
            else: yaxis_layout['nticks'] = min(10, int(max_v / (max_v/10 if max_v > 50 else 5)) +1 if max_v > 0 else 5 )
    fig.update_layout(title_text=title, xaxis_title=final_xaxis_title, yaxis=yaxis_layout, height=final_height, hovermode="x unified", legend=dict(traceorder='normal'))
    return fig

def plot_bar_chart(df_input: pd.DataFrame, x_col: str, y_col: str, title: str, color_col: Optional[str] = None, barmode: str = 'group', orientation: str = 'v', y_axis_title: Optional[str] = None, x_axis_title: Optional[str] = None, height: Optional[int] = None, text_auto: bool = True, sort_values_by: Optional[str] = None, ascending: bool = True, text_format: Optional[str] = None, y_is_count: bool = False, color_discrete_map: Optional[Dict] = None) -> go.Figure:
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT
    if df_input is None or df_input.empty or x_col not in df_input.columns or y_col not in df_input.columns: return _create_empty_figure(title, final_height)
    df = df_input.copy(); df[x_col] = df[x_col].astype(str)
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    if y_is_count: df[y_col] = df[y_col].round().astype('Int64') # Use Pandas nullable integer
    df.dropna(subset=[x_col, y_col], inplace=True)
    if df.empty: return _create_empty_figure(title, final_height, f"No valid data for x='{x_col}', y='{y_col}'.")
    
    final_text_format_str = text_format if text_format is not None else ('d' if y_is_count else ',.1f') # default based on count
    final_y_title = y_axis_title if y_axis_title else y_col.replace('_', ' ').title(); final_x_title = x_axis_title if x_axis_title else x_col.replace('_', ' ').title()
    
    if sort_values_by and sort_values_by in df.columns:
        try: 
            # Use pd.to_numeric for key in sort_values if the sort_by column is numeric
            sort_key_func = lambda cs: pd.to_numeric(cs, errors='ignore') if pd.api.types.is_numeric_dtype(df[sort_values_by]) else cs.astype(str)
            df.sort_values(by=sort_values_by, ascending=ascending, inplace=True, na_position='last', key=sort_key_func)
        except Exception as e: logger.warning(f"Bar chart sort failure by '{sort_values_by}': {e}")
    
    legend_title_text = color_col.replace('_',' ').title() if color_col and color_col in df.columns else None
    final_color_map_resolved = color_discrete_map
    if color_col and color_col in df.columns and color_discrete_map is None:
        if any(k in color_col.lower() for k in ['condition','disease','test_type','status','gender']):
             unique_vals = df[color_col].dropna().unique()
             final_color_map_resolved = { str(v): app_config.DISEASE_COLORS.get(str(v), _get_theme_color(abs(hash(str(v))) % 10)) for v in unique_vals if hasattr(app_config, 'DISEASE_COLORS')}
             if not final_color_map_resolved: final_color_map_resolved = None

    fig = px.bar(df, x=x_col, y=y_col, title=title, color=color_col, barmode=barmode, orientation=orientation, height=final_height, labels={y_col:final_y_title, x_col:final_x_title, color_col:legend_title_text if legend_title_text else ""}, text_auto=text_auto, color_discrete_map=final_color_map_resolved)
    
    # Define hover format based on whether the axis represents counts
    # If y_is_count, the Y-axis (if vertical) or X-axis (if horizontal) shows counts
    hover_val_fmt = 'd' if y_is_count else final_text_format_str
    base_hover_str = f'<b>{final_x_title}</b>: %{{x}}<br><b>{final_y_title}</b>: %{{y:{hover_val_fmt}}}' if orientation=='v' else f'<b>{final_y_title}</b>: %{{y}}<br><b>{final_x_title}</b>: %{{x:{hover_val_fmt}}}'
    hover_template_str = base_hover_str + (f'<br><b>{legend_title_text}</b>: %{{customdata[0]}}<extra></extra>' if color_col and color_col in df.columns and df[color_col].notna().any() else '<extra></extra>') # df[[color_col]] might error if color_col has bad chars
    
    plotly_text_specifier = final_text_format_str.split(':')[-1].lstrip('.,%') if any(c in final_text_format_str for c in [':','.',',','%']) else final_text_format_str
    texttemplate_str = (f'%{{y:{plotly_text_specifier}}}' if text_auto and orientation=='v' else (f'%{{x:{plotly_text_specifier}}}' if text_auto and orientation=='h' else None))
    
    customdata_for_hover = df[[color_col]] if color_col and color_col in df.columns else None # Ensure column exists for customdata
    fig.update_traces(marker_line_width=0.7, marker_line_color='rgba(30,30,30,0.6)', textfont_size=10, textangle=0, textposition='auto' if orientation == 'v' else 'outside', cliponaxis=False, texttemplate=texttemplate_str, hovertemplate=hover_template_str, customdata=customdata_for_hover)
    
    # Initialize axis configurations
    yaxis_config_bar = {'title_text': final_y_title}
    xaxis_config_bar = {'title_text': final_x_title}
    
    current_value_axis_config = yaxis_config_bar if orientation == 'v' else xaxis_config_bar
    current_category_axis_config = xaxis_config_bar if orientation == 'v' else yaxis_config_bar
    main_val_data = df[y_col]

    if y_is_count: # This applies to the axis showing bar lengths (y_col)
        current_value_axis_config['tickformat'] = 'd'; current_value_axis_config['rangemode'] = 'tozero'
        max_v = main_val_data.max(); min_v = main_val_data.min()
        if pd.notna(max_v) and max_v > 0:
            if max_v <=1 and pd.notna(min_v) and min_v >=0: current_value_axis_config['dtick'] = 0.5
            elif max_v <= 10: current_value_axis_config['dtick'] = 1
            elif max_v <= 50: current_value_axis_config['dtick'] = 5 if (max_v / 5) >= 3 else (2 if max_v > 5 else 1) # Prefer intervals of 1, 2, or 5
            else: current_value_axis_config['nticks'] = min(10, int(max_v/10)+1 if max_v/10 > 1 else (int(max_v/5)+1 if max_v/5 > 1 else 5)) # Ensure reasonable number of ticks
            
    # Apply sorted category order if DataFrame was sorted by categories
    category_data_col_name = x_col if orientation == 'v' else y_col
    if sort_values_by == category_data_col_name : 
        current_category_axis_config['categoryorder'] = 'array'; current_category_axis_config['categoryarray'] = df[category_data_col_name].tolist()
    elif orientation == 'h' and (not sort_values_by or sort_values_by == y_col): # Default sort for horizontal if categories are on y-axis
        current_category_axis_config['categoryorder']='total ascending' if ascending else 'total descending'
        
    fig.update_layout(yaxis=yaxis_config_bar, xaxis=xaxis_config_bar, uniformtext_minsize=8, uniformtext_mode='hide', legend_title_text=legend_title_text)
    return fig

def plot_donut_chart(data_df_input: pd.DataFrame, labels_col: str, values_col: str, title: str, height: Optional[int] = None, color_discrete_map: Optional[Dict] = None, pull_segments: float = 0.03, center_text: Optional[str] = None, values_are_counts: bool = True) -> go.Figure:
    final_height = height if height is not None else app_config.COMPACT_PLOT_HEIGHT + 40
    if data_df_input is None or data_df_input.empty or labels_col not in data_df_input.columns or values_col not in data_df_input.columns: return _create_empty_figure(title, final_height)
    df = data_df_input.copy(); df[values_col] = pd.to_numeric(df[values_col], errors='coerce').fillna(0)
    if values_are_counts: df[values_col] = df[values_col].round().astype('Int64')
    df = df[df[values_col] > 0];
    if df.empty: return _create_empty_figure(title, final_height, "No positive data to display.")
    df.sort_values(by=values_col, ascending=False, inplace=True); df[labels_col] = df[labels_col].astype(str)
    plot_colors_final = None
    if color_discrete_map: plot_colors_final = [color_discrete_map.get(str(lbl), _get_theme_color(i)) for i, lbl in enumerate(df[labels_col])]
    elif hasattr(app_config,"DISEASE_COLORS") and any(str(lbl) in app_config.DISEASE_COLORS for lbl in df[labels_col]): plot_colors_final = [_get_theme_color(str(lbl), color_type="disease", fallback_color=_get_theme_color(i)) for i,lbl in enumerate(df[labels_col])]
    hover_val_fmt = 'd' if values_are_counts else '.2f'; hovertemplate_donut = f'<b>%{{label}}</b><br>Value: %{{value:{hover_val_fmt}}}<br>Percent: %{{percent}}<extra></extra>'
    fig = go.Figure(data=[go.Pie(labels=df[labels_col], values=df[values_col], hole=0.50, pull=[pull_segments if i < 3 else 0 for i in range(len(df))], textinfo='label+percent', hoverinfo='label+value+percent', hovertemplate=hovertemplate_donut, insidetextorientation='radial', marker=dict(colors=plot_colors_final, line=dict(color='#ffffff', width=2)), sort=False)]) 
    annotations_list_donut = [dict(text=str(center_text), x=0.5, y=0.5, font_size=16, showarrow=False)] if center_text else None
    fig.update_layout(title_text=title, height=final_height, showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.15, traceorder="normal"), annotations=annotations_list_donut, margin=dict(l=20, r=100, t=60, b=20))
    return fig

def plot_heatmap(matrix_df_input: pd.DataFrame, title: str, height: Optional[int] = None, colorscale: str = "RdBu_r", zmid: Optional[float] = 0, text_auto: bool = True, text_format: str = ".2f", show_colorbar: bool = True) -> go.Figure:
    final_height = height if height is not None else app_config.DEFAULT_PLOT_HEIGHT + 80
    if not isinstance(matrix_df_input, pd.DataFrame) or matrix_df_input.empty: return _create_empty_figure(title, final_height, "Invalid data for Heatmap.")
    df_numeric = matrix_df_input.copy().apply(pd.to_numeric, errors='coerce')
    if df_numeric.isnull().all().all(): return _create_empty_figure(title, final_height, "All data non-numeric or empty after coercion.")
    df_plot_heatmap = df_numeric 
    z_vals = df_plot_heatmap.values; text_vals_heatmap = None
    if text_auto and not df_plot_heatmap.empty:
        decimals_from_format = 2 # Default
        try:
            if text_format.endswith('f') and text_format[-2].isdigit(): decimals_from_format = int(text_format[-2])
            elif text_format.lower() == 'd': decimals_from_format = 0
        except: pass # Keep default decimals_from_format if parsing fails
        text_vals_heatmap = np.around(z_vals, decimals=decimals_from_format)

    z_flat_no_nan = z_vals[~np.isnan(z_vals)]; zmid_final = zmid
    if len(z_flat_no_nan) > 0: # Only set zmid if there's actual data
        if not (np.any(z_flat_no_nan < 0) and np.any(z_flat_no_nan > 0)): zmid_final = None # Disable zmid if all values are same sign
    else: zmid_final = None # Disable zmid if all NaNs or empty
        
    fig = go.Figure(data=go.Heatmap(
        z=z_vals, x=df_plot_heatmap.columns.astype(str).tolist(), 
        y=df_plot_heatmap.index.astype(str).tolist(), 
        colorscale=colorscale, zmid=zmid_final, 
        text=text_vals_heatmap if text_auto else None, 
        texttemplate=f"%{{text:{text_format}}}" if text_auto and text_vals_heatmap is not None else "", 
        hoverongaps=False, xgap=1.5, ygap=1.5, 
        colorbar=dict(thickness=18, len=0.85, tickfont_size=10, title_side="right", outlinewidth=0.8, outlinecolor=_get_theme_color(0,"#ced4da")) if show_colorbar else None 
    ))
    max_x_label_len = max((len(str(c)) for c in df_plot_heatmap.columns if c is not None), default=0)
    rotate_x = -40 if len(df_plot_heatmap.columns) > 7 or max_x_label_len > 8 else 0
    fig.update_layout(title_text=title, height=final_height, xaxis_showgrid=False, yaxis_showgrid=False, xaxis_tickangle=rotate_x, yaxis_autorange='reversed', plot_bgcolor='rgba(0,0,0,0)')
    return fig
