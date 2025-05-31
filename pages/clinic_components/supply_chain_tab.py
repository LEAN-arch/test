# test/pages/clinic_components/supply_chain_tab.py
import streamlit as st
import pandas as pd
import logging
from config import app_config
from utils.core_data_processing import get_supply_forecast_data
from utils.ai_analytics_engine import SupplyForecastingModel
from utils.ui_visualization_helpers import plot_annotated_line_chart

logger = logging.getLogger(__name__)

def render_supply_chain(health_df_clinic_main, filtered_health_df_clinic): # Pass main for full history for forecast
    st.subheader("💊 Medical Supply Levels & Consumption Forecast")
    
    use_ai_forecast = st.checkbox("Use Advanced AI Supply Forecast (Beta)", value=False, key="clinic_ai_supply_forecast_toggle_v2")

    # Forecasting model needs the full history available in health_df_clinic_main for rates
    if health_df_clinic_main is not None and not health_df_clinic_main.empty and \
       all(c in health_df_clinic_main.columns for c in ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']):
        
        if use_ai_forecast:
            logger.info("Clinic Dashboard: Using AI Supply Forecasting Model for Supply Tab.")
            supply_model_ai = SupplyForecastingModel()
            supply_forecast_df = supply_model_ai.forecast_supply_levels_advanced(health_df_clinic_main, forecast_days_out=30)
        else:
            logger.info("Clinic Dashboard: Using Linear Supply Forecasting for Supply Tab.")
            supply_forecast_df = get_supply_forecast_data(health_df_clinic_main, forecast_days_out=30)

        if supply_forecast_df is not None and not supply_forecast_df.empty:
            key_drug_items_for_select = sorted(list(supply_forecast_df['item'].unique()))
            if not key_drug_items_for_select:
                 st.info("No forecast data available for any supply items based on historical data.")
            else:
                default_select_options = [item for item in key_drug_items_for_select if any(sub.lower() in str(item).lower() for sub in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY)]
                default_selection_idx = 0
                if default_select_options:
                    try: default_selection_idx = key_drug_items_for_select.index(default_select_options[0])
                    except ValueError: pass 

                selected_drug_for_forecast = st.selectbox(
                    "Select Item for Forecast Details:", key_drug_items_for_select,
                    index=default_selection_idx,
                    key="clinic_supply_item_forecast_selector_comp_v1", # Unique key
                    help="View the forecasted days of supply remaining for the selected item."
                )
                if selected_drug_for_forecast:
                    item_specific_forecast_df = supply_forecast_df[supply_forecast_df['item'] == selected_drug_for_forecast].copy()
                    if not item_specific_forecast_df.empty:
                        item_specific_forecast_df.sort_values('date', inplace=True)
                        current_info = item_specific_forecast_df.iloc[0]
                        forecast_title = (f"Forecast: {selected_drug_for_forecast}<br>"
                                          f"<sup_>Stock@Start: {current_info.get('current_stock',0):.0f} | Base Use: {current_info.get('consumption_rate',0):.1f}/d | Est. Stockout: {pd.to_datetime(current_info.get('estimated_stockout_date')).strftime('%d %b %Y') if pd.notna(current_info.get('estimated_stockout_date')) else 'N/A'}</sup>")
                        plot_series = item_specific_forecast_df.set_index('date')['forecast_days']
                        lc_series, uc_series = (item_specific_forecast_df.set_index('date').get('lower_ci'), item_specific_forecast_df.set_index('date').get('upper_ci')) if not use_ai_forecast else (None, None)
                        show_ci_flag = (lc_series is not None and not lc_series.empty and uc_series is not None and not uc_series.empty and not use_ai_forecast)
                        st.plotly_chart(plot_annotated_line_chart(data_series=plot_series, title=forecast_title, y_axis_title="Forecasted Days of Supply", target_line=app_config.CRITICAL_SUPPLY_DAYS, target_label=f"Critical ({app_config.CRITICAL_SUPPLY_DAYS} Days)", show_ci=show_ci_flag, lower_bound_series=lc_series, upper_bound_series=uc_series, height=app_config.DEFAULT_PLOT_HEIGHT + 60, show_anomalies=False), use_container_width=True)
                        if use_ai_forecast: st.caption("*Advanced forecast uses a simulated AI model.*")
                    else: st.info(f"No forecast data for {selected_drug_for_forecast}.")
        else:
            st.warning("Supply forecast data could not be generated from historical records.")
    elif not (health_df_clinic_main is not None and not health_df_clinic_main.empty): # If health_df_clinic_main was the issue
        st.warning("Supply forecasts cannot be generated as overall health records data is unavailable.")
    else: # Specific columns missing
        st.error("Health data is missing essential columns for supply forecasts (item, encounter_date, item_stock_agg_zone, consumption_rate_per_day).")
