# test/pages/clinic_components/supply_chain_tab.py
import streamlit as st
import pandas as pd
import logging # Added
from config import app_config
from utils.core_data_processing import get_supply_forecast_data
from utils.ai_analytics_engine import SupplyForecastingModel
from utils.ui_visualization_helpers import plot_annotated_line_chart

logger = logging.getLogger(__name__) # Added logger

def render_supply_chain(health_df_clinic_main, filtered_health_df_clinic): # Pass main for full history for forecast rates
    st.subheader("💊 Medical Supply Levels & Consumption Forecast")
    
    use_ai_forecast = st.checkbox("Use Advanced AI Supply Forecast (Beta)", value=False, key="clinic_ai_supply_forecast_toggle_comp_v1")

    # Forecasting model needs the full history available in health_df_clinic_main for rates
    # Ensure health_df_clinic_main is not None and not empty before proceeding
    if health_df_clinic_main is None or health_df_clinic_main.empty:
        st.warning("Supply forecasts cannot be generated as overall health records data (needed for historical consumption rates and stock levels) is unavailable.")
        return

    if not all(c in health_df_clinic_main.columns for c in ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']):
        st.error("Health data is missing essential columns for supply forecasts (item, encounter_date, item_stock_agg_zone, consumption_rate_per_day).")
        return
        
    if use_ai_forecast:
        logger.info("Clinic Supply Tab: Using AI Supply Forecasting Model.")
        supply_model_ai = SupplyForecastingModel()
        # AI model should ideally also use the full history for training/predicting rates
        supply_forecast_df = supply_model_ai.forecast_supply_levels_advanced(health_df_clinic_main, forecast_days_out=30)
    else:
        logger.info("Clinic Supply Tab: Using Linear Supply Forecasting from core_data_processing.")
        supply_forecast_df = get_supply_forecast_data(health_df_clinic_main, forecast_days_out=30)

    if supply_forecast_df is not None and not supply_forecast_df.empty:
        # Get all unique items from the forecast for selection
        all_forecasted_items = sorted(list(supply_forecast_df['item'].unique()))
        
        if not all_forecasted_items:
             st.info("No forecast data available for any supply items based on historical data and selected model.")
             return

        # For default selection, prefer items that are considered key drugs
        key_drug_items_in_forecast = [item for item in all_forecasted_items if any(sub.lower() in str(item).lower() for sub in app_config.KEY_DRUG_SUBSTRINGS_SUPPLY)]
        
        default_selection_index = 0
        if key_drug_items_in_forecast: # If any key drugs are in the forecast, pick the first one
            try:
                default_selection_index = all_forecasted_items.index(key_drug_items_in_forecast[0])
            except ValueError: # Should not happen if logic is correct, but safe
                pass 
        elif all_forecasted_items : # If no key drugs, but other items, pick first of all
            pass # default_selection_index is already 0
        else: # Should be caught by "not all_forecasted_items" above
            st.info("No items available for forecast.")
            return


        selected_drug_for_forecast = st.selectbox(
            "Select Item for Forecast Details:", all_forecasted_items,
            index=default_selection_index,
            key="clinic_supply_item_forecast_selector_comp_v2", # Incremented key
            help="View the forecasted days of supply remaining for the selected item."
        )
        if selected_drug_for_forecast:
            item_specific_forecast_df = supply_forecast_df[supply_forecast_df['item'] == selected_drug_for_forecast].copy()
            if not item_specific_forecast_df.empty:
                item_specific_forecast_df.sort_values('date', inplace=True)
                # Use the first row of the forecast data for 'current_stock' (stock at start of forecast)
                # and 'consumption_rate' (base historical rate used for forecast)
                current_info_for_title = item_specific_forecast_df.iloc[0] 
                
                est_stockout_date_str = "N/A"
                if pd.notna(current_info_for_title.get('estimated_stockout_date')):
                     est_stockout_date_str = pd.to_datetime(current_info_for_title['estimated_stockout_date']).strftime('%d %b %Y')

                forecast_plot_title = (
                    f"Forecast: {selected_drug_for_forecast}<br>"
                    f"<sup_>Stock at Forecast Start: {current_info_for_title.get('current_stock',0):.0f} | "
                    f"Base Daily Use (Hist.): {current_info_for_title.get('consumption_rate',0):.1f} | "
                    f"Est. Stockout: {est_stockout_date_str}</sup>"
                )
                
                plot_data_series = item_specific_forecast_df.set_index('date')['forecast_days']
                lower_ci_series = item_specific_forecast_df.set_index('date').get('lower_ci', None) if not use_ai_forecast else None
                upper_ci_series = item_specific_forecast_df.set_index('date').get('upper_ci', None) if not use_ai_forecast else None
                
                show_ci_in_plot = (lower_ci_series is not None and not lower_ci_series.empty and 
                                   upper_ci_series is not None and not upper_ci_series.empty and 
                                   not use_ai_forecast)

                st.plotly_chart(plot_annotated_line_chart(
                    data_series=plot_data_series, title=forecast_plot_title,
                    y_axis_title="Forecasted Days of Supply",
                    target_line=app_config.CRITICAL_SUPPLY_DAYS, target_label=f"Critical Level ({app_config.CRITICAL_SUPPLY_DAYS} Days)",
                    show_ci=show_ci_in_plot, 
                    lower_bound_series=lower_ci_series, upper_bound_series=upper_ci_series,
                    height=app_config.DEFAULT_PLOT_HEIGHT + 60, 
                    show_anomalies=False # Anomalies typically not relevant for supply forecast display
                ), use_container_width=True)
                if use_ai_forecast: 
                    st.caption("*Advanced forecast uses a simulated AI model with seasonal and trend components. Confidence intervals are not shown for this beta model.*")
            else: 
                st.info(f"No forecast data found for the selected item: {selected_drug_for_forecast}.")
    else: # Could not generate any forecast_df (either due to missing data or model error)
        st.warning("Supply forecast data could not be generated. Ensure historical health records contain necessary supply information (item, date, stock, consumption rate).")
