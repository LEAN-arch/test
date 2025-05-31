# test/utils/ai_analytics_engine.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from config import app_config # app_config for thresholds, model parameters if any

logger = logging.getLogger(__name__)

class RiskPredictionModel:
    """
    Simulates a pre-trained patient risk prediction model.
    Uses rule-based logic with weights and factors for features.
    """
    def __init__(self):
        self.base_risk_factors = {
            # Feature: {weight, threshold_high, factor_high, threshold_low, factor_low, is_flag, flag_value, factor_true, is_condition_match, condition_substring}
            'age': {'weight': 0.5, 'threshold_high': 60, 'factor_high': 10, 'threshold_low': 18, 'factor_low': -2}, # Minor penalty for very young adult
            'min_spo2_pct': {'weight': 2.0, 'threshold_low': app_config.SPO2_CRITICAL_THRESHOLD_PCT, 'factor_low': 25, 'mid_threshold_low': app_config.SPO2_LOW_THRESHOLD_PCT, 'factor_mid_low': 15},
            'vital_signs_temperature_celsius': {'weight': 1.5, 'threshold_high': app_config.SKIN_TEMP_FEVER_THRESHOLD_C, 'factor_high': 15, 'super_high_threshold': 39.5, 'factor_super_high': 25},
            'stress_level_score': {'weight': 0.4, 'threshold_high': app_config.STRESS_LEVEL_HIGH_THRESHOLD, 'factor_high': 10},
            'tb_contact_traced': {'weight': 1.0, 'is_flag': True, 'flag_value': 1, 'factor_true': 12}, # 1 if is a traced contact
            'encounter_type': {'weight': 0.3, 'is_value_match': True, 'value_to_match': 'Emergency Visit', 'factor_true': 15}, # example if data had 'Emergency Visit'
            'referral_status': {'weight': 0.5, 'is_value_match': True, 'value_to_match': 'Pending Urgent Referral', 'factor_true': 18}, # hypothetical
        }
        self.condition_base_scores = { # Base risk points for certain diagnosed conditions
            "Pneumonia": 20, "TB": 25, "HIV-Positive": 22, "Malaria": 15, "Diabetes": 15, "Hypertension": 12,
            "STI-Gonorrhea": 10, "STI-Syphilis": 10, "Anemia": 8, "Dengue": 18, "Sepsis": 40, "Severe Dehydration": 30, # More severe examples
            "Wellness Visit": -5, "Follow-up Health": -2 # Negative risk for wellness
        }
        logger.info("Simulated RiskPredictionModel initialized with rule-based logic.")

    def _get_condition_base_score(self, condition_str: Optional[str]) -> float:
        if pd.isna(condition_str) or condition_str == "Unknown": return 0.0
        base_score = 0.0
        # Handle multiple conditions if semi-colon separated in condition_str
        conditions = [c.strip() for c in str(condition_str).split(';')]
        for cond in conditions:
            for known_cond, score_val in self.condition_base_scores.items():
                if known_cond.lower() in cond.lower(): # Partial match for variants
                    base_score += score_val
                    break # Add first matched score for a given part
        return base_score

    def predict_risk_score(self, patient_features: pd.Series) -> float:
        """Simulates risk prediction for a single patient record."""
        calculated_risk = 0.0
        calculated_risk += self._get_condition_base_score(patient_features.get('condition'))

        for feature, params in self.base_risk_factors.items():
            if feature in patient_features and pd.notna(patient_features[feature]):
                value = patient_features[feature]
                weight = params.get('weight', 1.0)

                if params.get('is_flag'):
                    if value == params.get('flag_value', 1): calculated_risk += params.get('factor_true', 0) * weight
                elif params.get('is_value_match'):
                    if isinstance(value, str) and params.get('value_to_match','').lower() == value.lower():
                         calculated_risk += params.get('factor_true', 0) * weight
                else: # Numeric feature with thresholds
                    if 'super_high_threshold' in params and value >= params['super_high_threshold']: calculated_risk += params.get('factor_super_high', 0) * weight
                    elif 'threshold_high' in params and value >= params['threshold_high']: calculated_risk += params.get('factor_high', 0) * weight
                    
                    if 'mid_threshold_low' in params and value < params['mid_threshold_low'] and (not 'threshold_low' in params or value >= params['threshold_low']): calculated_risk += params.get('factor_mid_low', 0) * weight
                    elif 'threshold_low' in params and value < params['threshold_low']: calculated_risk += params.get('factor_low', 0) * weight
        
        chronic_summary = patient_features.get('key_chronic_conditions_summary', 'Unknown')
        if isinstance(chronic_summary, str) and chronic_summary != "Unknown" and chronic_summary:
            num_chronic = len([c for c in chronic_summary.split(';') if c.strip()])
            calculated_risk += num_chronic * 8 # Increased points per chronic condition

        adherence = patient_features.get('medication_adherence_self_report', 'Unknown')
        if adherence == 'Poor': calculated_risk += 15
        elif adherence == 'Fair': calculated_risk += 7
            
        # Ensure risk is within 0-100
        normalized_risk = np.clip(calculated_risk, 0, 100)
        return float(normalized_risk)

    def predict_bulk_risk_scores(self, health_df: pd.DataFrame) -> pd.Series:
        """Applies risk prediction to a DataFrame of patient records."""
        if 'ai_risk_score' not in health_df.columns: health_df['ai_risk_score'] = 0.0
        
        required_cols = list(self.base_risk_factors.keys()) + ['condition', 'key_chronic_conditions_summary', 'medication_adherence_self_report']
        for col in required_cols: # Ensure all input columns for the model exist, even if as NaN/Unknown
            if col not in health_df.columns:
                 health_df[col] = "Unknown" if isinstance(health_df.get(col, "string"), str) else np.nan
        
        # Fallback: if no patient features are useful, this could result in 0. Better to handle this.
        # Consider returning NaNs if insufficient data for a meaningful score, and handle downstream.
        # For this simulation, we proceed with applying rules which results in 0 if no rule hits.
        return health_df.apply(lambda row: self.predict_risk_score(row), axis=1)

class FollowUpPrioritizer:
    """Simulates an AI model or rule-set for prioritizing patient follow-ups."""
    def __init__(self):
        self.priority_weights = { # Max contribution from each factor is weight * 100 (approx)
            'ai_risk_score_component': 0.40, # Normalized AI risk contributes up to 40 points
            'critical_vital_alert_points': 30,  # Flat 30 points if any critical vital
            'days_since_last_contact_factor': 0.5, # Max (0.5 * 30 days) = 15 points
            'pending_critical_referral_points': 25, # Flat 25 points
            'specific_high_alert_condition_points': 20 # E.g., newly diagnosed critical communicable disease
        }
        logger.info("Simulated FollowUpPrioritizer initialized.")

    def _has_critical_vitals(self, patient_features: pd.Series) -> bool:
        if pd.notna(patient_features.get('min_spo2_pct')) and patient_features['min_spo2_pct'] < app_config.SPO2_CRITICAL_THRESHOLD_PCT: return True
        temp_col = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in patient_features and pd.notna(patient_features['vital_signs_temperature_celsius']) else 'max_skin_temp_celsius'
        if pd.notna(patient_features.get(temp_col)) and patient_features[temp_col] >= 39.5 : return True # Very high fever
        if pd.notna(patient_features.get('fall_detected_today')) and patient_features['fall_detected_today'] > 0: return True
        return False

    def _has_pending_critical_referral(self, patient_features: pd.Series) -> bool:
        if patient_features.get('referral_status', 'Unknown').lower() == 'pending':
            crit_conds_for_ref = ["tb", "hiv", "pneumonia", "sepsis", "cardiac", "cancer", "severe dehydration", "emergency"] # keywords
            condition = str(patient_features.get('condition','')).lower()
            reason = str(patient_features.get('referral_reason','')).lower()
            if any(keyword in condition for keyword in crit_conds_for_ref) or \
               any(keyword in reason for keyword in crit_conds_for_ref):
                return True
        return False
        
    def _has_specific_high_alert_condition(self, patient_features: pd.Series) -> bool:
        # Example: New active TB, new HIV positive with symptoms
        condition = str(patient_features.get('condition','')).lower()
        notes = str(patient_features.get('notes','')).lower()
        if ("tb" in condition and "new" in notes) or \
           ("hiv-positive" in condition and ("new art" in notes or patient_features.get('ai_risk_score',0) > 80)): # Example combined logic
            return True
        return False


    def calculate_priority_score(self, patient_features: pd.Series, health_history_df: Optional[pd.DataFrame] = None) -> float:
        """
        Calculates a follow-up priority score (0-100).
        `health_history_df` should be all records FOR THIS PATIENT sorted by date.
        """
        score = 0.0
        
        ai_risk = patient_features.get('ai_risk_score', 0.0)
        score += (ai_risk / 100.0) * (self.priority_weights['ai_risk_score_component'] * 100)

        if self._has_critical_vitals(patient_features): score += self.priority_weights['critical_vital_alert_points']
        if self._has_pending_critical_referral(patient_features): score += self.priority_weights['pending_critical_referral_points']
        if self._has_specific_high_alert_condition(patient_features): score += self.priority_weights['specific_high_alert_condition_points']
            
        days_since_last_contact = 0
        current_encounter_date = pd.to_datetime(patient_features.get('encounter_date'), errors='coerce')
        if pd.notna(current_encounter_date) and health_history_df is not None and not health_history_df.empty:
            prev_encs = health_history_df[pd.to_datetime(health_history_df['encounter_date']) < current_encounter_date]
            if not prev_encs.empty:
                days_since_last_contact = (current_encounter_date - prev_encs['encounter_date'].max()).days
        
        score += min(days_since_last_contact, 60) * self.priority_weights['days_since_last_contact_factor'] # Max 30 points from this

        return np.clip(score, 0, 100) # Ensure score is between 0 and 100

    def generate_followup_priorities(self, health_df: pd.DataFrame) -> pd.Series:
        if health_df.empty: return pd.Series(dtype='float64')
        
        priorities = []
        if 'patient_id' not in health_df.columns or 'encounter_date' not in health_df.columns:
             logger.warning("Follow-up priority: 'patient_id' or 'encounter_date' missing. Returning zeros.")
             return pd.Series(0.0, index=health_df.index)

        # Sort by patient and date to correctly identify previous encounters for 'days_since_last_contact'
        sorted_df = health_df.sort_values(['patient_id', 'encounter_date'])
        
        for patient_id, group in sorted_df.groupby('patient_id'):
            # Pass the historical group for each row being processed for that patient
            for index, row in group.iterrows():
                # Pass `group` (all history for this patient) to calculate_priority_score
                priorities.append({'original_index': index, 'score': self.calculate_priority_score(row, group)})
        
        if not priorities: return pd.Series(0.0, index=health_df.index) # No scores calculated

        priority_df = pd.DataFrame(priorities).set_index('original_index')
        return priority_df['score'].reindex(health_df.index).fillna(0.0) # Realign with original DF order and fill any misses


class SupplyForecastingModel:
    """Simulates an AI/statistical model for supply forecasting (e.g., ARIMA, Prophet)."""
    def __init__(self):
        self.item_seasonality_params = { # Example: month_factor[month-1]
            "ACT Tablets": {"coeffs": [0.8, 0.8, 0.9, 1.0, 1.2, 1.5, 1.6, 1.4, 1.1, 0.9, 0.8, 0.8], "trend": 0.02, "noise_std": 0.15},
            "TB-Regimen A": {"coeffs": [1.0]*12, "trend": 0.01, "noise_std": 0.05}, # Stable
            "Amoxicillin Syrup": {"coeffs": [1.2, 1.2, 1.1, 1.0, 0.9, 0.8, 0.8, 0.9, 1.0, 1.1, 1.1, 1.2], "trend": 0.015, "noise_std": 0.20}, # Winter peak
        }
        logger.info("Simulated AI SupplyForecastingModel initialized.")

    def _get_item_params(self, item_name: str) -> Dict:
        return self.item_seasonality_params.get(item_name, {"coeffs": [1.0]*12, "trend": 0.005, "noise_std": 0.1}) # Default params

    def predict_daily_consumption(self, base_consumption_rate: float, item_name: str, forecast_date: pd.Timestamp, num_periods_from_start: int) -> float:
        """Predicts consumption for one day using simulated model components."""
        if pd.isna(base_consumption_rate) or base_consumption_rate <= 0: return 0.0
        
        params = self._get_item_params(item_name)
        month_idx = forecast_date.month - 1
        seasonal_factor = params["coeffs"][month_idx]
        
        # Simple linear trend: base * (1 + trend_per_period * num_periods)
        # For a daily forecast, num_periods_from_start can be days.
        # trend could be a small daily growth factor.
        trend_factor = (1 + params["trend"]/30) ** (num_periods_from_start) # Compounding monthly trend adjusted daily
        
        noise = np.random.normal(1.0, params["noise_std"])
        predicted_rate = base_consumption_rate * seasonal_factor * trend_factor * noise
        return max(0.01, predicted_rate) # Ensure non-negative, small floor

    def forecast_supply_levels_advanced(self, health_df: pd.DataFrame, forecast_days_out: int = 30, item_filter_list: Optional[List[str]] = None) -> pd.DataFrame:
        forecasts = []
        if health_df is None or health_df.empty or not all(c in health_df.columns for c in ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']):
            return pd.DataFrame()

        supply_status_df = health_df.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last') # Use most recent record for initial stock/rate
        if item_filter_list: supply_status_df = supply_status_df[supply_status_df['item'].isin(item_filter_list)]
        if supply_status_df.empty: return pd.DataFrame()

        for _, row in supply_status_df.iterrows():
            item_name = row['item']
            current_stock_on_hand = row['item_stock_agg_zone']
            base_daily_consumption = row['consumption_rate_per_day'] # This is historical average
            last_known_date = pd.to_datetime(row['encounter_date'])

            if pd.isna(current_stock_on_hand) or current_stock_on_hand < 0: continue # Cant forecast without stock
            if pd.isna(base_daily_consumption) or base_daily_consumption <=0: # if no historical use, assume current stock lasts indefinitely or handle as zero consumption
                base_daily_consumption = 0.0

            temp_stock = current_stock_on_hand
            estimated_stockout_date = pd.NaT
            
            for i in range(forecast_days_out):
                current_forecast_date = last_known_date + pd.Timedelta(days=i + 1)
                if base_daily_consumption > 0: # Only predict if there's some base consumption
                    daily_predicted_consumption = self.predict_daily_consumption(base_daily_consumption, item_name, current_forecast_date, i + 1)
                else:
                    daily_predicted_consumption = 0.0
                
                previous_day_stock = temp_stock
                temp_stock -= daily_predicted_consumption
                temp_stock = max(0, temp_stock)

                days_of_supply_at_date = (temp_stock / daily_predicted_consumption) if daily_predicted_consumption > 0.01 else (np.inf if temp_stock > 0 else 0)

                if temp_stock <= 0 and pd.isna(estimated_stockout_date) and previous_day_stock > 0: # Only set stockout once
                    # More precise stockout point (fraction of the day it stocked out)
                    if daily_predicted_consumption > 0.01 :
                        fraction_of_day_to_stockout = previous_day_stock / daily_predicted_consumption
                        estimated_stockout_date = last_known_date + pd.Timedelta(days=i + fraction_of_day_to_stockout)
                    else: # Stocked out but zero consumption predicted this day - unusual, use start of day
                        estimated_stockout_date = current_forecast_date
                
                forecasts.append({
                    'item': item_name, 'date': current_forecast_date,
                    'current_stock': current_stock_on_hand, 'consumption_rate': base_daily_consumption,
                    'forecast_stock': temp_stock, 'forecast_days': days_of_supply_at_date,
                    'predicted_daily_consumption': daily_predicted_consumption,
                    'estimated_stockout_date': estimated_stockout_date
                })
            
            if pd.isna(estimated_stockout_date) and base_daily_consumption > 0.01 : # If never stocked out, estimate based on average of predicted consumption
                avg_predicted_consumption_period = pd.Series([f['predicted_daily_consumption'] for f in forecasts if f['item']==item_name]).mean()
                if avg_predicted_consumption_period > 0.01:
                     days_to_stockout_overall = current_stock_on_hand / avg_predicted_consumption_period
                     final_est_stockout = last_known_date + pd.to_timedelta(days_to_stockout_overall, unit='D')
                     for entry in forecasts: # Update all rows for this item if still NaT
                         if entry['item'] == item_name and pd.isna(entry['estimated_stockout_date']):
                             entry['estimated_stockout_date'] = final_est_stockout
        if not forecasts: return pd.DataFrame()
        forecast_df = pd.DataFrame(forecasts)
        forecast_df['estimated_stockout_date'] = pd.to_datetime(forecast_df['estimated_stockout_date'], errors='coerce')
        return forecast_df


# --- Main function to apply AI models to DataFrame ---
def apply_ai_models(health_df: pd.DataFrame) -> pd.DataFrame:
    if health_df is None or health_df.empty:
        logger.warning("apply_ai_models: Input health_df is empty. Skipping AI processing.")
        return pd.DataFrame(columns=health_df.columns if health_df is not None else [])


    df = health_df.copy()
    logger.info(f"Starting AI model application to DataFrame with {len(df)} rows.")

    risk_model = RiskPredictionModel()
    df['ai_risk_score'] = risk_model.predict_bulk_risk_scores(df)
    logger.info("Applied simulated AI patient risk scoring.")

    priority_model = FollowUpPrioritizer()
    df['ai_followup_priority_score'] = priority_model.generate_followup_priorities(df)
    logger.info("Applied simulated AI follow-up prioritization.")
    
    logger.info("AI model application complete.")
    return df
