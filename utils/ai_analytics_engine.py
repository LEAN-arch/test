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
    In a real system, this class would load a serialized model (e.g., scikit-learn, TensorFlow)
    and its associated preprocessor.
    """
    def __init__(self):
        # Placeholder for model loading and feature engineering parameters
        # For now, we use a rule-based simulation.
        self.base_risk_factors = {
            'age': {'weight': 0.5, 'threshold_high': 60, 'factor_high': 10},
            'min_spo2_pct': {'weight': 1.5, 'threshold_low': 92, 'factor_low': 15},
            'vital_signs_temperature_celsius': {'weight': 1.0, 'threshold_high': 38.5, 'factor_high': 12},
            'stress_level_score': {'weight': 0.3, 'threshold_high': 7, 'factor_high': 8},
            'tb_contact_traced': {'weight': 0.8, 'is_flag': True, 'flag_value': 1, 'factor_true': 10}, # if 1 (traced contact)
            'hiv_positive_cases': {'weight': 1.2, 'is_condition_match': True, 'condition_substring': "HIV-Positive", 'factor_true': 20},
            'active_tb_cases': {'weight': 1.0, 'is_condition_match': True, 'condition_substring': "TB", 'factor_true': 18},
        }
        self.condition_severity_score = { # Base points for certain conditions
            "Pneumonia": 15, "TB": 20, "HIV-Positive": 18, "Malaria": 10, "Diabetes": 12, "Hypertension": 10,
            "STI-Gonorrhea": 8, "Anemia": 5, "Dengue": 12
        }
        logger.info("Simulated RiskPredictionModel initialized.")

    def _get_condition_base_score(self, condition_str: Optional[str]) -> int:
        if pd.isna(condition_str) or condition_str == "Unknown":
            return 0
        for cond, score in self.condition_severity_score.items():
            if cond.lower() in condition_str.lower():
                return score
        return 0


    def predict_risk_score(self, patient_features: pd.Series) -> float:
        """
        Simulates risk prediction based on patient features.
        Returns a score between 0 and 100.
        """
        calculated_risk = 0

        # Base score from primary condition
        calculated_risk += self._get_condition_base_score(patient_features.get('condition'))

        # Iterate through defined risk factors
        for feature, params in self.base_risk_factors.items():
            if feature in patient_features and pd.notna(patient_features[feature]):
                value = patient_features[feature]
                weight = params.get('weight', 1.0)

                if params.get('is_flag'):
                    if value == params.get('flag_value', 1): # Default flag value is 1
                        calculated_risk += params.get('factor_true', 5) * weight
                elif params.get('is_condition_match'):
                    if isinstance(value, str) and params.get('condition_substring','').lower() in value.lower():
                         calculated_risk += params.get('factor_true', 5) * weight
                else: # Numeric feature with thresholds
                    if 'threshold_high' in params and value >= params['threshold_high']:
                        calculated_risk += params.get('factor_high', 5) * weight
                    elif 'threshold_low' in params and value <= params['threshold_low']:
                        calculated_risk += params.get('factor_low', 5) * weight
        
        # Add points for number of chronic conditions (example)
        chronic_summary = patient_features.get('key_chronic_conditions_summary', 'Unknown')
        if isinstance(chronic_summary, str) and chronic_summary != "Unknown":
            num_chronic = len(chronic_summary.split(';'))
            calculated_risk += num_chronic * 5 # Add 5 points per chronic condition listed

        # Medication adherence (example)
        adherence = patient_features.get('medication_adherence_self_report', 'Unknown')
        if adherence == 'Poor':
            calculated_risk += 15
        elif adherence == 'Fair':
            calculated_risk += 7
            
        # Normalize to 0-100 range (rough normalization based on typical max sum of factors)
        # This is a simplification; real models have calibrated outputs.
        normalized_risk = min(max(calculated_risk, 0), 100)
        return float(normalized_risk)

    def predict_bulk_risk_scores(self, health_df: pd.DataFrame) -> pd.Series:
        """Applies risk prediction to a DataFrame of patient records."""
        if 'ai_risk_score' not in health_df.columns:
             health_df['ai_risk_score'] = 0.0 # Ensure column exists

        # Use existing ai_risk_score as a base if present, or generate new
        # For simulation, we'll just call predict_risk_score for each
        # In reality, model inputs might be more complex (e.g., sequences of vitals)
        
        # Required columns for this simulated model:
        model_input_cols = list(self.base_risk_factors.keys()) + ['condition', 'key_chronic_conditions_summary', 'medication_adherence_self_report']
        missing_cols = [col for col in model_input_cols if col not in health_df.columns]
        if missing_cols:
            logger.warning(f"Risk model cannot run: Missing input columns: {missing_cols}. Returning existing or zero scores.")
            return health_df.get('ai_risk_score', pd.Series(0.0, index=health_df.index))

        return health_df.apply(lambda row: self.predict_risk_score(row), axis=1)


class FollowUpPrioritizer:
    """
    Simulates an AI model or rule-set for prioritizing patient follow-ups.
    """
    def __init__(self):
        self.priority_weights = {
            'ai_risk_score': 0.4,
            'critical_vital_alert': 0.3, # A flag indicating presence of a critical vital sign
            'days_since_last_contact': 0.1, # More days = higher priority
            'is_post_hospitalization': 0.15, # Flag for recent hospitalization
            'pending_critical_referral': 0.25 # Flag if critical referral is pending
        }
        logger.info("Simulated FollowUpPrioritizer initialized.")

    def _check_critical_vitals(self, patient_features: pd.Series) -> bool:
        """Checks for critical vital signs based on config."""
        if pd.notna(patient_features.get('min_spo2_pct')) and patient_features['min_spo2_pct'] < app_config.SPO2_CRITICAL_THRESHOLD_PCT:
            return True
        
        temp_col = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in patient_features and pd.notna(patient_features['vital_signs_temperature_celsius']) else 'max_skin_temp_celsius'
        if pd.notna(patient_features.get(temp_col)) and patient_features[temp_col] >= app_config.SKIN_TEMP_FEVER_THRESHOLD_C + 1.0: # High fever
             return True
        if pd.notna(patient_features.get('fall_detected_today')) and patient_features['fall_detected_today'] > 0:
            return True
        return False

    def calculate_priority_score(self, patient_features: pd.Series, encounter_df_for_patient: Optional[pd.DataFrame] = None) -> float:
        """
        Calculates a follow-up priority score (0-100).
        `encounter_df_for_patient` can be used to derive days_since_last_contact.
        """
        score = 0.0
        
        # AI Risk Score component
        ai_risk = patient_features.get('ai_risk_score', 0)
        score += ai_risk * self.priority_weights['ai_risk_score']

        # Critical Vitals component
        if self._check_critical_vitals(patient_features):
            score += 100 * self.priority_weights['critical_vital_alert'] # Max contribution if critical

        # Days since last contact (example logic, requires more data context)
        # This would typically come from longitudinal data. For this row, assume 0 if no history.
        # If 'encounter_df_for_patient' is provided (all encounters for this patient):
        if encounter_df_for_patient is not None and not encounter_df_for_patient.empty and 'encounter_date' in encounter_df_for_patient.columns:
            # Ensure patient_features has an encounter_date to compare against
            current_encounter_date = pd.to_datetime(patient_features.get('encounter_date'), errors='coerce')
            if pd.notna(current_encounter_date):
                previous_encounters = encounter_df_for_patient[pd.to_datetime(encounter_df_for_patient['encounter_date']) < current_encounter_date]
                if not previous_encounters.empty:
                    days_since = (current_encounter_date - previous_encounters['encounter_date'].max()).days
                    score += min(days_since, 30) * self.priority_weights['days_since_last_contact'] # Cap contribution

        # Post-hospitalization (requires a flag like 'was_recently_hospitalized')
        if patient_features.get('was_recently_hospitalized', False): # Assume boolean flag
            score += 100 * self.priority_weights['is_post_hospitalization']
            
        # Pending critical referral
        is_pending_critical_referral = False
        if patient_features.get('referral_status') == 'Pending':
            # Simple check: if any of the high-impact conditions has a pending referral
            critical_conditions_for_referral_check = ["TB", "HIV-Positive", "Pneumonia", "Suspected Cancer", "Cardiac Event"]
            if any(cond.lower() in str(patient_features.get('condition','')).lower() for cond in critical_conditions_for_referral_check):
                is_pending_critical_referral = True
            if any(reason.lower() in str(patient_features.get('referral_reason','')).lower() for reason in ["urgent", "critical", "emergency"]):
                 is_pending_critical_referral = True
        if is_pending_critical_referral:
             score += 100 * self.priority_weights['pending_critical_referral']


        return min(max(score, 0), 100) # Normalize to 0-100

    def generate_followup_priorities(self, health_df: pd.DataFrame) -> pd.Series:
        """Generates follow-up priority scores for a DataFrame of patient records."""
        if 'ai_followup_priority_score' not in health_df.columns:
            health_df['ai_followup_priority_score'] = 0.0
            
        # To calculate 'days_since_last_contact', we need to group by patient_id and sort by date
        # This is more complex for a direct .apply() if not pre-calculating this feature
        # For simulation, we'll simplify and assume 'days_since_last_contact' feature could be pre-computed or
        # we pass only the current row's data and the logic for days_since_last_contact becomes less accurate.
        
        # If 'encounter_date' is the *current* encounter for this row being processed
        # we would need a helper function or prior step to calculate `days_since_previous_encounter`
        
        # Simplified apply (more accurate version would involve grouping or pre-calculation)
        def apply_priority_calc(row):
            # In a real scenario, you might pass historical encounters for this patient:
            # patient_history_df = health_df[health_df['patient_id'] == row['patient_id']]
            return self.calculate_priority_score(row, None) # Passing None for patient_history_df for now

        return health_df.apply(apply_priority_calc, axis=1)


class SupplyForecastingModel:
    """
    Simulates a more advanced supply forecasting model (e.g., time series based).
    """
    def __init__(self):
        # In a real system, load a model per item or item_group, or a multi-variate model.
        # Parameters for simulation (e.g., seasonality, trend factors)
        self.item_seasonality_factor = { # Peak month (1-12) and strength (0-1)
            "ACT Tablets": {"peak_month": 7, "strength": 0.3}, # Malaria season
            "Amoxicillin Syrup": {"peak_month": 1, "strength": 0.2}, # Common infections winter
        }
        self.base_trend_factor = 1.02 # Slight general increase in demand
        logger.info("Simulated SupplyForecastingModel initialized.")

    def _get_seasonal_adjustment(self, item_name: str, date: pd.Timestamp) -> float:
        """Returns a seasonal adjustment factor (e.g., 0.8 to 1.2)."""
        if item_name in self.item_seasonality_factor:
            params = self.item_seasonality_factor[item_name]
            month_diff = abs(date.month - params['peak_month'])
            # Simple triangular seasonality
            adjustment = 1.0 + params['strength'] * (1 - min(month_diff, 12 - month_diff) / 6.0)
            return adjustment
        return 1.0

    def predict_consumption(self, base_consumption_rate: float, item_name: str, date: pd.Timestamp, days_since_start: int) -> float:
        """
        Predicts consumption for a given item on a specific date, applying simulated factors.
        `days_since_start` is from the beginning of the forecast period.
        """
        if pd.isna(base_consumption_rate) or base_consumption_rate <= 0:
            return 0.0
        
        seasonal_adj = self._get_seasonal_adjustment(item_name, date)
        # Trend adjustment - e.g., small % increase per period from start
        trend_adj = self.base_trend_factor ** (days_since_start / 30.0) # Compounded monthly if days_since_start is daily
        
        # Add random walk / noise component to simulate uncaptured variance
        random_noise = np.random.normal(1.0, 0.1) # Multiplicative noise, mean 1, std dev 10%
        
        predicted_rate = base_consumption_rate * seasonal_adj * trend_adj * random_noise
        return max(0.05, predicted_rate) # Ensure consumption is not zero if base rate was >0


    def forecast_supply_levels_advanced(self, health_df: pd.DataFrame, forecast_days_out: int = 30,
                                    item_filter_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Uses the simulated AI model to forecast supply.
        This overrides the simpler linear forecast in core_data_processing.
        """
        forecasts = []
        if health_df is None or health_df.empty or not all(c in health_df.columns for c in ['item', 'encounter_date', 'item_stock_agg_zone', 'consumption_rate_per_day']):
            logger.warning("AI Supply forecast: Missing essential columns in health_df.")
            return pd.DataFrame()

        supply_status_df = health_df.sort_values('encounter_date').drop_duplicates(subset=['item'], keep='last')
        if item_filter_list:
            supply_status_df = supply_status_df[supply_status_df['item'].isin(item_filter_list)]
        if supply_status_df.empty:
             return pd.DataFrame()

        for _, row in supply_status_df.iterrows():
            item_name = row['item']
            current_stock_on_hand = row['item_stock_agg_zone']
            base_daily_consumption = row['consumption_rate_per_day']
            last_known_date = row['encounter_date']

            if pd.isna(current_stock_on_hand) or pd.isna(base_daily_consumption) or current_stock_on_hand < 0:
                continue

            temp_stock = current_stock_on_hand
            estimated_stockout_date = None
            
            for i in range(forecast_days_out):
                current_forecast_date = last_known_date + pd.Timedelta(days=i + 1)
                # Daily predicted consumption using the "model"
                daily_predicted_consumption = self.predict_consumption(base_daily_consumption, item_name, current_forecast_date, days_since_start=i)
                
                temp_stock -= daily_predicted_consumption
                temp_stock = max(0, temp_stock) # Stock cannot be negative

                days_of_supply_at_date = (temp_stock / daily_predicted_consumption) if daily_predicted_consumption > 0 else (np.inf if temp_stock > 0 else 0)

                if temp_stock <= 0 and estimated_stockout_date is None:
                    # Estimate fraction of day for stockout
                    stock_before_this_day = temp_stock + daily_predicted_consumption
                    fraction_of_day = (stock_before_this_day / daily_predicted_consumption) if daily_predicted_consumption > 0 else 0
                    estimated_stockout_date = last_known_date + pd.Timedelta(days=i + fraction_of_day)

                forecasts.append({
                    'item': item_name,
                    'date': current_forecast_date,
                    'current_stock': current_stock_on_hand, # Initial stock at start of forecast period
                    'consumption_rate': base_daily_consumption, # Base rate for reference
                    'forecast_stock': temp_stock,
                    'forecast_days': days_of_supply_at_date, # Days of supply *at that forecasted date*
                    'predicted_daily_consumption': daily_predicted_consumption,
                    'estimated_stockout_date': estimated_stockout_date # Propagates if set
                })
            
            # If never stocked out in forecast_days_out period
            if estimated_stockout_date is None and base_daily_consumption > 0:
                # Extrapolate with base rate if AI model is too noisy for long term stockout
                # This is a simplification. A robust model would provide a more consistent stockout date.
                days_to_stockout_simple = current_stock_on_hand / base_daily_consumption
                final_stockout_date = last_known_date + pd.to_timedelta(days_to_stockout_simple, unit='D')
                for entry in forecasts:
                    if entry['item'] == item_name and pd.isna(entry['estimated_stockout_date']):
                         entry['estimated_stockout_date'] = final_stockout_date
            elif estimated_stockout_date is None and base_daily_consumption <=0 and current_stock_on_hand > 0:
                 for entry in forecasts:
                    if entry['item'] == item_name and pd.isna(entry['estimated_stockout_date']):
                        entry['estimated_stockout_date'] = pd.NaT # Indefinite


        if not forecasts: return pd.DataFrame()
        forecast_df = pd.DataFrame(forecasts)
        forecast_df['estimated_stockout_date'] = pd.to_datetime(forecast_df['estimated_stockout_date'], errors='coerce')
        logger.info(f"AI based supply forecast generated for {forecast_df['item'].nunique()} items.")
        return forecast_df


# --- Main function to apply AI models and add features to DataFrame ---
def apply_ai_models(health_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies various AI models/simulations to the health data.
    This would be called after initial data loading and cleaning.
    """
    if health_df is None or health_df.empty:
        logger.warning("apply_ai_models: Input health_df is empty. Skipping AI processing.")
        return health_df

    df = health_df.copy()

    # 1. Patient Risk Scoring
    risk_model = RiskPredictionModel()
    # `predict_bulk_risk_scores` will now calculate based on its internal rules
    df['ai_risk_score'] = risk_model.predict_bulk_risk_scores(df)
    logger.info("Applied simulated AI patient risk scoring.")

    # 2. Follow-up Prioritization
    priority_model = FollowUpPrioritizer()
    # `generate_followup_priorities` uses the newly calculated ai_risk_score and other features
    df['ai_followup_priority_score'] = priority_model.generate_followup_priorities(df)
    logger.info("Applied simulated AI follow-up prioritization.")
    
    # Supply forecasting is usually a separate process generating its own DataFrame,
    # not typically adding columns to health_df directly for each patient encounter.
    # It's called separately when needed, e.g., for the clinic supply tab.

    # Other AI-driven features or alerts could be added here.
    # For example, detecting anomalies in patient trajectories,
    # predicting likelihood of referral completion, etc.

    return df

# Example of how the supply forecast might be called and cached if used frequently
# @st.cache_data(ttl=app_config.CACHE_TTL_SECONDS)
# def get_ai_supply_forecast(health_df, forecast_days, items=None):
#     supply_model = SupplyForecastingModel()
#     return supply_model.forecast_supply_levels_advanced(health_df, forecast_days, items)
