# test/utils/ai_analytics_engine.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from config import app_config

logger = logging.getLogger(__name__)

class RiskPredictionModel:
    """
    Simulates a pre-trained patient risk prediction model.
    Uses rule-based logic with weights and factors for features.
    """
    def __init__(self):
        self.base_risk_factors = {
            'age': {'weight': 0.5, 'threshold_high': 60, 'factor_high': 10, 'threshold_low': 18, 'factor_low': -2},
            'min_spo2_pct': {'weight': 2.0, 'threshold_low': app_config.SPO2_CRITICAL_THRESHOLD_PCT, 'factor_low': 25, 'mid_threshold_low': app_config.SPO2_LOW_THRESHOLD_PCT, 'factor_mid_low': 15},
            'vital_signs_temperature_celsius': {'weight': 1.5, 'threshold_high': app_config.SKIN_TEMP_FEVER_THRESHOLD_C, 'factor_high': 15, 'super_high_threshold': 39.5, 'factor_super_high': 25},
            'stress_level_score': {'weight': 0.4, 'threshold_high': app_config.STRESS_LEVEL_HIGH_THRESHOLD, 'factor_high': 10},
            'tb_contact_traced': {'weight': 1.0, 'is_flag': True, 'flag_value': 1, 'factor_true': 12},
            'encounter_type': {'weight': 0.3, 'is_value_match': True, 'value_to_match': 'Emergency Visit', 'factor_true': 15}, # Example for 'Emergency Visit' type
            'referral_status': {'weight': 0.5, 'is_value_match': True, 'value_to_match': 'Pending Urgent Referral', 'factor_true': 18}, # Hypothetical specific referral status
        }
        self.condition_base_scores = {
            "Pneumonia": 20, "TB": 25, "HIV-Positive": 22, "Malaria": 15, "Diabetes": 15, "Hypertension": 12,
            "STI-Gonorrhea": 10, "STI-Syphilis": 10, "Anemia": 8, "Dengue": 18, "Sepsis": 40, "Severe Dehydration": 30,
            "Wellness Visit": -5, "Follow-up Health": -2
        }
        logger.info("Simulated RiskPredictionModel initialized with rule-based logic.")

    def _get_condition_base_score(self, condition_str: Optional[str]) -> float:
        if pd.isna(condition_str) or str(condition_str).lower() == "unknown": return 0.0
        base_score = 0.0
        conditions = [c.strip().lower() for c in str(condition_str).split(';')]
        for cond in conditions:
            for known_cond, score_val in self.condition_base_scores.items():
                if known_cond.lower() in cond: # Partial match to catch variants
                    base_score += score_val; break 
        return base_score

    def predict_risk_score(self, patient_features: pd.Series) -> float:
        calculated_risk = self._get_condition_base_score(patient_features.get('condition'))
        for feature, params in self.base_risk_factors.items():
            if feature in patient_features and pd.notna(patient_features[feature]):
                value = patient_features[feature]; weight = params.get('weight', 1.0)
                if params.get('is_flag'):
                    if value == params.get('flag_value', 1): calculated_risk += params.get('factor_true', 0) * weight
                elif params.get('is_value_match'): # check string equality
                    if isinstance(value, str) and params.get('value_to_match','').lower() == value.lower(): calculated_risk += params.get('factor_true', 0) * weight
                else: # Numeric with thresholds
                    if 'super_high_threshold' in params and value >= params['super_high_threshold']: calculated_risk += params.get('factor_super_high', 0) * weight
                    elif 'threshold_high' in params and value >= params['threshold_high']: calculated_risk += params.get('factor_high', 0) * weight
                    if 'mid_threshold_low' in params and value < params['mid_threshold_low'] and (not 'threshold_low' in params or value >= params['threshold_low']): calculated_risk += params.get('factor_mid_low', 0) * weight
                    elif 'threshold_low' in params and value < params['threshold_low']: calculated_risk += params.get('factor_low', 0) * weight
        
        chronic_summary = patient_features.get('key_chronic_conditions_summary', "Unknown")
        if isinstance(chronic_summary, str) and chronic_summary.lower() != "unknown" and chronic_summary:
            num_chronic = len([c for c in chronic_summary.split(';') if c.strip() and c.lower() != "none"]) # Count actual conditions
            calculated_risk += num_chronic * 8
        adherence = patient_features.get('medication_adherence_self_report', "Unknown")
        if isinstance(adherence, str): # Ensure adherence is string before lower()
            if adherence.lower() == 'poor': calculated_risk += 15
            elif adherence.lower() == 'fair': calculated_risk += 7
        return float(np.clip(calculated_risk, 0, 100))

    def predict_bulk_risk_scores(self, health_df: pd.DataFrame) -> pd.Series:
        if health_df.empty: return pd.Series(dtype='float64')
        # Ensure all required columns exist, filling with neutral defaults if not
        model_input_cols_check = list(self.base_risk_factors.keys()) + ['condition', 'key_chronic_conditions_summary', 'medication_adherence_self_report']
        temp_df = health_df.copy() # Work on a copy
        for col in model_input_cols_check:
            if col not in temp_df.columns:
                temp_df[col] = "Unknown" if isinstance(temp_df.get(col, "string"), str) else np.nan # Appropriate NaN or "Unknown"
        return temp_df.apply(lambda row: self.predict_risk_score(row), axis=1)

class FollowUpPrioritizer:
    def __init__(self):
        self.priority_weights = {'ai_risk_score_component': 0.40, 'critical_vital_alert_points': 30, 'days_since_last_contact_factor': 0.5, 'pending_critical_referral_points': 25, 'specific_high_alert_condition_points': 20 }
        logger.info("Simulated FollowUpPrioritizer initialized.")

    def _has_critical_vitals(self, patient_features: pd.Series) -> bool:
        if pd.notna(patient_features.get('min_spo2_pct')) and patient_features['min_spo2_pct'] < app_config.SPO2_CRITICAL_THRESHOLD_PCT: return True
        temp_col_prio = 'vital_signs_temperature_celsius' if 'vital_signs_temperature_celsius' in patient_features and pd.notna(patient_features['vital_signs_temperature_celsius']) else 'max_skin_temp_celsius'
        if pd.notna(patient_features.get(temp_col_prio)) and patient_features[temp_col_prio] >= 39.5 : return True
        if pd.notna(patient_features.get('fall_detected_today')) and patient_features['fall_detected_today'] > 0: return True
        return False

    def _has_pending_critical_referral(self, patient_features: pd.Series) -> bool:
        if str(patient_features.get('referral_status', 'Unknown')).lower() == 'pending':
            crit_keywords = ["tb", "hiv", "pneumonia", "sepsis", "cardiac", "cancer", "emergency", "urgent"]
            if any(keyword in str(patient_features.get('condition','')).lower() for keyword in crit_keywords) or \
               any(keyword in str(patient_features.get('referral_reason','')).lower() for keyword in crit_keywords):
                return True
        return False
        
    def _has_specific_high_alert_condition(self, patient_features: pd.Series) -> bool:
        condition_str = str(patient_features.get('condition','')).lower()
        notes_str = str(patient_features.get('notes','')).lower()
        ai_risk = patient_features.get('ai_risk_score', 0.0)
        if ("tb" in condition_str and "new" in notes_str) or \
           ("hiv-positive" in condition_str and ("new art" in notes_str or (pd.notna(ai_risk) and ai_risk > 80))):
            return True
        return False

    def calculate_priority_score(self, patient_features: pd.Series, health_history_df: Optional[pd.DataFrame] = None) -> float:
        score = 0.0
        ai_risk_val = patient_features.get('ai_risk_score', 0.0)
        if pd.notna(ai_risk_val): score += (ai_risk_val / 100.0) * (self.priority_weights['ai_risk_score_component'] * 100)
        if self._has_critical_vitals(patient_features): score += self.priority_weights['critical_vital_alert_points']
        if self._has_pending_critical_referral(patient_features): score += self.priority_weights['pending_critical_referral_points']
        if self._has_specific_high_alert_condition(patient_features): score += self.priority_weights['specific_high_alert_condition_points']
        days_since = 0
        current_enc_date = pd.to_datetime(patient_features.get('encounter_date'), errors='coerce')
        if pd.notna(current_enc_date) and health_history_df is not None and not health_history_df.empty and 'encounter_date' in health_history_df.columns:
            prev_encs = health_history_df[pd.to_datetime(health_history_df['encounter_date'], errors='coerce') < current_enc_date]
            if not prev_encs.empty: days_since = (current_enc_date - prev_encs['encounter_date'].max()).days
        score += min(days_since, 60) * self.priority_weights['days_since_last_contact_factor']
        return float(np.clip(score, 0, 100))

    def generate_followup_priorities(self, health_df: pd.DataFrame) -> pd.Series:
        if health_df.empty: return pd.Series(dtype='float64')
        if 'patient_id' not in health_df.columns or 'encounter_date' not in health_df.columns:
             return pd.Series(0.0, index=health_df.index) # Default if critical columns missing
        
        # Ensure encounter_date is datetime for sorting
        health_df_sorted = health_df.copy()
        health_df_sorted['encounter_date'] = pd.to_datetime(health_df_sorted['encounter_date'], errors='coerce')
        health_df_sorted.dropna(subset=['encounter_date'], inplace=True) # Remove if date cannot be parsed
        if health_df_sorted.empty: return pd.Series(0.0, index=health_df.index)

        health_df_sorted.sort_values(['patient_id', 'encounter_date'], inplace=True)
        
        all_scores = []
        for patient_id, group_df in health_df_sorted.groupby('patient_id', sort=False): # sort=False because already sorted
            patient_scores = []
            for index, row in group_df.iterrows():
                # Pass the group (historical encounters for this patient up to current sort order)
                patient_scores.append({'original_index': index, 'score': self.calculate_priority_score(row, group_df.loc[:index])})
            all_scores.extend(patient_scores)
        
        if not all_scores: return pd.Series(0.0, index=health_df.index)
        priority_df = pd.DataFrame(all_scores).set_index('original_index')
        return priority_df['score'].reindex(health_df.index).fillna(0.0) # Realign and fill if any original rows were dropped

class SupplyForecastingModel:
    def __init__(self):
        self.item_params = {
            "ACT Tablets": {"coeffs": [0.8,0.8,0.9,1.0,1.2,1.5,1.6,1.4,1.1,0.9,0.8,0.8], "trend": 0.02, "noise_std": 0.15},
            "TB-Regimen A": {"coeffs": [1.0]*12, "trend": 0.01, "noise_std": 0.05},
            "Amoxicillin Syrup": {"coeffs": [1.2,1.2,1.1,1.0,0.9,0.8,0.8,0.9,1.0,1.1,1.1,1.2], "trend": 0.015, "noise_std": 0.20},
        }
        logger.info("Simulated AI SupplyForecastingModel initialized.")

    def _get_item_params(self, item_name: str) -> Dict: return self.item_params.get(item_name, {"coeffs": [1.0]*12, "trend": 0.005, "noise_std": 0.1})

    def predict_daily_consumption(self, base_cons_rate: float, item: str, fc_date: pd.Timestamp, days_from_start: int) -> float:
        if pd.isna(base_cons_rate) or base_cons_rate <= 0: return 0.0
        p = self._get_item_params(item); season_adj = p["coeffs"][fc_date.month-1]
        trend_adj = (1 + p["trend"]/30) ** (days_from_start); noise = np.random.normal(1.0, p["noise_std"])
        return max(0.01, base_cons_rate * season_adj * trend_adj * noise)

    def forecast_supply_levels_advanced(self, health_df: pd.DataFrame, forecast_days_out: int = 30, item_filter_list: Optional[List[str]] = None) -> pd.DataFrame:
        forecasts = []; default_cols = ['item','date','current_stock','consumption_rate','forecast_stock','forecast_days','estimated_stockout_date','predicted_daily_consumption']
        if health_df is None or health_df.empty or not all(c in health_df.columns for c in ['item','encounter_date','item_stock_agg_zone','consumption_rate_per_day']): return pd.DataFrame(columns=default_cols)
        
        health_df_copy = health_df.copy()
        health_df_copy['encounter_date'] = pd.to_datetime(health_df_copy['encounter_date'], errors='coerce')
        health_df_copy.dropna(subset=['encounter_date'], inplace=True)
        if health_df_copy.empty: return pd.DataFrame(columns=default_cols)

        supply_status_df = health_df_copy.loc[health_df_copy.groupby('item')['encounter_date'].idxmax()]
        if item_filter_list: supply_status_df = supply_status_df[supply_status_df['item'].isin(item_filter_list)]
        if supply_status_df.empty: return pd.DataFrame(columns=default_cols)

        for _, r in supply_status_df.iterrows():
            item_name, stock, base_cons, last_date = r['item'], r['item_stock_agg_zone'], r['consumption_rate_per_day'], r['encounter_date']
            if pd.isna(stock) or pd.isna(last_date) or stock < 0: continue
            base_cons = 0.0 if pd.isna(base_cons) or base_cons < 0 else base_cons # Ensure base_cons is non-negative
            
            current_fc_stock, est_stockout_date = stock, pd.NaT
            for i in range(forecast_days_out):
                fc_date = last_date + pd.Timedelta(days=i + 1)
                daily_pred_cons = self.predict_daily_consumption(base_cons, item_name, fc_date, i + 1) if base_cons > 0 else 0.0
                stock_before_consumption = current_fc_stock
                current_fc_stock -= daily_pred_cons; current_fc_stock = max(0, current_fc_stock)
                days_of_supply = (current_fc_stock / daily_pred_cons) if daily_pred_cons > 0.01 else (np.inf if current_fc_stock > 0 else 0)
                if current_fc_stock <= 0 and pd.isna(est_stockout_date) and stock_before_consumption > 0: # Stocked out on this day
                    frac_day = (stock_before_consumption / daily_pred_cons) if daily_pred_cons > 0.01 else 0.0
                    est_stockout_date = last_date + pd.Timedelta(days=i + frac_day)
                forecasts.append({'item':item_name, 'date':fc_date, 'current_stock':stock, 'consumption_rate':base_cons, 'forecast_stock':current_fc_stock, 'forecast_days':days_of_supply, 'predicted_daily_consumption':daily_pred_cons, 'estimated_stockout_date':est_stockout_date})
            if pd.isna(est_stockout_date) and base_cons > 0.01: # If not stocked out in period, extrapolate based on average predicted
                avg_pred_cons_period = pd.Series([f['predicted_daily_consumption'] for f in forecasts if f['item']==item_name and f['date'] <= fc_date]).mean()
                if avg_pred_cons_period > 0.01: days_to_stockout = stock / avg_pred_cons_period; final_est_stockout = last_date + pd.to_timedelta(days_to_stockout,unit='D')
                else: final_est_stockout = pd.NaT # Cannot estimate if avg predicted cons is zero
                for entry in forecasts:
                    if entry['item']==item_name and pd.isna(entry['estimated_stockout_date']): entry['estimated_stockout_date'] = final_est_stockout
        if not forecasts: return pd.DataFrame(columns=default_cols)
        fc_df = pd.DataFrame(forecasts); fc_df['estimated_stockout_date'] = pd.to_datetime(fc_df['estimated_stockout_date'], errors='coerce')
        return fc_df

def apply_ai_models(health_df: pd.DataFrame) -> pd.DataFrame:
    if health_df is None or health_df.empty:
        logger.warning("apply_ai_models: Input health_df is empty. Skipping AI processing.")
        return pd.DataFrame(columns=health_df.columns if health_df is not None else [])
    df = health_df.copy(); logger.info(f"Starting AI model application to DataFrame with {len(df)} rows.")
    risk_model = RiskPredictionModel(); df['ai_risk_score'] = risk_model.predict_bulk_risk_scores(df)
    logger.info("Applied simulated AI patient risk scoring.")
    priority_model = FollowUpPrioritizer(); df['ai_followup_priority_score'] = priority_model.generate_followup_priorities(df)
    logger.info("Applied simulated AI follow-up prioritization.")
    logger.info("AI model application complete.")
    return df
