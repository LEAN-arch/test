# test/pages/clinic_components/patient_focus_tab.py
import streamlit as st
import pandas as pd
import numpy as np # Make sure numpy is imported for np.nan
from config import app_config
from utils.core_data_processing import get_patient_alerts_for_clinic # Used by this tab
from utils.ui_visualization_helpers import plot_bar_chart # For patient load

def render_patient_focus(filtered_health_df_clinic):
    st.subheader("🧍 Patient Load & High-Risk Case Identification (Period)")
    
    if filtered_health_df_clinic.empty:
        st.info("No health data available for selected period for Patient Load or alerts.")
        return

    # Patient Load by Key Condition
    if all(c in filtered_health_df_clinic.columns for c in ['condition', 'encounter_date', 'patient_id']):
        conditions_load_chart = app_config.KEY_CONDITIONS_FOR_TRENDS
        load_src_df = filtered_health_df_clinic[
            filtered_health_df_clinic['condition'].isin(conditions_load_chart) &
            (filtered_health_df_clinic['patient_id'].astype(str).str.lower() != 'unknown')
        ].copy()

        if not load_src_df.empty:
            # Ensure encounter_date is datetime for Grouper
            if not pd.api.types.is_datetime64_ns_dtype(load_src_df['encounter_date']):
                load_src_df['encounter_date'] = pd.to_datetime(load_src_df['encounter_date'], errors='coerce')
            load_src_df.dropna(subset=['encounter_date'], inplace=True) # Drop rows where date conversion failed

            if not load_src_df.empty:
                daily_load_summary = load_src_df.groupby(
                    [pd.Grouper(key='encounter_date', freq='D'), 'condition']
                )['patient_id'].nunique().reset_index()
                daily_load_summary.rename(columns={'patient_id': 'unique_patients', 'encounter_date':'date'}, inplace=True)

                if not daily_load_summary.empty:
                    st.plotly_chart(plot_bar_chart(
                        daily_load_summary, x_col='date', y_col='unique_patients',
                        title="Daily Unique Patient Encounters by Key Condition", color_col='condition',
                        barmode='stack', height=app_config.DEFAULT_PLOT_HEIGHT + 70,
                        y_axis_title="Unique Patients/Day", x_axis_title="Date",
                        color_discrete_map=app_config.DISEASE_COLORS, text_auto=False
                    ), use_container_width=True)
                else: st.caption("No patient load data for key conditions in period after grouping.")
            else: st.caption("No valid date data for patient load chart after filtering.")
        else: st.caption("No patients with encounters for key conditions in selected period.")
    else:
        st.info("Patient Load chart: Missing 'condition', 'encounter_date', or 'patient_id' columns.")

    st.markdown("---"); st.markdown("###### **Flagged Patient Cases for Clinical Review (Selected Period)**")
    flagged_patients_df = get_patient_alerts_for_clinic(
        filtered_health_df_clinic, risk_threshold_moderate=app_config.RISK_THRESHOLDS['moderate']
    )
    
    if flagged_patients_df is not None and not flagged_patients_df.empty:
        st.markdown(f"Found **{len(flagged_patients_df)}** unique patient encounters flagged for review.")
        
        cols_for_alert_table = [
            'patient_id', 'encounter_date', 'condition',
            'ai_risk_score', 'ai_followup_priority_score',
            'alert_reason', 'test_result', 'test_type',
            'hiv_viral_load_copies_ml', 'min_spo2_pct',
            'priority_score' # This is the aggregated score from get_patient_alerts_for_clinic
        ]
        existing_cols_display = [col for col in cols_for_alert_table if col in flagged_patients_df.columns]
        display_df = flagged_patients_df[existing_cols_display].copy()
        
        if 'priority_score' in display_df.columns and display_df['priority_score'].notna().any():
            display_df_sorted = display_df.sort_values(by='priority_score', ascending=False)
        else:
            display_df_sorted = display_df

        df_to_show_st = display_df_sorted.head(25).copy()

        # --- THOROUGH PRE-SERIALIZATION CLEANUP for st.dataframe ---
        safe_df_for_streamlit = pd.DataFrame()
        for col in df_to_show_st.columns:
            if col == 'encounter_date':
                safe_df_for_streamlit[col] = pd.to_datetime(df_to_show_st[col], errors='coerce')
                if pd.api.types.is_datetime64tz_dtype(safe_df_for_streamlit[col]):
                    safe_df_for_streamlit[col] = safe_df_for_streamlit[col].dt.tz_localize(None)
            elif col in ['ai_risk_score', 'ai_followup_priority_score', 'priority_score', 'hiv_viral_load_copies_ml', 'min_spo2_pct']:
                safe_df_for_streamlit[col] = pd.to_numeric(df_to_show_st[col], errors='coerce')
                # Note: For ProgressColumn, NaNs usually render as empty. For NumberColumn, as blank.
            else: # Aggressively convert all other columns to strings
                # First, fill NAs with a placeholder string, then convert to string type.
                safe_df_for_streamlit[col] = df_to_show_st[col].fillna('N/A').astype(str)
                # Replace common stringified NaN/None representations.
                safe_df_for_streamlit[col] = safe_df_for_streamlit[col].replace(['nan', 'None', 'NaT', '<NA>'], 'N/A', regex=False)
        
        # Define column configuration
        column_config_alerts = {
            "encounter_date": st.column_config.DateColumn("Encounter Date", format="YYYY-MM-DD"),
            "ai_risk_score": st.column_config.ProgressColumn("AI Risk", format="%d", min_value=0, max_value=100),
            "ai_followup_priority_score": st.column_config.ProgressColumn("AI Prio.", format="%d", min_value=0, max_value=100),
            "priority_score": st.column_config.NumberColumn("Overall Alert Prio.", format="%d", help="Aggregated priority score."),
            "alert_reason": st.column_config.TextColumn("Alert Reason(s)", width="large", help="Reasons this case is flagged."),
            "hiv_viral_load_copies_ml": st.column_config.NumberColumn("HIV VL (cp/mL)", format="%.0f"), # Ensure format allows NaNs gracefully
            "min_spo2_pct": st.column_config.NumberColumn("Min SpO2 (%)", format="%d%%"),
            # Add other string columns if specific TextColumn features are needed, otherwise they default to text
            "patient_id": st.column_config.TextColumn("Patient ID"),
            "condition": st.column_config.TextColumn("Condition"),
            "test_result": st.column_config.TextColumn("Test Result"),
            "test_type": st.column_config.TextColumn("Test Type"),
        }
        # Filter config to only include columns present in the safe_df_for_streamlit
        active_column_config_alerts = {k: v for k, v in column_config_alerts.items() if k in safe_df_for_streamlit.columns}

        st.dataframe(
            safe_df_for_streamlit, 
            use_container_width=True, 
            column_config=active_column_config_alerts, 
            height=450, 
            hide_index=True 
        )
    else: 
        st.info("No specific patient cases flagged for clinical review in the selected period based on current criteria.")
