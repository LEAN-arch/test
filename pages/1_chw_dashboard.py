# /pages/chw_components/strategic_overview_tab.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render(period_df, program_targets, chw_monthly_cost):
    """
    Renders the entire strategic overview tab.
    
    Args:
        period_df (pd.DataFrame): The dataframe filtered for the selected strategic period.
        program_targets (dict): A dictionary containing program goals.
        chw_monthly_cost (float): The average monthly cost per CHW.
    """
    st.header("📈 Strategic Program Overview")
    st.markdown(f"Analysis for period: **{period_df['encounter_date_obj'].min().strftime('%d %b %Y')}** to **{period_df['encounter_date_obj'].max().strftime('%d %b %Y')}**")

    if period_df.empty:
        st.warning("No data available for the selected period to generate a strategic overview.")
        return

    # --- 1. Top-Line Impact & Cost-Effectiveness ---
    st.subheader("💰 Program Impact & Cost-Effectiveness")
    
    # Calculate core metrics
    total_visits = period_df['encounter_id'].nunique()
    unique_chws_in_period = period_df['chw_id'].nunique()
    num_months_in_period = (period_df['encounter_date_obj'].max() - period_df['encounter_date_obj'].min()).days / 30.44
    if num_months_in_period < 1: num_months_in_period = 1 # Avoid division by zero for short periods
    
    # Estimate total cost for the period based on active CHWs
    estimated_total_cost = unique_chws_in_period * chw_monthly_cost * num_months_in_period
    
    high_risk_patients_identified = period_df[period_df['ai_risk_score'] > 75]['patient_id'].nunique()
    critical_alerts_generated = len(period_df[period_df['alert_type'] == 'Critical Vitals'])

    cost_per_visit = estimated_total_cost / total_visits if total_visits > 0 else 0
    cost_per_hr_patient = estimated_total_cost / high_risk_patients_identified if high_risk_patients_identified > 0 else 0
    cost_per_critical_alert = estimated_total_cost / critical_alerts_generated if critical_alerts_generated > 0 else 0
    
    # Display KPIs
    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        target_cpv = program_targets.get('target_cost_per_visit_usd', 0)
        delta_cpv = cost_per_visit - target_cpv if target_cpv > 0 else None
        st.metric(
            label="Cost per Visit (USD)", 
            value=f"${cost_per_visit:.2f}", 
            delta=f"${delta_cpv:.2f} vs Target" if delta_cpv is not None else None, 
            delta_color="inverse"
        )
    with kpi_cols[1]:
        st.metric(label="Cost per High-Risk Patient Identified", value=f"${cost_per_hr_patient:.2f}")
    with kpi_cols[2]:
        st.metric(label="Investment per Life-Saving Alert", value=f"${cost_per_critical_alert:.2f}")

    st.markdown("---")

    # --- 2. Health Equity & Coverage ---
    st.subheader("🌍 Health Equity & Coverage Analysis")
    st.markdown("Verifying that our services reach the most vulnerable populations.")
    
    equity_cols = st.columns([2, 1])
    with equity_cols[0]:
        st.write("**Patient Encounter Distribution by Geographic Zone**")
        if 'latitude' in period_df.columns and 'longitude' in period_df.columns:
            map_df = period_df.dropna(subset=['latitude', 'longitude', 'ai_risk_score']).copy()
            if not map_df.empty:
                # Add a 'size' column for better visualization on the map
                map_df['size_for_map'] = map_df['ai_risk_score'] * 0.5 
                st.map(map_df, latitude='latitude', longitude='longitude', size='size_for_map', color='#ff000088')
            else:
                st.caption("No location data to display on map for this period.")
        else:
            st.caption("Latitude/Longitude data not available in the dataset.")
            
    with equity_cols[1]:
        st.write("**Reach by Socioeconomic Tier**")
        if 'socioeconomic_tier' in period_df.columns:
            ses_counts = period_df.drop_duplicates(subset=['patient_id'])['socioeconomic_tier'].value_counts().reset_index()
            ses_counts.columns = ['Tier', 'Unique Patients']
            fig = px.pie(ses_counts, names='Tier', values='Unique Patients', hole=0.4)
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Socioeconomic data not available.")

    st.markdown("---")

    # --- 3. Program Performance vs. Targets ---
    st.subheader("🎯 Program Performance vs. Targets")
    
    perf_cols = st.columns(2)
    with perf_cols[0]:
        st.write("**CHW Caseload & Activity**")
        avg_visits_per_chw = total_visits / unique_chws_in_period if unique_chws_in_period > 0 else 0
        target_caseload = program_targets.get('target_monthly_caseload', 100) # Assuming 100 as target
        
        fig_caseload = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = avg_visits_per_chw,
            title = {'text': "Avg. Visits per CHW (in period)"},
            delta = {'reference': target_caseload * num_months_in_period, 'reference': target_caseload},
            gauge = {
                'axis': {'range': [None, target_caseload * 1.5]},
                'steps' : [
                    {'range': [0, target_caseload * 0.5], 'color': "lightgray"},
                    {'range': [target_caseload * 0.5, target_caseload], 'color': "gray"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': target_caseload}
            }))
        fig_caseload.update_layout(height=250, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_caseload, use_container_width=True)

    with perf_cols[1]:
        st.write("**High-Risk Patient Follow-up Rate**")
        if 'high_risk_follow_up' in period_df.columns:
            follow_up_rate = period_df['high_risk_follow_up'].mean()
            target_follow_up = program_targets.get('target_high_risk_follow_up_rate', 0.90)
            
            fig_followup = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = follow_up_rate * 100,
                number = {'suffix': '%'},
                title = {'text': "Follow-up within 7 days"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': target_follow_up * 100}
                }))
            fig_followup.update_layout(height=250, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_followup, use_container_width=True)
        else:
            st.caption("Follow-up data not available.")
