# tests/conftest.py
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
from config import app_config

@pytest.fixture(scope="session")
def sample_health_records_df_main():
    """
    Comprehensive DataFrame of health records, aligning with 'health_records.csv'
    and dashboard functionalities. Includes a wider variety of data.
    Uses column names matching the expanded CSV after cleaning (lowercase, underscore).
    """
    data = {
        'encounter_id': [f'ENC{i:03d}' for i in range(1, 43)], # Added from expanded schema
        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P001', 'P002', 'P009', 'P010', 'P011', 'P012', 'P013', 'P014', 'P015', 'P101', 'P102', 'P103', 'P104', 'P105', 'P106', 'P107', 'P108', 'P109', 'P110', 'P111', 'P112', 'P113', 'P114', 'P115', 'P116', 'P117', 'P118', 'P119', 'P120', 'P121', 'P122', 'P123', 'P124', 'P125'],
        'encounter_date': pd.to_datetime([ # Renamed from 'date'
            '2023-10-01', '2023-10-01', '2023-10-02', '2023-10-02', '2023-10-03',
            '2023-10-03', '2023-10-04', '2023-10-05', '2023-10-07', '2023-10-08',
            '2023-10-09', '2023-10-10', '2023-10-11', '2023-10-12', '2023-10-13',
            '2023-10-14', '2023-10-15',
            '2025-03-05', '2025-03-10', '2025-03-15', '2025-03-20', '2025-03-25',
            '2025-04-02', '2025-04-08', '2025-04-12', '2025-04-18', '2025-04-25',
            '2025-05-01', '2025-05-05', '2025-05-10', '2025-05-15', '2025-05-20',
            '2025-05-25', '2025-05-28', '2025-05-29', '2025-05-30', '2025-05-30',
            '2025-05-02', '2025-05-08', '2025-05-16', '2025-05-22', '2025-05-28'
        ]),
        'encounter_type': ['Clinic Visit', 'CHW Home Visit'] * 21, # Added from expanded
        'condition': ['TB', 'Malaria', 'Wellness Visit', 'Pneumonia', 'TB', 'STI-Syphilis', 'Anemia', 'HIV-Positive', 'TB', 'Malaria', 'Hypertension', 'Diabetes', 'Wellness Visit', 'Dengue', 'STI-Gonorrhea', 'Anemia', 'TB', 'TB', 'Malaria', 'Wellness Visit', 'Pneumonia', 'Hypertension', 'Diabetes', 'Anemia', 'STI-Gonorrhea', 'HIV-Positive', 'Wellness Visit', 'TB', 'Malaria', 'Hypertension', 'Pneumonia', 'STI-Chlamydia', 'Diabetes', 'Dengue', 'Anemia', 'TB', 'HIV-Positive', 'Wellness Visit', 'STI-Syphilis', 'TB', 'Malaria', 'Pneumonia'],
        'diagnosis_code_icd10': ['A15.0', 'B54', 'Z00.0', 'J18.9'] * 10 + ['A15.0', 'B54'], # Added
        'test_type': ['Sputum-AFB', 'RDT-Malaria', 'HIV-Rapid', 'Chest X-Ray', 'Sputum-GeneXpert', 'RPR', 'Hemoglobin Test', 'HIV-ViralLoad', 'Follow-up TB', 'RDT-Malaria', 'BP Check', 'Glucose Test', 'PapSmear', 'NS1 Antigen','NAAT-GC', 'Ferritin Test', 'Sputum-AFB', 'Sputum-AFB', 'RDT-Malaria', 'HIV-Rapid', 'Chest X-Ray', 'BP Check', 'Glucose Test', 'Hemoglobin Test', 'NAAT-GC', 'HIV-ViralLoad', 'PapSmear', 'Sputum-GeneXpert', 'Microscopy-Malaria', 'BP Check', 'Chest X-Ray', 'Test of Cure', 'Eye Exam', 'IgM Test', 'Follow-up Health', 'Sputum-AFB', 'HIV-ViralLoad', 'BP Check', 'RPR', 'Sputum-AFB', 'RDT-Malaria', 'Chest X-Ray'],
        'test_result': ['Positive', 'Positive', 'Negative', 'Positive', 'Positive', 'Positive', 'Low', "2500", 'N/A', 'Negative', 'High', 'High', 'Normal', 'Positive', 'Positive', 'Low', 'Negative', 'Positive', 'Positive', 'Negative', 'Rejected Sample', 'High', 'High', 'Low', 'Positive', '1500', 'Normal', 'Positive', 'Negative', 'Stable', 'Improving', 'Negative', 'No Retinopathy', 'Positive', 'Normal', 'Pending', '<50>', 'Normal', 'Rejected Sample', 'Positive', 'Pending', 'Indeterminate'],
        'sample_collection_date': pd.to_datetime(['2023-09-30', '2023-10-01'] * 21), # Added
        'sample_registered_lab_date': pd.to_datetime(['2023-09-28', '2023-10-01', '2023-10-01', '2023-09-30', '2023-10-01', '2023-10-02', '2023-10-04', '2023-09-28', '2023-10-07', '2023-10-08', '2023-10-09', '2023-10-09', '2023-10-06', '2023-10-11', '2023-10-10', '2023-10-12', '2023-10-13', '2025-03-03', '2025-03-10', '2025-03-14', '2025-03-19', '2025-03-25', '2025-04-01', '2025-04-08', '2025-04-09', '2025-04-12', '2025-04-21', '2025-04-30', '2025-05-04', '2025-05-10', '2025-05-14', '2025-05-17', '2025-05-20', '2025-05-26', '2025-05-29', '2025-05-30', '2025-05-23', '2025-05-02', '2025-05-06', '2025-05-13', '2025-05-22', '2025-05-27']), # Was 'test_date'
        'test_turnaround_days': [3,0,1,2,2,1,0,7,0,0,0,1,5,1,3,2,2,  2,0,1,1,0,1,0,3,6,4,1,1,0,1,3,5,2,0,0,7,  0,2,3,0,1],
        'item': ['TB-Regimen A', 'ACT Tablets', 'ARV-Regimen B', 'Amoxicillin Syrup', 'TB-Regimen A', 'Penicillin G', 'Iron-Folate Tabs', 'ARV-Regimen B', 'TB-Regimen A', 'ACT Tablets', 'Amlodipine', 'Metformin', 'Wellness Pack', 'Paracetamol', 'Ceftriaxone Inj', 'Iron Sucrose IV', 'TB-Regimen B', 'TB-Regimen A', 'ACT Tablets', 'Multivitamins', 'Amoxicillin Syrup', 'Amlodipine', 'Metformin', 'Iron-Folate Tabs', 'Ceftriaxone Inj', 'ARV-Regimen C', 'Calcium Supplement', 'TB-Regimen B', 'Prophylaxis', 'Lisinopril', 'Co-amoxiclav', '', '', 'ORS Sachets', 'Iron-Folate Tabs', '', 'ARV-Regimen D', 'Folic Acid', 'Penicillin G Procaine', 'TB-Regimen A', '', 'Azithromycin'],
        'quantity_dispensed': [1,1,0,1,0,1,1,1,0,0,1,1,0,1,1,1,1, 1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,0,1, 1,0,1,0,1],
        'item_stock_agg_zone': [100,50,75,200,98,30,150,73,96,48,120,90,0,80,18,25,60, 120,60,200,150,90,100,180,25,80,50,70,30,100,40,0,0,50,120,0,90, 60,45,110,0,75], # Was stock_on_hand
        'consumption_rate_per_day': [2,1,0.5,5,2,0.2,3,0.5,2,1,1.5,3,0,1.5,0.1,0.3,1.2, 2.1,1.5,0.5,4.5,1.0,2.5,2.8,0.3,0.8,0.2,1.9,0.0,1.2,3.0,0,0,1.0,3.0,0,0.7, 0.1,0.4,2.0,0,3.5],
        'zone_id': ['ZoneA','ZoneB','ZoneA','ZoneC','ZoneA','ZoneB','ZoneC','ZoneA','ZoneA','ZoneB','ZoneD','ZoneB','ZoneE','ZoneF','ZoneC','ZoneA','ZoneD', 'ZoneA','ZoneB','ZoneA','ZoneC','ZoneD','ZoneE','ZoneF','ZoneA','ZoneB','ZoneC','ZoneD','ZoneE','ZoneF','ZoneA','ZoneB','ZoneC','ZoneD','ZoneE','ZoneA','ZoneF', 'ZoneB','ZoneC','ZoneE','ZoneF','ZoneA'],
        'clinic_id': ['C01','C02','C01','C03','C01','C02','C03','C01','C01','C02','C04','C02','C05','C06','C03','C01','C04', 'C01','C02','C01','C03','C04','C05','C06','C01','C02','C03','C04','C05','C06','C01','C02','C03','C04','C05','C01','C06', 'C02','C03','C05','C06','C01'],
        'physician_id': ['DOC001','DOC002','DOC001','DOC003','DOC001','DOC002','DOC003','DOC001','DOC001','DOC002','DOC004','DOC002','DOC005','DOC006','DOC003','DOC001','DOC004', 'DOC001','DOC002','DOC001','DOC003','DOC004','DOC005','DOC006','DOC001','DOC002','DOC003','DOC004','DOC005','DOC006','DOC001','DOC002','DOC003','DOC004','DOC005','DOC001','DOC006','DOC002','DOC003','DOC005','DOC006','DOC001'],
        'notes': ['Follow up','','','','Resistant strain?','','','Adherence counseling','','','New diagnosis HTN','Dietary advice given','Routine screen','Supportive care','Contact tracing','IV Iron','Resolved', 'New TB case','Fever and chills','Annual checkup','Image unclear_retake','Routine BP check','Follow-up diabetes','Pale and fatigued','Partner referral','New ART initiation','Routine screening','Persistent cough','Travel history','BP check','Follow-up X-ray','Cured','Annual eye check','High_fever_joint_pain','Hemoglobin normalized','Contact of P111','Undetectable', 'Prenatal visit','Sample hemolyzed_repeat','New case_contact','Suspected malaria','Needs radiologist review'],
        'patient_reported_symptoms': ['Cough', 'Fever;Chills', 'None'] * 14, # Added
        'vital_signs_bp_systolic': [120, 130, 125] * 14, # Added
        'vital_signs_bp_diastolic': [80, 85, 75] * 14, # Added
        'vital_signs_temperature_celsius': [36.5, 38.2, 37.0] * 14, # Added
        'min_spo2_pct': [95,96,98,93,94,95,97,94,95,97,96,95,98,96,95,93,97, 94,97,98,92,96,95,95,93,92,98,93,97,96,96,98,97,94,98,93,97, 98,96,94,97,93],
        'max_skin_temp_celsius': [37.5,37.2,36.8,38.5,38.1,37.0,36.9,38.2,37.3,37.1,37.2,37.3,36.7,37.4,37.6,37.5,37.0, 37.8,37.1,36.5,38.6,37.0,37.2,36.8,37.7,38.1,36.6,37.9,37.0,36.9,37.3,36.7,37.1,38.2,36.6,38.0,37.0, 36.9,37.3,37.6,37.0,38.1],
        'avg_spo2': [97,98,99,95,96,97,98,96,97,98,98,97,99,97,96,95,98, 96,98,99,94,97,96,97,95,94,99,95,98,97,97,99,98,96,99,95,98, 99,97,96,98,95],
        'avg_daily_steps': [3500,6200,8100,4500,3400,5800,7500,4200,3700,7800,5500,6300,9500,5300,4900,4000,6100, 4200,5800,9500,3900,6700,5300,7100,4800,3500,8800,3800,6900,7600,5200,8500,6100,4300,9000,3000,7800, 7900,5100,3300,6000,4100],
        'resting_heart_rate': [65,70,60,75,68,72,66,78,64,62,80,70,58,70,74,79,66, 70,68,60,78,75,72,65,70,80,62,73,67,68,70,63,66,77,61,74,64, 62,71,69,67,76],
        'avg_hrv': [50,45,55,40,48,42,52,38,51,53,35,44,60,46,41,36,53, 45,50,58,38,40,43,52,46,35,55,41,53,49,44,57,51,39,59,42,54, 57,44,47,51,39],
        'avg_sleep_duration_hrs': [6.5,7.2,8.0,5.0,6.0,7.5,6.8,5.5,6.7,7.8,5.8,7.1,8.2,7.0,6.2,5.7,7.0, 6.2,7.5,8.1,5.5,6.9,6.4,7.1,5.9,5.2,7.8,6.0,7.3,6.8,6.1,8.0,7.2,5.7,8.3,5.4,7.6, 7.9,6.3,6.6,7.0,5.8],
        'sleep_score_pct': [70,78,85,60,65,80,72,58,73,82,55,77,90,75,68,57,76, 68,82,88,60,75,70,78,65,58,85,67,80,76,69,88,79,62,90,59,83, 87,71,72,77,64],
        'stress_level_score': [5,3,2,6,7,4,3,8,4,2,7,5,1,4,6,7,3, 6,3,2,7,5,6,4,7,8,2,6,3,4,5,2,4,7,1,8,3, 2,5,6,4,7],
        'fall_detected_today': [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
        'ai_risk_score': [85,70,30,65,88,75,55,92,80,40,72,60,25,68,78,70,35, 78,65,25,72,60,68,58,75,85,20,82,35,55,45,30,40,70,28,80,30, 18,72,86,50,60],
        'ai_followup_priority_score': [90, 75, 20] * 14, # Added
        'gender': ['Male','Female','Male','Female','Male','Female','Male','Female','Male','Female','Female','Male','Female','Male','Female','Female','Male', 'Male','Female','Female','Male','Female','Male','Female','Male','Female','Female','Male','Female','Male','Female','Male','Female','Male','Female','Male','Female', 'Female','Male','Female','Male','Female'],
        'age': [45,30,62,25,55,38,28,50,45,30,58,67,42,33,29,39,41, 52,28,45,68,59,61,33,27,39,50,48,31,66,72,24,55,19,37,33,41, 29,35,42,22,58],
        'screening_hpv_status': ['Negative','Negative','Negative','Negative','Negative','Negative','Negative','Positive','Negative','Negative','Unknown','Unknown','Normal','Unknown','Unknown','Negative','Unknown', 'Negative','Negative','Negative','Negative','Negative','Negative','Negative','Negative','Positive','Normal','Negative','Negative','Negative','Negative','Negative','Negative','Unknown','Negative','Negative','Positive', 'Negative','Negative','Negative','Unknown','Negative'], # was 'hpv_status'
        'hiv_viral_load_copies_ml': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,2500.0,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,1500.0,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,50.0,  np.nan,np.nan,np.nan,np.nan,np.nan], # was hiv_viral_load
        'key_chronic_conditions_summary': ['Hypertension', 'None', 'Diabetes;Hypertension'] * 14, # Added
        'medication_adherence_self_report': ['Good', 'Fair', 'Poor'] * 14, # Added
        'chw_visit': [1,1,0,1,1,1,0,1,1,0,1,1,0,1,1,1,0, 1,1,0,1,0,0,1,1,1,0,1,0,0,0,0,0,1,0,1,1, 0,1,1,1,0],
        'tb_contact_traced': [0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0, 0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0, 0,0,0,0,0],
        'referral_status': ['Pending','Completed','N/A','Pending','Initiated','Completed','N/A','Pending','Follow-up','Completed','Pending','Completed','N/A','Pending','Initiated','Completed','N/A', 'Pending','Completed','N/A','Pending','Follow-up','Initiated','Completed','Pending','Initiated','N/A','Pending','Completed','Follow-up','Completed','Completed','N/A','Pending','Completed','Pending','Completed', 'N/A','Pending','Pending','Pending','Pending'],
        'referral_date': pd.to_datetime([None,'2023-09-25',None,'2023-10-01','2023-10-02','2023-09-28',None,'2023-10-04','2023-10-06','2023-10-07',None,'2023-10-09',None,'2023-10-11','2023-10-12',None,None, None,'2025-03-08',None,'2025-03-20','2025-03-24','2025-04-02','2025-04-07','2025-04-12','2025-04-15',None,'2025-05-01','2025-05-04','2025-05-09','2025-05-13','2025-05-18',None,'2025-05-28','2025-05-28','2025-05-30','2025-05-27', None,'2025-05-07','2025-05-16','2025-05-22','2025-05-28']),
        'referral_reason': ['TB Test', 'Further Investigation', 'N/A'] * 14, # Added
        'referred_to_facility_id': ['CL001', 'CL002', ''] * 14, # Added
        'referral_outcome': ['Attended', 'Pending', 'Missed Appointment'] * 14, # Added
        'referral_outcome_date': pd.to_datetime(['2023-09-28', None] * 21), # Added
        'sample_status': ['Accepted','Accepted','Accepted','Rejected','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted', 'Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Rejected','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted','Accepted', 'Accepted','Rejected','Accepted','Accepted','Accepted'],
        'rejection_reason': [np.nan,np.nan,np.nan,'Image Unclear',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'Slide Damaged',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, np.nan,'Hemolyzed',np.nan,np.nan,np.nan],
        'patient_latitude': [32.5 + np.random.rand()*0.1 for _ in range(42)], # Added
        'patient_longitude': [-117.0 + np.random.rand()*0.1 for _ in range(42)], # Added
    }
    df = pd.DataFrame(data)
    # Ensure hiv_viral_load_copies_ml is numeric, coercing non-numeric like '<50>'
    df['hiv_viral_load_copies_ml'] = pd.to_numeric(df['hiv_viral_load_copies_ml'], errors='coerce')

    # Numeric cols expected by core_data_processing
    numeric_cols = [
        'test_turnaround_days', 'quantity_dispensed', 'item_stock_agg_zone',
        'consumption_rate_per_day', 'ai_risk_score', 'ai_followup_priority_score',
        'vital_signs_bp_systolic', 'vital_signs_bp_diastolic', 'vital_signs_temperature_celsius',
        'min_spo2_pct', 'max_skin_temp_celsius', 'avg_spo2', 'avg_daily_steps',
        'resting_heart_rate', 'avg_hrv', 'avg_sleep_duration_hrs', 'sleep_score_pct',
        'stress_level_score', 'fall_detected_today', 'age', 'chw_visit', 'tb_contact_traced',
        'patient_latitude', 'patient_longitude'
    ]
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

    string_like_cols = [
        'encounter_id', 'patient_id', 'encounter_type', 'condition', 'diagnosis_code_icd10',
        'test_type', 'test_result', 'item', 'zone_id', 'clinic_id', 'physician_id',
        'notes', 'patient_reported_symptoms', 'gender', 'screening_hpv_status',
        'key_chronic_conditions_summary', 'medication_adherence_self_report',
        'referral_status', 'referral_reason', 'referred_to_facility_id',
        'referral_outcome', 'sample_status', 'rejection_reason'
    ]
    for col in string_like_cols:
        if col in df.columns:
             df[col] = df[col].fillna("Unknown").astype(str).str.strip()
             df.loc[df[col].isin(['', 'nan', 'None', 'N/A', '#N/A']), col] = "Unknown"
        else:
            df[col] = "Unknown"
    return df

@pytest.fixture(scope="session")
def sample_iot_clinic_df_main():
    data = {
        'timestamp': pd.to_datetime(['2023-10-01T09:00:00Z', '2023-10-01T10:00:00Z', '2023-10-01T09:00:00Z', '2023-10-01T09:30:00Z', '2023-10-02T09:00:00Z', '2023-10-02T14:00:00Z', '2023-10-03T10:00:00Z', '2025-03-05T08:00:00Z', '2025-04-10T13:00:00Z']),
        'clinic_id': ['C01', 'C01', 'C02', 'C03', 'C01', 'C04', 'C01', 'C05', 'C06'],
        'room_name': ['Waiting Room A', 'Waiting Room A', 'Consultation 1', 'TB Clinic Waiting', 'Consultation 1', 'General Waiting Area', 'Waiting Room A', 'Waiting Room E', 'Screening Tent F'],
        'avg_co2_ppm': [650, 700, 550, 950, 680, 700, 660, 720, 900],
        'max_co2_ppm': [700, 750, 600, 1100, 720, 780, 710, 790, 1050],
        'avg_pm25': [10.5, 12.1, 8.2, 22.8, 11.0, 13.5, 10.8, 12.5, 18.7],
        'voc_index': [120,130,100,180, 125, 135, 122, 133, 170],
        'avg_temp_celsius': [24.5, 24.7, 23.0, 24.0, 24.3, 24.6, 24.4, 24.1, 26.1],
        'avg_humidity_rh': [55,56,50,58, 54, 56, 55, 53, 62],
        'avg_noise_db': [50,52,45,53, 51, 50, 50, 49, 60],
        'waiting_room_occupancy': [5, 7, 2, 12, 6, 4, 5, 3, 11],
        'patient_throughput_per_hour': [8,9,10,5, 8, 6, 9, 7, 4],
        'sanitizer_dispenses_per_hour': [2,4,1,2, 3, 4, 3, 2, 4],
        'zone_id': ['ZoneA', 'ZoneA', 'ZoneB', 'ZoneC', 'ZoneA', 'ZoneD', 'ZoneA', 'ZoneE', 'ZoneF']
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_zone_attributes_df_main():
    data = {'zone_id': ['ZoneA', 'ZoneB', 'ZoneC', 'ZoneD', 'ZoneE', 'ZoneF'],
            'zone_display_name': ['Northwood District', 'Southville Area', 'Eastgate Community', 'Westend Borough', 'Central City Sector', 'Riverdale Precinct'],
            'population': [12500, 21000, 17500, 8500, 35000, 15000],
            'socio_economic_index': [0.65, 0.42, 0.78, 0.55, 0.85, 0.30],
            'num_clinics': [2, 1, 3, 1, 4, 1],
            'avg_travel_time_clinic_min': [15, 28, 12, 22, 8, 35]}
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_zone_geometries_gdf_main():
    features = [{"zone_id": "ZoneA", "name_geom": "Northwood District Geo", "polygon": Polygon([[0,0],[0,10],[10,10],[10,0],[0,0]])},
                {"zone_id": "ZoneB", "name_geom": "Southville Area Geo", "polygon": Polygon([[10,0],[10,10],[20,10],[20,0],[10,0]])},
                {"zone_id": "ZoneC", "name_geom": "Eastgate Community Geo", "polygon": Polygon([[0,-10],[0,0],[10,0],[10,-10],[0,-10]])},
                {"zone_id": "ZoneD", "name_geom": "Westend Borough Geo", "polygon": Polygon([[10,-10],[10,0],[20,0],[20,-10],[10,-10]])},
                {"zone_id": "ZoneE", "name_geom": "Central City Sector Geo", "polygon": Polygon([[-10,0],[-10,10],[0,10],[0,0],[-10,0]])},
                {"zone_id": "ZoneF", "name_geom": "Riverdale Precinct Geo", "polygon": Polygon([[-10,-10],[-10,0],[0,0],[0,-10],[-10,-10]])}]
    df = pd.DataFrame(features)
    return gpd.GeoDataFrame(df, geometry='polygon', crs=app_config.DEFAULT_CRS)

@pytest.fixture(scope="session")
def sample_enriched_gdf_main(sample_zone_geometries_gdf_main, sample_zone_attributes_df_main, sample_health_records_df_main, sample_iot_clinic_df_main):
    from utils.core_data_processing import load_zone_data, enrich_zone_geodata_with_health_aggregates
    
    # Simulate the result of load_zone_data which merges attributes and geometries
    # and sets the 'name' column.
    base_gdf = sample_zone_geometries_gdf_main.merge(
        sample_zone_attributes_df_main.rename(columns={'zone_display_name': 'name'}), # use display name as 'name'
        on="zone_id", how="left"
    )
    # Ensure 'name' column is present, using 'name_geom' or 'zone_id' as fallback if 'name' wasn't merged from attributes
    if 'name' not in base_gdf.columns:
        if 'name_geom' in base_gdf.columns: base_gdf.rename(columns={'name_geom': 'name'}, inplace=True)
        else: base_gdf['name'] = base_gdf['zone_id'] # Fallback
    
    # Make sure columns required by enrichment function are present (e.g. from zone_attributes_df)
    for col_attr in ['population', 'socio_economic_index', 'num_clinics']:
        if col_attr not in base_gdf.columns and col_attr in sample_zone_attributes_df_main.columns:
            # This situation shouldn't occur if merge was correct, but as safety for test.
            base_gdf = base_gdf.merge(sample_zone_attributes_df_main[['zone_id', col_attr]], on="zone_id", how="left", suffixes=('', '_attr_fill'))
            base_gdf[col_attr] = base_gdf[col_attr].fillna(base_gdf.get(f"{col_attr}_attr_fill"))
            base_gdf.drop(columns=[f"{col_attr}_attr_fill"], errors='ignore', inplace=True)
        elif col_attr not in base_gdf.columns: # If still not there
             base_gdf[col_attr] = 0 if col_attr == 'population' else (0.5 if col_attr == 'socio_economic_index' else 0)


    enriched_gdf = enrich_zone_geodata_with_health_aggregates(base_gdf, sample_health_records_df_main, sample_iot_clinic_df_main)
    return enriched_gdf

@pytest.fixture(scope="session")
def sample_series_data():
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07'])
    values = [10, 12, 9, 15, 13, 18, 16]; return pd.Series(values, index=dates, name="Daily Count")

@pytest.fixture(scope="session")
def sample_bar_df():
    return pd.DataFrame({'category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'], 'value': [20, 15, 25, 30, 18, 22, 28, 12], 'group': ['G1', 'G1', 'G2', 'G1', 'G2', 'G2', 'G1', 'G1']})

@pytest.fixture(scope="session")
def sample_donut_df():
    return pd.DataFrame({'status': ['Critical', 'Warning', 'Okay', 'Unknown'], 'count': [5, 12, 30, 3]})

@pytest.fixture(scope="session")
def sample_heatmap_df():
    np.random.seed(42); data = np.random.rand(5, 5) * 2 - 1
    df = pd.DataFrame(data, columns=[f'Var{i+1}' for i in range(5)], index=[f'Factor{j+1}' for j in range(5)])
    for i in range(5): df.iloc[i,i] = 1.0; return df

@pytest.fixture(scope="session")
def sample_choropleth_gdf(sample_zone_geometries_gdf_main, sample_zone_attributes_df_main):
    gdf = sample_zone_geometries_gdf_main.merge(sample_zone_attributes_df_main.rename(columns={'zone_display_name':'name'}), on="zone_id", how="left")
    if 'name_x' in gdf.columns and 'name_y' in gdf.columns: # Handle merge suffixes
        gdf['name'] = gdf['name_y'].fillna(gdf['name_x'])
        gdf.drop(columns=['name_x', 'name_y'], inplace=True)
    elif 'name_x' in gdf.columns : gdf.rename(columns={'name_x':'name'}, inplace=True)
    elif 'name_y' in gdf.columns : gdf.rename(columns={'name_y':'name'}, inplace=True)
    
    np.random.seed(10); gdf['risk_score'] = np.random.randint(30, 90, size=len(gdf))
    gdf['population'] = pd.to_numeric(gdf.get('population'), errors='coerce').fillna(0) #Ensure numeric pop
    return gdf

@pytest.fixture
def empty_health_df_with_schema(): # Based on expanded schema after cleaning
    return pd.DataFrame(columns=[
        'encounter_id', 'patient_id', 'encounter_date', 'encounter_type', 'condition',
        'diagnosis_code_icd10', 'test_type', 'test_result', 'sample_collection_date',
        'sample_registered_lab_date', 'test_turnaround_days', 'item', 'quantity_dispensed',
        'item_stock_agg_zone', 'consumption_rate_per_day', 'zone_id', 'clinic_id',
        'physician_id', 'notes', 'patient_reported_symptoms', 'vital_signs_bp_systolic',
        'vital_signs_bp_diastolic', 'vital_signs_temperature_celsius', 'min_spo2_pct',
        'max_skin_temp_celsius', 'avg_spo2', 'avg_daily_steps', 'resting_heart_rate',
        'avg_hrv', 'avg_sleep_duration_hrs', 'sleep_score_pct', 'stress_level_score',
        'fall_detected_today', 'ai_risk_score', 'ai_followup_priority_score', 'gender', 'age',
        'screening_hpv_status', 'hiv_viral_load_copies_ml', 'key_chronic_conditions_summary',
        'medication_adherence_self_report', 'chw_visit', 'tb_contact_traced',
        'referral_status', 'referral_date', 'referral_reason', 'referred_to_facility_id',
        'referral_outcome', 'referral_outcome_date', 'sample_status', 'rejection_reason',
        'patient_latitude', 'patient_longitude'
    ])

@pytest.fixture
def empty_iot_df_with_schema(): # After cleaning
    return pd.DataFrame(columns=[
        'timestamp', 'clinic_id', 'room_name', 'avg_co2_ppm', 'max_co2_ppm',
        'avg_pm25', 'voc_index', 'avg_temp_celsius', 'avg_humidity_rh', 'avg_noise_db',
        'waiting_room_occupancy', 'patient_throughput_per_hour',
        'sanitizer_dispenses_per_hour', 'zone_id'
    ])

@pytest.fixture
def empty_zone_attributes_df_with_schema(): # After cleaning
     return pd.DataFrame(columns=[
         'zone_id', 'name', 'population', 'socio_economic_index',
         'num_clinics', 'avg_travel_time_clinic_min'
     ]) # 'name' replaces 'zone_display_name'

@pytest.fixture
def empty_gdf_with_schema(): # Schema of enriched_gdf after processing
    return gpd.GeoDataFrame(columns=[
        'zone_id', 'name', 'geometry', 'population', 'socio_economic_index', 'num_clinics',
        'avg_travel_time_clinic_min', 'total_population_health_data', 'avg_risk_score',
        'total_patient_encounters', 'active_tb_cases', 'active_malaria_cases',
        'hiv_positive_cases', 'pneumonia_cases', 'total_referrals_made',
        'successful_referrals', 'avg_test_turnaround_critical',
        'perc_critical_tests_tat_met', 'prevalence_per_1000',
        'total_active_key_infections', 'avg_daily_steps_zone', 'zone_avg_co2',
        'facility_coverage_score'
        ], geometry='geometry', crs=app_config.DEFAULT_CRS)
