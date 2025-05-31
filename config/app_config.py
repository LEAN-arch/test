# health_hub/config/app_config.py
import os
import pandas as pd

# Base directory of the application
# In health_hub/config/app_config.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
APP_LOGO = os.path.join(ASSETS_DIR, "DNA-DxBrand.png") 

# Data sources directory
DATA_SOURCES_DIR = os.path.join(BASE_DIR, "data_sources")
HEALTH_RECORDS_CSV = os.path.join(DATA_SOURCES_DIR, "health_records.csv")
ZONE_ATTRIBUTES_CSV = os.path.join(DATA_SOURCES_DIR, "zone_attributes.csv")
ZONE_GEOMETRIES_GEOJSON = os.path.join(DATA_SOURCES_DIR, "zone_geometries.geojson")
IOT_CLINIC_ENVIRONMENT_CSV = os.path.join(DATA_SOURCES_DIR, "iot_clinic_environment.csv")

# Assets directory
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STYLE_CSS_PATH = os.path.join(ASSETS_DIR, "style.css") 
APP_LOGO = os.path.join(ASSETS_DIR, "DNA-DxBrand.png") 

# App Settings
APP_TITLE = "Community Health Intelligence Hub"
APP_VERSION = "2.1.0" # Version update for testing insights enhancement
APP_FOOTER = f"© {pd.Timestamp('now').year} Health Informatics Initiative. All Rights Reserved."
CONTACT_EMAIL = "support@healthhub-demo.com"
CACHE_TTL_SECONDS = 3600 

# Dashboard specific settings
DEFAULT_DATE_RANGE_DAYS_VIEW = 1 
DEFAULT_DATE_RANGE_DAYS_TREND = 30 
RISK_THRESHOLDS = {
    "high": 75, "moderate": 60, "low": 40, 
    "chw_alert_high": 80, "chw_alert_moderate": 65, 
    "district_zone_high_risk": 70 
}
CRITICAL_SUPPLY_DAYS = 10 
TARGET_PATIENT_RISK_SCORE = 50 

# Disease-Specific Targets/Thresholds
TARGET_TB_CASE_DETECTION_RATE = 85; TARGET_MALARIA_POSITIVITY_RATE = 5 
TARGET_HIV_LINKAGE_TO_CARE = 90; TARGET_HPV_SCREENING_COVERAGE = 70 
TARGET_ANEMIA_PREVALENCE_WOMEN = 15; PNEUMONIA_CASE_FATALITY_TARGET = 5 
STI_SYNDROMIC_MANAGEMENT_ACCURACY = 90 

# --- ENHANCEMENTS FOR TESTING INSIGHTS ---
KEY_TEST_TYPES_FOR_ANALYSIS = { # Dictionary to define properties for key tests
    "Sputum-AFB": {"disease_group": "TB", "target_tat_days": 2, "critical": True},
    "Sputum-GeneXpert": {"disease_group": "TB", "target_tat_days": 1, "critical": True},
    "RDT-Malaria": {"disease_group": "Malaria", "target_tat_days": 0.5, "critical": True}, # 0.5 days = 12 hours
    "Microscopy-Malaria": {"disease_group": "Malaria", "target_tat_days": 1, "critical": False},
    "HIV-Rapid": {"disease_group": "HIV", "target_tat_days": 0.25, "critical": True}, # 0.25 days = 6 hours
    "HIV-ViralLoad": {"disease_group": "HIV", "target_tat_days": 7, "critical": True},
    "RPR": {"disease_group": "STI", "test_for": "Syphilis", "target_tat_days": 1, "critical": True},
    "NAAT-GC": {"disease_group": "STI", "test_for": "Gonorrhea", "target_tat_days": 3, "critical": True},
    "PapSmear": {"disease_group": "CervicalCancer", "test_for": "HPV/Cancer", "target_tat_days": 14, "critical": False},
    "Glucose Test": {"disease_group": "Diabetes", "target_tat_days": 0.25, "critical": False},
    "Hemoglobin Test": {"disease_group": "Anemia", "target_tat_days": 0.25, "critical": False},
    "Chest X-Ray": {"disease_group": "TB/Pneumonia", "target_tat_days": 1, "critical": False}
}
# Derive some lists from the above dict for convenience
CRITICAL_TESTS_LIST = [test for test, props in KEY_TEST_TYPES_FOR_ANALYSIS.items() if props.get("critical")]
TARGET_OVERALL_TESTS_MEETING_TAT_PCT = 85 # Overall target: % of all critical tests meeting their specific TAT
TARGET_SAMPLE_REJECTION_RATE_PCT = 5 # Target to keep sample rejection rate below this
OVERDUE_PENDING_TEST_DAYS = 7 # Tests pending for more than this many days are "critically overdue"
# --- END OF ENHANCEMENTS FOR TESTING INSIGHTS ---


KEY_CONDITIONS_FOR_TRENDS = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'STI-Syphilis', 'STI-Gonorrhea', 'Anemia', 'Dengue', 'Hypertension', 'Diabetes', 'Wellness Visit']
KEY_DRUG_SUBSTRINGS_SUPPLY = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'Penicillin', 'Ceftriaxone', 'Iron-Folate', 'Amoxicillin', 'Metformin', 'Amlodipine', 'Insulin']

# Wearable/Sensor-Based Thresholds/Targets
SKIN_TEMP_FEVER_THRESHOLD_C = 38.0; SPO2_LOW_THRESHOLD_PCT = 94
SPO2_CRITICAL_THRESHOLD_PCT = 90; TARGET_DAILY_STEPS = 8000
TARGET_SLEEP_HOURS = 7.0; TARGET_SLEEP_SCORE_PCT = 75 
STRESS_LEVEL_HIGH_THRESHOLD = 7 

# Clinic Environment IoT Thresholds/Targets
CO2_LEVEL_ALERT_PPM = 1000; CO2_LEVEL_IDEAL_PPM = 800 
PM25_ALERT_UGM3 = 25; PM25_IDEAL_UGM3 = 12 
VOC_INDEX_ALERT = 200; NOISE_LEVEL_ALERT_DB = 65 
TARGET_WAITING_ROOM_OCCUPANCY = 10 
TARGET_PATIENT_THROUGHPUT_PER_HOUR = 8 
TARGET_SANITIZER_DISPENSES_PER_HOUR_PER_DISPENSER = 5 

# Intervention thresholds
INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD = 60 
INTERVENTION_TB_BURDEN_HIGH_THRESHOLD = 5 
INTERVENTION_MALARIA_BURDEN_HIGH_THRESHOLD = 10 
INTERVENTION_PREVALENCE_HIGH_PERCENTILE = 0.75 

# Plotly specific settings
DEFAULT_PLOT_HEIGHT = 400; COMPACT_PLOT_HEIGHT = 320 
MAP_PLOT_HEIGHT = 600 

# Map Configurations
TIJUANA_CENTER_LAT = 32.5149; TIJUANA_CENTER_LON = -117.0382
TIJUANA_DEFAULT_ZOOM = 10; MAP_DEFAULT_CENTER_LAT = TIJUANA_CENTER_LAT
MAP_DEFAULT_CENTER_LON = TIJUANA_CENTER_LON; MAP_DEFAULT_ZOOM = TIJUANA_DEFAULT_ZOOM
DEFAULT_CRS = "EPSG:4326"; MAPBOX_STYLE = "carto-positron"  

# Logging Configuration
LOG_LEVEL = "DEBUG" 
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Color Palette
DISEASE_COLORS = {"TB": "#EF4444", "Malaria": "#F59E0B", "HIV": "#8B5CF6", "Pneumonia": "#3B82F6", "Anemia": "#10B981", "STI": "#EC4899", "Dengue": "#6366F1", "Hypertension": "#F97316", "Diabetes": "#0EA5E9", "Wellness Visit": "#84CC16", "CervicalCancer": "#d6336c", "Other": "#6B7280" }
RISK_STATUS_COLORS = {"High": "#EF4444", "Moderate": "#F59E0B", "Low": "#10B981", "Neutral": "#6B7280"}
