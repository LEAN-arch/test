# health_hub/config/app_config.py
import os
import pandas as pd

# Base directory of the application
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # health_hub directory

# Data sources directory
DATA_SOURCES_DIR = os.path.join(BASE_DIR, "data_sources")
HEALTH_RECORDS_CSV = os.path.join(DATA_SOURCES_DIR, "health_records.csv")
ZONE_ATTRIBUTES_CSV = os.path.join(DATA_SOURCES_DIR, "zone_attributes.csv")
ZONE_GEOMETRIES_GEOJSON = os.path.join(DATA_SOURCES_DIR, "zone_geometries.geojson")
IOT_CLINIC_ENVIRONMENT_CSV = os.path.join(DATA_SOURCES_DIR, "iot_clinic_environment.csv")

# Assets directory
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STYLE_CSS_PATH = os.path.join(ASSETS_DIR, "style.css") # Renamed to avoid conflict if STYLE_CSS becomes a dict
APP_LOGO = os.path.join(ASSETS_DIR, "logo.png") # Ensure this logo.png exists in assets

# App Settings
APP_TITLE = "Community Health Intelligence Hub"
APP_VERSION = "2.0.2" # Incremented version
APP_FOOTER = f"© {pd.Timestamp('now').year} Health Informatics Initiative. All Rights Reserved."
CONTACT_EMAIL = "synergydx.mystrikinglycom"
CACHE_TTL_SECONDS = 3600 # Cache data for 1 hour

# Dashboard specific settings
DEFAULT_DATE_RANGE_DAYS_VIEW = 1 # For single day views like CHW daily tasks
DEFAULT_DATE_RANGE_DAYS_TREND = 30 # For trend charts
RISK_THRESHOLDS = {
    "high": 75,
    "moderate": 60,
    "low": 40, 
    "chw_alert_high": 80, 
    "chw_alert_moderate": 65, 
    "district_zone_high_risk": 70 
}
CRITICAL_SUPPLY_DAYS = 10 
TARGET_TEST_TURNAROUND_DAYS = 2 
TARGET_PATIENT_RISK_SCORE = 50 

# Disease-Specific Targets/Thresholds
TARGET_TB_CASE_DETECTION_RATE = 85 
TARGET_MALARIA_POSITIVITY_RATE = 5 
TARGET_HIV_LINKAGE_TO_CARE = 90 
TARGET_HPV_SCREENING_COVERAGE = 70 
TARGET_ANEMIA_PREVALENCE_WOMEN = 15 
PNEUMONIA_CASE_FATALITY_TARGET = 5 
STI_SYNDROMIC_MANAGEMENT_ACCURACY = 90 

KEY_CONDITIONS_FOR_TRENDS = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'STI-Syphilis', 'STI-Gonorrhea', 'Anemia', 'Dengue', 'Hypertension', 'Diabetes', 'Wellness Visit']
CRITICAL_TESTS_PENDING = ['Sputum-AFB', 'Sputum-GeneXpert', 'HIV-Rapid', 'HIV-ViralLoad', 'RPR', 'NAAT-GC', 'PapSmear', 'Glucose Test']
KEY_DRUG_SUBSTRINGS_SUPPLY = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'Penicillin', 'Ceftriaxone', 'Iron-Folate', 'Amoxicillin', 'Metformin', 'Amlodipine', 'Insulin']

# Wearable/Sensor-Based Thresholds/Targets
SKIN_TEMP_FEVER_THRESHOLD_C = 38.0
SPO2_LOW_THRESHOLD_PCT = 94
SPO2_CRITICAL_THRESHOLD_PCT = 90 
TARGET_DAILY_STEPS = 8000
TARGET_SLEEP_HOURS = 7.0
TARGET_SLEEP_SCORE_PCT = 75 
STRESS_LEVEL_HIGH_THRESHOLD = 7 

# Clinic Environment IoT Thresholds/Targets
CO2_LEVEL_ALERT_PPM = 1000
CO2_LEVEL_IDEAL_PPM = 800 
PM25_ALERT_UGM3 = 25 
PM25_IDEAL_UGM3 = 12 
VOC_INDEX_ALERT = 200 
NOISE_LEVEL_ALERT_DB = 65 
TARGET_WAITING_ROOM_OCCUPANCY = 10 
TARGET_PATIENT_THROUGHPUT_PER_HOUR = 8 
TARGET_SANITIZER_DISPENSES_PER_HOUR_PER_DISPENSER = 5 

# Intervention thresholds
INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD = 60 
INTERVENTION_TB_BURDEN_HIGH_THRESHOLD = 5 
INTERVENTION_MALARIA_BURDEN_HIGH_THRESHOLD = 10 
INTERVENTION_PREVALENCE_HIGH_PERCENTILE = 0.75 

# Plotly specific settings
DEFAULT_PLOT_HEIGHT = 400 
COMPACT_PLOT_HEIGHT = 320 
MAP_PLOT_HEIGHT = 600 

# Map Configurations
TIJUANA_CENTER_LAT = 32.5149
TIJUANA_CENTER_LON = -117.0382
TIJUANA_DEFAULT_ZOOM = 10 
MAP_DEFAULT_CENTER_LAT = TIJUANA_CENTER_LAT
MAP_DEFAULT_CENTER_LON = TIJUANA_CENTER_LON
MAP_DEFAULT_ZOOM = TIJUANA_DEFAULT_ZOOM
DEFAULT_CRS = "EPSG:4326" 
MAPBOX_STYLE = "carto-positron"  

# Logging Configuration
LOG_LEVEL = "DEBUG" # Set to DEBUG to get more verbose logs for troubleshooting
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s' # Added funcName
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Color Palette
DISEASE_COLORS = {
    "TB": "#EF4444", "Malaria": "#F59E0B", "HIV-Positive": "#8B5CF6", "Pneumonia": "#3B82F6", 
    "Anemia": "#10B981", "STI": "#EC4899", "Dengue": "#6366F1", "Hypertension": "#F97316", 
    "Diabetes": "#0EA5E9", "Wellness Visit": "#84CC16", "Other": "#6B7280" 
}
RISK_STATUS_COLORS = {
    "High": "#EF4444", "Moderate": "#F59E0B", "Low": "#10B981", "Neutral": "#6B7280"
}
