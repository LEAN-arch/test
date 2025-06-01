# test/config/app_config.py
import os
import pandas as pd

# Base directory calculation: Assumes this config file is in 'test/config/'
# So, os.path.dirname(__file__) is 'test/config/'
# os.path.dirname(os.path.dirname(__file__)) is 'test/' which is our project root here.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data sources directory relative to BASE_DIR (which is 'test/')
DATA_SOURCES_DIR = os.path.join(BASE_DIR, "data_sources")
HEALTH_RECORDS_CSV = os.path.join(DATA_SOURCES_DIR, "health_records_expanded.csv")
ZONE_ATTRIBUTES_CSV = os.path.join(DATA_SOURCES_DIR, "zone_attributes.csv")
ZONE_GEOMETRIES_GEOJSON = os.path.join(DATA_SOURCES_DIR, "zone_geometries.geojson")
IOT_CLINIC_ENVIRONMENT_CSV = os.path.join(DATA_SOURCES_DIR, "iot_clinic_environment.csv")

# Assets directory relative to BASE_DIR
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STYLE_CSS_PATH = os.path.join(ASSETS_DIR, "style.css")
APP_LOGO = os.path.join(ASSETS_DIR, "DNA-DxBrand.png") # Make sure this image exists or use placeholder

# App Settings
APP_TITLE = "Community Health Intelligence Hub"
APP_VERSION = "2.1.1"
APP_FOOTER = f"© {pd.Timestamp('now').year} SynergyDx Health Informatics Initiative. All Rights Reserved. For Demonstration Purposes Only."
CONTACT_EMAIL = "https://synergydx.mystrikingly.com/"
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

TARGET_TEST_TURNAROUND_DAYS = 2
TARGET_PATIENT_RISK_SCORE = 50

TARGET_TB_CASE_DETECTION_RATE = 85
TARGET_MALARIA_POSITIVITY_RATE = 5
TARGET_HIV_LINKAGE_TO_CARE = 90
TARGET_HPV_SCREENING_COVERAGE = 70
TARGET_ANEMIA_PREVALENCE_WOMEN = 15
PNEUMONIA_CASE_FATALITY_TARGET = 5
STI_SYNDROMIC_MANAGEMENT_ACCURACY = 90

KEY_TEST_TYPES_FOR_ANALYSIS = {
    "Sputum-AFB": {"disease_group": "TB", "target_tat_days": 2, "critical": True, "display_name": "TB Sputum (AFB)"},
    "Sputum-GeneXpert": {"disease_group": "TB", "target_tat_days": 1, "critical": True, "display_name": "TB GeneXpert"},
    "RDT-Malaria": {"disease_group": "Malaria", "target_tat_days": 0.5, "critical": True, "display_name": "Malaria RDT"},
    "Microscopy-Malaria": {"disease_group": "Malaria", "target_tat_days": 1, "critical": False, "display_name": "Malaria Microscopy"},
    "HIV-Rapid": {"disease_group": "HIV", "target_tat_days": 0.25, "critical": True, "display_name": "HIV Rapid Test"},
    "HIV-ViralLoad": {"disease_group": "HIV", "target_tat_days": 7, "critical": True, "display_name": "HIV Viral Load"},
    "RPR": {"disease_group": "STI", "test_for": "Syphilis", "target_tat_days": 1, "critical": True, "display_name": "Syphilis RPR"},
    "NAAT-GC": {"disease_group": "STI", "test_for": "Gonorrhea", "target_tat_days": 3, "critical": True, "display_name": "Gonorrhea NAAT"},
    "PapSmear": {"disease_group": "CervicalCancer", "test_for": "HPV/Cancer", "target_tat_days": 14, "critical": False, "display_name": "Pap Smear"},
    "Glucose Test": {"disease_group": "Diabetes", "target_tat_days": 0.25, "critical": False, "display_name": "Glucose Test"},
    "Hemoglobin Test": {"disease_group": "Anemia", "target_tat_days": 0.25, "critical": False, "display_name": "Hemoglobin Test"},
    "Chest X-Ray": {"disease_group": "TB/Pneumonia", "target_tat_days": 1, "critical": False, "display_name": "Chest X-Ray"},
    "CD4 Count": {"disease_group": "HIV", "target_tat_days": 3, "critical": False, "display_name": "CD4 Count"},
    "Follow-up TB": {"disease_group": "TB", "target_tat_days": 0, "critical": False, "display_name": "Follow-up TB"},
    "General Checkup": {"disease_group": "Wellness", "target_tat_days": 1, "critical": False, "display_name": "General Checkup"},
    "Mental Health Screen": {"disease_group": "MentalHealth", "target_tat_days": 0.5, "critical": False, "display_name": "Mental Health Screen"},
    "Random Blood Sugar": {"disease_group": "Diabetes", "target_tat_days": 0.25, "critical": False, "display_name": "Random Blood Sugar"},
    "BP Check": {"disease_group": "Hypertension", "target_tat_days": 0, "critical": False, "display_name": "BP Check"},
}
CRITICAL_TESTS_LIST = [test_key for test_key, props in KEY_TEST_TYPES_FOR_ANALYSIS.items() if props.get("critical")] # Use keys
TARGET_OVERALL_TESTS_MEETING_TAT_PCT = 85
TARGET_SAMPLE_REJECTION_RATE_PCT = 5
OVERDUE_PENDING_TEST_DAYS = 7 # General fallback if test-specific TAT isn't found or not applicable

KEY_CONDITIONS_FOR_TRENDS = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'STI-Syphilis', 'STI-Gonorrhea', 'Anemia', 'Dengue', 'Hypertension', 'Diabetes', 'Wellness Visit', 'Anxiety', 'New Patient']
KEY_DRUG_SUBSTRINGS_SUPPLY = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'Penicillin', 'Ceftriaxone', 'Iron-Folate', 'Amoxicillin', 'Metformin', 'Amlodipine', 'Insulin', 'Co-amoxiclav', 'Calcium Supplement', 'Multivitamins', 'Iron Supplement']

SKIN_TEMP_FEVER_THRESHOLD_C = 38.0
SPO2_LOW_THRESHOLD_PCT = 94
SPO2_CRITICAL_THRESHOLD_PCT = 90
TARGET_DAILY_STEPS = 8000
TARGET_SLEEP_HOURS = 7.0
TARGET_SLEEP_SCORE_PCT = 75
STRESS_LEVEL_HIGH_THRESHOLD = 7

CO2_LEVEL_ALERT_PPM = 1000
CO2_LEVEL_IDEAL_PPM = 800
PM25_ALERT_UGM3 = 25
PM25_IDEAL_UGM3 = 12
VOC_INDEX_ALERT = 200
NOISE_LEVEL_ALERT_DB = 65
TARGET_WAITING_ROOM_OCCUPANCY = 10
TARGET_PATIENT_THROUGHPUT_PER_HOUR = 8
TARGET_SANITIZER_DISPENSES_PER_HOUR_PER_DISPENSER = 5

INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD = 60
INTERVENTION_TB_BURDEN_HIGH_THRESHOLD = 5
INTERVENTION_MALARIA_BURDEN_HIGH_THRESHOLD = 10
INTERVENTION_PREVALENCE_HIGH_PERCENTILE = 0.75

DEFAULT_PLOT_HEIGHT = 400
COMPACT_PLOT_HEIGHT = 320
MAP_PLOT_HEIGHT = 600

TIJUANA_CENTER_LAT = 32.5149
TIJUANA_CENTER_LON = -117.0382
TIJUANA_DEFAULT_ZOOM = 10
MAP_DEFAULT_CENTER_LAT = TIJUANA_CENTER_LAT
MAP_DEFAULT_CENTER_LON = TIJUANA_CENTER_LON
MAP_DEFAULT_ZOOM = TIJUANA_DEFAULT_ZOOM
DEFAULT_CRS = "EPSG:4326"
MAPBOX_STYLE = "carto-positron"

LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

DISEASE_COLORS = {
    "TB": "#EF4444", "Malaria": "#F59E0B", "HIV-Positive": "#8B5CF6",
    "Pneumonia": "#3B82F6", "Anemia": "#10B981", "STI": "#EC4899", "Dengue": "#6366F1",
    "Hypertension": "#F97316", "Diabetes": "#0EA5E9", "Wellness Visit": "#84CC16",
    "CervicalCancer": "#d6336c", "MentalHealth": "#fbbf24", "Anxiety": "#fbbf24",
    "New Patient": "#4ade80",
    "STI-Syphilis": "#c026d3", "STI-Gonorrhea": "#db2777", "STI-Chlamydia": "#e11d48",
    "Other": "#6B7280",
}
RISK_STATUS_COLORS = {"High": "#EF4444", "Moderate": "#F59E0B", "Low": "#10B981", "Neutral": "#6B7280"}
