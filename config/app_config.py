# health_hub/config/app_config.py
import os
import pandas as pd

# Base directory of the application
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # health_hub directory (or test/ in this structure)

# Data sources directory
DATA_SOURCES_DIR = os.path.join(BASE_DIR, "data_sources")
# Ensure this points to the EXPANDED CSV for the full app functionality
HEALTH_RECORDS_CSV = os.path.join(DATA_SOURCES_DIR, "health_records_expanded.csv") # Path to the expanded CSV
ZONE_ATTRIBUTES_CSV = os.path.join(DATA_SOURCES_DIR, "zone_attributes.csv")
ZONE_GEOMETRIES_GEOJSON = os.path.join(DATA_SOURCES_DIR, "zone_geometries.geojson")
IOT_CLINIC_ENVIRONMENT_CSV = os.path.join(DATA_SOURCES_DIR, "iot_clinic_environment.csv")

# Assets directory
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STYLE_CSS_PATH = os.path.join(ASSETS_DIR, "style.css")
APP_LOGO = os.path.join(ASSETS_DIR, "DNA-DxBrand.png") # Replace with your actual logo filename if different

# App Settings
APP_TITLE = "Community Health Intelligence Hub"
APP_VERSION = "2.1.1" # Updated version
APP_FOOTER = f"© {pd.Timestamp('now').year} Health Informatics Initiative. All Rights Reserved. For Demonstration Purposes Only."
CONTACT_EMAIL = "support@healthhub-demo.com"
CACHE_TTL_SECONDS = 3600 # 1 hour

# Dashboard specific settings
DEFAULT_DATE_RANGE_DAYS_VIEW = 1 # For daily views like CHW dashboard's specific date
DEFAULT_DATE_RANGE_DAYS_TREND = 30 # For trend charts
RISK_THRESHOLDS = {
    "high": 75, "moderate": 60, "low": 40, # General AI risk categories
    "chw_alert_high": 80, "chw_alert_moderate": 65, # CHW specific thresholds for general AI risk based alerts
    "district_zone_high_risk": 70 # Threshold for a zone to be considered "high-risk" by AI score
}
CRITICAL_SUPPLY_DAYS = 10 # Days of supply below which an item is considered critically low

TARGET_TEST_TURNAROUND_DAYS = 2 # Global default Target TAT in Days for non-specified tests
TARGET_PATIENT_RISK_SCORE = 50 # Ideal average patient risk score to aim below (example)

# Disease-Specific Targets/Thresholds (examples)
TARGET_TB_CASE_DETECTION_RATE = 85 # % (Hypothetical target for a program)
TARGET_MALARIA_POSITIVITY_RATE = 5 # % (Max acceptable rate in low transmission)
TARGET_HIV_LINKAGE_TO_CARE = 90 # % (Patients linked to care after diagnosis)
TARGET_HPV_SCREENING_COVERAGE = 70 # % (Target population screened)
TARGET_ANEMIA_PREVALENCE_WOMEN = 15 # % (Max acceptable prevalence in reproductive age women)
PNEUMONIA_CASE_FATALITY_TARGET = 5 # % (Max acceptable case fatality for pneumonia)
STI_SYNDROMIC_MANAGEMENT_ACCURACY = 90 # % (Correct management based on syndrome)


# Key Test Types for Analysis (Crucial for Clinic Dashboard and TAT calculations)
# 'display_name' is what users see in UI selectors.
# 'types_in_group' (optional): if display_name groups multiple raw test_type values. If not present, assumes display_name maps to original key.
# 'target_tat_days' is specific for this test/group.
# 'critical' flag indicates if it's a high-priority test for TAT monitoring.
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
    "CD4 Count": {"disease_group": "HIV", "target_tat_days": 3, "critical": False, "display_name": "CD4 Count"}, # Added from expanded data
    "Follow-up TB": {"disease_group": "TB", "target_tat_days": 0, "critical": False, "display_name": "Follow-up TB"}, # No TAT if just checkup
    "General Checkup": {"disease_group": "Wellness", "target_tat_days": 1, "critical": False, "display_name": "General Checkup"},
    "Mental Health Screen": {"disease_group": "MentalHealth", "target_tat_days": 0.5, "critical": False, "display_name": "Mental Health Screen"},
    "Random Blood Sugar": {"disease_group": "Diabetes", "target_tat_days": 0.25, "critical": False, "display_name": "Random Blood Sugar"},
}
# List of critical test original names (keys from above)
CRITICAL_TESTS_LIST = [test for test, props in KEY_TEST_TYPES_FOR_ANALYSIS.items() if props.get("critical")]
TARGET_OVERALL_TESTS_MEETING_TAT_PCT = 85 # Target for % of CRITICAL tests meeting their individual TATs
TARGET_SAMPLE_REJECTION_RATE_PCT = 5 # Max acceptable overall sample rejection rate
OVERDUE_PENDING_TEST_DAYS = 7 # General threshold for how many days a non-critical test can be pending before flagged (critical tests use their specific TAT + buffer)

KEY_CONDITIONS_FOR_TRENDS = ['TB', 'Malaria', 'HIV-Positive', 'Pneumonia', 'STI-Syphilis', 'STI-Gonorrhea', 'Anemia', 'Dengue', 'Hypertension', 'Diabetes', 'Wellness Visit', 'Anxiety', 'New Patient'] # Added from expanded data
KEY_DRUG_SUBSTRINGS_SUPPLY = ['TB-Regimen', 'ACT', 'ARV-Regimen', 'Penicillin', 'Ceftriaxone', 'Iron-Folate', 'Amoxicillin', 'Metformin', 'Amlodipine', 'Insulin', 'Co-amoxiclav', 'Calcium Supplement', 'Multivitamins', 'Iron Supplement'] # Added from expanded data

# CHW specific device/metric thresholds (from wearable data or CHW direct measurements)
SKIN_TEMP_FEVER_THRESHOLD_C = 38.0
SPO2_LOW_THRESHOLD_PCT = 94 # Alert if SpO2 drops below this
SPO2_CRITICAL_THRESHOLD_PCT = 90 # Critical alert for SpO2
TARGET_DAILY_STEPS = 8000
TARGET_SLEEP_HOURS = 7.0
TARGET_SLEEP_SCORE_PCT = 75
STRESS_LEVEL_HIGH_THRESHOLD = 7 # Example stress score (0-10 scale)

# Clinic Environment IoT Thresholds
CO2_LEVEL_ALERT_PPM = 1000 # Alert if CO2 exceeds this
CO2_LEVEL_IDEAL_PPM = 800  # Ideal max for CO2
PM25_ALERT_UGM3 = 25       # Alert if PM2.5 exceeds this (WHO guideline for 24h mean)
PM25_IDEAL_UGM3 = 12       # Ideal max for PM2.5
VOC_INDEX_ALERT = 200      # VOC Index alert threshold (scales vary, e.g. 0-500 from Sensirion)
NOISE_LEVEL_ALERT_DB = 65  # Alert for sustained noise in quiet areas
TARGET_WAITING_ROOM_OCCUPANCY = 10 # Max ideal persons for a standard waiting room
TARGET_PATIENT_THROUGHPUT_PER_HOUR = 8 # Per clinician/consult room
TARGET_SANITIZER_DISPENSES_PER_HOUR_PER_DISPENSER = 5 # Example

# District Dashboard / Intervention Planning Thresholds
INTERVENTION_FACILITY_COVERAGE_LOW_THRESHOLD = 60 # % score below which facility coverage is considered low
INTERVENTION_TB_BURDEN_HIGH_THRESHOLD = 5         # Number of absolute TB cases in a zone to be considered high burden for intervention
INTERVENTION_MALARIA_BURDEN_HIGH_THRESHOLD = 10   # Number of absolute Malaria cases in a zone
INTERVENTION_PREVALENCE_HIGH_PERCENTILE = 0.75    # Zones in top (1-X)% prevalence for key infections are flagged

# Plotting & Map Configuration
DEFAULT_PLOT_HEIGHT = 400
COMPACT_PLOT_HEIGHT = 320
MAP_PLOT_HEIGHT = 600 # Increased for better map display

# Map defaults (example: Tijuana) - adjust to your area of interest
TIJUANA_CENTER_LAT = 32.5149
TIJUANA_CENTER_LON = -117.0382
TIJUANA_DEFAULT_ZOOM = 10 # Zoom level for the general Tijuana area
MAP_DEFAULT_CENTER_LAT = TIJUANA_CENTER_LAT # App default can be set to a specific region
MAP_DEFAULT_CENTER_LON = TIJUANA_CENTER_LON
MAP_DEFAULT_ZOOM = TIJUANA_DEFAULT_ZOOM
DEFAULT_CRS = "EPSG:4326" # WGS84 standard for GeoJSON lat/lon
MAPBOX_STYLE = "carto-positron" # Light, open style; alternatives: "open-street-map", "stamen-terrain" or Mapbox custom styles if token set

# Logging Configuration
LOG_LEVEL = "INFO" # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# UI Theme Colors (for plot consistency if not overridden by CSS or Plotly theme)
# These are used by ui_visualization_helpers._get_theme_color if no template colorway is found
DISEASE_COLORS = {
    "TB": "#EF4444", "Malaria": "#F59E0B", "HIV-Positive": "#8B5CF6", # HIV distinct from generic STI
    "Pneumonia": "#3B82F6", "Anemia": "#10B981", "STI": "#EC4899", "Dengue": "#6366F1",
    "Hypertension": "#F97316", "Diabetes": "#0EA5E9", "Wellness Visit": "#84CC16",
    "CervicalCancer": "#d6336c", "MentalHealth": "#fbbf24", "Anxiety": "#fbbf24", # Added for expanded data
    "New Patient": "#4ade80",
    "STI-Syphilis": "#c026d3", "STI-Gonorrhea": "#db2777", "STI-Chlamydia": "#e11d48", # More specific STI colors
    "Other": "#6B7280", # Default fallback color
}
RISK_STATUS_COLORS = {"High": "#EF4444", "Moderate": "#F59E0B", "Low": "#10B981", "Neutral": "#6B7280"}
