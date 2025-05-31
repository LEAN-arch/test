# test/app_home.py
import streamlit as st
import os
import pandas as pd
from config import app_config
import logging

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title=f"{app_config.APP_TITLE} - Home",
    page_icon=app_config.APP_LOGO if os.path.exists(app_config.APP_LOGO) else "❤️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': f"mailto:{app_config.CONTACT_EMAIL}?subject=Help Request - {app_config.APP_TITLE}",
        'Report a bug': f"mailto:{app_config.CONTACT_EMAIL}?subject=Bug Report - {app_config.APP_TITLE} v{app_config.APP_VERSION}",
        'About': f"""
        ### {app_config.APP_TITLE}
        **Version:** {app_config.APP_VERSION}

        {app_config.APP_FOOTER}

        This application is a demonstration platform for public health intelligence,
        utilizing synthetic data to showcase potential analytics and visualizations.
        It is intended for illustrative purposes only and should not be used for
        actual medical decision-making or clinical care.
        """
    }
)

# --- Logging Setup ---
logging.basicConfig(
    level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
    format=app_config.LOG_FORMAT,
    datefmt=app_config.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Function to load CSS ---
@st.cache_resource
def load_css(css_file_path: str):
    logger.debug(f"load_css trying to access CSS from: {css_file_path}")
    if os.path.exists(css_file_path):
        try:
            with open(css_file_path) as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
                logger.info(f"Successfully loaded CSS from {css_file_path}")
        except Exception as e:
            logger.error(f"Error reading CSS file {css_file_path}: {e}")
    else:
        logger.warning(f"CSS file not found at {css_file_path}. Styling may be affected.")
load_css(app_config.STYLE_CSS_PATH)

# --- App Header ---
header_cols = st.columns([0.08, 0.92])
with header_cols[0]:
    if os.path.exists(app_config.APP_LOGO): st.image(app_config.APP_LOGO, width=70)
    else: logger.warning(f"Header logo not found at: {app_config.APP_LOGO}"); st.markdown("❤️", unsafe_allow_html=True)
with header_cols[1]:
    st.title(app_config.APP_TITLE)
    st.caption(f"Version {app_config.APP_VERSION}  |  Empowering Health Decisions with Data-Driven Insights")
st.markdown("---")

# --- App Introduction ---
st.markdown("""
    #### Welcome to the Health Intelligence Hub!
    This platform is designed to provide actionable, real-time intelligence to enhance community well-being
    and optimize public health interventions. Our suite of tools leverages advanced data analytics and
    interactive visualizations to support informed decision-making across all levels of the health system.
""")
st.success("👈 **Please select a specialized dashboard from the sidebar navigation to explore tailored insights and tools.**")

st.subheader("Explore Role-Specific Dashboards:")
# ... (Expander sections for CHW, Clinic, DHO dashboards - unchanged from previous version)
with st.expander("🧑‍⚕️ **Community Health Worker (CHW) Dashboard** - Daily field operations and patient prioritization.", expanded=False):
    st.markdown("""
    - **Focus:** Field-level insights, patient tracking, AI-driven alert prioritization, and efficient task management for proactive community care. Includes local epidemiology snippets.
    - **Key Features:** Daily task lists, AI-based high-risk patient identification & follow-up priority, wellness monitoring (SpO2, fever, activity), referral tracking, simple symptom cluster indicators, and activity trends.
    - **Objective:** Equip CHWs with timely, actionable information to deliver targeted interventions and improve patient outcomes at the household level.
    """)
    if st.button("Go to CHW Dashboard", key="nav_chw_home_button_v6", type="primary"): st.switch_page("pages/1_chw_dashboard.py")

with st.expander("🏥 **Clinic Operations Dashboard** - Facility-level performance and resource management.", expanded=False):
    st.markdown("""
    - **Focus:** Monitoring operational efficiency, diagnostic performance, supply chain integrity (with optional AI forecasting), quality of care metrics, and local epidemiological patterns.
    - **Key Features:** Test positivity rates & trends, turnaround times, critical drug stock levels, AI-enhanced patient load analysis, clinic environmental health monitoring, syndromic surveillance, and demographic breakdown of cases.
    - **Objective:** Enable clinic managers to optimize resource utilization, enhance service delivery quality, identify local disease patterns, and ensure patient safety.
    """)
    if st.button("Go to Clinic Dashboard", key="nav_clinic_home_button_v6", type="primary"): st.switch_page("pages/2_clinic_dashboard.py")

with st.expander("🗺️ **District Health Officer (DHO) Dashboard** - Strategic population health oversight.", expanded=False):
    st.markdown("""
    - **Focus:** Analyzing population health trends, disease hotspots using AI-aggregated risk, intervention impacts, and strategic resource allocation across the district.
    - **Key Features:** Geospatial mapping of health indicators, zonal comparisons, burden of disease analysis, district-wide incidence trends, and data-driven tools for intervention planning.
    - **Objective:** Provide DHOs with a comprehensive strategic overview to guide public health policy, optimize resource deployment, and lead targeted health initiatives.
    """)
    if st.button("Go to District Dashboard", key="nav_dho_home_button_v6", type="primary"): st.switch_page("pages/3_district_dashboard.py")

with st.expander("📊 **Population Analytics Dashboard** - Deep-dive epidemiological and systems analysis.", expanded=True): # Expanded by default now
    st.markdown("""
    - **Focus:** Comprehensive analysis of demographic patterns, social determinants of health (SDOH), clinical and diagnostic trends, health system performance, and equity considerations across the entire population dataset.
    - **Key Features:** Stratified views of disease burden, AI risk score distributions by demographics/SDOH, aggregated test positivity trends, comorbidity analysis, and referral pathway overviews.
    - **Objective:** Offer a macro-level analytical tool for epidemiologists, program managers, and policymakers to understand population health dynamics and identify areas for strategic focus and research.
    """)
    if st.button("Go to Population Analytics", key="nav_pop_analytics_home_button_v1", type="primary"): st.switch_page("pages/4_population_analytics_dashboard.py")


st.markdown("---")

# --- UPDATED Platform Capabilities Overview ---
st.subheader("Platform Capabilities at a Glance")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 📈 **Epidemiological Intelligence**")
    st.markdown("<small>Track disease incidence and prevalence trends, monitor syndromic patterns, analyze test positivity rates, and understand demographic distributions of health conditions. View localized epi-snippets for CHWs and clinic-level patterns.</small>", unsafe_allow_html=True)
    
    st.markdown("##### 🗺️ **Geospatial Analysis**")
    st.markdown("<small>Visualize disease hotspots, map AI-derived population risk, analyze resource distribution, identify access barriers, and understand environmental health factors using interactive maps. Identify high-risk zones for targeted action.</small>", unsafe_allow_html=True)

with col2:
    st.markdown("##### 🤖 **AI-Powered Prioritization & Alerts**")
    st.markdown("<small>Leverage simulated AI models for patient risk stratification, follow-up prioritization for CHWs and clinics, and identification of high-risk cases needing clinical review. Receive alerts for critical health indicators.</small>", unsafe_allow_html=True)
    
    st.markdown("##### 💊 **Resource & Supply Chain Management**")
    st.markdown("<small>Monitor critical medical supply levels, utilize linear and AI-enhanced (simulated) forecasting for consumption, and identify potential stockouts to ensure continuity of care.</small>", unsafe_allow_html=True)
    
with col3:
    st.markdown("##### 🎯 **Targeted Intervention & Systems Insights**")
    st.markdown("<small>Utilize data-driven criteria to identify priority zones for public health interventions. Analyze referral pathway performance and explore contextual health systems data for strategic planning.</small>", unsafe_allow_html=True)

    st.markdown("##### ❤️ **Equity-Focused Analytics**")
    st.markdown("<small>Explore health outcomes and risk distributions in context of social determinants (e.g., socio-economic status by zone) to identify disparities and inform equitable health strategies (feature in Population Analytics).</small>", unsafe_allow_html=True)


# --- Sidebar Content Customization ---
# ... (Sidebar content remains unchanged from previous version) ...
st.sidebar.header(f"{app_config.APP_TITLE.split(' ')[0]} Hub Navigation")
st.sidebar.caption(f"Version {app_config.APP_VERSION}")
st.sidebar.markdown("---")
st.sidebar.markdown("### About This Platform")
st.sidebar.info("This platform provides tools for health professionals. Select a dashboard from the options above or the main list to get started.")
st.sidebar.markdown("---")
st.sidebar.caption(app_config.APP_FOOTER)
st.sidebar.markdown(f"Need help? Contact Support:<br/>[{app_config.CONTACT_EMAIL}](mailto:{app_config.CONTACT_EMAIL})", unsafe_allow_html=True)

logger.info(f"Application home page ({app_config.APP_TITLE}) loaded successfully.")
