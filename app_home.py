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
header_cols = st.columns([0.12, 0.88]) # Adjusted column ratio slightly for a potentially larger logo
with header_cols[0]:
    if os.path.exists(app_config.APP_LOGO):
        st.image(app_config.APP_LOGO, width=180) # << INCREASED LOGO WIDTH
    else:
        logger.warning(f"Header logo not found at: {app_config.APP_LOGO}")
        st.markdown("❤️", unsafe_allow_html=True) # Fallback emoji icon

with header_cols[1]:
    st.title(app_config.APP_TITLE)
    st.caption(f"Version {app_config.APP_VERSION}  |  Empowering Health Decisions with Data-Driven Insights")

st.markdown("---") 

# --- App Introduction ---
st.markdown("""
    #### Welcome to the Health Intelligence Hub!
    This platform provides actionable, real-time intelligence to enhance community well-being
    and optimize public health interventions using advanced data analytics and interactive visualizations.
""")
st.success("👈 **Select a specialized dashboard from the sidebar to explore tailored insights.**")

st.subheader("Explore Role-Specific Dashboards:")
with st.expander("🧑‍⚕️ **CHW Dashboard** - Daily field operations & patient prioritization.", expanded=False):
    st.markdown("""
    - **Focus:** Field-level insights, AI-driven alert prioritization, and task management for proactive community care. Includes local epi-snippets & activity trends.
    - **Key Features:** Daily task lists, AI high-risk patient identification & follow-up priority, wellness monitoring, referral tracking.
    - **Objective:** Equip CHWs with timely information for targeted interventions.
    """)
    if st.button("Go to CHW Dashboard", key="nav_chw_home_v7", type="primary"): st.switch_page("pages/1_chw_dashboard.py")

with st.expander("🏥 **Clinic Dashboard** - Facility performance & resource management.", expanded=False):
    st.markdown("""
    - **Focus:** Operational efficiency, diagnostic performance, supply chain (with AI forecast option), quality of care, and local epidemiological patterns.
    - **Key Features:** Test positivity & TAT trends, critical drug stocks, AI-enhanced patient load analysis, clinic environment monitoring, syndromic surveillance, demographic case breakdowns.
    - **Objective:** Enable clinic managers to optimize resources, enhance service quality, and identify local disease patterns.
    """)
    if st.button("Go to Clinic Dashboard", key="nav_clinic_home_v7", type="primary"): st.switch_page("pages/2_clinic_dashboard.py")

with st.expander("🗺️ **DHO Dashboard** - Strategic population health oversight.", expanded=False):
    st.markdown("""
    - **Focus:** Population health trends, AI-aggregated risk hotspots, intervention impacts, and strategic resource allocation.
    - **Key Features:** Geospatial mapping, zonal comparisons, disease burden analysis, district incidence trends, intervention planning tools.
    - **Objective:** Provide DHOs with a strategic overview for public health policy and resource deployment.
    """)
    if st.button("Go to District Dashboard", key="nav_dho_home_v7", type="primary"): st.switch_page("pages/3_district_dashboard.py")

with st.expander("📊 **Population Analytics Dashboard** - Deep-dive epidemiological & systems analysis.", expanded=True):
    st.markdown("""
    - **Focus:** Comprehensive analysis of demographic patterns, SDOH, clinical/diagnostic trends, health system performance, and equity insights.
    - **Key Features:** Stratified disease burden, AI risk by demographics/SDOH, test positivity trends, comorbidity analysis, referral overviews.
    - **Objective:** Offer macro-level analytical tools for epidemiologists, program managers, and policymakers.
    """)
    if st.button("Go to Population Analytics", key="nav_pop_analytics_home_v2", type="primary"): st.switch_page("pages/4_population_analytics_dashboard.py")
st.markdown("---")

st.subheader("Platform Capabilities at a Glance")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("##### 📈 **Epidemiological Intelligence**")
    st.markdown("<small>Track disease incidence & prevalence, monitor syndromic patterns, analyze test positivity rates, and understand demographic distributions of health conditions. View localized epi-snippets and clinic/district level patterns.</small>", unsafe_allow_html=True)
    st.markdown("##### 🗺️ **Geospatial Analysis**")
    st.markdown("<small>Visualize disease hotspots, map AI-derived population risk, analyze resource distribution, identify access barriers, and understand environmental health factors using interactive maps. Identify high-risk zones for targeted action.</small>", unsafe_allow_html=True)
with col2:
    st.markdown("##### 🤖 **AI-Powered Prioritization & Alerts**")
    st.markdown("<small>Leverage simulated AI for patient risk stratification, follow-up prioritization, and identification of high-risk cases needing clinical review. Receive alerts for critical health indicators.</small>", unsafe_allow_html=True)
    st.markdown("##### 💊 **Resource & Supply Chain Management**")
    st.markdown("<small>Monitor critical medical supply levels, utilize linear and AI-enhanced (simulated) forecasting for consumption, and identify potential stockouts to ensure continuity of care.</small>", unsafe_allow_html=True)
with col3:
    st.markdown("##### 🎯 **Targeted Intervention & Systems Insights**")
    st.markdown("<small>Utilize data-driven criteria to identify priority zones for interventions. Analyze referral pathway performance and explore contextual health systems data for strategic planning.</small>", unsafe_allow_html=True)
    st.markdown("##### ❤️ **Equity-Focused Analytics**")
    st.markdown("<small>Explore health outcomes and risk distributions in context of social determinants (e.g., socio-economic status by zone) to identify disparities and inform equitable health strategies.</small>", unsafe_allow_html=True)

# --- Sidebar Content Customization ---
st.sidebar.header(f"{app_config.APP_TITLE.split(' ')[0]} Hub Navigation")
st.sidebar.caption(f"Version {app_config.APP_VERSION}")
st.sidebar.markdown("---")
st.sidebar.markdown("### About This Platform")
st.sidebar.info("This platform provides tools for health professionals. Select a dashboard from the main page or sidebar navigation to get started.")
st.sidebar.markdown("---")
st.sidebar.caption(app_config.APP_FOOTER)
st.sidebar.markdown(f"Need help? Contact Support:<br/>[{app_config.CONTACT_EMAIL}](mailto:{app_config.CONTACT_EMAIL})", unsafe_allow_html=True)

logger.info(f"Application home page ({app_config.APP_TITLE}) loaded successfully.")
