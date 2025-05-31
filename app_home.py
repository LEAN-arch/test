# test/app_home.py
import streamlit as st
import os
import pandas as pd # Though not used directly in this version's display logic
from config import app_config 
import logging

# --- Page Configuration ---
# Define a main page logo path in app_config, or hardcode here for demo
# For this example, let's assume MAIN_PAGE_APP_LOGO could be different
# If it's the same, app_config.APP_LOGO would be used.
MAIN_PAGE_APP_LOGO_PATH = os.path.join(app_config.ASSETS_DIR, "main_page_brand_logo.png") # Example for a different/larger logo
# Fallback to standard logo if main page specific one doesn't exist
if not os.path.exists(MAIN_PAGE_APP_LOGO_PATH):
    MAIN_PAGE_APP_LOGO_PATH = app_config.APP_LOGO


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
        This application is a demonstration platform... 
        """ # Shortened for brevity
    }
)

# --- Logging Setup ---
logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO), format=app_config.LOG_FORMAT, datefmt=app_config.LOG_DATE_FORMAT, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__) 

# --- CSS Loading ---
@st.cache_resource
def load_css(css_file_path: str):
    if os.path.exists(css_file_path):
        try:
            with open(css_file_path) as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info(f"CSS loaded from {css_file_path}")
        except Exception as e: logger.error(f"Error reading CSS {css_file_path}: {e}")
    else: logger.warning(f"CSS file not found: {css_file_path}.")
load_css(app_config.STYLE_CSS_PATH)

# --- App Header (with potentially different/larger main page logo) ---
header_cols = st.columns([0.15, 0.85]) # Give a bit more space for potentially larger logo
with header_cols[0]:
    if os.path.exists(MAIN_PAGE_APP_LOGO_PATH):
        st.image(MAIN_PAGE_APP_LOGO_PATH, width=120) # Larger width for main page logo
    else:
        logger.warning(f"Main page header logo not found at: {MAIN_PAGE_APP_LOGO_PATH}. Falling back to default or icon.")
        if os.path.exists(app_config.APP_LOGO):
            st.image(app_config.APP_LOGO, width=100) # Fallback to standard logo
        else:
            st.markdown("❤️", unsafe_allow_html=True)      
with header_cols[1]:
    st.title(app_config.APP_TITLE)
    st.caption(f"Version {app_config.APP_VERSION}  |  Empowering Health Decisions with Data-Driven Insights")
st.markdown("---") 

# --- App Introduction & Navigation Expanders ---
# (Content of Introduction unchanged)
st.markdown("""
    #### Welcome to the Health Intelligence Hub!
    This platform is designed to provide actionable, real-time intelligence...
""")
st.success("👈 **Please select a specialized dashboard from the sidebar navigation to explore tailored insights and tools.**")
st.subheader("Explore Role-Specific Dashboards:")

# (Expanders for CHW, Clinic, DHO dashboards remain the same)
with st.expander("🧑‍⚕️ **Community Health Worker (CHW) Dashboard** ...", expanded=False):
    # ... (content)
    if st.button("Go to CHW Dashboard", key="nav_chw_home_v8", type="primary"): st.switch_page("pages/1_chw_dashboard.py")

with st.expander("🏥 **Clinic Operations Dashboard** ...", expanded=False):
    # ... (content)
    if st.button("Go to Clinic Dashboard", key="nav_clinic_home_v8", type="primary"): st.switch_page("pages/2_clinic_dashboard.py")

with st.expander("🗺️ **District Health Officer (DHO) Dashboard** ...", expanded=False):
    # ... (content)
    if st.button("Go to District Dashboard", key="nav_dho_home_v8", type="primary"): st.switch_page("pages/3_district_dashboard.py")

# Updated Expander for the renamed Population Dashboard
with st.expander("📊 **Population Dashboard** - Deep-dive epidemiological & systems analysis.", expanded=True):
    st.markdown("""
    - **Focus:** Comprehensive analysis of demographic patterns, social determinants of health (SDOH), clinical and diagnostic trends, health system performance, and equity considerations across the entire population dataset.
    - **Key Features:** Stratified views of disease burden, AI risk score distributions by demographics/SDOH, aggregated test positivity trends, comorbidity analysis, and referral pathway overviews.
    - **Objective:** Offer a macro-level analytical tool for epidemiologists, program managers, and policymakers to understand population health dynamics and identify areas for strategic focus and research.
    """)
    if st.button("Go to Population Dashboard", key="nav_pop_dashboard_home_v2", type="primary"): # Updated key for new name
        st.switch_page("pages/4_Population_Dashboard.py") # Ensure this matches your new filename

st.markdown("---")
# ... (Platform Capabilities section unchanged from last good version) ...
st.subheader("Platform Capabilities at a Glance")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("##### 📈 **Epidemiological Intelligence**"); st.markdown("<small>Track disease incidence & prevalence...</small>", unsafe_allow_html=True)
    st.markdown("##### 🗺️ **Geospatial Analysis**"); st.markdown("<small>Visualize disease hotspots...</small>", unsafe_allow_html=True)
with col2:
    st.markdown("##### 🤖 **AI-Powered Prioritization & Alerts**"); st.markdown("<small>Leverage simulated AI for patient risk stratification...</small>", unsafe_allow_html=True)
    st.markdown("##### 💊 **Resource & Supply Chain Management**"); st.markdown("<small>Monitor critical medical supply levels...</small>", unsafe_allow_html=True)
with col3:
    st.markdown("##### 🎯 **Targeted Intervention & Systems Insights**"); st.markdown("<small>Utilize data-driven criteria...</small>", unsafe_allow_html=True)
    st.markdown("##### ❤️ **Equity-Focused Analytics**"); st.markdown("<small>Explore health outcomes and risk distributions...</small>", unsafe_allow_html=True)


# --- Sidebar Content Customization for app_home.py ---
st.sidebar.header(f"{app_config.APP_TITLE.split(' ')[0]} Hub Navigation") # General title
# Add the APP_LOGO to the app_home.py sidebar
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, width=230) # Using the smaller, consistent sidebar logo size
st.sidebar.caption(f"Version {app_config.APP_VERSION}")
st.sidebar.markdown("---")
st.sidebar.markdown("### About This Platform")
st.sidebar.info(
    "This is the main home page. Select a dashboard from the options above (or the auto-generated list from 'pages' folder) to get started."
)
st.sidebar.markdown("---")
st.sidebar.caption(app_config.APP_FOOTER)
st.sidebar.markdown(f"Need help? Contact Support:<br/>[{app_config.CONTACT_EMAIL}](mailto:{app_config.CONTACT_EMAIL})", unsafe_allow_html=True)

logger.info(f"Application home page ({app_config.APP_TITLE}) loaded successfully.")
