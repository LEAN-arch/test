# health_hub/app_home.py
import streamlit as st
import os
import pandas as pd 
from config import app_config 
import logging 

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title=f"{app_config.APP_TITLE} - Home",
   
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
logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO),
                    format=app_config.LOG_FORMAT,
                    datefmt=app_config.LOG_DATE_FORMAT) 
logger = logging.getLogger(__name__) 

# --- Function to load CSS ---
@st.cache_resource
def load_css(_config_obj_page): # Use a unique param name if desired
    if os.path.exists(_config_obj_page.STYLE_CSS_PATH):
        with open(_config_obj_page.STYLE_CSS_PATH) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.info(f"{_config_obj_page.APP_TITLE} Page: CSS loaded.") # Example of using another attr
    else:
        logger.warning(f"CSS file not found at {_config_obj_page.STYLE_CSS_PATH} for this page.")

# Load custom CSS from configured path
load_css(app_config) # <--- CORRECTED HERE


# --- App Header ---
header_cols = st.columns([0.08, 0.92]) 

       

with header_cols[1]:
    st.title(app_config.APP_TITLE)
    st.caption(f"Version {app_config.APP_VERSION}  |  Empowering Health Decisions with Data-Driven Insights")

st.markdown("---") 

# --- App Introduction ---
st.markdown(
    """
    #### Welcome to the Health Intelligence Hub!
    This platform is designed to provide actionable, real-time intelligence to enhance community well-being
    and optimize public health interventions. Our suite of tools leverages advanced data analytics and
    interactive visualizations to support informed decision-making across all levels of the health system.
    """
)
st.success("👈 **Please select a specialized dashboard from the sidebar navigation to explore tailored insights and tools.**")


st.subheader("Explore Role-Specific Dashboards:")

with st.expander("🧑‍⚕️ **Community Health Worker (CHW) Dashboard** - Daily field operations and patient prioritization.", expanded=False):
    st.markdown("""
    - **Focus:** Field-level insights, patient tracking, alert prioritization, and efficient task management for proactive community care.
    - **Key Features:** Daily task lists, high-risk patient identification, wellness monitoring (SpO2, fever, activity levels), and referral tracking.
    - **Objective:** Equip CHWs with timely, actionable information to deliver targeted interventions and improve patient outcomes at the household level.
    """)
    if st.button("Go to CHW Dashboard", key="nav_chw_home_button_final_v2", type="primary"): # Key incremented
        st.switch_page("pages/1_chw_dashboard.py")


with st.expander("🏥 **Clinic Operations Dashboard** - Facility-level performance and resource management.", expanded=False):
    st.markdown("""
    - **Focus:** Monitoring operational efficiency, diagnostic performance, supply chain integrity, and quality of care metrics within health facilities.
    - **Key Features:** Test positivity rates, turnaround times, critical drug stock levels, patient load analysis, and clinic environmental health monitoring.
    - **Objective:** Enable clinic managers to optimize resource utilization, enhance service delivery quality, and ensure patient safety and satisfaction.
    """)
    if st.button("Go to Clinic Dashboard", key="nav_clinic_home_button_final_v2", type="primary"):
        st.switch_page("pages/2_clinic_dashboard.py")

with st.expander("🗺️ **District Health Officer (DHO) Dashboard** - Strategic population health oversight.", expanded=False):
    st.markdown("""
    - **Focus:** Analyzing population health trends, disease hotspots, intervention impacts, and strategic resource allocation across the district.
    - **Key Features:** Geospatial mapping of health indicators, zonal comparisons, burden of disease analysis, and data-driven tools for intervention planning.
    - **Objective:** Provide DHOs with a comprehensive strategic overview to guide public health policy, optimize resource deployment, and lead targeted health initiatives.
    """)
    if st.button("Go to District Dashboard", key="nav_dho_home_button_final_v2", type="primary"):
        st.switch_page("pages/3_district_dashboard.py")


st.markdown("---")

# --- Platform Capabilities Overview ---
st.subheader("Platform Capabilities at a Glance")
capabilities_cols = st.columns(3)
with capabilities_cols[0]:
    st.markdown("##### 📈 Trend Analysis & Forecasting")
    st.markdown("<small>Visualize health indicators over time, identify emerging patterns, and forecast future needs for proactive planning.</small>", unsafe_allow_html=True)
    st.markdown("##### 📊 Comparative Analytics")
    st.markdown("<small>Benchmark performance across zones, clinics, or demographic groups to identify best practices and areas for improvement.</small>", unsafe_allow_html=True)
with capabilities_cols[1]:
    st.markdown("##### 🗺️ Geospatial Intelligence")
    st.markdown("<small>Map disease hotspots, resource distribution, access barriers, and environmental factors affecting health outcomes.</small>", unsafe_allow_html=True)
    st.markdown("##### 💊 Resource Management")
    st.markdown("<small>Monitor and forecast essential medical supplies, track equipment utilization, and optimize staffing levels based on demand.</small>", unsafe_allow_html=True)
with capabilities_cols[2]:
    st.markdown("##### 🔔 Proactive Alerting Systems")
    st.markdown("<small>Identify high-risk patients, critical supply shortages, and emerging public health threats for timely intervention.</small>", unsafe_allow_html=True)
    st.markdown("##### 🎯 Intervention Planning & Evaluation")
    st.markdown("<small>Utilize data to design, target, monitor, and evaluate the impact of public health interventions effectively and efficiently.</small>", unsafe_allow_html=True)


# --- Sidebar Content Customization ---
st.sidebar.markdown("---") 
st.sidebar.image("assets//DNA-DxBrand.png", width=300)
if os.path.exists(app_config.APP_LOGO):
    st.sidebar.image(app_config.APP_LOGO, width=60, use_column_width='auto')
    st.sidebar.image("assets//DNA-DxBrand.png", width=400)
st.sidebar.header(f"{app_config.APP_TITLE.split(' ')[0]} Hub") 
st.sidebar.caption(f"Version {app_config.APP_VERSION}")
st.sidebar.markdown("---")
st.sidebar.caption(app_config.APP_FOOTER)
st.sidebar.markdown(f"Need help? Contact Support:<br/>[{app_config.CONTACT_EMAIL}](mailto:{app_config.CONTACT_EMAIL})", unsafe_allow_html=True)
logger.info("Application home page loaded successfully.")
