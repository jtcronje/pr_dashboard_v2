import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add the current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from utils.data import load_or_generate_data, apply_filters
from components.filters import render_filter_pane
from utils.charts import create_kpi_summary, create_headcount_trend, create_worker_type_breakdown, create_farm_map

# Configure the Streamlit page
st.set_page_config(
    page_title="People Radar Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for styling
st.markdown("""
<style>
    :root {
        --background-color: #F8F9FA;
        --text-color: #212121;
        --primary-color: #0D47A1;
        --secondary-color: #1976D2;
        --accent-color: #2196F3;
        --axis-color: #000000;  /* Changed to black for all axes */
        --axis-title-color: #000000;  /* Black for axis titles */
        --table-header-bg: #0D47A1;  /* Primary color for table headers */
        --table-header-text: #FFFFFF;  /* White text for table headers */
        --table-border: #E0E0E0;  /* Light gray for table borders */
        --table-row-hover: #F5F9FF;  /* Very light blue for row hover */
        --table-row-alt: #F8FAFD;  /* Slightly blue-tinted white for alternating rows */
    }
    
    /* Global text color override */
    * {
        color: #000000 !important;
    }
    
    /* Body styling */
    body {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: 'Roboto', sans-serif;
    }
    
    /* Main content area */
    .main {
        background-color: var(--background-color);
    }
    
    /* Sidebar styling - make background white */
    .sidebar .sidebar-content,
    [data-testid="stSidebar"],
    .st-emotion-cache-16txtl3,
    .st-emotion-cache-1cypcdb,
    .st-emotion-cache-1k8n3ey,
    .st-emotion-cache-1wbqh2d {
        background-color: white !important;
    }
    
    /* Modern Table Styling */
    /* Table container styling */
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"],
    .stDataFrame,
    .stTable,
    div.stDataFrame,
    div.stTable,
    div[data-testid="dataFrameVizContainer"] {
        background-color: white !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05) !important;
        border: none !important;
    }
    
    /* Table styling */
    table, 
    .dataframe, 
    [data-testid="StyledDataFrame"],
    div[data-testid="stTable"] table {
        background-color: white !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border: 1px solid var(--table-border) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        width: 100% !important;
        margin-bottom: 1rem !important;
    }
    
    /* Table headers */
    table th,
    .dataframe th, 
    [data-testid="StyledDataFrame"] th, 
    div[data-testid="stTable"] th,
    .stDataFrame th,
    .stTable th,
    div.stDataFrame th,
    div.stTable th,
    div[data-testid="stDataFrame"] th {
        background-color: var(--primary-color) !important;
        color: white !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.5px !important;
        padding: 12px 15px !important;
        text-align: left !important;
        border: none !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 1 !important;
    }
    
    /* Override the global color for table headers to ensure white text */
    table th *,
    .dataframe th *, 
    [data-testid="StyledDataFrame"] th *, 
    div[data-testid="stTable"] th *,
    .stDataFrame th *,
    .stTable th *,
    div.stDataFrame th *,
    div.stTable th *,
    div[data-testid="stDataFrame"] th * {
        color: white !important;
    }
    
    /* Table cells */
    table td,
    .dataframe td, 
    [data-testid="StyledDataFrame"] td, 
    div[data-testid="stTable"] td,
    .stDataFrame td,
    .stTable td,
    div.stDataFrame td,
    div.stTable td,
    div[data-testid="stDataFrame"] td {
        background-color: white !important;
        color: var(--text-color) !important;
        padding: 10px 15px !important;
        border-bottom: 1px solid var(--table-border) !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
        font-size: 0.9rem !important;
        vertical-align: middle !important;
    }
    
    /* Alternating row colors for better readability */
    table tr:nth-child(even) td,
    .dataframe tr:nth-child(even) td,
    [data-testid="StyledDataFrame"] tr:nth-child(even) td,
    div[data-testid="stTable"] tr:nth-child(even) td,
    .stDataFrame tr:nth-child(even) td,
    .stTable tr:nth-child(even) td,
    div.stDataFrame tr:nth-child(even) td,
    div.stTable tr:nth-child(even) td,
    div[data-testid="stDataFrame"] tr:nth-child(even) td {
        background-color: var(--table-row-alt) !important;
    }
    
    /* Hover effect for rows */
    table tr:hover td,
    .dataframe tr:hover td,
    [data-testid="StyledDataFrame"] tr:hover td,
    div[data-testid="stTable"] tr:hover td,
    .stDataFrame tr:hover td,
    .stTable tr:hover td,
    div.stDataFrame tr:hover td,
    div.stTable tr:hover td,
    div[data-testid="stDataFrame"] tr:hover td {
        background-color: var(--table-row-hover) !important;
        transition: background-color 0.2s ease !important;
    }
    
    /* Ensure table containers have white background */
    div[data-testid="dataFrameVizContainer"],
    div[data-testid="dataFrameVizContainer"] > div {
        background-color: white !important;
    }
    
    /* Chart styling - ensure axes and titles are black */
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text {
        fill: #000000 !important;
        color: #000000 !important;
        font-family: inherit !important;
        font-weight: 500 !important;
    }
    
    /* Axis title styling */
    .js-plotly-plot .plotly .xaxis-title,
    .js-plotly-plot .plotly .yaxis-title {
        fill: #000000 !important;
        color: #000000 !important;
        font-family: inherit !important;
        font-weight: 700 !important;
    }
    
    /* Chart title styling */
    .js-plotly-plot .plotly .gtitle {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Make chart backgrounds transparent */
    .js-plotly-plot .plotly .main-svg {
        background-color: transparent !important;
    }
    
    .js-plotly-plot .plotly .plot-container {
        background-color: transparent !important;
    }
    
    .js-plotly-plot .plotly .svg-container {
        background-color: transparent !important;
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color) !important;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton button {
        background-color: white;
        border: 1px solid #EEEEEE;
        color: var(--text-color) !important;
    }
    
    .stButton button:hover {
        border-color: var(--primary-color);
        color: var(--primary-color) !important;
    }
    
    /* Streamlit elements background fix */
    div[data-testid="stToolbar"], div[data-testid="stDecoration"] {
        background-color: var(--background-color);
    }
    
    /* Sidebar navigation buttons */
    .stSidebar [data-testid="stVerticalBlock"] button {
        color: var(--text-color) !important;
    }
    
    /* Ensure text inputs and selects have proper contrast */
    .stTextInput input, .stSelectbox select {
        color: var(--text-color) !important;
    }
    
    /* Ensure all text in the app is visible */
    body, .stApp {
        color-scheme: light !important;
    }
    
    /* Additional CSS reset for Streamlit styled tables */
    div.stDataFrame table,
    div[data-testid="stDataFrame"] table,
    div[data-testid="dataFrameVizContainer"] table,
    div.stDataFrame tbody,
    div[data-testid="stDataFrame"] tbody,
    div[data-testid="dataFrameVizContainer"] tbody,
    div.stDataFrame tr,
    div[data-testid="stDataFrame"] tr,
    div[data-testid="dataFrameVizContainer"] tr,
    div.stDataFrame th,
    div[data-testid="stDataFrame"] th,
    div[data-testid="dataFrameVizContainer"] th,
    div.stDataFrame td,
    div[data-testid="stDataFrame"] td,
    div[data-testid="dataFrameVizContainer"] td,
    div.stDataFrame div,
    div[data-testid="stDataFrame"] div,
    div[data-testid="dataFrameVizContainer"] div {
        background-color: transparent !important;
        color: #000000 !important;
    }
    
    /* Force all Streamlit iframe contents to have transparent background */
    iframe.stIframe,
    iframe.stIframe html,
    iframe.stIframe body,
    iframe.stIframe div {
        background-color: transparent !important;
    }
    
    /* Additional styling for pandas styled DataFrames */
    div[data-testid="dataFrameVizContainer"] table,
    div[data-testid="stDataFrame"] table {
        background-color: white !important;
        border: 1px solid #000000 !important;
    }
    
    /* Override any pandas styling classes */
    [class*="col"], [class*="row"], [class*="level"] {
        background-color: white !important;
        color: #000000 !important;
    }
</style>

<!-- JavaScript to enhance table styling -->
<script>
    // Function to apply consistent styling to all tables
    function enhanceTableStyling() {
        // Wait for tables to be rendered
        setTimeout(function() {
            const tables = document.querySelectorAll('table');
            
            tables.forEach(table => {
                // Add a class for custom styling
                table.classList.add('enhanced-table');
                
                // Ensure header cells have proper styling
                const headerCells = table.querySelectorAll('th');
                headerCells.forEach(cell => {
                    cell.style.backgroundColor = '#0D47A1';
                    cell.style.color = 'white';
                    cell.style.fontWeight = '500';
                    cell.style.textTransform = 'uppercase';
                    cell.style.fontSize = '0.85rem';
                    cell.style.letterSpacing = '0.5px';
                    cell.style.padding = '12px 15px';
                    
                    // Make sure all child elements have white text
                    const childElements = cell.querySelectorAll('*');
                    childElements.forEach(el => {
                        el.style.color = 'white';
                    });
                });
                
                // Style table rows for zebra striping and hover effects
                const rows = table.querySelectorAll('tbody tr');
                rows.forEach((row, index) => {
                    // Add hover event listeners
                    row.addEventListener('mouseenter', function() {
                        const cells = this.querySelectorAll('td');
                        cells.forEach(cell => {
                            cell.style.backgroundColor = '#F5F9FF';
                            cell.style.transition = 'background-color 0.2s ease';
                        });
                    });
                    
                    row.addEventListener('mouseleave', function() {
                        const cells = this.querySelectorAll('td');
                        cells.forEach((cell, cellIndex) => {
                            if (index % 2 === 1) {
                                cell.style.backgroundColor = '#F8FAFD';
                            } else {
                                cell.style.backgroundColor = 'white';
                            }
                        });
                    });
                    
                    // Apply zebra striping
                    if (index % 2 === 1) {
                        const cells = row.querySelectorAll('td');
                        cells.forEach(cell => {
                            cell.style.backgroundColor = '#F8FAFD';
                        });
                    }
                });
            });
        }, 1000); // Wait 1 second for tables to render
    }
    
    // Run the function when the DOM is fully loaded
    document.addEventListener('DOMContentLoaded', enhanceTableStyling);
    
    // Also run when Streamlit redraws the page
    const observer = new MutationObserver(function(mutations) {
        enhanceTableStyling();
    });
    
    // Start observing the document body for changes
    observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Overview'

# Define pages
PAGES = [
    'Overview',
    'Demographics',
    'Payroll Analysis',
    'Leave Analysis',
    'Productivity Analysis',
    'Employee Details',
    'Workforce Scenario',
    'Employee Barometer'
]

def load_data():
    """
    Load or generate the data needed for the dashboard.
    Returns a dictionary of DataFrames.
    """
    # Check if data is already in session state
    if 'data' not in st.session_state:
        with st.spinner('Loading data...'):
            # Load or generate data
            data_dict = load_or_generate_data(num_employees=1000)
            
            # Store in session state
            st.session_state.data = data_dict
            
            st.success('Data loaded successfully!')
    
    return st.session_state.data

def render_sidebar():
    """
    Render the sidebar navigation and filters.
    Returns the selected filters.
    """
    # Title and logo
    st.sidebar.title("People Radar Dashboard")
    st.sidebar.image("PeopleRadarLogo.jpeg", use_column_width=True)
    
    # Navigation buttons
    st.sidebar.markdown("## Navigation")
    
    # Create navigation buttons
    for page in PAGES:
        if st.sidebar.button(page, key=f"nav_{page}", 
                            use_container_width=True,
                            help=f"Navigate to {page} page"):
            st.session_state.current_page = page
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Render filter pane and get selected filters
    filters = render_filter_pane(st.session_state.data)
    
    return filters

# Import page modules
from pages.overview import render_overview_page
from pages.demographics import render_demographics_page
from pages.payroll import render_payroll_page
from pages.productivity import render_productivity_page
from pages.employee_detail import render_employee_detail_page
from pages.workforce_scenario import render_workforce_scenario_page
from pages.barometer import render_barometer_page
from pages.leave import render_leave_page

def render_header(title):
    """Render the page header with logo."""
    header_col1, header_col2 = st.columns([3, 1])
    
    with header_col1:
        st.title(title)
    
    with header_col2:
        st.image("PeopleRadarLogo.jpeg", width=150)

def main():
    """Main application function."""
    # Load data
    data_dict = load_data()
    
    # Render sidebar and get filters
    filters = render_sidebar()
    
    # Render selected page
    if st.session_state.current_page == 'Overview':
        # render_header("People Radar")
        render_overview_page(data_dict, filters)
    elif st.session_state.current_page == 'Demographics':
        render_header("Demographic Overview")
        render_demographics_page(data_dict, filters)
    elif st.session_state.current_page == 'Payroll Analysis':
        render_header("Payroll Analysis")
        render_payroll_page(data_dict, filters)
    elif st.session_state.current_page == 'Leave Analysis':
        render_header("Leave Analysis")
        render_leave_page(data_dict, filters)
    elif st.session_state.current_page == 'Productivity Analysis':
        render_header("Productivity Analysis")
        render_productivity_page(data_dict, filters)
    elif st.session_state.current_page == 'Employee Details':
        render_header("Employee Details")
        render_employee_detail_page(data_dict, filters)
    elif st.session_state.current_page == 'Workforce Scenario':
        render_header("Workforce Scenario")
        render_workforce_scenario_page(data_dict, filters)
    elif st.session_state.current_page == 'Employee Barometer':
        render_header("Employee Barometer")
        render_barometer_page(data_dict, filters)

if __name__ == "__main__":
    main()