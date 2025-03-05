import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Add the parent directory to sys.path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import get_unique_values

def initialize_session_state():
    """Initialize session state variables for filters if they don't exist."""
    # Date range - default to last 12 months
    if 'date_range' not in st.session_state:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        st.session_state.date_range = (start_date, end_date)
    
    # Filter selections
    if 'selected_departments' not in st.session_state:
        st.session_state.selected_departments = []
    
    if 'selected_locations' not in st.session_state:
        st.session_state.selected_locations = []
    
    if 'selected_job_titles' not in st.session_state:
        st.session_state.selected_job_titles = []
    
    if 'selected_genders' not in st.session_state:
        st.session_state.selected_genders = []
    
    if 'selected_ethnicities' not in st.session_state:
        st.session_state.selected_ethnicities = []
    
    if 'selected_worker_types' not in st.session_state:
        st.session_state.selected_worker_types = []
    
    if 'age_range' not in st.session_state:
        st.session_state.age_range = (18, 65)
    
    if 'tenure_range' not in st.session_state:
        st.session_state.tenure_range = (0, 15)  # 0-15 years
    
    # Filter visibility toggles
    if 'show_demographic_filters' not in st.session_state:
        st.session_state.show_demographic_filters = False
    
    if 'show_employment_filters' not in st.session_state:
        st.session_state.show_employment_filters = False
    
    # Store selected filter state to detect changes
    if 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = False


def render_filter_pane(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Render the global filter pane in the sidebar.
    
    Args:
        data_dict: Dictionary containing all dataframes
        
    Returns:
        Dictionary containing all selected filter values
    """
    # Remove the "Filters" title
    # st.sidebar.title("Filters")
    
    # Initialize session state if needed
    initialize_session_state()
    
    # Get reference data frames
    employees_df = data_dict['employees']
    
    # CORE DATE FILTER
    # -----------------
    st.sidebar.subheader("Date Range")
    
    # Date range selection
    date_range = st.sidebar.date_input(
        "Select period",
        value=st.session_state.date_range,
        min_value=datetime.now() - timedelta(days=365*5),  # 5 years back
        max_value=datetime.now(),
        key="date_filter"
    )
    
    # Handle single date selection edge case
    if isinstance(date_range, datetime) or len(date_range) == 1:
        date_range = (date_range, datetime.now())
    elif len(date_range) >= 2:
        date_range = (date_range[0], date_range[1])
    
    # Store in session state
    st.session_state.date_range = date_range
    
    # LOCATION FILTERS
    # -----------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Location Filters")
    
    # Get unique locations
    all_locations = get_unique_values(employees_df, 'location')
    
    # Default to all locations if none selected
    if not st.session_state.selected_locations:
        st.session_state.selected_locations = all_locations
    
    selected_locations = st.sidebar.multiselect(
        "Select locations",
        options=all_locations,
        default=st.session_state.selected_locations,
        key="location_filter"
    )
    
    # Update session state
    st.session_state.selected_locations = selected_locations if selected_locations else all_locations
    
    # EMPLOYMENT FILTERS
    # ------------------
    st.sidebar.markdown("---")
    employment_expander = st.sidebar.expander("Employment Filters", expanded=st.session_state.show_employment_filters)
    
    with employment_expander:
        # Update expander state
        st.session_state.show_employment_filters = True
        
        # Get unique departments
        all_departments = get_unique_values(employees_df, 'department')
        
        # Default to all departments if none selected
        if not st.session_state.selected_departments:
            st.session_state.selected_departments = all_departments
        
        selected_departments = st.multiselect(
            "Select departments",
            options=all_departments,
            default=st.session_state.selected_departments,
            key="dept_filter"
        )
        
        # Update session state
        st.session_state.selected_departments = selected_departments if selected_departments else all_departments
        
        # Get unique job titles (can be filtered by department)
        dept_filter = employees_df['department'].isin(st.session_state.selected_departments)
        filtered_employees = employees_df[dept_filter]
        all_job_titles = get_unique_values(filtered_employees, 'job_title')
        
        # Default to all job titles if none selected
        if not st.session_state.selected_job_titles or not set(st.session_state.selected_job_titles).issubset(set(all_job_titles)):
            st.session_state.selected_job_titles = all_job_titles
        
        selected_job_titles = st.multiselect(
            "Select job titles",
            options=all_job_titles,
            default=st.session_state.selected_job_titles,
            key="job_title_filter"
        )
        
        # Update session state
        st.session_state.selected_job_titles = selected_job_titles if selected_job_titles else all_job_titles
        
        # Worker types
        all_worker_types = get_unique_values(employees_df, 'worker_type')
        
        # Default to all worker types if none selected
        if not st.session_state.selected_worker_types:
            st.session_state.selected_worker_types = all_worker_types
        
        selected_worker_types = st.multiselect(
            "Select worker types",
            options=all_worker_types,
            default=st.session_state.selected_worker_types,
            key="worker_type_filter"
        )
        
        # Update session state
        st.session_state.selected_worker_types = selected_worker_types if selected_worker_types else all_worker_types
        
        # Tenure range (years)
        tenure_min, tenure_max = st.session_state.tenure_range
        tenure_range = st.slider(
            "Tenure range (years)",
            min_value=0,
            max_value=15,
            value=(tenure_min, tenure_max),
            step=1,
            key="tenure_slider"
        )
        
        # Update session state
        st.session_state.tenure_range = tenure_range
    
    # DEMOGRAPHIC FILTERS
    # -------------------
    st.sidebar.markdown("---")
    demographic_expander = st.sidebar.expander("Demographic Filters", expanded=st.session_state.show_demographic_filters)
    
    with demographic_expander:
        # Update expander state
        st.session_state.show_demographic_filters = True
        
        # Gender filter
        all_genders = get_unique_values(employees_df, 'gender')
        
        # Default to all genders if none selected
        if not st.session_state.selected_genders:
            st.session_state.selected_genders = all_genders
        
        selected_genders = st.multiselect(
            "Select genders",
            options=all_genders,
            default=st.session_state.selected_genders,
            key="gender_filter"
        )
        
        # Update session state
        st.session_state.selected_genders = selected_genders if selected_genders else all_genders
        
        # Ethnicity filter
        all_ethnicities = get_unique_values(employees_df, 'ethnicity')
        
        # Default to all ethnicities if none selected
        if not st.session_state.selected_ethnicities:
            st.session_state.selected_ethnicities = all_ethnicities
        
        selected_ethnicities = st.multiselect(
            "Select ethnicities",
            options=all_ethnicities,
            default=st.session_state.selected_ethnicities,
            key="ethnicity_filter"
        )
        
        # Update session state
        st.session_state.selected_ethnicities = selected_ethnicities if selected_ethnicities else all_ethnicities
        
        # Age range
        age_min, age_max = st.session_state.age_range
        age_range = st.slider(
            "Age range",
            min_value=18,
            max_value=65,
            value=(age_min, age_max),
            step=1,
            key="age_slider"
        )
        
        # Update session state
        st.session_state.age_range = age_range
        
        # Disability filter option
        show_disability = st.checkbox(
            "Include disability status", 
            value=False,
            key="disability_checkbox"
        )
    
    # FILTER ACTION BUTTONS
    # --------------------
    st.sidebar.markdown("---")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        # Apply button
        if st.button("Apply Filters", key="apply_filters"):
            st.session_state.filters_applied = True
    
    with col2:
        # Reset button
        if st.button("Reset All", key="reset_filters"):
            # Reset all filters to default
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            st.session_state.date_range = (start_date, end_date)
            st.session_state.selected_departments = all_departments
            st.session_state.selected_locations = all_locations
            st.session_state.selected_job_titles = all_job_titles
            st.session_state.selected_genders = all_genders
            st.session_state.selected_ethnicities = all_ethnicities
            st.session_state.selected_worker_types = all_worker_types
            st.session_state.age_range = (18, 65)
            st.session_state.tenure_range = (0, 15)
            st.session_state.filters_applied = True
    
    # Construct filter dictionary
    filter_dict = {
        'date_range': st.session_state.date_range,
        'departments': st.session_state.selected_departments,
        'locations': st.session_state.selected_locations,
        'job_titles': st.session_state.selected_job_titles,
        'genders': st.session_state.selected_genders,
        'ethnicities': st.session_state.selected_ethnicities,
        'worker_types': st.session_state.selected_worker_types,
        'age_range': st.session_state.age_range,
        'tenure_range': st.session_state.tenure_range,
        'show_disability': show_disability if 'show_disability' in locals() else False
    }
    
    return filter_dict