import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from copy import deepcopy

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.charts import (
    create_kpi_summary,
    create_pie_chart,
    create_bar_chart,
    create_stacked_bar_chart,
    create_comparison_chart,
    format_currency,
    apply_common_layout
)
from utils.data import apply_filters

def render_workforce_scenario_page(data_dict, filters):
    """
    Render the workforce scenario analysis dashboard page.
    
    Args:
        data_dict: Dictionary containing all data frames
        filters: Dictionary containing selected filter values
    """
    
    # Filter data based on selected filters
    employees_df = apply_filters(data_dict['employees'], filters)
    payroll_df = apply_filters(data_dict['payroll'], filters) if 'payroll' in data_dict else pd.DataFrame()
    
    # Initialize session state for scenario data if not exists
    if 'scenario_data' not in st.session_state:
        st.session_state.scenario_data = {
            'permanent_changes': [],
            'seasonal_workers': {
                'count': 0,
                'avg_cost': 0,
                'duration_months': 0
            },
            'calculated': False
        }
    
    # CURRENT WORKFORCE SUMMARY
    # =====================
    st.markdown("### Current Workforce Summary")
    
    # Calculate current workforce metrics
    total_employees = len(employees_df)
    
    # Worker types
    worker_type_counts = employees_df['worker_type'].value_counts()
    permanent_count = worker_type_counts.get('Permanent', 0)
    seasonal_count = worker_type_counts.get('Seasonal', 0)
    contract_count = worker_type_counts.get('Contract', 0)
    
    # Calculate monthly payroll cost
    monthly_cost = 0
    if not payroll_df.empty:
        latest_month = payroll_df['month'].max()
        latest_payroll = payroll_df[payroll_df['month'] == latest_month]
        monthly_cost = latest_payroll['gross_pay'].sum()
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_kpi_summary(
            total_employees,
            "Total Workforce",
            delta=None
        )
    
    with col2:
        create_kpi_summary(
            permanent_count,
            "Permanent Workers",
            delta=round(permanent_count / total_employees * 100 if total_employees > 0 else 0, 1),
            delta_description="% of workforce",
            format_type="percent"
        )
    
    with col3:
        create_kpi_summary(
            seasonal_count,
            "Seasonal Workers",
            delta=round(seasonal_count / total_employees * 100 if total_employees > 0 else 0, 1),
            delta_description="% of workforce",
            format_type="percent"
        )
    
    with col4:
        create_kpi_summary(
            monthly_cost,
            "Monthly Payroll Cost",
            delta=None,
            format_type="currency"
        )
    
    # Current distribution by department and location
    col1, col2 = st.columns(2)
    
    with col1:
        # Department distribution
        dept_counts = employees_df['department'].value_counts().reset_index()
        dept_counts.columns = ['department', 'count']
        
        st.plotly_chart(
            create_pie_chart(
                dept_counts,
                values_col='count',
                names_col='department',
                title="Current Department Distribution"
            ),
            use_container_width=True
        )
    
    with col2:
        # Location distribution
        location_counts = employees_df['location'].value_counts().reset_index()
        location_counts.columns = ['location', 'count']
        
        st.plotly_chart(
            create_pie_chart(
                location_counts,
                values_col='count',
                names_col='location',
                title="Current Location Distribution"
            ),
            use_container_width=True
        )
    
    # SCENARIO INPUT FORM
    # ================
    st.markdown("### Workforce Change Scenario")
    
    # Tabs for different types of changes
    scenario_tab, seasonal_tab = st.tabs(["Permanent Staff Changes", "Seasonal Workers"])
    
    with scenario_tab:
        st.markdown("#### Add or Remove Permanent Workers")
        st.markdown("Enter the details of workers to add (positive numbers) or remove (negative numbers):")
        
        # Define available options from the data
        available_locations = sorted(employees_df['location'].unique().tolist())
        available_departments = sorted(employees_df['department'].unique().tolist())
        
        # Create a container for the form
        scenario_container = st.container()
        
        # Function to update job titles based on department
        def get_job_titles_for_dept(dept):
            if dept:
                job_titles = employees_df[employees_df['department'] == dept]['job_title'].unique().tolist()
                return sorted(job_titles)
            return []
        
        # Initialize session state for values if needed
        if 'permanent_form_values' not in st.session_state:
            st.session_state.permanent_form_values = [
                {'count': 0, 'location': '', 'department': '', 'job_title': '', 'salary': 0} 
                for _ in range(10)
            ]
        
        # Create 10 input lines
        permanent_changes = []
        
        with scenario_container:
            for i in range(10):  # 10 input lines
                col1, col2, col3, col4, col5 = st.columns([1, 1.5, 1.5, 1.5, 1])
                
                with col1:
                    count = st.number_input(
                        f"Count #{i+1}",
                        min_value=-100,
                        max_value=100,
                        value=st.session_state.permanent_form_values[i]['count'],
                        step=1,
                        key=f"count_{i}"
                    )
                    # Update session state
                    st.session_state.permanent_form_values[i]['count'] = count
                
                with col2:
                    location = st.selectbox(
                        f"Location #{i+1}",
                        options=[""] + available_locations,
                        index=0 if not st.session_state.permanent_form_values[i]['location'] else ([""] + available_locations).index(st.session_state.permanent_form_values[i]['location']),
                        key=f"location_{i}"
                    )
                    # Update session state
                    st.session_state.permanent_form_values[i]['location'] = location
                
                with col3:
                    department = st.selectbox(
                        f"Department #{i+1}",
                        options=[""] + available_departments,
                        index=0 if not st.session_state.permanent_form_values[i]['department'] else ([""] + available_departments).index(st.session_state.permanent_form_values[i]['department']),
                        key=f"department_{i}"
                    )
                    # Update session state immediately
                    st.session_state.permanent_form_values[i]['department'] = department
                
                # Get job titles for the selected department
                job_titles = get_job_titles_for_dept(department) if department else []
                
                with col4:
                    job_title_options = [""] + job_titles
                    # Find index of current job title or default to 0
                    current_job_title = st.session_state.permanent_form_values[i]['job_title']
                    job_title_index = 0
                    if current_job_title in job_title_options:
                        job_title_index = job_title_options.index(current_job_title)
                    
                    job_title = st.selectbox(
                        f"Job Title #{i+1}",
                        options=job_title_options,
                        index=job_title_index,
                        key=f"job_title_{i}"
                    )
                    # Update session state
                    st.session_state.permanent_form_values[i]['job_title'] = job_title
                
                with col5:
                    # Default salary based on job title if available
                    default_salary = st.session_state.permanent_form_values[i]['salary']
                    if job_title and not payroll_df.empty and default_salary == 0:
                        job_salaries = payroll_df.merge(
                            employees_df[['employee_id', 'job_title']],
                            on='employee_id'
                        )
                        relevant_salaries = job_salaries[job_salaries['job_title'] == job_title]['base_salary']
                        if not relevant_salaries.empty:
                            default_salary = int(relevant_salaries.mean())
                    
                    salary = st.number_input(
                        f"Salary #{i+1}",
                        min_value=0,
                        max_value=100000,
                        value=default_salary,
                        step=500,
                        key=f"salary_{i}"
                    )
                    # Update session state
                    st.session_state.permanent_form_values[i]['salary'] = salary
                
                # Add to changes if all required fields are filled
                if count != 0 and location and department and job_title:
                    permanent_changes.append({
                        'count': count,
                        'location': location,
                        'department': department,
                        'job_title': job_title,
                        'salary': salary
                    })
            
            # Submit button (outside of form now)
            if st.button("Apply Permanent Changes"):
                st.session_state.scenario_data['permanent_changes'] = permanent_changes
                st.session_state.scenario_data['calculated'] = False
                st.success(f"Added {len([c for c in permanent_changes if c['count'] > 0])} worker additions and {len([c for c in permanent_changes if c['count'] < 0])} worker reductions to the scenario.")
    
    with seasonal_tab:
        st.markdown("#### Seasonal Workers Scenario")
        st.markdown("Enter details for seasonal workers:")
        
        # Initialize session state for seasonal values if needed
        if 'seasonal_form_values' not in st.session_state:
            st.session_state.seasonal_form_values = {
                'count': st.session_state.scenario_data['seasonal_workers']['count'],
                'avg_cost': st.session_state.scenario_data['seasonal_workers']['avg_cost'],
                'duration_months': st.session_state.scenario_data['seasonal_workers']['duration_months'] if st.session_state.scenario_data['seasonal_workers']['duration_months'] > 0 else 3,
                'location': '',
                'department': ''
            }
        
        # Input fields for seasonal workers
        col1, col2, col3 = st.columns(3)
        
        with col1:
            seasonal_count = st.number_input(
                "Number of Seasonal Workers",
                min_value=0,
                max_value=1000,
                value=st.session_state.seasonal_form_values['count'],
                step=10
            )
            st.session_state.seasonal_form_values['count'] = seasonal_count
        
        with col2:
            # Calculate average seasonal salary as default
            default_seasonal_salary = 0
            if not payroll_df.empty and not employees_df.empty:
                seasonal_salaries = payroll_df.merge(
                    employees_df[['employee_id', 'worker_type']],
                    on='employee_id'
                )
                # Check if worker_type column exists before filtering
                if 'worker_type' in seasonal_salaries.columns:
                    relevant_salaries = seasonal_salaries[seasonal_salaries['worker_type'] == 'Seasonal']['base_salary']
                    if not relevant_salaries.empty:
                        default_seasonal_salary = int(relevant_salaries.mean())
                else:
                    # Fall back to using all salaries if worker_type doesn't exist
                    if 'base_salary' in seasonal_salaries.columns:
                        default_seasonal_salary = int(seasonal_salaries['base_salary'].mean())
                    
            # If we have a saved value and it's not 0, use it; otherwise use the default
            if st.session_state.seasonal_form_values['avg_cost'] > 0:
                default_value = st.session_state.seasonal_form_values['avg_cost']
            elif default_seasonal_salary > 0:
                default_value = default_seasonal_salary
            else:
                default_value = 0
                
            seasonal_cost = st.number_input(
                "Average Monthly Cost per Worker",
                min_value=0,
                max_value=50000,
                value=default_value,
                step=500
            )
            st.session_state.seasonal_form_values['avg_cost'] = seasonal_cost
        
        with col3:
            seasonal_duration = st.number_input(
                "Duration of Employment (months)",
                min_value=1,
                max_value=12,
                value=st.session_state.seasonal_form_values['duration_months'],
                step=1
            )
            st.session_state.seasonal_form_values['duration_months'] = seasonal_duration
        
        # Location and department selection for seasonal workers
        col1, col2 = st.columns(2)
        
        with col1:
            seasonal_location = st.selectbox(
                "Primary Location",
                options=available_locations,
                index=0 if not st.session_state.seasonal_form_values['location'] or st.session_state.seasonal_form_values['location'] not in available_locations else available_locations.index(st.session_state.seasonal_form_values['location'])
            )
            st.session_state.seasonal_form_values['location'] = seasonal_location
        
        with col2:
            seasonal_dept = st.selectbox(
                "Primary Department",
                options=available_departments,
                index=0 if not st.session_state.seasonal_form_values['department'] or st.session_state.seasonal_form_values['department'] not in available_departments else available_departments.index(st.session_state.seasonal_form_values['department'])
            )
            st.session_state.seasonal_form_values['department'] = seasonal_dept
        
        # Submit button for seasonal workers
        if st.button("Apply Seasonal Changes"):
            st.session_state.scenario_data['seasonal_workers'] = {
                'count': seasonal_count,
                'avg_cost': seasonal_cost,
                'duration_months': seasonal_duration,
                'location': seasonal_location,
                'department': seasonal_dept
            }
            st.session_state.scenario_data['calculated'] = False
            
            if seasonal_count > 0:
                st.success(f"Added {seasonal_count} seasonal workers to the scenario.")
            else:
                st.info("No seasonal workers added to the scenario.")
    
    # Button to calculate scenario results
    calculate_btn = st.button("Calculate Scenario Results", type="primary")
    
    if calculate_btn:
        st.session_state.scenario_data['calculated'] = True
    
    # SCENARIO RESULTS SECTION
    # =====================
    if st.session_state.scenario_data['calculated']:
        st.markdown("### Scenario Results")
        
        # Calculate the scenario workforce
        scenario_workforce, cost_impact = calculate_scenario_workforce(
            employees_df, 
            payroll_df,
            st.session_state.scenario_data['permanent_changes'],
            st.session_state.scenario_data['seasonal_workers']
        )
        
        # Scenario summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Total workforce change
            workforce_change = len(scenario_workforce) - total_employees
            workforce_pct_change = (workforce_change / total_employees * 100) if total_employees > 0 else 0
            
            create_kpi_summary(
                len(scenario_workforce),
                "New Total Workforce",
                delta=workforce_change,
                delta_description="workers change",
                format_type=None
            )
        
        with col2:
            # Permanent worker change
            scenario_permanent = len(scenario_workforce[scenario_workforce['worker_type'] == 'Permanent'])
            permanent_change = scenario_permanent - permanent_count
            
            create_kpi_summary(
                scenario_permanent,
                "New Permanent Count",
                delta=permanent_change,
                delta_description="workers change",
                format_type=None
            )
        
        with col3:
            # Seasonal worker change
            scenario_seasonal = len(scenario_workforce[scenario_workforce['worker_type'] == 'Seasonal'])
            seasonal_change = scenario_seasonal - seasonal_count
            
            create_kpi_summary(
                scenario_seasonal,
                "New Seasonal Count",
                delta=seasonal_change,
                delta_description="workers change",
                format_type=None
            )
        
        with col4:
            # Monthly cost change
            create_kpi_summary(
                cost_impact['monthly_cost'],
                "New Monthly Cost",
                delta=cost_impact['monthly_change'],
                delta_description="cost change",
                format_type="currency"
            )
        
        # Cost implications section
        st.markdown("#### Cost Implications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly cost breakdown
            st.markdown("##### Monthly Cost Breakdown")
            
            monthly_breakdown = pd.DataFrame([
                {"Category": "Current Monthly Cost", "Amount": monthly_cost},
                {"Category": "Changes Cost", "Amount": cost_impact['changes_cost']},
                {"Category": "New Monthly Cost", "Amount": cost_impact['monthly_cost']}
            ])
            
            # Add percentage change
            pct_change = (cost_impact['monthly_change'] / monthly_cost * 100) if monthly_cost > 0 else 0
            
            # Display as a table with formatting
            st.dataframe(
                monthly_breakdown,
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown(f"**Monthly Cost Change:** {format_currency(cost_impact['monthly_change'])} ({pct_change:.1f}%)")
        
        with col2:
            # Annual cost projection
            st.markdown("##### Annual Cost Projection")
            
            # Calculate annual costs
            annual_current = monthly_cost * 12
            
            # For seasonal workers, consider their duration
            seasonal_duration = st.session_state.scenario_data['seasonal_workers']['duration_months']
            seasonal_total_cost = (st.session_state.scenario_data['seasonal_workers']['count'] * 
                                  st.session_state.scenario_data['seasonal_workers']['avg_cost'] * 
                                  seasonal_duration)
            
            # Annual cost of permanent changes
            permanent_annual = cost_impact['permanent_monthly'] * 12
            
            # Total annual cost with scenario
            annual_new = annual_current + permanent_annual + seasonal_total_cost
            annual_change = annual_new - annual_current
            
            annual_breakdown = pd.DataFrame([
                {"Category": "Current Annual Cost", "Amount": annual_current},
                {"Category": "Permanent Changes (Annual)", "Amount": permanent_annual},
                {"Category": "Seasonal Workers (Duration-based)", "Amount": seasonal_total_cost},
                {"Category": "New Annual Cost", "Amount": annual_new}
            ])
            
            # Display as a table with formatting
            st.dataframe(
                annual_breakdown,
                use_container_width=True,
                hide_index=True
            )
            
            # Annual change percentage
            annual_pct_change = (annual_change / annual_current * 100) if annual_current > 0 else 0
            st.markdown(f"**Annual Cost Impact:** {format_currency(annual_change)} ({annual_pct_change:.1f}%)")
        
        # Visual comparisons
        st.markdown("#### Workforce Distribution Comparison")
        
        # Compare department distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Department distribution before/after
            current_dept = employees_df['department'].value_counts().reset_index()
            current_dept.columns = ['department', 'count']
            
            scenario_dept = scenario_workforce['department'].value_counts().reset_index()
            scenario_dept.columns = ['department', 'count']
            
            st.plotly_chart(
                create_comparison_chart(
                    current_dept,
                    scenario_dept,
                    compare_col='department',
                    value_col='count',
                    title="Department Distribution: Before vs After",
                    orientation='h'
                ),
                use_container_width=True
            )
        
        with col2:
            # Worker type distribution before/after
            current_type = employees_df['worker_type'].value_counts().reset_index()
            current_type.columns = ['worker_type', 'count']
            
            scenario_type = scenario_workforce['worker_type'].value_counts().reset_index()
            scenario_type.columns = ['worker_type', 'count']
            
            st.plotly_chart(
                create_comparison_chart(
                    current_type,
                    scenario_type,
                    compare_col='worker_type',
                    value_col='count',
                    title="Worker Type Distribution: Before vs After"
                ),
                use_container_width=True
            )
        
        # Detailed changes breakdown
        st.markdown("#### Detailed Changes Breakdown")
        
        # Create a table summarizing all changes
        changes_summary = []
        
        # Permanent changes
        for change in st.session_state.scenario_data['permanent_changes']:
            if change['count'] != 0:
                changes_summary.append({
                    'Change Type': 'Add' if change['count'] > 0 else 'Remove',
                    'Worker Type': 'Permanent',
                    'Count': abs(change['count']),
                    'Department': change['department'],
                    'Location': change['location'],
                    'Job Title': change['job_title'],
                    'Monthly Cost Impact': change['salary'] * change['count']
                })
        
        # Seasonal workers
        seasonal_data = st.session_state.scenario_data['seasonal_workers']
        if seasonal_data['count'] > 0:
            changes_summary.append({
                'Change Type': 'Add',
                'Worker Type': 'Seasonal',
                'Count': seasonal_data['count'],
                'Department': seasonal_data.get('department', 'Various'),
                'Location': seasonal_data.get('location', 'Various'),
                'Job Title': 'Seasonal Worker',
                'Monthly Cost Impact': seasonal_data['count'] * seasonal_data['avg_cost'],
                'Duration (Months)': seasonal_data['duration_months']
            })
        
        # Display the changes summary
        if changes_summary:
            changes_df = pd.DataFrame(changes_summary)
            
            # Format the cost column
            changes_df['Monthly Cost Impact'] = changes_df['Monthly Cost Impact'].apply(
                lambda x: ('+' if x > 0 else '') + format_currency(x, currency='')
            )
            
            st.dataframe(changes_df, use_container_width=True, hide_index=True)
        else:
            st.info("No changes were specified in the scenario.")
        
        # Reset button
        if st.button("Reset Scenario"):
            st.session_state.scenario_data = {
                'permanent_changes': [],
                'seasonal_workers': {
                    'count': 0,
                    'avg_cost': 0,
                    'duration_months': 0
                },
                'calculated': False
            }
            st.rerun()

def calculate_scenario_workforce(employees_df, payroll_df, permanent_changes, seasonal_workers):
    """
    Calculate the workforce and costs after applying scenario changes.
    
    Args:
        employees_df: DataFrame with current employee data
        payroll_df: DataFrame with payroll data
        permanent_changes: List of dictionaries with permanent staff changes
        seasonal_workers: Dictionary with seasonal worker scenario
        
    Returns:
        Tuple of (scenario_workforce_df, cost_impact_dict)
    """
    # Create a copy of the current workforce
    scenario_workforce = employees_df.copy()
    
    # Initialize a counter for new employee IDs
    if not scenario_workforce.empty:
        max_id = scenario_workforce['employee_id'].max()
    else:
        max_id = 0
    
    # Calculate monthly cost of current workforce
    monthly_cost = 0
    if not payroll_df.empty:
        latest_month = payroll_df['month'].max()
        latest_payroll = payroll_df[payroll_df['month'] == latest_month]
        monthly_cost = latest_payroll['gross_pay'].sum()
    
    # Apply permanent changes
    changes_cost = 0
    permanent_monthly_impact = 0
    
    for change in permanent_changes:
        if change['count'] > 0:  # Adding workers
            # Create new employees
            for i in range(change['count']):
                max_id += 1
                
                # Create a new employee record
                new_employee = {
                    'employee_id': max_id,
                    'department': change['department'],
                    'location': change['location'],
                    'job_title': change['job_title'],
                    'worker_type': 'Permanent',
                    'salary': change['salary'],
                    'join_date': datetime.now()
                }
                
                # Add other required fields with dummy values
                if 'first_name' in scenario_workforce.columns:
                    new_employee['first_name'] = f"New{i+1}"
                
                if 'last_name' in scenario_workforce.columns:
                    new_employee['last_name'] = f"Employee{max_id}"
                
                if 'gender' in scenario_workforce.columns:
                    new_employee['gender'] = 'Not Specified'
                
                if 'date_of_birth' in scenario_workforce.columns:
                    new_employee['date_of_birth'] = datetime.now() - timedelta(days=365*30)  # Approx 30 years old
                
                if 'ethnicity' in scenario_workforce.columns:
                    new_employee['ethnicity'] = 'Not Specified'
                
                if 'has_disability' in scenario_workforce.columns:
                    new_employee['has_disability'] = False
                
                if 'manager_id' in scenario_workforce.columns:
                    # Find a potential manager in the same department
                    potential_managers = scenario_workforce[
                        (scenario_workforce['department'] == change['department']) & 
                        (scenario_workforce['job_title'].str.contains('Manager|Supervisor|Lead', case=False, na=False))
                    ]
                    
                    if not potential_managers.empty:
                        new_employee['manager_id'] = potential_managers.iloc[0]['employee_id']
                    else:
                        new_employee['manager_id'] = None
                
                # Add to scenario workforce
                scenario_workforce = pd.concat([scenario_workforce, pd.DataFrame([new_employee])], ignore_index=True)
                
                # Add to cost impact
                changes_cost += change['salary']
                permanent_monthly_impact += change['salary']
                
        elif change['count'] < 0:  # Removing workers
            # Find employees matching the criteria to remove
            employees_to_remove = scenario_workforce[
                (scenario_workforce['department'] == change['department']) & 
                (scenario_workforce['location'] == change['location']) & 
                (scenario_workforce['job_title'] == change['job_title']) & 
                (scenario_workforce['worker_type'] == 'Permanent')
            ]
            
            # Limit to the number specified
            count_to_remove = min(abs(change['count']), len(employees_to_remove))
            
            if count_to_remove > 0:
                # Get the IDs to remove
                ids_to_remove = employees_to_remove.iloc[:count_to_remove]['employee_id'].tolist()
                
                # Remove from workforce
                scenario_workforce = scenario_workforce[~scenario_workforce['employee_id'].isin(ids_to_remove)]
                
                # Calculate cost impact (negative for savings)
                changes_cost -= count_to_remove * change['salary']
                permanent_monthly_impact -= count_to_remove * change['salary']
    
    # Apply seasonal workers
    seasonal_count = seasonal_workers['count']
    seasonal_cost = seasonal_workers['avg_cost']
    seasonal_monthly_impact = seasonal_count * seasonal_cost
    
    # Only add seasonal workers if count > 0
    if seasonal_count > 0:
        # Get department and location
        seasonal_dept = seasonal_workers.get('department', scenario_workforce['department'].iloc[0] if not scenario_workforce.empty else 'Production')
        seasonal_location = seasonal_workers.get('location', scenario_workforce['location'].iloc[0] if not scenario_workforce.empty else 'Main Farm')
        
        # Create seasonal workers
        for i in range(seasonal_count):
            max_id += 1
            
            # Create a new employee record
            new_employee = {
                'employee_id': max_id,
                'department': seasonal_dept,
                'location': seasonal_location,
                'job_title': 'Seasonal Worker',
                'worker_type': 'Seasonal',
                'salary': seasonal_cost,
                'join_date': datetime.now()
            }
            
            # Add other required fields with dummy values
            if 'first_name' in scenario_workforce.columns:
                new_employee['first_name'] = f"Seasonal{i+1}"
            
            if 'last_name' in scenario_workforce.columns:
                new_employee['last_name'] = f"Worker{max_id}"
            
            if 'gender' in scenario_workforce.columns:
                new_employee['gender'] = 'Not Specified'
            
            if 'date_of_birth' in scenario_workforce.columns:
                new_employee['date_of_birth'] = datetime.now() - timedelta(days=365*30)  # Approx 30 years old
            
            if 'ethnicity' in scenario_workforce.columns:
                new_employee['ethnicity'] = 'Not Specified'
            
            if 'has_disability' in scenario_workforce.columns:
                new_employee['has_disability'] = False
            
            if 'manager_id' in scenario_workforce.columns:
                # Find a potential manager in the same department
                potential_managers = scenario_workforce[
                    (scenario_workforce['department'] == seasonal_dept) & 
                    (scenario_workforce['job_title'].str.contains('Manager|Supervisor|Lead', case=False, na=False))
                ]
                
                if not potential_managers.empty:
                    new_employee['manager_id'] = potential_managers.iloc[0]['employee_id']
                else:
                    new_employee['manager_id'] = None
            
            # Add to scenario workforce
            scenario_workforce = pd.concat([scenario_workforce, pd.DataFrame([new_employee])], ignore_index=True)
        
        # Add to cost impact
        changes_cost += seasonal_monthly_impact
    
    # Calculate final monthly cost
    new_monthly_cost = monthly_cost + changes_cost
    
    # Prepare cost impact dictionary
    cost_impact = {
        'monthly_cost': new_monthly_cost,
        'monthly_change': changes_cost,
        'changes_cost': changes_cost,
        'permanent_monthly': permanent_monthly_impact,
        'seasonal_monthly': seasonal_monthly_impact
    }
    
    return scenario_workforce, cost_impact