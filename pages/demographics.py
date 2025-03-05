import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.charts import (
    create_kpi_summary,
    create_pie_chart,
    create_bar_chart,
    create_stacked_bar_chart,
    create_age_distribution,
    create_farm_map,
    create_headcount_trend,
    create_heatmap,
    WORKER_TYPE_COLORS
)
from utils.data import apply_filters

def render_demographics_page(data_dict, filters):
    """
    Render the demographics dashboard page with detailed demographic breakdowns.
    
    Args:
        data_dict: Dictionary containing all data frames
        filters: Dictionary containing selected filter values
    """
    # Filter data based on selected filters
    employees_df = apply_filters(data_dict['employees'], filters)
    payroll_df = apply_filters(data_dict['payroll'], filters)
    
    # Display date range context
    start_date, end_date = filters['date_range']
    st.markdown(f"**Showing data from:** {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")
    
    # WORKFORCE SIZE METRICS
    # =====================
    st.markdown("### Workforce Size Metrics")
    
    total_employees = len(employees_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_kpi_summary(
            total_employees,
            "Total Workforce",
            delta=None
        )
    
    with col2:
        # Count by worker type
        permanent_count = len(employees_df[employees_df['worker_type'] == 'Permanent'])
        seasonal_count = len(employees_df[employees_df['worker_type'] == 'Seasonal'])
        contract_count = len(employees_df[employees_df['worker_type'] == 'Contract'])
        
        # Permanent percentage
        permanent_pct = (permanent_count / total_employees * 100) if total_employees > 0 else 0
        
        create_kpi_summary(
            permanent_count,
            "Permanent Workers",
            delta=round(permanent_pct, 1),
            delta_description="% of workforce"
        )
    
    with col3:
        # Seasonal percentage
        seasonal_pct = (seasonal_count / total_employees * 100) if total_employees > 0 else 0
        
        create_kpi_summary(
            seasonal_count,
            "Seasonal Workers",
            delta=round(seasonal_pct, 1),
            delta_description="% of workforce"
        )
    
    with col4:
        # Contract percentage
        contract_pct = (contract_count / total_employees * 100) if total_employees > 0 else 0
        
        create_kpi_summary(
            contract_count,
            "Contract Workers",
            delta=round(contract_pct, 1),
            delta_description="% of workforce"
        )
    
    # LOCATION DISTRIBUTION
    # ===================
    st.markdown("### Workforce by Location")
    
    # Prepare location data
    location_counts = employees_df['location'].value_counts().reset_index()
    location_counts.columns = ['location', 'count']
    
    # Add percentage
    location_counts['percentage'] = (location_counts['count'] / total_employees * 100).round(1) if total_employees > 0 else 0
    
    # Map visualization
    st.plotly_chart(
        create_farm_map(location_counts, 'count', title="Workforce Distribution by Location"),
        use_container_width=True
    )
    
    # DEPARTMENT DISTRIBUTION
    # =====================
    st.markdown("### Distribution by Department")
    
    # Department counts
    dept_counts = employees_df['department'].value_counts().reset_index()
    dept_counts.columns = ['department', 'count']
    dept_counts['percentage'] = (dept_counts['count'] / total_employees * 100).round(1) if total_employees > 0 else 0
    
    # Worker type by department
    dept_worker_type = pd.crosstab(
        employees_df['department'], 
        employees_df['worker_type'],
        normalize='index'
    ).reset_index() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Department distribution pie chart
        st.plotly_chart(
            create_pie_chart(
                dept_counts,
                values_col='count',
                names_col='department',
                title="Department Distribution",
                show_legend=True
            ),
            use_container_width=True
        )
    
    with col2:
        # Stacked bar chart of worker types by department
        if 'department' in employees_df and 'worker_type' in employees_df:
            worker_dept_df = employees_df.groupby(['department', 'worker_type']).size().reset_index(name='count')
            
            st.plotly_chart(
                create_stacked_bar_chart(
                    worker_dept_df,
                    x_col='department',
                    y_col='count',
                    color_col='worker_type',
                    title="Worker Types by Department",
                    color_map=WORKER_TYPE_COLORS,
                    show_legend=True
                ),
                use_container_width=True
            )
    
    # GENDER DISTRIBUTION
    # =================
    st.markdown("### Gender Demographics")
    
    # Gender counts
    gender_counts = employees_df['gender'].value_counts().reset_index()
    gender_counts.columns = ['gender', 'count']
    gender_counts['percentage'] = (gender_counts['count'] / total_employees * 100).round(1) if total_employees > 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution pie chart
        st.plotly_chart(
            create_pie_chart(
                gender_counts,
                values_col='count',
                names_col='gender',
                title="Gender Distribution",
                show_legend=True
            ),
            use_container_width=True
        )
    
    with col2:
        # Gender by department
        if 'department' in employees_df and 'gender' in employees_df:
            gender_dept_df = employees_df.groupby(['department', 'gender']).size().reset_index(name='count')
            
            st.plotly_chart(
                create_stacked_bar_chart(
                    gender_dept_df,
                    x_col='department',
                    y_col='count',
                    color_col='gender',
                    title="Gender Distribution by Department",
                    show_legend=True
                ),
                use_container_width=True
            )
    
    # ETHNICITY DISTRIBUTION
    # ====================
    st.markdown("### Ethnicity Demographics")
    
    # Ethnicity counts
    ethnicity_counts = employees_df['ethnicity'].value_counts().reset_index()
    ethnicity_counts.columns = ['ethnicity', 'count']
    ethnicity_counts['percentage'] = (ethnicity_counts['count'] / total_employees * 100).round(1) if total_employees > 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Ethnicity distribution pie chart
        st.plotly_chart(
            create_pie_chart(
                ethnicity_counts,
                values_col='count',
                names_col='ethnicity',
                title="Ethnicity Distribution",
                show_legend=True
            ),
            use_container_width=True
        )
    
    with col2:
        # Ethnicity by department
        if 'department' in employees_df and 'ethnicity' in employees_df:
            ethnicity_dept_df = employees_df.groupby(['department', 'ethnicity']).size().reset_index(name='count')
            
            st.plotly_chart(
                create_stacked_bar_chart(
                    ethnicity_dept_df,
                    x_col='department',
                    y_col='count',
                    color_col='ethnicity',
                    title="Ethnicity Distribution by Department",
                    show_legend=True
                ),
                use_container_width=True
            )
    
    # AGE DISTRIBUTION
    # ==============
    st.markdown("### Age Demographics")
    
    # Calculate age
    today = datetime.now()
    employees_df = employees_df.copy()
    employees_df['age'] = ((today - employees_df['date_of_birth']).dt.days / 365.25).round(0).astype(int)
    
    # Create age groups
    age_bins = [18, 25, 35, 45, 55, 65, 100]
    age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    employees_df['age_group'] = pd.cut(employees_df['age'], bins=age_bins, labels=age_labels, right=False)
    
    # Age group counts
    age_counts = employees_df['age_group'].value_counts().reset_index()
    age_counts.columns = ['age_group', 'count']
    
    # Sort by age group properly
    age_counts['sort_order'] = age_counts['age_group'].apply(lambda x: age_labels.index(x))
    age_counts = age_counts.sort_values('sort_order')
    age_counts = age_counts.drop('sort_order', axis=1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution histogram
        st.plotly_chart(
            create_age_distribution(
                employees_df,
                title="Age Distribution"
            ),
            use_container_width=True
        )
    
    with col2:
        # Age groups bar chart
        st.plotly_chart(
            create_bar_chart(
                age_counts,
                x_col='age_group',
                y_col='count',
                title="Age Group Distribution",
                color_col=None,
                category_orders={"age_group": age_labels},
                show_legend=True
            ),
            use_container_width=True
        )
    
    # DISABILITY STATUS
    # ===============
    if 'has_disability' in employees_df.columns:
        st.markdown("### Employees with Disabilities")
        
        # Count disability status
        disability_counts = employees_df['has_disability'].value_counts().reset_index()
        disability_counts.columns = ['has_disability', 'count']
        
        # Replace boolean with text
        disability_counts['has_disability'] = disability_counts['has_disability'].apply(
            lambda x: 'With Disability' if x else 'Without Disability'
        )
        
        # Calculate percentages
        disability_counts['percentage'] = (disability_counts['count'] / total_employees * 100).round(1) if total_employees > 0 else 0
        
        # Workers with disability count
        with_disability_count = disability_counts.loc[disability_counts['has_disability'] == 'With Disability', 'count'].sum() if len(disability_counts) > 0 else 0
        with_disability_pct = disability_counts.loc[disability_counts['has_disability'] == 'With Disability', 'percentage'].sum() if len(disability_counts) > 0 else 0
        
        # Display metrics and chart
        col1, col2 = st.columns([1, 3])
        
        with col1:
            create_kpi_summary(
                with_disability_count,
                "Employees with Disabilities",
                delta=round(with_disability_pct, 1),
                delta_description="% of workforce"
            )
        
        with col2:
            # Disability by department
            if 'department' in employees_df:
                disability_dept_df = employees_df.groupby(['department', 'has_disability']).size().reset_index(name='count')
                disability_dept_df['has_disability'] = disability_dept_df['has_disability'].apply(
                    lambda x: 'With Disability' if x else 'Without Disability'
                )
                
                st.plotly_chart(
                    create_stacked_bar_chart(
                        disability_dept_df,
                        x_col='department',
                        y_col='count',
                        color_col='has_disability',
                        title="Disability Status by Department",
                        color_map={
                            'With Disability': '#5C6BC0',
                            'Without Disability': '#90CAF9'
                        },
                        show_legend=True
                    ),
                    use_container_width=True
                )
    
    # TENURE HEATMAP SECTION
    # ====================
    st.markdown("### Average Tenure by Department and Location")
    
    # Calculate tenure in years based on join_date and selected end date
    end_datetime = pd.to_datetime(end_date)
    employees_df['tenure_years'] = ((end_datetime - pd.to_datetime(employees_df['join_date'])).dt.days / 365.25).round(1)
    
    # Create pivot table for average tenure by department and location
    tenure_heatmap = pd.pivot_table(
        employees_df,
        values='tenure_years',
        index='department',
        columns='location',
        aggfunc='mean',
        fill_value=0
    ).round(1)
    
    # Display heatmap
    st.plotly_chart(
        create_heatmap(
            tenure_heatmap,
            x_col='location',
            y_col='department',
            value_col='tenure_years',
            title="Average Tenure (Years)",
            height=400,
            text_format=None  # Will show actual values
        ),
        use_container_width=True
    )
    
    # RETIREMENT PREDICTION
    # ====================
    st.markdown("### Retirement Prediction")
    
    # Calculate retirement year based on date of birth + 65 years
    employees_df['retirement_year'] = pd.to_datetime(employees_df['date_of_birth']).dt.year + 65
    
    # Get the min and max retirement years in the dataset
    min_retirement_year = employees_df['retirement_year'].min()
    max_retirement_year = employees_df['retirement_year'].max()
    
    # Year selector for retirement prediction
    selected_retirement_year = st.select_slider(
        "Select retirement year to view",
        options=range(min_retirement_year, max_retirement_year + 1),
        value=datetime.now().year  # Default to current year
    )
    
    # Filter employees retiring in the selected year
    retiring_employees = employees_df[employees_df['retirement_year'] == selected_retirement_year].copy()
    
    # Display count of retiring employees
    st.markdown(f"##### Employees retiring in {selected_retirement_year}: {len(retiring_employees)}")
    
    if len(retiring_employees) > 0:
        # Add retirement month based on date of birth
        retiring_employees['retirement_month'] = pd.to_datetime(retiring_employees['date_of_birth']).dt.month_name()
        
        # Prepare data for display
        retirement_display = retiring_employees[['employee_id', 'first_name', 'last_name', 'department', 'location', 'job_title', 'retirement_month']].copy()
        retirement_display.columns = ['Employee ID', 'First Name', 'Last Name', 'Department', 'Location', 'Job Title', 'Retirement Month']
        
        # Sort by retirement month
        month_order = {month: i for i, month in enumerate(['January', 'February', 'March', 'April', 'May', 'June', 
                                                          'July', 'August', 'September', 'October', 'November', 'December'])}
        retirement_display['Month Order'] = retirement_display['Retirement Month'].map(month_order)
        retirement_display = retirement_display.sort_values('Month Order')
        retirement_display = retirement_display.drop(columns=['Month Order'])
        
        # Display as a table
        st.dataframe(retirement_display, use_container_width=True)
        
        # Display department breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            # Department breakdown of retiring employees
            dept_count = retiring_employees['department'].value_counts().reset_index()
            dept_count.columns = ['Department', 'Count']
            
            st.markdown("##### Retiring Employees by Department")
            st.plotly_chart(
                create_bar_chart(
                    dept_count,
                    x_col='Department',
                    y_col='Count',
                    title="",
                    height=300,
                    show_legend=False
                ),
                use_container_width=True
            )
        
        with col2:
            # Job title breakdown of retiring employees
            job_count = retiring_employees['job_title'].value_counts().reset_index()
            job_count.columns = ['Job Title', 'Count']
            
            st.markdown("##### Retiring Employees by Job Title")
            st.plotly_chart(
                create_bar_chart(
                    job_count,
                    x_col='Job Title',
                    y_col='Count',
                    title="",
                    height=300,
                    show_legend=False
                ),
                use_container_width=True
            )
    else:
        st.info(f"No employees are predicted to retire in {selected_retirement_year}.")
    
    # DEMOGRAPHIC TRENDS
    # ================
    st.markdown("### Demographic Trends")
    
    # For trend analysis, we need time series data
    # We'll use the payroll data which has monthly information
    if len(payroll_df) > 0:
        # Monthly workforce size by worker type
        worker_type_trend = payroll_df.groupby(['month', 'worker_type']).agg(
            count=('employee_id', 'nunique')
        ).reset_index()
        
        # Sort by month
        worker_type_trend = worker_type_trend.sort_values('month')
        
        # Display chart
        st.plotly_chart(
            create_headcount_trend(
                worker_type_trend,
                time_col='month',
                count_col='count',
                group_col='worker_type',
                title="Workforce Trend by Worker Type",
                show_legend=True
            ),
            use_container_width=True
        )
        
        # Now look at department trends if department info is in the payroll data
        if 'department' in payroll_df.columns:
            dept_trend = payroll_df.groupby(['month', 'department']).agg(
                count=('employee_id', 'nunique')
            ).reset_index()
            
            # Sort by month
            dept_trend = dept_trend.sort_values('month')
            
            # Display chart
            st.plotly_chart(
                create_headcount_trend(
                    dept_trend,
                    time_col='month',
                    count_col='count',
                    group_col='department',
                    title="Workforce Trend by Department",
                    show_legend=True
                ),
                use_container_width=True
            )
    else:
        st.info("No trend data available for the selected filters.")