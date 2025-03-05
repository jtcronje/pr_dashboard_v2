import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
import calendar

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.charts import (
    create_kpi_summary,
    create_pie_chart,
    create_bar_chart,
    create_stacked_bar_chart,
    create_line_chart,
    create_heatmap,
    apply_common_layout
)
from utils.data import apply_filters

def render_leave_page(data_dict, filters):
    """
    Render the leave analysis dashboard page.
    
    Args:
        data_dict: Dictionary containing all data frames
        filters: Dictionary containing selected filter values
    """
    # Filter data based on selected filters
    leave_df = apply_filters(data_dict['leave'], filters) if 'leave' in data_dict else pd.DataFrame()
    employees_df = apply_filters(data_dict['employees'], filters) if 'employees' in data_dict else pd.DataFrame()
    
    # Check if we have leave data
    if leave_df.empty:
        st.warning("No leave data is available for the selected filters.")
        return
    
    # Display date range context
    start_date, end_date = filters['date_range']
    st.markdown(f"**Showing data from:** {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")
    
    # Preprocess leave data to extract additional information
    leave_df = preprocess_leave_data(leave_df)
    
    # LEAVE METRICS SECTION
    # ==================
    st.markdown("### Leave Metrics")
    
    # Calculate total leave days
    total_leave_days = leave_df['duration_days'].sum()
    
    # Calculate employee count
    employee_count = len(employees_df) if not employees_df.empty else len(leave_df['employee_id'].unique())
    
    # Calculate average leave days per employee
    avg_leave_days = total_leave_days / employee_count if employee_count > 0 else 0
    
    # Calculate leave utilization (if we have entitlement data)
    # For now, use a placeholder value or estimate based on worker type
    leave_entitlement = 0
    if 'worker_type' in employees_df.columns:
        # Estimate entitlement based on worker type distribution
        permanent_count = len(employees_df[employees_df['worker_type'] == 'Permanent'])
        contract_count = len(employees_df[employees_df['worker_type'] == 'Contract'])
        seasonal_count = len(employees_df[employees_df['worker_type'] == 'Seasonal'])
        
        # Estimated entitlement (Permanent: 21 days, Contract: 15 days, Seasonal: 7 days)
        leave_entitlement = (permanent_count * 21 + contract_count * 15 + seasonal_count * 7) / employee_count if employee_count > 0 else 0
    else:
        # Default estimate (15 days per year)
        leave_entitlement = 15
    
    # Calculate utilization percentage
    leave_utilization = (avg_leave_days / leave_entitlement * 100) if leave_entitlement > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_kpi_summary(
            total_leave_days,
            "Total Leave Days",
            delta=None
        )
    
    with col2:
        create_kpi_summary(
            round(avg_leave_days, 1),
            "Avg. Leave Days per Employee",
            delta=None
        )
    
    with col3:
        create_kpi_summary(
            round(leave_utilization, 1),
            "Leave Utilization",
            delta=None,
            delta_description="of annual entitlement",
            format_type="percent"
        )
    
    # LEAVE DISTRIBUTION ANALYSIS
    # ========================
    st.markdown("### Leave Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Leave type breakdown pie chart
        leave_by_type = leave_df.groupby('leave_type')['duration_days'].sum().reset_index()
        
        st.plotly_chart(
            create_pie_chart(
                leave_by_type,
                values_col='duration_days',
                names_col='leave_type',
                title="Leave Days by Type"
            ),
            use_container_width=True
        )
    
    with col2:
        # Leave days by department bar chart (if department data available)
        if 'department' in leave_df.columns:
            leave_by_dept = leave_df.groupby('department')['duration_days'].sum().reset_index()
            leave_by_dept = leave_by_dept.sort_values('duration_days', ascending=False)
            
            st.plotly_chart(
                create_bar_chart(
                    leave_by_dept,
                    x_col='department',
                    y_col='duration_days',
                    title="Leave Days by Department"
                ),
                use_container_width=True
            )
        else:
            st.info("Department data is not available for leave analysis.")
    
    # Leave days by month heat map (seasonality)
    if 'month' in leave_df.columns and 'leave_type' in leave_df.columns:
        # Group by month and leave type
        monthly_leave = leave_df.groupby(['month', 'leave_type'])['duration_days'].sum().reset_index()
        
        # Create a pivot table for the heatmap
        leave_pivot = monthly_leave.pivot(index='leave_type', columns='month', values='duration_days').fillna(0)
        
        # Sort months chronologically if they are in YYYY-MM format
        if all(m.count('-') == 1 for m in leave_pivot.columns):
            leave_pivot = leave_pivot[sorted(leave_pivot.columns)]
        
        # Convert to long format for heatmap
        leave_long = pd.melt(
            leave_pivot.reset_index(), 
            id_vars=['leave_type'], 
            var_name='month', 
            value_name='days'
        )
        
        st.plotly_chart(
            create_heatmap(
                leave_long,
                x_col='month',
                y_col='leave_type',
                value_col='days',
                title="Leave Patterns by Month and Type",
                height=400
            ),
            use_container_width=True
        )
    
    # OUTSTANDING LEAVE BALANCES
    # =======================
    st.markdown("### Outstanding Leave Balances")
    
    # For this section, we would ideally have leave balance data
    # In absence of that, we can estimate based on entitlement and used leave
    if not employees_df.empty and 'worker_type' in employees_df.columns:
        # Create a dataframe with leave entitlement based on worker type
        employees_with_entitlement = employees_df.copy()
        employees_with_entitlement['annual_entitlement'] = employees_with_entitlement['worker_type'].map({
            'Permanent': 21,
            'Contract': 15,
            'Seasonal': 7
        }).fillna(15)  # Default to 15 days
        
        # Calculate pro-rated entitlement based on join date
        current_year = datetime.now().year
        year_start = datetime.strptime(f"{current_year}-01-01", '%Y-%m-%d')
        year_end = datetime.strptime(f"{current_year}-12-31", '%Y-%m-%d')
        
        if 'join_date' in employees_with_entitlement.columns:
            # For employees who joined this year, prorate their entitlement
            joined_this_year = employees_with_entitlement['join_date'] > year_start
            days_in_year = (year_end - year_start).days + 1
            
            employees_with_entitlement.loc[joined_this_year, 'prorated_entitlement'] = (
                employees_with_entitlement.loc[joined_this_year, 'annual_entitlement'] * 
                ((year_end - employees_with_entitlement.loc[joined_this_year, 'join_date']).dt.days + 1) / 
                days_in_year
            ).round(1)
            
            # For employees who joined before this year, use full entitlement
            employees_with_entitlement.loc[~joined_this_year, 'prorated_entitlement'] = employees_with_entitlement.loc[~joined_this_year, 'annual_entitlement']
        else:
            # If join date is not available, use full entitlement for all
            employees_with_entitlement['prorated_entitlement'] = employees_with_entitlement['annual_entitlement']
        
        # Calculate leave taken this year (if we have date information)
        if 'year' in leave_df.columns:
            current_year_leave = leave_df[leave_df['year'] == current_year]
        else:
            # Assume all leave in the dataset is for the current year
            current_year_leave = leave_df
        
        # Aggregate leave taken by employee
        leave_taken = current_year_leave.groupby('employee_id')['duration_days'].sum().reset_index()
        leave_taken.columns = ['employee_id', 'leave_taken']
        
        # Merge with employee data
        employees_with_leave = pd.merge(
            employees_with_entitlement,
            leave_taken,
            on='employee_id',
            how='left'
        )
        
        # Fill NaN leave taken with 0
        employees_with_leave['leave_taken'] = employees_with_leave['leave_taken'].fillna(0)
        
        # Calculate remaining leave
        employees_with_leave['remaining_leave'] = employees_with_leave['prorated_entitlement'] - employees_with_leave['leave_taken']
        
        # Department-wise leave balance summary
        if 'department' in employees_with_leave.columns:
            dept_leave_summary = employees_with_leave.groupby('department').agg(
                avg_entitlement=('prorated_entitlement', 'mean'),
                avg_taken=('leave_taken', 'mean'),
                avg_remaining=('remaining_leave', 'mean'),
                employee_count=('employee_id', 'count')
            ).reset_index()
            
            # Calculate utilization percentage
            dept_leave_summary['utilization_pct'] = (dept_leave_summary['avg_taken'] / dept_leave_summary['avg_entitlement'] * 100).round(1)
            
            # Sort by remaining leave (descending)
            dept_leave_summary = dept_leave_summary.sort_values('avg_remaining', ascending=False)
            
            # Display as table
            st.subheader("Department-wise Leave Balance Summary")
            
            # Format for display
            display_summary = dept_leave_summary.copy()
            display_summary.columns = ['Department', 'Avg. Entitlement', 'Avg. Taken', 'Avg. Remaining', 'Employee Count', 'Utilization %']
            
            # Round decimal values
            for col in ['Avg. Entitlement', 'Avg. Taken', 'Avg. Remaining']:
                display_summary[col] = display_summary[col].round(1)
            
            # Add % sign to utilization
            display_summary['Utilization %'] = display_summary['Utilization %'].apply(lambda x: f"{x}%")
            
            st.dataframe(display_summary, use_container_width=True, hide_index=True)
        
        # Employees with high leave balances
        st.subheader("Employees with High Leave Balances")
        
        # Sort by remaining leave (descending)
        high_balance_employees = employees_with_leave.sort_values('remaining_leave', ascending=False).head(10)
        
        if not high_balance_employees.empty:
            # Select columns to display
            display_cols = ['employee_id', 'first_name', 'last_name', 'department', 
                           'prorated_entitlement', 'leave_taken', 'remaining_leave']
            
            # Keep only columns that exist
            display_cols = [col for col in display_cols if col in high_balance_employees.columns]
            
            # Rename columns for display
            column_names = {
                'employee_id': 'Employee ID',
                'first_name': 'First Name',
                'last_name': 'Last Name',
                'department': 'Department',
                'prorated_entitlement': 'Entitlement',
                'leave_taken': 'Leave Taken',
                'remaining_leave': 'Remaining Leave'
            }
            
            # Select and rename columns
            display_df = high_balance_employees[display_cols].copy()
            display_df.columns = [column_names.get(col, col) for col in display_cols]
            
            # Round decimal values
            for col in ['Entitlement', 'Leave Taken', 'Remaining Leave']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(1)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No employee data available for leave balance analysis.")
    else:
        st.info("Employee data with worker types is needed for leave balance analysis.")
    
    # SEASONALITY TRENDS
    # ===============
    st.markdown("### Seasonality Trends")
    
    # Monthly leave patterns
    if 'month' in leave_df.columns:
        monthly_totals = leave_df.groupby('month')['duration_days'].sum().reset_index()
        
        # Sort chronologically if months are in YYYY-MM format
        if all(m.count('-') == 1 for m in monthly_totals['month']):
            monthly_totals = monthly_totals.sort_values('month')
        
        st.plotly_chart(
            create_line_chart(
                monthly_totals,
                x_col='month',
                y_col='duration_days',
                title="Monthly Leave Patterns",
                height=400,
                yaxis_title="Total Leave Days"
            ),
            use_container_width=True
        )
    
    # Day of week frequency
    if 'day_of_week' in leave_df.columns:
        # Group by day of week
        weekday_counts = leave_df.groupby('day_of_week')['duration_days'].sum().reset_index()
        
        # Map day numbers to names and sort
        day_mapping = {i: day for i, day in enumerate(calendar.day_name)}
        weekday_counts['day_name'] = weekday_counts['day_of_week'].map(day_mapping)
        weekday_counts = weekday_counts.sort_values('day_of_week')
        
        st.plotly_chart(
            create_bar_chart(
                weekday_counts,
                x_col='day_name',
                y_col='duration_days',
                title="Leave Days by Day of Week",
                height=400
            ),
            use_container_width=True
        )
    elif 'start_date' in leave_df.columns:
        # Calculate day of week from start date
        leave_df['day_of_week'] = leave_df['start_date'].dt.weekday
        leave_df['day_name'] = leave_df['day_of_week'].apply(lambda x: calendar.day_name[x])
        
        # Group by day of week
        weekday_counts = leave_df.groupby('day_name')['duration_days'].sum().reset_index()
        
        # Set correct order
        weekday_order = list(calendar.day_name)
        weekday_counts['day_order'] = weekday_counts['day_name'].apply(lambda x: weekday_order.index(x))
        weekday_counts = weekday_counts.sort_values('day_order')
        
        st.plotly_chart(
            create_bar_chart(
                weekday_counts,
                x_col='day_name',
                y_col='duration_days',
                title="Leave Days by Day of Week",
                height=400
            ),
            use_container_width=True
        )
    else:
        st.info("Date information is not available for day of week analysis.")
    
    # DRILL-DOWN FUNCTIONALITY
    # =====================
    st.markdown("### Leave Records Drill-Down")
    
    # Create filters for drill-down
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Employee filter (optional)
        if 'employee_id' in leave_df.columns:
            # Allow searching by employee ID
            employees = sorted(leave_df['employee_id'].unique())
            
            # Add employee names if available
            if not employees_df.empty and 'employee_id' in employees_df.columns:
                if all(col in employees_df.columns for col in ['first_name', 'last_name']):
                    employee_names = {
                        row['employee_id']: f"{row['employee_id']} - {row['first_name']} {row['last_name']}"
                        for _, row in employees_df.iterrows()
                    }
                    employee_options = [employee_names.get(emp_id, str(emp_id)) for emp_id in employees]
                else:
                    employee_options = [str(emp_id) for emp_id in employees]
            else:
                employee_options = [str(emp_id) for emp_id in employees]
            
            selected_employee = st.selectbox(
                "Filter by Employee",
                options=["All Employees"] + employee_options
            )
            
            if selected_employee != "All Employees":
                # Extract employee ID from selection
                if " - " in selected_employee:
                    selected_employee_id = int(selected_employee.split(" - ")[0])
                else:
                    selected_employee_id = int(selected_employee)
                
                # Filter leave data
                leave_df = leave_df[leave_df['employee_id'] == selected_employee_id]
    
    with col2:
        # Department filter (if available)
        if 'department' in leave_df.columns:
            departments = ["All Departments"] + sorted(leave_df['department'].unique())
            selected_department = st.selectbox("Filter by Department", departments)
            
            if selected_department != "All Departments":
                leave_df = leave_df[leave_df['department'] == selected_department]
    
    with col3:
        # Leave type filter
        leave_types = ["All Types"] + sorted(leave_df['leave_type'].unique())
        selected_leave_type = st.selectbox("Filter by Leave Type", leave_types)
        
        if selected_leave_type != "All Types":
            leave_df = leave_df[leave_df['leave_type'] == selected_leave_type]
    
    # Display filtered leave records
    if not leave_df.empty:
        # Sort by start date (descending)
        leave_records = leave_df.sort_values('start_date', ascending=False)
        
        # Select columns to display
        display_cols = ['employee_id', 'start_date', 'end_date', 'duration_days', 
                       'leave_type', 'status', 'department', 'location']
        
        # Keep only columns that exist
        display_cols = [col for col in display_cols if col in leave_records.columns]
        
        # Add employee name if available
        if not employees_df.empty and all(col in employees_df.columns for col in ['employee_id', 'first_name', 'last_name']):
            leave_records = pd.merge(
                leave_records,
                employees_df[['employee_id', 'first_name', 'last_name']],
                on='employee_id',
                how='left'
            )
            
            leave_records['employee_name'] = leave_records['first_name'] + ' ' + leave_records['last_name']
            display_cols = ['employee_id', 'employee_name'] + [col for col in display_cols if col not in ['employee_id']]
        
        # Format dates for display
        if 'start_date' in leave_records.columns:
            leave_records['start_date'] = leave_records['start_date'].dt.strftime('%d %b %Y')
        
        if 'end_date' in leave_records.columns:
            leave_records['end_date'] = leave_records['end_date'].dt.strftime('%d %b %Y')
        
        # Rename columns for display
        column_names = {
            'employee_id': 'Employee ID',
            'employee_name': 'Employee Name',
            'start_date': 'Start Date',
            'end_date': 'End Date',
            'duration_days': 'Duration (Days)',
            'leave_type': 'Leave Type',
            'status': 'Status',
            'department': 'Department',
            'location': 'Location'
        }
        
        display_df = leave_records[display_cols].copy()
        display_df.columns = [column_names.get(col, col) for col in display_cols]
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No leave records match the selected filters.")

def preprocess_leave_data(leave_df):
    """
    Preprocess leave data to extract additional information.
    
    Args:
        leave_df: DataFrame with leave records
        
    Returns:
        Processed DataFrame with additional columns
    """
    # Make a copy to avoid modifying the original
    df = leave_df.copy()
    
    # Extract year and month from start_date if available
    if 'start_date' in df.columns and pd.api.types.is_datetime64_dtype(df['start_date']):
        df['year'] = df['start_date'].dt.year
        df['month'] = df['start_date'].dt.strftime('%Y-%m')
        df['day_of_week'] = df['start_date'].dt.weekday
    
    return df