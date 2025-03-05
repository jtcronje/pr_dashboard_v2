import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.charts import (
    create_kpi_summary,
    create_bar_chart,
    create_stacked_bar_chart,
    create_line_chart,
    create_distribution_chart,
    format_currency,
    apply_common_layout
)
from utils.data import apply_filters

def render_payroll_page(data_dict, filters):
    """
    Render the payroll analysis dashboard page with salary and payment visualizations.
    
    Args:
        data_dict: Dictionary containing all data frames
        filters: Dictionary containing selected filter values
    """
    # Filter data based on selected filters
    payroll_df = apply_filters(data_dict['payroll'], filters)
    employees_df = apply_filters(data_dict['employees'], filters)
    
    # Merge employee data with payroll for more detailed analysis
    if not payroll_df.empty and not employees_df.empty:
        # Make copies to avoid modifying original data
        payroll_df = payroll_df.copy()
        employees_df = employees_df.copy()
        
        # Merge relevant employee data into payroll
        employee_cols = ['employee_id', 'job_title', 'department', 'location', 'worker_type', 'join_date']
        avail_cols = [col for col in employee_cols if col in employees_df.columns]
        
        if 'employee_id' in avail_cols:
            # Get latest payroll data per employee
            payroll_with_employee = pd.merge(
                payroll_df,
                employees_df[avail_cols],
                on='employee_id',
                how='left',
                suffixes=('', '_employee')
            )
            
            # Use employee data where payroll data might be missing
            for col in ['department', 'location', 'worker_type']:
                if f"{col}_employee" in payroll_with_employee.columns:
                    mask = payroll_with_employee[col].isna()
                    payroll_with_employee.loc[mask, col] = payroll_with_employee.loc[mask, f"{col}_employee"]
                    payroll_with_employee = payroll_with_employee.drop(f"{col}_employee", axis=1)
            
            payroll_df = payroll_with_employee
    
    # Display date range context
    start_date, end_date = filters['date_range']
    st.markdown(f"**Showing data from:** {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")
    
    if len(payroll_df) == 0:
        st.warning("No payroll data available for the selected filters.")
        return
    
    # TOP PAYROLL METRICS
    # ==================
    st.markdown("### Payroll Metrics")
    
    # Get the most recent month's data
    latest_month = payroll_df['month'].max()
    latest_payroll = payroll_df[payroll_df['month'] == latest_month]
    
    # Calculate year-to-date figures
    current_year = str(datetime.now().year)
    ytd_payroll = payroll_df[payroll_df['month'].str.startswith(current_year)]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Total monthly payroll
        total_monthly_payroll = latest_payroll['gross_pay'].sum()
        
        # Compare to previous month
        prev_month = sorted(payroll_df['month'].unique())[-2] if len(payroll_df['month'].unique()) > 1 else None
        
        delta = None
        if prev_month:
            prev_month_payroll = payroll_df[payroll_df['month'] == prev_month]['gross_pay'].sum()
            if prev_month_payroll > 0:
                delta = ((total_monthly_payroll - prev_month_payroll) / prev_month_payroll) * 100
        
        create_kpi_summary(
            total_monthly_payroll,
            "Monthly Payroll",
            delta=delta,
            delta_description="vs previous month",
            format_type="currency"
        )
    
    with col2:
        # Year to date payroll
        ytd_total = ytd_payroll['gross_pay'].sum()
        
        # Calculate monthly average
        monthly_avg = ytd_total / len(ytd_payroll['month'].unique()) if len(ytd_payroll['month'].unique()) > 0 else 0
        
        create_kpi_summary(
            ytd_total,
            "Year-to-Date Payroll",
            delta=monthly_avg,
            delta_description="monthly average",
            format_type="currency"
        )
    
    with col3:
        # Average salary
        avg_salary = latest_payroll['base_salary'].mean()
        
        # Overtime as percentage
        overtime_total = latest_payroll['overtime_pay'].sum()
        overtime_pct = overtime_total / total_monthly_payroll * 100 if total_monthly_payroll > 0 else 0
        
        create_kpi_summary(
            avg_salary,
            "Average Salary",
            delta=round(overtime_pct, 1),
            delta_description="% overtime",
            format_type="currency"
        )
    
    # SALARY ANALYSIS
    # =============
    st.markdown("### Salary Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average salary by department
        if 'department' in payroll_df.columns:
            dept_salary = latest_payroll.groupby('department')['base_salary'].mean().reset_index()
            dept_salary = dept_salary.sort_values('base_salary', ascending=False)
            
            st.plotly_chart(
                create_bar_chart(
                    dept_salary,
                    x_col='department',
                    y_col='base_salary',
                    title="Average Salary by Department",
                    text_format="currency",
                    height=400
                ),
                use_container_width=True
            )
    
    with col2:
        # Salary distribution
        st.plotly_chart(
            create_distribution_chart(
                latest_payroll,
                x_col='base_salary',
                title="Salary Distribution",
                height=400,
                x_title="Monthly Salary (R)"
            ),
            use_container_width=True
        )
    
    # Job title salary ranges
    if 'job_title' in payroll_df.columns:
        st.markdown("### Salary Ranges by Job Title", help="""
        How to interpret this boxplot:
        - The middle line in each box represents the median salary
        - The box represents the middle 50% of salaries (interquartile range)
        - The whiskers extend to the min and max values (excluding outliers)
        - Individual points represent outliers (unusually high or low salaries)
        
        This visualization helps identify salary ranges and variations within job titles.
        """)
        
        # Filter to top 10 job titles by count for readability
        job_counts = latest_payroll['job_title'].value_counts()
        top_jobs = job_counts[job_counts > 1].nlargest(10).index.tolist()
        
        if top_jobs:
            # Filter data to only include the top jobs
            job_salary_df = latest_payroll[latest_payroll['job_title'].isin(top_jobs)]
            
            # Create proper box plot
            fig = px.box(
                job_salary_df,
                x='job_title',
                y='base_salary',
                title="Salary Ranges by Job Title",
                color_discrete_sequence=['#1E88E5'],
                height=450,
                points='outliers'  # Only show outlier points
            )
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Job Title",
                yaxis_title="Monthly Salary (R)",
                boxmode='group',
                height=450
            )
            
            # Apply common styling
            fig = apply_common_layout(fig)
            
            # Format y-axis to currency
            fig.update_yaxes(tickprefix="R ", tickformat=",")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to display salary ranges by job title.")
    
    # PAYROLL TRENDS
    # ============
    st.markdown("### Payroll Trends")
    
    # Monthly total payroll trend
    monthly_payroll = payroll_df.groupby('month').agg(
        total_payroll=('gross_pay', 'sum'),
        total_base=('base_salary', 'sum'),
        total_overtime=('overtime_pay', 'sum'),
        employee_count=('employee_id', 'nunique')
    ).reset_index()
    
    # Sort by month
    monthly_payroll = monthly_payroll.sort_values('month')
    
    # Calculate growth rates
    monthly_payroll['payroll_growth'] = monthly_payroll['total_payroll'].pct_change() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly payroll trend
        fig = go.Figure()
        
        # Add total payroll line
        fig.add_trace(go.Scatter(
            x=monthly_payroll['month'],
            y=monthly_payroll['total_payroll'],
            mode='lines+markers',
            name='Gross Payroll',
            line=dict(color='#1E88E5', width=3),
            marker=dict(size=8)
        ))
        
        # Add base salary line
        fig.add_trace(go.Scatter(
            x=monthly_payroll['month'],
            y=monthly_payroll['total_base'],
            mode='lines+markers',
            name='Base Salaries',
            line=dict(color='#90CAF9', width=2, dash='dot'),
            marker=dict(size=6)
        ))
        
        # Add overtime line
        fig.add_trace(go.Scatter(
            x=monthly_payroll['month'],
            y=monthly_payroll['total_overtime'],
            mode='lines+markers',
            name='Overtime Pay',
            line=dict(color='#F57C00', width=2),
            marker=dict(size=6)
        ))
        
        # Apply styling
        fig.update_layout(
            title="Monthly Payroll Components",
            xaxis_title="Month",
            yaxis_title="Amount (R)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            height=400
        )
        
        # Format y-axis to currency
        fig.update_yaxes(tickprefix="R ", tickformat=",")
        
        # Apply common styling
        fig = apply_common_layout(fig)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Payroll growth rate trend
        fig = go.Figure()
        
        # Add growth rate line
        fig.add_trace(go.Scatter(
            x=monthly_payroll['month'],
            y=monthly_payroll['payroll_growth'],
            mode='lines+markers',
            name='Growth Rate',
            line=dict(color='#1E88E5', width=3),
            marker=dict(size=8)
        ))
        
        # Add zero line reference
        fig.add_shape(
            type="line",
            x0=monthly_payroll['month'].iloc[0],
            y0=0,
            x1=monthly_payroll['month'].iloc[-1],
            y1=0,
            line=dict(
                color="red",
                width=1,
                dash="dash",
            )
        )
        
        # Apply styling
        fig.update_layout(
            title="Monthly Payroll Growth Rate",
            xaxis_title="Month",
            yaxis_title="Growth Rate (%)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            height=400
        )
        
        # Apply common styling
        fig = apply_common_layout(fig)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # DEPARTMENT COMPARISON
    # ===================
    st.markdown("### Department Comparison")
    
    if 'department' in payroll_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Total payroll by department
            dept_payroll = latest_payroll.groupby('department').agg(
                total_payroll=('gross_pay', 'sum'),
                employee_count=('employee_id', 'nunique')
            ).reset_index()
            
            # Sort by total payroll
            dept_payroll = dept_payroll.sort_values('total_payroll', ascending=False)
            
            # Create payroll by department chart
            st.plotly_chart(
                create_bar_chart(
                    dept_payroll,
                    x_col='department',
                    y_col='total_payroll',
                    title="Total Payroll by Department",
                    text_format="currency",
                    height=400
                ),
                use_container_width=True
            )
        
        with col2:
            # Calculate per-employee payroll by department
            dept_payroll['per_employee'] = dept_payroll['total_payroll'] / dept_payroll['employee_count']
            
            # Create per-employee payroll chart
            st.plotly_chart(
                create_bar_chart(
                    dept_payroll,
                    x_col='department',
                    y_col='per_employee',
                    title="Average Payroll per Employee by Department",
                    text_format="currency",
                    height=400
                ),
                use_container_width=True
            )
    
    # WORKER TYPE ANALYSIS
    # =================
    st.markdown("### Worker Type Analysis")
    
    if 'worker_type' in payroll_df.columns:
        # Payroll by worker type
        worker_payroll = latest_payroll.groupby('worker_type').agg(
            total_payroll=('gross_pay', 'sum'),
            employee_count=('employee_id', 'nunique'),
            avg_salary=('base_salary', 'mean')
        ).reset_index()
        
        # Compare average salaries by worker type
        st.plotly_chart(
            create_bar_chart(
                worker_payroll,
                x_col='worker_type',
                y_col='avg_salary',
                title="Average Salary by Worker Type",
                text_format="currency",
                height=400
            ),
            use_container_width=True
        )
    
    # COST PER EMPLOYEE TREND
    # =====================
    st.markdown("### Cost per Employee Trend")
    
    # Calculate cost per employee over time
    monthly_payroll['cost_per_employee'] = monthly_payroll['total_payroll'] / monthly_payroll['employee_count']
    
    # Create line chart
    st.plotly_chart(
        create_line_chart(
            monthly_payroll,
            x_col='month',
            y_col='cost_per_employee',
            title="Average Cost per Employee Over Time",
            height=400,
            yaxis_title="Cost per Employee (R)",
            text_format="currency"
        ),
        use_container_width=True
    )
    
    # INDIVIDUAL SALARY BREAKDOWN
    # =========================
    st.markdown("### Individual Salary Analysis")
    
    # Allow drilling down to individual level
    with st.expander("View Individual Salary Details"):
        # Create a searchable table of employee salaries
        if 'job_title' in latest_payroll.columns:
            search_term = st.text_input("Search by Employee ID, Name, or Job Title", "")
            
            # Get the latest payroll data with names if available
            if 'first_name' in employees_df.columns and 'last_name' in employees_df.columns:
                latest_with_names = pd.merge(
                    latest_payroll,
                    employees_df[['employee_id', 'first_name', 'last_name']],
                    on='employee_id',
                    how='left'
                )
                
                # Add full name column
                latest_with_names['full_name'] = latest_with_names['first_name'] + ' ' + latest_with_names['last_name']
                salary_data = latest_with_names
            else:
                salary_data = latest_payroll
            
            # Filter based on search term
            if search_term:
                filtered_data = salary_data[
                    (salary_data['employee_id'].astype(str).str.contains(search_term, case=False)) |
                    (salary_data['job_title'].str.contains(search_term, case=False, na=False)) |
                    (salary_data.get('full_name', '').str.contains(search_term, case=False, na=False))
                ]
            else:
                filtered_data = salary_data
            
            # Select columns to display
            display_cols = ['employee_id', 'full_name', 'job_title', 'department', 
                           'base_salary', 'overtime_pay', 'gross_pay', 'net_pay']
            
            # Keep only columns that exist
            display_cols = [col for col in display_cols if col in filtered_data.columns]
            
            # Format currency columns
            for col in ['base_salary', 'overtime_pay', 'gross_pay', 'net_pay']:
                if col in filtered_data.columns:
                    filtered_data[col] = filtered_data[col].apply(format_currency)
            
            # Display as table
            if len(filtered_data) > 0:
                st.dataframe(filtered_data[display_cols], use_container_width=True)
            else:
                st.info("No matching records found.")