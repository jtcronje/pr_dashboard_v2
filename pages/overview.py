import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.charts import (
    create_kpi_summary, 
    create_headcount_trend, 
    create_worker_type_breakdown,
    create_farm_map, 
    create_bar_chart, 
    create_pie_chart,
    create_tenure_distribution,
    create_heatmap
)
from utils.data import apply_filters

def render_overview_page(data_dict, filters):
    """
    Render the overview dashboard page with KPIs and high-level summary visualizations.
    
    Args:
        data_dict: Dictionary containing all data frames
        filters: Dictionary containing selected filter values
    """
    # Custom CSS for navy blue title
    st.markdown("""
        <style>
        .main .block-container h1 {
            color: #000080;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create two columns for title and logo
    title_col, logo_col = st.columns([3, 1])
    
    with title_col:
        st.title("PeopleRadar", help="Farm Workforce Management Dashboard")
    
    with logo_col:
        st.image("PeopleRadarLogo.jpeg", width=150)
    
    # Filter data based on selected filters
    employees_df = apply_filters(data_dict['employees'], filters)
    payroll_df = apply_filters(data_dict['payroll'], filters)
    
    # Display date range context
    start_date, end_date = filters['date_range']
    st.markdown(f"**Showing data from:** {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")
    
    # TOP METRICS SECTION
    # ==================
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Total workforce count
        total_employees = len(employees_df)
        
        # Calculate MoM change if possible
        prev_month_filters = filters.copy()
        prev_month_start = start_date.replace(month=start_date.month-1 if start_date.month > 1 else 12)
        prev_month_end = end_date.replace(month=end_date.month-1 if end_date.month > 1 else 12)
        prev_month_filters['date_range'] = (prev_month_start, prev_month_end)
        
        prev_employees_df = apply_filters(data_dict['employees'], prev_month_filters)
        prev_total = len(prev_employees_df)
        
        delta = None
        if prev_total > 0:
            delta = ((total_employees - prev_total) / prev_total) * 100
        
        create_kpi_summary(
            total_employees,
            "Total Workforce",
            delta=delta,
            delta_description="vs previous month"
        )
    
    with col2:
        # Worker type breakdown
        permanent_count = len(employees_df[employees_df['worker_type'] == 'Permanent'])
        seasonal_count = len(employees_df[employees_df['worker_type'] == 'Seasonal'])
        contract_count = len(employees_df[employees_df['worker_type'] == 'Contract'])
        
        # Calculate ratio
        permanent_ratio = permanent_count / total_employees * 100 if total_employees > 0 else 0
        
        create_kpi_summary(
            permanent_count,
            "Permanent Workers",
            delta=round(permanent_ratio, 1),
            delta_description="% of total workforce"
        )
    
    with col3:
        # Seasonal worker count and ratio
        seasonal_ratio = seasonal_count / total_employees * 100 if total_employees > 0 else 0
        
        create_kpi_summary(
            seasonal_count,
            "Seasonal Workers", 
            delta=round(seasonal_ratio, 1),
            delta_description="% of total workforce"
        )
    
    with col4:
        # Calculate average tenure in years
        today = datetime.now()
        tenure_days = (today - employees_df['join_date']).dt.days
        avg_tenure = tenure_days.mean() / 365.25 if len(tenure_days) > 0 else 0
        
        create_kpi_summary(
            round(avg_tenure, 1),
            "Avg. Tenure (Years)",
            delta=None
        )
    
    # FARM LOCATIONS SECTION
    # =====================
    st.markdown("### Farm Locations and Workforce Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Location distribution
        location_counts = employees_df['location'].value_counts().reset_index()
        location_counts.columns = ['location', 'count']
        
        # Add percentage column
        location_counts['percentage'] = (location_counts['count'] / total_employees * 100).round(1)
        
        st.plotly_chart(
            create_farm_map(
                location_counts, 
                'count', 
                title="Workers by Location",
                show_legend=False
            ), 
            use_container_width=True
        )
    
    with col2:
        # Worker type breakdown
        st.plotly_chart(
            create_worker_type_breakdown(
                employees_df, 
                title="Worker Type Distribution",
                show_legend=False
            ),
            use_container_width=True
        )
    
    # WORKFORCE COMPOSITION SECTION
    # ===========================
    st.markdown("### Workforce Composition")
    col1, col2 = st.columns(2)
    
    with col1:
        # Department distribution
        dept_counts = employees_df['department'].value_counts().reset_index()
        dept_counts.columns = ['department', 'count']
        
        # Calculate percentage of total
        dept_counts['percentage'] = (dept_counts['count'] / total_employees * 100).round(1)
        
        # Sort by count descending
        dept_counts = dept_counts.sort_values('count', ascending=False)
        
        # Display pie chart
        st.plotly_chart(
            create_pie_chart(
                dept_counts,
                values_col='count',
                names_col='department',
                title="Distribution by Department",
                show_legend=False
            ),
            use_container_width=True
        )
    
    with col2:
        # Gender distribution
        gender_counts = employees_df['gender'].value_counts().reset_index()
        gender_counts.columns = ['gender', 'count']
        
        # Calculate percentage of total
        gender_counts['percentage'] = (gender_counts['count'] / total_employees * 100).round(1)
        
        # Display pie chart
        st.plotly_chart(
            create_pie_chart(
                gender_counts,
                values_col='count',
                names_col='gender',
                title="Gender Distribution",
                show_legend=False
            ),
            use_container_width=True
        )
    
    # WORKFORCE TRENDS SECTION
    # ======================
    st.markdown("### Workforce Trends")
    
    # Monthly headcount trend
    if len(payroll_df) > 0:
        # Group by month to get headcount
        headcount_trend = payroll_df.groupby('month').agg(
            count=('employee_id', 'nunique')
        ).reset_index()
        
        # Sort by month chronologically
        headcount_trend = headcount_trend.sort_values('month')
        
        # Display chart
        st.plotly_chart(
            create_headcount_trend(
                headcount_trend, 
                time_col='month', 
                count_col='count', 
                title="Monthly Headcount Trend",
                show_legend=False
            ),
            use_container_width=True
        )
    else:
        st.info("No trend data available for the selected filters.")
    
    # DEPARTMENT BREAKDOWN SECTION
    # ==========================
    st.markdown("### Department Breakdown")
    
    # Format table with percentages
    display_df = dept_counts.copy()
    display_df['percentage'] = display_df['percentage'].apply(lambda x: f"{x}%")
    
    # Display full-width table with department breakdown
    st.markdown("##### Workforce by Department")
    st.dataframe(
        display_df.set_index('department'),
        use_container_width=True,
        height=400  # Increased height for better visibility
    )
    
    # TENURE DISTRIBUTION SECTION
    # =========================
    st.markdown("### Employee Tenure")
    
    # Display tenure distribution
    st.plotly_chart(
        create_tenure_distribution(
            employees_df,
            title="Employee Tenure Distribution",
            show_legend=False
        ),
        use_container_width=True
    )
    
    # TENURE HEATMAP SECTION
    # ====================
    st.markdown("### Average Tenure by Department and Location")
    
    # Convert end_date to datetime for compatibility
    end_datetime = pd.to_datetime(end_date)
    
    # Calculate tenure in years based on join_date and selected end date
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
    
    # RECENT HIRES & TERMINATIONS
    # =========================
    st.markdown("### Recent Workforce Changes")
    
    # Calculate recent hires (joined in last 90 days)
    recent_cutoff = datetime.now() - pd.Timedelta(days=90)
    recent_hires = employees_df[employees_df['join_date'] >= recent_cutoff]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"##### Recent Hires (Last 90 Days): {len(recent_hires)}")
        
        if len(recent_hires) > 0:
            # Group by department
            recent_by_dept = recent_hires['department'].value_counts().reset_index()
            recent_by_dept.columns = ['department', 'count']
            
            # Display as bar chart
            st.plotly_chart(
                create_bar_chart(
                    recent_by_dept,
                    x_col='department',
                    y_col='count',
                    title="Recent Hires by Department",
                    height=300
                ),
                use_container_width=True
            )
        else:
            st.info("No recent hires in the selected period.")
    
    with col2:
        # Note: In a real application, we would have termination data
        # For now, just display a placeholder
        st.markdown("##### Recent Terminations (Last 90 Days)")
        st.info("Termination data not available in the current dataset.")
    
    # PAYROLL OVERVIEW
    # ==============
    if len(payroll_df) > 0:
        st.markdown("### Payroll Overview")
        
        # Calculate payroll metrics
        latest_month = payroll_df['month'].max()
        latest_payroll = payroll_df[payroll_df['month'] == latest_month]
        
        total_payroll = latest_payroll['gross_pay'].sum()
        avg_salary = latest_payroll['base_salary'].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_kpi_summary(
                total_payroll,
                "Total Monthly Payroll",
                format_type="currency"
            )
        
        with col2:
            create_kpi_summary(
                avg_salary,
                "Average Salary",
                format_type="currency"
            )
        
        with col3:
            # Calculate overtime percentage
            overtime_total = latest_payroll['overtime_pay'].sum()
            overtime_pct = overtime_total / total_payroll * 100 if total_payroll > 0 else 0
            
            create_kpi_summary(
                overtime_total,
                "Total Overtime Pay",
                delta=round(overtime_pct, 1),
                delta_description="% of total payroll",
                format_type="currency"
            )
    
    # OPTIONAL: ADD FARM PRODUCTIVITY PREVIEW
    # =====================================
    productivity_df = apply_filters(data_dict['productivity'], filters)
    
    if len(productivity_df) > 0:
        st.markdown("### Productivity Preview")
        
        
        # Calculate average performance score by department
        perf_by_dept = productivity_df.groupby('department')['performance_score'].mean().reset_index()
        perf_by_dept['performance_score'] = perf_by_dept['performance_score'].round(1)
        
        # Sort by performance score
        perf_by_dept = perf_by_dept.sort_values('performance_score', ascending=False)
        
        # Display as bar chart
        st.plotly_chart(
            create_bar_chart(
                perf_by_dept,
                x_col='department',
                y_col='performance_score',
                title="Average Performance Score by Department",
                height=300
            ),
            use_container_width=True
        )