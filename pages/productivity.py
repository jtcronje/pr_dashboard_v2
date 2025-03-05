import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
import json
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.charts import (
    create_kpi_summary,
    create_bar_chart,
    create_pie_chart,
    create_line_chart,
    create_scatter_plot,
    create_heatmap,
    create_distribution_chart,
    apply_common_layout
)
from utils.data import apply_filters

def render_productivity_page(data_dict, filters):
    """
    Render the productivity analysis dashboard page.
    
    Args:
        data_dict: Dictionary containing all data frames
        filters: Dictionary containing selected filter values
    """
    # Filter data based on selected filters
    productivity_df = apply_filters(data_dict['productivity'], filters)
    employees_df = apply_filters(data_dict['employees'], filters)
    
    # Display date range context
    start_date, end_date = filters['date_range']
    st.markdown(f"**Showing data from:** {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")
    
    if len(productivity_df) == 0:
        st.warning("No productivity data available for the selected filters.")
        return
    
    # Get the most recent month's data
    latest_month = productivity_df['month'].max()
    latest_productivity = productivity_df[productivity_df['month'] == latest_month]
    
    # PRODUCTIVITY METRICS
    # ==================
    st.markdown("### Productivity Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Average output per worker
        avg_output = latest_productivity['productivity_level'].mean()
        
        # Compare to previous month
        prev_month = sorted(productivity_df['month'].unique())[-2] if len(productivity_df['month'].unique()) > 1 else None
        
        delta = None
        if prev_month:
            prev_month_output = productivity_df[productivity_df['month'] == prev_month]['productivity_level'].mean()
            if prev_month_output > 0:
                delta = ((avg_output - prev_month_output) / prev_month_output) * 100
        
        # Add explanation tooltip
        st.markdown("**Average Output per Worker** ℹ️", help="""
        Average Output per Worker measures the typical quantity of work completed by employees.
        
        The value represents:
        - For farm production: Average kilograms harvested per day
        - For pack house: Average number of boxes packed per day
        - For cattle farming: Average number of activities completed
        - For other departments: Average tasks completed
        
        This metric is affected by:
        • Worker experience (tenure with the company)
        • Seasonal factors (peak harvest vs. off-season)
        • Department-specific base outputs
        """)
        
        create_kpi_summary(
            round(avg_output, 1),
            "",  # Empty title since we added it above with the tooltip
            delta=round(delta, 1) if delta is not None else None,
            delta_description="vs previous month",
            format_type="percent" if delta is not None else None
        )
    
    with col2:
        # Days worked average
        avg_days_worked = latest_productivity['days_worked'].mean()
        max_working_days = 21  # Assuming 21 working days per month
        attendance_rate = (avg_days_worked / max_working_days) * 100 if max_working_days > 0 else 0
        
        create_kpi_summary(
            round(avg_days_worked, 1),
            "Average Days Worked",
            delta=round(attendance_rate, 1),
            delta_description="attendance rate",
            format_type="percent"
        )
    
    with col3:
        # Average performance score
        avg_performance = latest_productivity['performance_score'].mean()
        
        # Compare to previous month
        delta = None
        if prev_month:
            prev_performance = productivity_df[productivity_df['month'] == prev_month]['performance_score'].mean()
            delta = avg_performance - prev_performance
        
        create_kpi_summary(
            round(avg_performance, 1),
            "Average Performance Score",
            delta=round(delta, 1) if delta is not None else None,
            delta_description="vs previous month"
        )
    
    # PRODUCTIVITY COMPARISON BY DEPARTMENT
    # ==================================
    st.markdown("### Productivity by Department")
    
    if 'department' in productivity_df.columns:
        # Department productivity comparison
        dept_productivity = latest_productivity.groupby('department').agg(
            avg_productivity=('productivity_level', 'mean'),
            avg_days_worked=('days_worked', 'mean'),
            avg_performance=('performance_score', 'mean'),
            worker_count=('employee_id', 'nunique')
        ).reset_index()
        
        # Sort by average productivity
        dept_productivity = dept_productivity.sort_values('avg_productivity', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Department productivity comparison
            st.plotly_chart(
                create_bar_chart(
                    dept_productivity,
                    x_col='department',
                    y_col='avg_productivity',
                    title="Average Output by Department",
                    height=400
                ),
                use_container_width=True,
                help="""
                This chart shows the average output by department, where:
                - Farm production depts (Citrus, Maize, Grape): Average kg harvested per day
                - Pack House: Average boxes packed per day
                - Cattle Farming: Average cattle activities completed per day
                - Other departments: Average tasks completed per day
                
                Higher values indicate greater productivity. Department outputs are affected by 
                the nature of work, seasonal factors, and worker experience.
                """
            )
        
        with col2:
            # Department performance comparison
            st.plotly_chart(
                create_bar_chart(
                    dept_productivity,
                    x_col='department',
                    y_col='avg_performance',
                    title="Average Performance Score by Department",
                    height=400
                ),
                use_container_width=True
            )
    
    # PRODUCTIVITY TRENDS
    # ================
    st.markdown("### Productivity Trends")
    
    # Aggregate by month
    monthly_productivity = productivity_df.groupby('month').agg(
        avg_productivity=('productivity_level', 'mean'),
        avg_days_worked=('days_worked', 'mean'),
        avg_performance=('performance_score', 'mean')
    ).reset_index()
    
    # Sort by month
    monthly_productivity = monthly_productivity.sort_values('month')
    
    # Create trend chart
    st.plotly_chart(
        create_line_chart(
            monthly_productivity,
            x_col='month',
            y_col='avg_productivity',
            title="Productivity Trend Over Time",
            height=400,
            yaxis_title="Average Output per Worker"
        ),
        use_container_width=True,
        help="""
        This chart shows the trend of average worker output over time.
        
        The values represent:
        - For farm production: Average kilograms harvested per day
        - For pack house: Average number of boxes packed per day
        - For cattle farming: Average number of activities completed
        - For other departments: Average tasks completed
        
        Seasonal patterns are normal, with higher productivity typically seen during 
        peak harvest seasons (Jan, Feb, Nov, Dec) and lower productivity during off-seasons.
        """
    )
    
    # ATTENDANCE METRICS
    # ===============
    st.markdown("### Attendance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Attendance by department
        if 'department' in productivity_df.columns:
            dept_attendance = latest_productivity.groupby('department').agg(
                avg_days_worked=('days_worked', 'mean')
            ).reset_index()
            
            # Calculate attendance rate
            dept_attendance['attendance_rate'] = (dept_attendance['avg_days_worked'] / max_working_days) * 100
            
            # Sort by attendance rate
            dept_attendance = dept_attendance.sort_values('attendance_rate', ascending=False)
            
            st.plotly_chart(
                create_bar_chart(
                    dept_attendance,
                    x_col='department',
                    y_col='attendance_rate',
                    title="Attendance Rate by Department",
                    text_format="percent",
                    height=400
                ),
                use_container_width=True
            )
    
    with col2:
        # Attendance trend over time
        st.plotly_chart(
            create_line_chart(
                monthly_productivity,
                x_col='month',
                y_col='avg_days_worked',
                title="Days Worked Trend",
                height=400,
                yaxis_title="Average Days Worked"
            ),
            use_container_width=True
        )
    
    # PERFORMANCE METRICS
    # ================
    st.markdown("### Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance score distribution
        st.plotly_chart(
            create_distribution_chart(
                latest_productivity,
                x_col='performance_score',
                title="Performance Score Distribution",
                bin_size=0.5,
                height=400,
                x_title="Performance Score (1-5)"
            ),
            use_container_width=True
        )
    
    with col2:
        # Performance trend over time
        st.plotly_chart(
            create_line_chart(
                monthly_productivity,
                x_col='month',
                y_col='avg_performance',
                title="Performance Score Trend",
                height=400,
                yaxis_title="Average Performance Score"
            ),
            use_container_width=True
        )
    
    # PERFORMANCE HEATMAP
    # ================
    st.markdown("### Performance Heatmap")
    
    # Check if we have the necessary data for the heatmap
    if 'department' in productivity_df.columns and 'location' in productivity_df.columns:
        # Create cross-tabulation of department and location with average performance
        perf_matrix = pd.pivot_table(
            latest_productivity,
            values='performance_score',
            index='department',
            columns='location',
            aggfunc='mean'
        ).round(1)
        
        # Convert to long format for heatmap
        perf_long = pd.DataFrame()
        
        # Only proceed if we have data
        if not perf_matrix.empty:
            perf_long = perf_matrix.reset_index().melt(
                id_vars='department',
                var_name='location',
                value_name='performance_score'
            )
            
            # Create heatmap
            heatmap_fig = create_heatmap(
                perf_long,
                x_col='location',
                y_col='department',
                value_col='performance_score',
                title="Performance Score by Department and Location",
                height=500
            )
            
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Store the current data for AI interpretation
            st.session_state['performance_heatmap_data'] = perf_matrix.to_dict()
            
            # Add AI Interpretation button
            if st.button("Generate AI Interpretation", key="interpret_performance_heatmap"):
                with st.spinner("Generating insights..."):
                    try:
                        # Set OpenAI API key from secrets
                        os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "your-openai-api-key-here")
                        
                        # Create the LangChain model with OpenAI
                        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
                        
                        # Prepare the data for interpretation
                        heatmap_data_str = json.dumps(perf_matrix.to_dict())
                        
                        # Create the prompt
                        prompt = f"""You are a workforce analytics expert analyzing a heatmap of performance scores.
                        
                        The heatmap shows performance scores (on a scale of 1-5) across different departments (y-axis) and locations (x-axis).
                        
                        Here's the data: {heatmap_data_str}
                        
                        Please provide a detailed interpretation of this performance heatmap, including:
                        1. Which departments perform best/worst overall?
                        2. Which locations show the strongest/weakest performance?
                        3. Are there any notable patterns or outliers in specific department-location combinations?
                        4. What actionable recommendations would you suggest based on this data?
                        
                        Focus on providing insights that would be valuable for workforce management.

                        Please limit the response to 250 words.
                        """
                        
                        # Get the interpretation
                        response = model.invoke([HumanMessage(content=prompt)])
                        
                        # Display the interpretation
                        st.markdown("### AI Interpretation")
                        st.markdown(response.content)
                    except Exception as e:
                        st.error(f"Error generating AI interpretation: {str(e)}")
                        st.info("Ensure that you have set up your OpenAI API key correctly in st.secrets or as an environment variable.")
        else:
            st.info("Not enough data for performance heatmap.")
    else:
        st.info("Department or location data is not available for heatmap creation.")
    
    # DISCIPLINARY ANALYSIS
    # ==================
    st.markdown("### Disciplinary Analysis")
    
    # Count incidents
    total_incidents = latest_productivity['incident_count'].sum()
    workers_with_incidents = len(latest_productivity[latest_productivity['incident_count'] > 0])
    incident_rate = (workers_with_incidents / len(latest_productivity)) * 100 if len(latest_productivity) > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_kpi_summary(
            total_incidents,
            "Total Incidents",
            delta=None
        )
    
    with col2:
        create_kpi_summary(
            workers_with_incidents,
            "Workers with Incidents",
            delta=round(incident_rate, 1),
            delta_description="% of workforce",
            format_type="percent"
        )
    
    with col3:
        # Average incidents per affected worker
        avg_incidents = total_incidents / workers_with_incidents if workers_with_incidents > 0 else 0
        
        create_kpi_summary(
            round(avg_incidents, 2),
            "Avg. Incidents per Affected Worker",
            delta=None
        )
    
    # Incident types breakdown
    if 'incident_type' in productivity_df.columns:
        # Filter to only records with incidents
        incidents_df = latest_productivity[latest_productivity['incident_count'] > 0].copy()
        
        if len(incidents_df) > 0:
            # Count by incident type
            incident_types = incidents_df['incident_type'].value_counts().reset_index()
            incident_types.columns = ['incident_type', 'count']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Incident types pie chart
                st.plotly_chart(
                    create_pie_chart(
                        incident_types,
                        values_col='count',
                        names_col='incident_type',
                        title="Incident Types Breakdown"
                    ),
                    use_container_width=True
                )
            
            with col2:
                # Incidents by department
                if 'department' in incidents_df.columns:
                    dept_incidents = incidents_df.groupby('department').agg(
                        total_incidents=('incident_count', 'sum'),
                        worker_count=('employee_id', 'nunique')
                    ).reset_index()
                    
                    # Calculate incident rate
                    dept_incidents['incident_rate'] = dept_incidents['total_incidents'] / dept_incidents['worker_count']
                    
                    # Sort by incident rate
                    dept_incidents = dept_incidents.sort_values('incident_rate', ascending=False)
                    
                    st.plotly_chart(
                        create_bar_chart(
                            dept_incidents,
                            x_col='department',
                            y_col='incident_rate',
                            title="Incident Rate by Department",
                            height=400
                        ),
                        use_container_width=True
                    )
        else:
            st.info("No incidents recorded for the selected period.")
    
    # DISCIPLINARY INCIDENT ANALYSIS
    # ===========================
    st.markdown("### Disciplinary Incident Analysis")
    
    # Incidents over time
    if 'location' in productivity_df.columns:
        # Aggregate by month and location
        monthly_incidents = productivity_df.groupby(['month', 'location']).agg(
            total_incidents=('incident_count', 'sum')
        ).reset_index()
        
        # Sort by month
        monthly_incidents = monthly_incidents.sort_values('month')
        
        # Create line chart
        st.plotly_chart(
            create_line_chart(
                monthly_incidents,
                x_col='month',
                y_col='total_incidents',
                color_col='location',
                title="Incidents Over Time by Location",
                height=400,
                yaxis_title="Total Incidents"
            ),
            use_container_width=True
        )
    
    # Incidents by type
    if 'incident_type' in productivity_df.columns:
        # Aggregate by incident type
        incident_types = productivity_df.groupby('incident_type').agg(
            total_incidents=('incident_count', 'sum')
        ).reset_index()
        
        # Sort by total incidents
        incident_types = incident_types.sort_values('total_incidents', ascending=False)
        
        # Create bar chart
        st.plotly_chart(
            create_bar_chart(
                incident_types,
                x_col='incident_type',
                y_col='total_incidents',
                title="Incidents by Type",
                height=400
            ),
            use_container_width=True
        )

    # Incidents by type
    if 'incident_type' in productivity_df.columns:
        # Aggregate by month and incident type
        monthly_incident_types = productivity_df.groupby(['month', 'incident_type']).agg(
            total_incidents=('incident_count', 'sum')
        ).reset_index()
        
        # Sort by month
        monthly_incident_types = monthly_incident_types.sort_values('month')
        
        # Create line chart showing incident types over time
        st.plotly_chart(
            create_line_chart(
                monthly_incident_types,
                x_col='month',
                y_col='total_incidents',
                color_col='incident_type',
                title="Incident Types Over Time",
                height=400,
                yaxis_title="Total Incidents"
            ),
            use_container_width=True
        )
    
    # AI Analysis
    if 'location' in productivity_df.columns and 'incident_type' in productivity_df.columns:
        # Prepare data for AI analysis
        ai_data = productivity_df[['month', 'location', 'department', 'incident_type', 'incident_count']].copy()
        
        # Check if 'month' is datetime type before formatting
        if pd.api.types.is_datetime64_any_dtype(ai_data['month']):
            ai_data['month'] = ai_data['month'].dt.strftime('%Y-%m')
        
        # Store the current data for AI interpretation
        st.session_state['disciplinary_incident_data'] = ai_data.to_dict()
        
        # Add AI Interpretation button
        if st.button("Generate AI Analysis", key="interpret_disciplinary_incidents"):
            with st.spinner("Generating insights..."):
                try:
                    # Set OpenAI API key from secrets
                    os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "your-openai-api-key-here")
                    
                    # Create the LangChain model with OpenAI
                    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
                    
                    # Prepare the data for interpretation
                    ai_data_str = json.dumps(ai_data.to_dict())
                    
                    # Get time trends (simplified approach - just report if incidents are increasing/decreasing by type)
                    if 'month' in ai_data.columns and len(ai_data['month'].unique()) > 1:
                        # Get earliest and latest periods for comparison
                        months = sorted(ai_data['month'].unique())
                        if len(months) > 4:
                            # Compare first 2 months vs last 2 months
                            early_months = months[:2]
                            late_months = months[-2:]
                        else:
                            # Compare first half vs second half
                            mid_point = len(months) // 2
                            early_months = months[:mid_point]
                            late_months = months[mid_point:]
                        
                        # Calculate trend by incident type
                        early_data = ai_data[ai_data['month'].isin(early_months)]
                        late_data = ai_data[ai_data['month'].isin(late_months)]
                        
                        early_counts = early_data.groupby('incident_type')['incident_count'].sum()
                        late_counts = late_data.groupby('incident_type')['incident_count'].sum()
                        
                        # Calculate percent change
                        trend_data = {}
                        for incident_type in set(early_counts.index) | set(late_counts.index):
                            early_count = early_counts.get(incident_type, 0)
                            late_count = late_counts.get(incident_type, 0)
                            if early_count > 0:
                                percent_change = ((late_count - early_count) / early_count) * 100
                            else:
                                percent_change = float('inf') if late_count > 0 else 0
                            
                            trend_data[incident_type] = {
                                'early_count': float(early_count),
                                'late_count': float(late_count),
                                'percent_change': round(percent_change, 1)
                            }
                        
                        time_trends_dict = {'trend_by_type': trend_data}
                    else:
                        time_trends_dict = {}
                    
                    # Prepare summarized data for interpretation instead of sending the full dataset
                    # Summarize by location
                    location_summary = ai_data.groupby('location')['incident_count'].sum().sort_values(ascending=False).to_dict()
                    
                    # Summarize by department
                    department_summary = ai_data.groupby('department')['incident_count'].sum().sort_values(ascending=False).to_dict()
                    
                    # Summarize by incident type
                    type_summary = ai_data.groupby('incident_type')['incident_count'].sum().sort_values(ascending=False).to_dict()
                    
                    # Create summarized data dictionary
                    summarized_data = {
                        "location_summary": location_summary,
                        "department_summary": department_summary,
                        "incident_type_summary": type_summary,
                        "recent_trends": time_trends_dict
                    }
                    
                    # Convert to JSON string (much smaller than the full dataset)
                    summarized_data_str = json.dumps(summarized_data)
                    
                    # Create the prompt
                    prompt = f"""Analyze this summarized disciplinary incident data: {summarized_data_str}
                    
                    Focus on:
                    1. Top/bottom locations and departments by incident rates
                    2. Which incident types are increasing or decreasing
                    3. Key recommendations for management
                    
                    Keep your response brief and actionable (150 words max).
                    """
                    
                    # Get the interpretation
                    response = model.invoke([HumanMessage(content=prompt)])
                    
                    # Display the interpretation
                    st.markdown("### AI Analysis")
                    st.markdown(response.content)
                except Exception as e:
                    st.error(f"Error generating AI analysis: {str(e)}")
                    st.info("Ensure that you have set up your OpenAI API key correctly in st.secrets or as an environment variable.")
    
    # DRILL-DOWN ANALYSIS
    # =================
    st.markdown("### Drill-Down Analysis")
    
    with st.expander("Departmental Drill-Down"):
        if 'department' in productivity_df.columns:
            # Department selector
            departments = sorted(productivity_df['department'].unique())
            selected_dept = st.selectbox("Select Department", departments)
            
            if selected_dept:
                # Filter data for selected department
                dept_data = latest_productivity[latest_productivity['department'] == selected_dept]
                
                # Show summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    dept_avg_productivity = dept_data['productivity_level'].mean()
                    overall_avg = latest_productivity['productivity_level'].mean()
                    productivity_vs_avg = ((dept_avg_productivity / overall_avg) - 1) * 100 if overall_avg > 0 else 0
                    
                    create_kpi_summary(
                        round(dept_avg_productivity, 1),
                        f"{selected_dept} Avg. Productivity",
                        delta=round(productivity_vs_avg, 1),
                        delta_description="vs. overall average",
                        format_type="percent"
                    )
                
                with col2:
                    dept_avg_attendance = dept_data['days_worked'].mean()
                    overall_attendance = latest_productivity['days_worked'].mean()
                    attendance_vs_avg = ((dept_avg_attendance / overall_attendance) - 1) * 100 if overall_attendance > 0 else 0
                    
                    create_kpi_summary(
                        round(dept_avg_attendance, 1),
                        f"{selected_dept} Avg. Days Worked",
                        delta=round(attendance_vs_avg, 1),
                        delta_description="vs. overall average",
                        format_type="percent"
                    )
                
                with col3:
                    dept_avg_performance = dept_data['performance_score'].mean()
                    overall_performance = latest_productivity['performance_score'].mean()
                    performance_vs_avg = dept_avg_performance - overall_performance
                    
                    create_kpi_summary(
                        round(dept_avg_performance, 1),
                        f"{selected_dept} Avg. Performance",
                        delta=round(performance_vs_avg, 1),
                        delta_description="vs. overall average"
                    )
                
                # Show employee-level data for the department
                st.markdown(f"#### {selected_dept} Employee Performance")
                
                # Merge with employee names if available
                if 'employee_id' in dept_data.columns:
                    if 'first_name' in employees_df.columns and 'last_name' in employees_df.columns:
                        dept_data = pd.merge(
                            dept_data,
                            employees_df[['employee_id', 'first_name', 'last_name', 'job_title']],
                            on='employee_id',
                            how='left'
                        )
                        
                        # Add full name column
                        dept_data['employee_name'] = dept_data['first_name'] + ' ' + dept_data['last_name']
                
                # Select columns to display
                display_cols = ['employee_id', 'employee_name', 'job_title', 
                               'productivity_level', 'days_worked', 'performance_score', 'incident_count']
                
                # Keep only columns that exist
                display_cols = [col for col in display_cols if col in dept_data.columns]
                
                # Sort by performance score descending
                dept_data_sorted = dept_data.sort_values('performance_score', ascending=False)
                
                # Display as table
                st.dataframe(dept_data_sorted[display_cols], use_container_width=True)
                
                # Optional: Show performance distribution within department
                st.markdown(f"#### {selected_dept} Performance Distribution")
                
                st.plotly_chart(
                    create_distribution_chart(
                        dept_data,
                        x_col='performance_score',
                        title=f"{selected_dept} Performance Score Distribution",
                        height=300,
                        x_title="Performance Score"
                    ),
                    use_container_width=True
                )
        else:
            st.info("Department data is not available for drill-down analysis.")
    
    # Optional: Job title drill-down
    with st.expander("Job Title Drill-Down"):
        if 'job_title' in productivity_df.columns:
            # Job title selector
            job_titles = sorted(productivity_df['job_title'].unique())
            selected_job = st.selectbox("Select Job Title", job_titles)
            
            if selected_job:
                # Filter data for selected job title
                job_data = latest_productivity[latest_productivity['job_title'] == selected_job]
                
                # Show summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    job_avg_productivity = job_data['productivity_level'].mean()
                    productivity_vs_avg = ((job_avg_productivity / overall_avg) - 1) * 100 if overall_avg > 0 else 0
                    
                    create_kpi_summary(
                        round(job_avg_productivity, 1),
                        f"{selected_job} Avg. Productivity",
                        delta=round(productivity_vs_avg, 1),
                        delta_description="vs. overall average",
                        format_type="percent"
                    )
                
                with col2:
                    job_avg_attendance = job_data['days_worked'].mean()
                    attendance_vs_avg = ((job_avg_attendance / overall_attendance) - 1) * 100 if overall_attendance > 0 else 0
                    
                    create_kpi_summary(
                        round(job_avg_attendance, 1),
                        f"{selected_job} Avg. Days Worked",
                        delta=round(attendance_vs_avg, 1),
                        delta_description="vs. overall average",
                        format_type="percent"
                    )
                
                with col3:
                    job_avg_performance = job_data['performance_score'].mean()
                    performance_vs_avg = job_avg_performance - overall_performance
                    
                    create_kpi_summary(
                        round(job_avg_performance, 1),
                        f"{selected_job} Avg. Performance",
                        delta=round(performance_vs_avg, 1),
                        delta_description="vs. overall average"
                    )
                
                # Display performance by department for this job title
                if 'department' in job_data.columns:
                    job_dept_perf = job_data.groupby('department').agg(
                        avg_performance=('performance_score', 'mean'),
                        count=('employee_id', 'nunique')
                    ).reset_index()
                    
                    # Sort by performance
                    job_dept_perf = job_dept_perf.sort_values('avg_performance', ascending=False)
                    
                    st.markdown(f"#### {selected_job} Performance by Department")
                    
                    st.plotly_chart(
                        create_bar_chart(
                            job_dept_perf,
                            x_col='department',
                            y_col='avg_performance',
                            title=f"{selected_job} Performance by Department",
                            height=300
                        ),
                        use_container_width=True
                    )
        else:
            st.info("Job title data is not available for drill-down analysis.")