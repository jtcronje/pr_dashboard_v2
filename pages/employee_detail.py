import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.charts import (
    create_kpi_summary,
    create_bar_chart,
    create_line_chart,
    format_currency,
    apply_common_layout
)
from utils.data import apply_filters

# Add a custom JSON encoder class after imports
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def render_employee_detail_page(data_dict, filters):
    """
    Render the employee detail page.
    
    Args:
        data_dict: Dictionary containing all data frames
        filters: Dictionary containing selected filter values
    """
    
    # Filter data based on selected filters - but keep all employees for search
    all_employees_df = data_dict['employees'].copy()
    
    # Check if we have employee data available
    if all_employees_df.empty:
        st.warning("No employee data available.")
        return
    
    # Set up filtered data for other aspects
    employees_df = apply_filters(all_employees_df, filters)
    payroll_df = apply_filters(data_dict['payroll'], filters) if 'payroll' in data_dict else pd.DataFrame()
    leave_df = apply_filters(data_dict['leave'], filters) if 'leave' in data_dict else pd.DataFrame()
    productivity_df = apply_filters(data_dict['productivity'], filters) if 'productivity' in data_dict else pd.DataFrame()
    
    # EMPLOYEE SEARCH SECTION
    # =====================
    st.markdown("### Employee Search")
    
    # Create search options with employee ID and name
    if 'first_name' in all_employees_df.columns and 'last_name' in all_employees_df.columns:
        all_employees_df['full_name'] = all_employees_df['first_name'] + ' ' + all_employees_df['last_name']
        search_options = all_employees_df[['employee_id', 'full_name']].apply(
            lambda x: f"{x['employee_id']} - {x['full_name']}", axis=1
        ).tolist()
    else:
        search_options = all_employees_df['employee_id'].astype(str).tolist()
    
    # Search box with autocomplete
    search_term = st.selectbox(
        "Search for an employee by ID or name",
        options=search_options,
        index=None,
        placeholder="Type to search..."
    )
    
    # If no employee is selected yet, show some instructions
    if not search_term:
        st.info("Select an employee from the dropdown above to view their detailed information.")
        
        # Show a table of employees as a reference
        with st.expander("Browse Employees"):
            # Determine which columns to show
            display_cols = ['employee_id', 'full_name', 'department', 'job_title', 'location', 'worker_type']
            display_cols = [col for col in display_cols if col in all_employees_df.columns]
            
            # Sort by employee ID
            browse_df = all_employees_df.sort_values('employee_id')
            
            # Display paginated table
            st.dataframe(browse_df[display_cols], use_container_width=True)
        
        return
    
    # Extract employee ID from search selection
    employee_id = int(search_term.split(' - ')[0])
    
    # Get the selected employee's data
    employee_data = all_employees_df[all_employees_df['employee_id'] == employee_id].iloc[0]
    
    # EMPLOYEE PROFILE SECTION
    # =====================
    st.markdown("### Employee Profile")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display employee photo or avatar
        st.image("https://via.placeholder.com/150", width=150, caption=employee_data.get('full_name', f"Employee #{employee_id}"))
        
        # Download report button
        if st.button("📥 Download Employee Report", type="primary"):
            pdf_buffer = generate_employee_report(
                employee_id, 
                employee_data, 
                payroll_df, 
                leave_df, 
                productivity_df
            )
            
            # Create a download link
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name=f"Employee_{employee_id}_Report.pdf",
                mime="application/pdf",
                key="download_report"
            )
    
    with col2:
        # Personal Information Card
        st.markdown("#### Personal Information")
        
        personal_info = {}
        
        # Add available personal fields
        if 'full_name' in employee_data:
            personal_info["Full Name"] = employee_data['full_name']
        elif 'first_name' in employee_data and 'last_name' in employee_data:
            personal_info["Full Name"] = f"{employee_data['first_name']} {employee_data['last_name']}"
        
        personal_info["Employee ID"] = employee_id
        
        if 'date_of_birth' in employee_data:
            personal_info["Date of Birth"] = employee_data['date_of_birth'].strftime('%d %b %Y')
            
            # Calculate age
            today = datetime.now()
            personal_info["Age"] = int((today - employee_data['date_of_birth']).days / 365.25)
        
        if 'gender' in employee_data:
            personal_info["Gender"] = employee_data['gender']
        
        if 'ethnicity' in employee_data:
            personal_info["Ethnicity"] = employee_data['ethnicity']
        
        if 'has_disability' in employee_data:
            personal_info["Disability Status"] = "Yes" if employee_data['has_disability'] else "No"
        
        # Display as a table
        personal_df = pd.DataFrame({
            'Field': personal_info.keys(),
            'Value': personal_info.values()
        })
        
        st.dataframe(personal_df, use_container_width=True, hide_index=True)
    
    # Employment Details Card
    st.markdown("#### Employment Details")
    
    employment_info = {}
    
    if 'job_title' in employee_data:
        employment_info["Job Title"] = employee_data['job_title']
    
    if 'department' in employee_data:
        employment_info["Department"] = employee_data['department']
    
    if 'location' in employee_data:
        employment_info["Location"] = employee_data['location']
    
    if 'worker_type' in employee_data:
        employment_info["Worker Type"] = employee_data['worker_type']
    
    if 'join_date' in employee_data:
        employment_info["Join Date"] = employee_data['join_date'].strftime('%d %b %Y')
        
        # Calculate tenure
        today = datetime.now()
        tenure_days = (today - employee_data['join_date']).days
        years = tenure_days // 365
        months = (tenure_days % 365) // 30
        employment_info["Tenure"] = f"{years} years, {months} months"
    
    if 'manager_id' in employee_data and not pd.isna(employee_data['manager_id']):
        manager_id = int(employee_data['manager_id'])
        manager_data = all_employees_df[all_employees_df['employee_id'] == manager_id]
        
        if not manager_data.empty:
            if 'full_name' in manager_data.iloc[0]:
                employment_info["Manager"] = f"{manager_data.iloc[0]['full_name']} (ID: {manager_id})"
            elif 'first_name' in manager_data.iloc[0] and 'last_name' in manager_data.iloc[0]:
                employment_info["Manager"] = f"{manager_data.iloc[0]['first_name']} {manager_data.iloc[0]['last_name']} (ID: {manager_id})"
            else:
                employment_info["Manager"] = f"Employee ID: {manager_id}"
    
    # Display as a table
    employment_df = pd.DataFrame({
        'Field': employment_info.keys(),
        'Value': employment_info.values()
    })
    
    st.dataframe(employment_df, use_container_width=True, hide_index=True)
    
    # COMPENSATION HISTORY
    # =================
    st.markdown("### Compensation History")
    
    if not payroll_df.empty:
        # Filter payroll data for this employee
        employee_payroll = payroll_df[payroll_df['employee_id'] == employee_id].copy()
        
        if not employee_payroll.empty:
            # Sort by month
            employee_payroll = employee_payroll.sort_values('month')
            
            # Salary history line chart
            st.plotly_chart(
                create_line_chart(
                    employee_payroll,
                    x_col='month',
                    y_col='base_salary',
                    title="Salary History",
                    height=400,
                    yaxis_title="Monthly Base Salary (R)",
                    text_format="currency"
                ),
                use_container_width=True
            )
            
            # Compensation changes table
            st.markdown("#### Compensation Details")
            
            # Calculate month-over-month changes
            employee_payroll['prev_salary'] = employee_payroll['base_salary'].shift(1)
            employee_payroll['salary_change'] = employee_payroll['base_salary'] - employee_payroll['prev_salary']
            employee_payroll['salary_change_pct'] = (employee_payroll['salary_change'] / employee_payroll['prev_salary'] * 100).fillna(0)
            
            # Select records where salary changed
            salary_changes = employee_payroll[employee_payroll['salary_change'] != 0].copy()
            
            if not salary_changes.empty:
                # Format for display
                display_changes = salary_changes[['month', 'base_salary', 'salary_change', 'salary_change_pct']].copy()
                display_changes['base_salary'] = display_changes['base_salary'].apply(format_currency)
                display_changes['salary_change'] = display_changes['salary_change'].apply(
                    lambda x: ('+' if x > 0 else '') + format_currency(x)
                )
                display_changes['salary_change_pct'] = display_changes['salary_change_pct'].apply(
                    lambda x: f"{'+' if x > 0 else ''}{x:.2f}%"
                )
                
                # Rename columns for display
                display_changes.columns = ['Month', 'Base Salary', 'Change Amount', 'Change %']
                
                st.dataframe(display_changes, use_container_width=True, hide_index=True)
            else:
                st.info("No salary changes recorded for this employee.")
        else:
            st.info("No payroll data available for this employee.")
    else:
        st.info("Payroll data is not available.")
    
    # PERFORMANCE HISTORY
    # ================
    st.markdown("### Performance History")
    
    if not productivity_df.empty:
        # Filter productivity data for this employee
        employee_performance = productivity_df[productivity_df['employee_id'] == employee_id].copy()
        
        if not employee_performance.empty:
            # Sort by month
            employee_performance = employee_performance.sort_values('month')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance score line chart
                st.plotly_chart(
                    create_line_chart(
                        employee_performance,
                        x_col='month',
                        y_col='performance_score',
                        title="Performance Score History",
                        height=350,
                        yaxis_title="Performance Score (1-5)",
                        range_y=[1, 5]
                    ),
                    use_container_width=True
                )
            
            with col2:
                # Productivity level line chart
                st.plotly_chart(
                    create_line_chart(
                        employee_performance,
                        x_col='month',
                        y_col='productivity_level',
                        title="Productivity History",
                        height=350,
                        yaxis_title="Productivity Level"
                    ),
                    use_container_width=True
                )
            
            # Performance summary table
            st.markdown("#### Performance Summary")
            
            # Calculate average metrics
            avg_performance = employee_performance['performance_score'].mean()
            avg_productivity = employee_performance['productivity_level'].mean()
            avg_days_worked = employee_performance['days_worked'].mean()
            total_incidents = employee_performance['incident_count'].sum()
            
            # Get department averages for comparison if department data is available
            dept_comparison = {}
            if 'department' in employee_data:
                department = employee_data['department']
                dept_performance = productivity_df[productivity_df['department'] == department]
                
                if not dept_performance.empty:
                    dept_avg_performance = dept_performance['performance_score'].mean()
                    dept_avg_productivity = dept_performance['productivity_level'].mean()
                    dept_avg_days = dept_performance['days_worked'].mean()
                    
                    performance_vs_dept = avg_performance - dept_avg_performance
                    productivity_vs_dept = ((avg_productivity / dept_avg_productivity) - 1) * 100 if dept_avg_productivity > 0 else 0
                    attendance_vs_dept = ((avg_days_worked / dept_avg_days) - 1) * 100 if dept_avg_days > 0 else 0
                    
                    dept_comparison = {
                        'Performance vs. Dept': f"{'+' if performance_vs_dept > 0 else ''}{performance_vs_dept:.2f} points",
                        'Productivity vs. Dept': f"{'+' if productivity_vs_dept > 0 else ''}{productivity_vs_dept:.2f}%",
                        'Attendance vs. Dept': f"{'+' if attendance_vs_dept > 0 else ''}{attendance_vs_dept:.2f}%"
                    }
            
            # Combine metrics
            performance_summary = {
                'Average Performance Score': f"{avg_performance:.2f} / 5.0",
                'Average Productivity Level': f"{avg_productivity:.2f}",
                'Average Days Worked per Month': f"{avg_days_worked:.1f} days",
                'Total Incidents': f"{total_incidents}"
            }
            
            # Add department comparisons if available
            performance_summary.update(dept_comparison)
            
            # Display as a table
            performance_df = pd.DataFrame({
                'Metric': performance_summary.keys(),
                'Value': performance_summary.values()
            })
            
            st.dataframe(performance_df, use_container_width=True, hide_index=True)
            
            # Detailed performance table by month
            with st.expander("View Monthly Performance Details"):
                monthly_perf = employee_performance[['month', 'performance_score', 'productivity_level', 
                                                   'days_worked', 'incident_count']].copy()
                
                monthly_perf.columns = ['Month', 'Performance Score', 'Productivity Level', 
                                      'Days Worked', 'Incidents']
                
                st.dataframe(monthly_perf.sort_values('Month', ascending=False), 
                           use_container_width=True, hide_index=True)
                
            # Add AI Insights for performance trends
            if len(employee_performance) > 1:
                st.markdown("#### AI Performance Insights")
                
                if st.button("Generate AI Insights", key="generate_insights"):
                    with st.spinner("Analyzing performance trends..."):
                        try:
                            # Prepare the data for the AI model
                            perf_data = employee_performance.sort_values('month')
                            
                            # Calculate trends
                            first_month = perf_data.iloc[0]['month']
                            last_month = perf_data.iloc[-1]['month']
                            
                            first_performance = float(perf_data.iloc[0]['performance_score'])
                            last_performance = float(perf_data.iloc[-1]['performance_score'])
                            perf_change = float(last_performance - first_performance)
                            
                            first_productivity = float(perf_data.iloc[0]['productivity_level'])
                            last_productivity = float(perf_data.iloc[-1]['productivity_level'])
                            prod_change = float(last_productivity - first_productivity)
                            
                            total_incidents = int(perf_data['incident_count'].sum())
                            avg_incidents = float(perf_data['incident_count'].mean())
                            recent_incidents = int(perf_data.iloc[-3:]['incident_count'].sum()) if len(perf_data) >= 3 else None
                            
                            # Convert the performance history data to ensure it's JSON serializable
                            performance_history = []
                            for _, row in perf_data[['month', 'performance_score', 'productivity_level', 'incident_count']].iterrows():
                                performance_history.append({
                                    'month': row['month'],
                                    'performance_score': float(row['performance_score']),
                                    'productivity_level': float(row['productivity_level']),
                                    'incident_count': int(row['incident_count'])
                                })
                            
                            # Prepare the data object to send to AI
                            analysis_data = {
                                "employee_info": {
                                    "id": int(employee_id),
                                    "name": employee_data.get('full_name', f"Employee {employee_id}"),
                                    "department": employee_data.get('department', 'Unknown'),
                                    "job_title": employee_data.get('job_title', 'Unknown'),
                                },
                                "performance_history": performance_history,
                                "summary_metrics": {
                                    "time_period": f"{first_month} to {last_month}",
                                    "performance_change": perf_change,
                                    "productivity_change": prod_change,
                                    "total_incidents": total_incidents,
                                    "avg_incidents_per_month": avg_incidents,
                                    "recent_incidents": recent_incidents
                                }
                            }
                            
                            # Create the prompt for OpenAI
                            prompt = f"""
                            You are a farm workforce analytics expert. Analyze this employee's performance data and provide insights.
                            
                            DATA:
                            {json.dumps(analysis_data, indent=2, cls=NumpyEncoder)}
                            
                            Please analyze:
                            1. Performance trend: Is the employee's performance score improving, declining, or stable? Provide specific numbers.
                            2. Productivity trend: Is the employee's productivity level improving, declining, or stable? Provide specific numbers.
                            3. Incident analysis: Evaluate the pattern of incidents. Is there cause for concern?
                            4. Overall assessment: Based on all metrics, is this employee's performance trajectory positive or concerning?
                            5. Recommendations: What specific actions would you recommend for management?

                            Combine the above insights into a single response. Please mention whether the employee is performing well or not and whether there performance have been incresing or remaining stable. 
                            Please also comment on whether rhere is cause for concern based on the incident data and declining productivity.
                            
                            Format your response with clear section headings. Be concise but thorough in your analysis.

                            Your response should not exceed 200 words.
                            """
                            
                            # Call the OpenAI API
                            try:
                                model = ChatOpenAI(
                                    model="gpt-4o-mini", 
                                    temperature=0.7
                                )
                                
                                response = model.invoke([
                                    HumanMessage(content=prompt)
                                ])
                                
                                # Display the AI insights
                                st.markdown(response.content)
                                
                            except Exception as ai_error:
                                st.error(f"Error generating AI insights: {str(ai_error)}")
                                st.info("Make sure your OpenAI API key is correctly set in .streamlit/secrets.toml")
                        
                        except Exception as e:
                            st.error(f"Error preparing data for analysis: {str(e)}")
        else:
            st.info("No performance data available for this employee.")
    else:
        st.info("Performance data is not available.")
    
    # DISCIPLINARY RECORD
    # ================
    st.markdown("### Disciplinary Record")
    
    if not productivity_df.empty:
        # Filter incidents for this employee
        employee_incidents = productivity_df[
            (productivity_df['employee_id'] == employee_id) & 
            (productivity_df['incident_count'] > 0)
        ].copy()
        
        if not employee_incidents.empty:
            # Prepare incident data
            incidents_table = employee_incidents[['month', 'incident_count', 'incident_type']].copy()
            incidents_table.columns = ['Month', 'Count', 'Type']
            
            # Sort by month descending
            incidents_table = incidents_table.sort_values('Month', ascending=False)
            
            st.dataframe(incidents_table, use_container_width=True, hide_index=True)
            
            # Show incident trend if there's enough data
            if len(employee_incidents) > 1:
                st.plotly_chart(
                    create_line_chart(
                        employee_incidents,
                        x_col='month',
                        y_col='incident_count',
                        title="Incident History",
                        height=300,
                        yaxis_title="Number of Incidents"
                    ),
                    use_container_width=True
                )
        else:
            st.success("No disciplinary incidents recorded for this employee.")
    else:
        st.info("Disciplinary data is not available.")
    
    # LEAVE HISTORY
    # ===========
    st.markdown("### Leave History")
    
    if not leave_df.empty:
        # Filter leave data for this employee
        employee_leave = leave_df[leave_df['employee_id'] == employee_id].copy()
        
        if not employee_leave.empty:
            # Calculate leave totals by type
            leave_by_type = employee_leave.groupby('leave_type')['duration_days'].sum().reset_index()
            leave_by_type.columns = ['Leave Type', 'Total Days']
            
            # Leave utilization bar chart
            st.plotly_chart(
                create_bar_chart(
                    leave_by_type,
                    x_col='Leave Type',
                    y_col='Total Days',
                    title="Leave Utilization by Type",
                    height=350
                ),
                use_container_width=True
            )
            
            # Leave history table
            st.markdown("#### Leave History Details")
            
            # Prepare leave history table
            leave_history = employee_leave[['start_date', 'end_date', 'duration_days', 'leave_type', 'status']].copy()
            
            # Format dates
            leave_history['start_date'] = leave_history['start_date'].dt.strftime('%d %b %Y')
            leave_history['end_date'] = leave_history['end_date'].dt.strftime('%d %b %Y')
            
            # Rename columns
            leave_history.columns = ['Start Date', 'End Date', 'Duration (Days)', 'Leave Type', 'Status']
            
            # Sort by start date descending
            leave_history = leave_history.sort_values('Start Date', ascending=False)
            
            st.dataframe(leave_history, use_container_width=True, hide_index=True)
        else:
            st.info("No leave records found for this employee.")
    else:
        st.info("Leave data is not available.")

def generate_employee_report(employee_id, employee_data, payroll_df, leave_df, productivity_df):
    """
    Generate a PDF report for the selected employee.
    
    Args:
        employee_id: The ID of the selected employee
        employee_data: Series containing employee information
        payroll_df: Payroll dataframe
        leave_df: Leave dataframe
        productivity_df: Productivity dataframe
        
    Returns:
        BytesIO buffer containing the PDF report
    """
    # Create a BytesIO buffer to save the PDF to
    buffer = BytesIO()
    
    # Set up the PDF with matplotlib
    with PdfPages(buffer) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        if 'full_name' in employee_data:
            employee_name = employee_data['full_name']
        elif 'first_name' in employee_data and 'last_name' in employee_data:
            employee_name = f"{employee_data['first_name']} {employee_data['last_name']}"
        else:
            employee_name = f"Employee #{employee_id}"
            
        ax.text(0.5, 0.9, "Employee Report", fontsize=24, weight='bold', ha='center')
        ax.text(0.5, 0.85, employee_name, fontsize=20, ha='center')
        ax.text(0.5, 0.8, f"ID: {employee_id}", fontsize=16, ha='center')
        
        # Date of report
        ax.text(0.5, 0.75, f"Report Date: {datetime.now().strftime('%d %b %Y')}", fontsize=12, ha='center')
        
        # Add company logo placeholder
        ax.text(0.5, 0.5, "Farm Group Logo", fontsize=16, ha='center', bbox=dict(facecolor='lightgray', alpha=0.5))
        
        # Add footer
        ax.text(0.5, 0.1, "CONFIDENTIAL - FOR INTERNAL USE ONLY", fontsize=10, ha='center', style='italic')
        ax.text(0.5, 0.05, "Generated by People Radar Dashboard", fontsize=8, ha='center')
        
        pdf.savefig(fig)
        plt.close()
        
        # Employee Profile page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Page title
        ax.text(0.5, 0.95, "Employee Profile", fontsize=16, weight='bold', ha='center')
        
        # Personal Information Section
        ax.text(0.1, 0.9, "Personal Information", fontsize=14, weight='bold')
        
        y_pos = 0.85
        personal_info = []
        
        personal_info.append(f"Name: {employee_name}")
        personal_info.append(f"Employee ID: {employee_id}")
        
        if 'date_of_birth' in employee_data:
            personal_info.append(f"Date of Birth: {employee_data['date_of_birth'].strftime('%d %b %Y')}")
            
            # Calculate age
            today = datetime.now()
            age = int((today - employee_data['date_of_birth']).days / 365.25)
            personal_info.append(f"Age: {age} years")
        
        if 'gender' in employee_data:
            personal_info.append(f"Gender: {employee_data['gender']}")
        
        if 'ethnicity' in employee_data:
            personal_info.append(f"Ethnicity: {employee_data['ethnicity']}")
        
        if 'has_disability' in employee_data:
            personal_info.append(f"Disability Status: {'Yes' if employee_data['has_disability'] else 'No'}")
        
        # Print personal info
        for info in personal_info:
            ax.text(0.1, y_pos, info, fontsize=10)
            y_pos -= 0.05
        
        # Employment Details Section
        ax.text(0.1, y_pos - 0.05, "Employment Details", fontsize=14, weight='bold')
        y_pos -= 0.1
        
        employment_info = []
        
        if 'job_title' in employee_data:
            employment_info.append(f"Job Title: {employee_data['job_title']}")
        
        if 'department' in employee_data:
            employment_info.append(f"Department: {employee_data['department']}")
        
        if 'location' in employee_data:
            employment_info.append(f"Location: {employee_data['location']}")
        
        if 'worker_type' in employee_data:
            employment_info.append(f"Worker Type: {employee_data['worker_type']}")
        
        if 'join_date' in employee_data:
            employment_info.append(f"Join Date: {employee_data['join_date'].strftime('%d %b %Y')}")
            
            # Calculate tenure
            today = datetime.now()
            tenure_days = (today - employee_data['join_date']).days
            years = tenure_days // 365
            months = (tenure_days % 365) // 30
            employment_info.append(f"Tenure: {years} years, {months} months")
        
        if 'manager_id' in employee_data and not pd.isna(employee_data['manager_id']):
            employment_info.append(f"Manager ID: {int(employee_data['manager_id'])}")
        
        # Print employment info
        for info in employment_info:
            ax.text(0.1, y_pos, info, fontsize=10)
            y_pos -= 0.05
        
        pdf.savefig(fig)
        plt.close()
        
        # Performance and Compensation page
        if not productivity_df.empty or not payroll_df.empty:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            
            # Page title
            ax.text(0.5, 0.95, "Performance & Compensation", fontsize=16, weight='bold', ha='center')
            
            # Compensation Section
            if not payroll_df.empty:
                ax.text(0.1, 0.9, "Compensation Summary", fontsize=14, weight='bold')
                
                # Filter payroll data for this employee
                employee_payroll = payroll_df[payroll_df['employee_id'] == employee_id].copy()
                
                if not employee_payroll.empty:
                    # Get latest salary
                    latest_month = employee_payroll['month'].max()
                    latest_salary = employee_payroll[employee_payroll['month'] == latest_month]['base_salary'].iloc[0]
                    
                    y_pos = 0.85
                    ax.text(0.1, y_pos, f"Current Monthly Salary: R {latest_salary:,.2f}", fontsize=10)
                    y_pos -= 0.05
                    
                    # Calculate average overtime
                    avg_overtime = employee_payroll['overtime_pay'].mean()
                    ax.text(0.1, y_pos, f"Average Monthly Overtime: R {avg_overtime:,.2f}", fontsize=10)
                    y_pos -= 0.05
                    
                    # Calculate salary growth
                    first_salary = employee_payroll.sort_values('month')['base_salary'].iloc[0]
                    salary_growth = ((latest_salary / first_salary) - 1) * 100 if first_salary > 0 else 0
                    ax.text(0.1, y_pos, f"Salary Growth: {salary_growth:.2f}%", fontsize=10)
                    y_pos -= 0.05
                    
                    # Plot salary history if there's enough data
                    if len(employee_payroll) > 1:
                        y_pos -= 0.05
                        salary_ax = fig.add_axes([0.1, y_pos - 0.3, 0.8, 0.3])
                        
                        # Sort by month
                        employee_payroll = employee_payroll.sort_values('month')
                        
                        # Convert month to datetime for better plotting
                        employee_payroll['month_dt'] = pd.to_datetime(employee_payroll['month'] + '-01')
                        
                        # Plot salary history
                        salary_ax.plot(employee_payroll['month_dt'], employee_payroll['base_salary'], 
                                     marker='o', linestyle='-', color='#1E88E5')
                        
                        # Format axes
                        salary_ax.set_title("Salary History", fontsize=12)
                        salary_ax.set_ylabel("Monthly Base Salary (R)", fontsize=10)
                        salary_ax.grid(True, linestyle='--', alpha=0.7)
                        
                        # Format x-axis as dates
                        salary_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                        plt.setp(salary_ax.get_xticklabels(), rotation=45, ha='right')
                        
                        # Format y-axis as currency
                        salary_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"R {x:,.0f}"))
                        
                        y_pos -= 0.35
            
            # Performance Section
            if not productivity_df.empty:
                ax.text(0.1, y_pos - 0.05, "Performance Summary", fontsize=14, weight='bold')
                y_pos -= 0.1
                
                # Filter productivity data for this employee
                employee_performance = productivity_df[productivity_df['employee_id'] == employee_id].copy()
                
                if not employee_performance.empty:
                    # Calculate average metrics
                    avg_performance = employee_performance['performance_score'].mean()
                    avg_productivity = employee_performance['productivity_level'].mean()
                    avg_days_worked = employee_performance['days_worked'].mean()
                    total_incidents = employee_performance['incident_count'].sum()
                    
                    ax.text(0.1, y_pos, f"Average Performance Score: {avg_performance:.2f} / 5.0", fontsize=10)
                    y_pos -= 0.05
                    ax.text(0.1, y_pos, f"Average Productivity Level: {avg_productivity:.2f}", fontsize=10)
                    y_pos -= 0.05
                    ax.text(0.1, y_pos, f"Average Days Worked per Month: {avg_days_worked:.1f} days", fontsize=10)
                    y_pos -= 0.05
                    ax.text(0.1, y_pos, f"Total Incidents: {total_incidents}", fontsize=10)
                    y_pos -= 0.05
                    
                    # Plot performance history if there's enough data
                    if len(employee_performance) > 1:
                        y_pos -= 0.05
                        perf_ax = fig.add_axes([0.1, y_pos - 0.3, 0.8, 0.3])
                        
                        # Sort by month
                        employee_performance = employee_performance.sort_values('month')
                        
                        # Convert month to datetime for better plotting
                        employee_performance['month_dt'] = pd.to_datetime(employee_performance['month'] + '-01')
                        
                        # Plot performance history
                        perf_ax.plot(employee_performance['month_dt'], employee_performance['performance_score'], 
                                   marker='o', linestyle='-', color='#1E88E5')
                        
                        # Format axes
                        perf_ax.set_title("Performance Score History", fontsize=12)
                        perf_ax.set_ylabel("Performance Score (1-5)", fontsize=10)
                        perf_ax.set_ylim([1, 5])
                        perf_ax.grid(True, linestyle='--', alpha=0.7)
                        
                        # Format x-axis as dates
                        perf_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                        plt.setp(perf_ax.get_xticklabels(), rotation=45, ha='right')
            
            pdf.savefig(fig)
            plt.close()
        
        # Leave History page
        if not leave_df.empty:
            # Filter leave data for this employee
            employee_leave = leave_df[leave_df['employee_id'] == employee_id].copy()
            
            if not employee_leave.empty:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                
                # Page title
                ax.text(0.5, 0.95, "Leave History", fontsize=16, weight='bold', ha='center')
                
                # Leave summary
                ax.text(0.1, 0.9, "Leave Summary", fontsize=14, weight='bold')
                
                # Calculate leave totals
                total_leave_days = employee_leave['duration_days'].sum()
                leave_by_type = employee_leave.groupby('leave_type')['duration_days'].sum()
                
                y_pos = 0.85
                ax.text(0.1, y_pos, f"Total Leave Days: {total_leave_days}", fontsize=10)
                y_pos -= 0.05
                
                # Print leave by type
                for leave_type, days in leave_by_type.items():
                    ax.text(0.1, y_pos, f"{leave_type}: {days} days", fontsize=10)
                    y_pos -= 0.05
                
                # Leave bar chart
                if len(leave_by_type) > 0:
                    y_pos -= 0.05
                    leave_ax = fig.add_axes([0.1, y_pos - 0.3, 0.8, 0.3])
                    
                    # Prepare data for chart
                    leave_types = leave_by_type.index.tolist()
                    leave_days = leave_by_type.values
                    
                    # Create bar chart
                    bars = leave_ax.bar(leave_types, leave_days, color='#1E88E5')
                    
                    # Add data labels
                    for bar in bars:
                        height = bar.get_height()
                        leave_ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f"{height:.0f}",
                                    ha='center', va='bottom', fontsize=9)
                    
                    # Format axes
                    leave_ax.set_title("Leave Days by Type", fontsize=12)
                    leave_ax.set_ylabel("Days", fontsize=10)
                    leave_ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                    
                    y_pos -= 0.35
                
                # Leave table
                y_pos -= 0.05
                ax.text(0.1, y_pos, "Recent Leave Records", fontsize=12, weight='bold')
                y_pos -= 0.05
                
                # Table headers
                col_labels = ['Start Date', 'End Date', 'Duration', 'Leave Type', 'Status']
                col_widths = [0.2, 0.2, 0.1, 0.25, 0.15]  # Proportional column widths
                
                # Calculate column positions
                col_positions = [0.1]  # Start position
                for width in col_widths[:-1]:
                    col_positions.append(col_positions[-1] + width)
                
                # Draw table headers
                for i, (label, pos) in enumerate(zip(col_labels, col_positions)):
                    ax.text(pos, y_pos, label, fontsize=9, weight='bold')
                
                y_pos -= 0.03
                ax.axhline(y=y_pos, xmin=0.1, xmax=0.9, color='black', linewidth=0.5)
                y_pos -= 0.03
                
                # Sort leave records by start date (descending)
                recent_leave = employee_leave.sort_values('start_date', ascending=False).head(10)
                
                # Draw table rows
                for _, row in recent_leave.iterrows():
                    start_date = row['start_date'].strftime('%d %b %Y')
                    end_date = row['end_date'].strftime('%d %b %Y')
                    duration = f"{row['duration_days']} days"
                    leave_type = row['leave_type']
                    status = row['status']
                    
                    # Print row values
                    ax.text(col_positions[0], y_pos, start_date, fontsize=8)
                    ax.text(col_positions[1], y_pos, end_date, fontsize=8)
                    ax.text(col_positions[2], y_pos, duration, fontsize=8)
                    ax.text(col_positions[3], y_pos, leave_type, fontsize=8)
                    ax.text(col_positions[4], y_pos, status, fontsize=8)
                    
                    y_pos -= 0.03
                    
                    # Stop if we're running out of space
                    if y_pos < 0.1:
                        break
                
                pdf.savefig(fig)
                plt.close()
        
        # Add a final page with legal disclaimer
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Page title
        ax.text(0.5, 0.95, "Legal Information", fontsize=16, weight='bold', ha='center')
        
        # Disclaimer
        disclaimer = [
            "This report contains confidential information and is intended solely for internal use.",
            "The information contained in this report may not be disclosed to any third party without prior consent.",
            "This report was generated automatically by the People Radar Dashboard system.",
            "The information is provided as-is and the company makes no guarantees about its accuracy or completeness.",
            "",
            f"Report generated on {datetime.now().strftime('%d %b %Y at %H:%M')}",
            f"Employee ID: {employee_id}"
        ]
        
        y_pos = 0.85
        for line in disclaimer:
            ax.text(0.1, y_pos, line, fontsize=10)
            y_pos -= 0.05
        
        pdf.savefig(fig)
        plt.close()
    
    # Reset buffer position to the beginning
    buffer.seek(0)
    
    return buffer