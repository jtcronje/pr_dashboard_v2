import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import os
import random
from typing import Dict, List, Tuple, Optional, Union

# Initialize Faker for generating realistic employee data
fake = Faker()
Faker.seed(42)  # For reproducibility
np.random.seed(42)

# Constants for data generation - Farm specific
FARM_OPERATIONS = [
    'Citrus Production', 'Maize Production', 'Grape Production', 'Cattle Farming',
    'Pack House Operations', 'Maintenance', 'Administration', 'Logistics'
]

FARM_LOCATIONS = [
    'Ceres', 'Dullstroom', 'Graaff-Reinet', 'Upington', 'Citrusdal', 
    'Ceres Pack House', 'Citrusdal Pack House'
]

# Map operations to specific locations (not all operations at all locations)
LOCATION_OPERATIONS = {
    'Ceres': ['Citrus Production', 'Pack House Operations', 'Maintenance', 'Administration', 'Logistics'],
    'Dullstroom': ['Cattle Farming', 'Maintenance', 'Administration'],
    'Graaff-Reinet': ['Maize Production', 'Maintenance', 'Administration', 'Logistics'],
    'Upington': ['Grape Production', 'Maintenance', 'Administration', 'Logistics'],
    'Citrusdal': ['Citrus Production', 'Maintenance', 'Administration', 'Logistics'],
    'Ceres Pack House': ['Pack House Operations', 'Maintenance', 'Administration', 'Logistics'],
    'Citrusdal Pack House': ['Pack House Operations', 'Maintenance', 'Administration', 'Logistics']
}

JOB_TITLES = {
    'Citrus Production': ['Farm Worker', 'Harvester', 'Irrigation Specialist', 'Tractor Driver', 'Farm Supervisor'],
    'Maize Production': ['Farm Worker', 'Harvester', 'Tractor Driver', 'Field Supervisor', 'Crop Specialist'],
    'Grape Production': ['Vineyard Worker', 'Harvester', 'Pruner', 'Vineyard Supervisor', 'Viticulturist'],
    'Cattle Farming': ['Farm Hand', 'Cattle Herder', 'Animal Care Specialist', 'Milking Technician', 'Livestock Supervisor'],
    'Pack House Operations': ['Sorter', 'Packer', 'Quality Inspector', 'Machine Operator', 'Floor Supervisor', 'Logistics Coordinator'],
    'Maintenance': ['Mechanic', 'Electrician', 'Maintenance Worker', 'Equipment Technician', 'Maintenance Supervisor'],
    'Administration': ['Admin Clerk', 'Payroll Officer', 'HR Assistant', 'Office Manager', 'Data Entry Clerk'],
    'Logistics': ['Driver', 'Forklift Operator', 'Dispatcher', 'Warehouse Worker', 'Logistics Supervisor']
}

# Management roles spanning multiple operations
MANAGEMENT_ROLES = [
    'Farm Manager', 'Operations Manager', 'Production Manager', 
    'HR Manager', 'Finance Manager', 'General Manager'
]

# Worker types
WORKER_TYPES = ['Permanent', 'Seasonal', 'Contract']
WORKER_TYPE_DISTRIBUTION = [0.4, 0.5, 0.1]  # 40% permanent, 50% seasonal, 10% contract

ETHNICITY_GROUPS = [
    'Black African', 'Coloured', 'White', 'Indian/Asian'
]

# Adjusted for South African demographic context
ETHNICITY_DISTRIBUTION = [0.76, 0.09, 0.09, 0.06]

GENDERS = ['Male', 'Female']
GENDER_DISTRIBUTION = [0.55, 0.45]  # Approximate distribution for farm workforce

LEAVE_TYPES = ['Annual Leave', 'Sick Leave', 'Family Responsibility', 'Maternity/Paternity', 'Unpaid Leave']
INCIDENT_TYPES = ['Safety Incident', 'Equipment Damage', 'Tardiness', 'Unauthorized Absence', 'Policy Violation']

# Base salary ranges by job level (in South African Rand)
SALARY_RANGES = {
    'Worker': (3500, 6000),          # Farm workers, entry level
    'Specialist': (6000, 12000),      # Specialized roles
    'Driver': (5000, 8000),           # Drivers, operators
    'Supervisor': (8000, 15000),      # Supervisory roles 
    'Coordinator': (10000, 16000),    # Coordinators
    'Manager': (18000, 40000),        # Management roles
    'Seasonal': (3000, 5000)          # Seasonal workers (monthly equivalent)
}

def get_job_level(job_title):
    """Determine salary level based on job title"""
    if 'Manager' in job_title:
        return 'Manager'
    elif 'Supervisor' in job_title:
        return 'Supervisor'
    elif 'Specialist' in job_title or 'Technician' in job_title:
        return 'Specialist'
    elif 'Coordinator' in job_title:
        return 'Coordinator'
    elif 'Driver' in job_title or 'Operator' in job_title:
        return 'Driver'
    elif job_title in ['Harvester', 'Sorter', 'Packer', 'Pruner', 'Farm Hand', 'Farm Worker', 'Vineyard Worker']:
        return 'Worker'
    else:
        return 'Worker'  # Default

def generate_employee_data(num_employees: int = 1000) -> pd.DataFrame:
    """
    Generate realistic employee data for a mega farm workforce.
    
    Args:
        num_employees: Number of employee records to generate
        
    Returns:
        DataFrame with employee demographic and employment data
    """
    employees = []
    
    # Calculate date ranges for realistic dates
    today = datetime.now()
    max_tenure_years = 15  # Maximum years of employment
    
    # Pre-generate managers for each location (about 5% of workforce)
    num_managers = int(num_employees * 0.05)
    manager_ids = list(range(1, num_managers + 1))
    
    # Generate managers first
    for employee_id in manager_ids:
        location = np.random.choice(FARM_LOCATIONS)
        
        # Select randomly from management roles
        job_title = np.random.choice(MANAGEMENT_ROLES)
        department = 'Management'
        worker_type = 'Permanent'  # Managers are permanent
        
        # Generate demographic data
        gender = np.random.choice(GENDERS, p=GENDER_DISTRIBUTION)
        first_name = fake.first_name_male() if gender == 'Male' else fake.first_name_female()
        last_name = fake.last_name()
        
        # Age between 30-60 for managers
        birth_date = fake.date_of_birth(minimum_age=30, maximum_age=60)
        
        # Ethnicity
        ethnicity = np.random.choice(ETHNICITY_GROUPS, p=ETHNICITY_DISTRIBUTION)
        
        # Small chance of disability
        has_disability = random.random() < 0.03
        
        # Join date (managers tend to have longer tenure)
        tenure_years = random.randint(2, max_tenure_years)
        join_date = today - timedelta(days=365 * tenure_years + random.randint(0, 364))
        
        # Salary based on management role (higher end)
        min_salary, max_salary = SALARY_RANGES['Manager']
        salary = int(np.random.uniform(min_salary, max_salary))
        
        # Manager's manager is null or a higher level manager
        if employee_id > num_managers // 3:  # Most managers report to someone
            manager_id = random.randint(1, num_managers // 3)  # Report to top-level managers
        else:
            manager_id = None  # Top managers don't have a manager in the dataset
        
        employees.append({
            'employee_id': employee_id,
            'first_name': first_name,
            'last_name': last_name,
            'date_of_birth': birth_date,
            'gender': gender,
            'ethnicity': ethnicity,
            'has_disability': has_disability,
            'join_date': join_date,
            'department': department,
            'job_title': job_title,
            'location': location,
            'worker_type': worker_type,
            'salary': salary,
            'manager_id': manager_id
        })
    
    # Now generate other employees
    for employee_id in range(num_managers + 1, num_employees + 1):
        # Determine if worker is permanent, seasonal, or contract
        worker_type = np.random.choice(WORKER_TYPES, p=WORKER_TYPE_DISTRIBUTION)
        
        # Select location
        location = np.random.choice(FARM_LOCATIONS)
        
        # Select department/operation available at this location
        available_operations = LOCATION_OPERATIONS[location]
        department = np.random.choice(available_operations)
        
        # Select job title based on department
        if department == 'Management':
            job_title = np.random.choice(MANAGEMENT_ROLES)
        else:
            job_title = np.random.choice(JOB_TITLES[department])
        
        # Seasonal workers are mostly in production departments
        if worker_type == 'Seasonal' and department not in ['Citrus Production', 'Maize Production', 'Grape Production', 'Pack House Operations']:
            # Reassign seasonal workers to production departments
            production_depts = [dept for dept in LOCATION_OPERATIONS[location] 
                              if dept in ['Citrus Production', 'Maize Production', 'Grape Production', 'Pack House Operations']]
            if production_depts:
                department = np.random.choice(production_depts)
                job_title = np.random.choice(JOB_TITLES[department])
        
        # Generate demographic data
        gender = np.random.choice(GENDERS, p=GENDER_DISTRIBUTION)
        first_name = fake.first_name_male() if gender == 'Male' else fake.first_name_female()
        last_name = fake.last_name()
        
        # Age distribution (18-65)
        if 'Supervisor' in job_title or 'Specialist' in job_title:
            # Supervisors and specialists tend to be older
            birth_date = fake.date_of_birth(minimum_age=25, maximum_age=60)
        else:
            birth_date = fake.date_of_birth(minimum_age=18, maximum_age=65)
        
        # Ethnicity
        ethnicity = np.random.choice(ETHNICITY_GROUPS, p=ETHNICITY_DISTRIBUTION)
        
        # Small chance of disability
        has_disability = random.random() < 0.03
        
        # Join date logic
        if worker_type == 'Permanent':
            # Permanent workers can have longer tenure
            tenure_years = random.randint(0, max_tenure_years)
            join_date = today - timedelta(days=365 * tenure_years + random.randint(0, 364))
        elif worker_type == 'Contract':
            # Contract workers usually more recent
            tenure_years = random.randint(0, 3)
            join_date = today - timedelta(days=365 * tenure_years + random.randint(0, 364))
        else:  # Seasonal
            # Seasonal workers very recent (within last year)
            join_date = today - timedelta(days=random.randint(7, 180))
        
        # Calculate salary
        job_level = get_job_level(job_title)
        
        if worker_type == 'Seasonal':
            min_salary, max_salary = SALARY_RANGES['Seasonal']
        else:
            min_salary, max_salary = SALARY_RANGES.get(job_level, SALARY_RANGES['Worker'])
            
        # Add some variation based on tenure for permanent workers
        if worker_type == 'Permanent':
            tenure_factor = min(1 + (tenure_years * 0.03), 1.3)  # Up to 30% increase for long tenure
            min_salary = int(min_salary * tenure_factor)
            max_salary = int(max_salary * tenure_factor)
            
        salary = int(np.random.uniform(min_salary, max_salary))
        
        # Assign manager from the same location
        location_managers = [
            e['employee_id'] for e in employees 
            if e['location'] == location and 'Manager' in e['job_title']
        ]
        
        if location_managers:
            manager_id = np.random.choice(location_managers)
        else:
            # Fallback to any manager if none at this location
            manager_id = np.random.choice(manager_ids)
        
        employees.append({
            'employee_id': employee_id,
            'first_name': first_name,
            'last_name': last_name,
            'date_of_birth': birth_date,
            'gender': gender,
            'ethnicity': ethnicity,
            'has_disability': has_disability,
            'join_date': join_date,
            'department': department,
            'job_title': job_title,
            'location': location,
            'worker_type': worker_type,
            'salary': salary,
            'manager_id': manager_id
        })
    
    return pd.DataFrame(employees)

def generate_monthly_payroll(employees_df: pd.DataFrame, start_date: datetime, months: int = 24) -> pd.DataFrame:
    """
    Generate monthly payroll data for all employees.
    
    Args:
        employees_df: DataFrame with employee data
        start_date: Start date for payroll history
        months: Number of months of history to generate
        
    Returns:
        DataFrame with monthly payroll data
    """
    payroll_data = []
    
    # For each month in the range
    for i in range(months):
        month_date = start_date + timedelta(days=30 * i)
        month_str = month_date.strftime('%Y-%m')
        
        # Filter employees who were employed in this month
        active_employees = employees_df[employees_df['join_date'] <= month_date].copy()
        
        # Handle seasonality - adjust number of seasonal workers
        if month_str[-2:] in ['01', '02', '11', '12']:  # Summer months in South Africa (harvest)
            seasonal_factor = 1.0  # Full seasonal workforce
        elif month_str[-2:] in ['03', '04', '10']:  # Shoulder season
            seasonal_factor = 0.6  # Reduced seasonal workforce
        else:  # Winter
            seasonal_factor = 0.2  # Minimal seasonal workforce
        
        # Apply seasonal adjustment to seasonal workers
        seasonal_mask = active_employees['worker_type'] == 'Seasonal'
        seasonal_employees = active_employees[seasonal_mask].copy()
        
        # Randomly select seasonal workers based on factor
        if not seasonal_employees.empty:
            num_seasonal = len(seasonal_employees)
            keep_count = int(num_seasonal * seasonal_factor)
            keep_indices = np.random.choice(seasonal_employees.index, size=keep_count, replace=False)
            drop_indices = list(set(seasonal_employees.index) - set(keep_indices))
            active_employees = active_employees.drop(drop_indices)
        
        # Calculate overtime and deductions
        for _, employee in active_employees.iterrows():
            # Base salary from employee record
            base_salary = employee['salary']
            
            # Random overtime (more likely for certain departments)
            overtime_prone = employee['department'] in ['Citrus Production', 'Maize Production', 'Grape Production', 'Pack House Operations']
            overtime_hours = 0
            
            if overtime_prone:
                # More overtime during harvest seasons
                if month_str[-2:] in ['01', '02', '11', '12']:
                    overtime_hours = int(np.random.exponential(15))  # More overtime in peak season
                else:
                    overtime_hours = int(np.random.exponential(5))  # Less in off-season
                
                overtime_hours = min(overtime_hours, 40)  # Cap overtime at realistic level
            
            # Calculate overtime pay (1.5x hourly rate)
            hourly_rate = base_salary / 160  # Assuming 160 working hours per month
            overtime_pay = overtime_hours * hourly_rate * 1.5
            
            # Calculate deductions (tax, benefits, etc.)
            tax_rate = 0.18  # Simplified tax rate
            benefits_rate = 0.05  # Simplified benefits rate
            
            gross_pay = base_salary + overtime_pay
            tax_deduction = gross_pay * tax_rate
            benefits_deduction = base_salary * benefits_rate
            
            # Net pay
            net_pay = gross_pay - tax_deduction - benefits_deduction
            
            payroll_data.append({
                'employee_id': employee['employee_id'],
                'month': month_str,
                'base_salary': base_salary,
                'overtime_hours': overtime_hours,
                'overtime_pay': overtime_pay,
                'gross_pay': gross_pay,
                'tax_deduction': tax_deduction,
                'benefits_deduction': benefits_deduction,
                'net_pay': net_pay,
                'department': employee['department'],
                'location': employee['location'],
                'worker_type': employee['worker_type']
            })
    
    return pd.DataFrame(payroll_data)

def generate_leave_data(employees_df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Generate leave records for employees.
    
    Args:
        employees_df: DataFrame with employee data
        start_date: Start date for leave history
        end_date: End date for leave history
        
    Returns:
        DataFrame with leave records
    """
    leave_records = []
    
    # Annual leave days based on worker type
    leave_entitlement = {
        'Permanent': 21,  # 21 days annual leave
        'Contract': 15,   # 15 days annual leave
        'Seasonal': 7     # 7 days annual leave (prorated)
    }
    
    for _, employee in employees_df.iterrows():
        # Skip if employee joined after end date
        if employee['join_date'] > end_date:
            continue
            
        # Calculate employee's start date for leave purposes
        employee_start = max(employee['join_date'], start_date)
        
        # Number of days between start and end
        days_employed = (end_date - employee_start).days
        
        # Calculate number of leave instances based on employment duration and type
        if employee['worker_type'] == 'Permanent':
            leave_frequency = max(1, days_employed // 60)  # More regular leave
        elif employee['worker_type'] == 'Contract':
            leave_frequency = max(1, days_employed // 90)  # Less frequent
        else:  # Seasonal
            leave_frequency = max(1, days_employed // 120)  # Minimal leave
        
        # Adjust for short employment periods
        leave_frequency = min(leave_frequency, 10)  # Cap number of leave instances
        
        # Generate leave records
        leave_days_used = 0
        annual_entitlement = leave_entitlement[employee['worker_type']]
        
        for _ in range(leave_frequency):
            # Determine leave type (weighted)
            if employee['worker_type'] == 'Seasonal':
                leave_weights = [0.4, 0.5, 0.1, 0, 0]  # Seasonal workers more likely to take sick leave
            else:
                leave_weights = [0.6, 0.25, 0.1, 0.03, 0.02]
                
            leave_type = np.random.choice(LEAVE_TYPES, p=leave_weights)
            
            # Determine leave duration
            if leave_type == 'Annual Leave':
                # Annual leave can be longer
                duration = random.randint(1, 5)
                
                # Ensure we don't exceed annual entitlement
                if leave_days_used + duration > annual_entitlement:
                    duration = max(1, annual_entitlement - leave_days_used)
                leave_days_used += duration
                
            elif leave_type == 'Sick Leave':
                duration = random.randint(1, 3)  # Typically shorter
            elif leave_type == 'Family Responsibility':
                duration = random.randint(1, 3)
            elif leave_type == 'Maternity/Paternity':
                duration = random.randint(30, 90) if employee['gender'] == 'Female' else random.randint(3, 10)
            else:  # Unpaid
                duration = random.randint(1, 5)
            
            # Generate random start date within employment period
            latest_start = end_date - timedelta(days=duration)
            if employee_start >= latest_start:
                continue  # Skip if employment period too short for this leave
                
            leave_start = employee_start + timedelta(days=random.randint(0, (latest_start - employee_start).days))
            leave_end = leave_start + timedelta(days=duration - 1)
            
            # Create leave record
            leave_records.append({
                'employee_id': employee['employee_id'],
                'leave_type': leave_type,
                'start_date': leave_start,
                'end_date': leave_end,
                'duration_days': duration,
                'status': 'Completed' if leave_end < datetime.now() else 'Ongoing',
                'department': employee['department'],
                'location': employee['location'],
                'worker_type': employee['worker_type']
            })
    
    return pd.DataFrame(leave_records)

def generate_productivity_data(employees_df: pd.DataFrame, start_date: datetime, months: int = 24) -> pd.DataFrame:
    """
    Generate productivity and performance data for employees.
    
    Args:
        employees_df: DataFrame with employee data
        start_date: Start date for productivity history
        months: Number of months of history to generate
        
    Returns:
        DataFrame with productivity metrics
    """
    productivity_data = []
    
    # Process each month
    for i in range(months):
        month_date = start_date + timedelta(days=30 * i)
        month_str = month_date.strftime('%Y-%m')
        
        # Filter active employees for this month
        active_employees = employees_df[employees_df['join_date'] <= month_date]
        
        # Seasonal factors affecting productivity (1.0 is baseline)
        month_num = int(month_str.split('-')[1])
        if month_num in [1, 2, 11, 12]:  # Peak harvest season
            seasonal_factor = 1.2  # 20% higher productivity in peak season
        elif month_num in [3, 4, 9, 10]:  # Shoulder seasons
            seasonal_factor = 1.0  # Normal productivity
        else:  # Off season
            seasonal_factor = 0.9  # 10% lower productivity
        
        for _, employee in active_employees.iterrows():
            # Base productivity depends on job and experience
            job_level = get_job_level(employee['job_title'])
            
            # Experience factor (increases with tenure)
            months_employed = (month_date.year - employee['join_date'].year) * 12 + (month_date.month - employee['join_date'].month)
            experience_factor = min(1.0 + (months_employed * 0.002), 1.3)  # Up to 30% boost for experienced workers
            
            # Base output depends on job type
            if employee['department'] in ['Citrus Production', 'Maize Production', 'Grape Production']:
                base_output = 100  # Farm production metrics (e.g., kg harvested per day)
            elif employee['department'] == 'Pack House Operations':
                base_output = 150  # Packing metrics (e.g., boxes packed)
            elif employee['department'] == 'Cattle Farming':
                base_output = 80   # Cattle metrics
            else:
                base_output = 90   # Other departments
                
            # Productivity metrics (with some randomness)
            productivity_level = base_output * experience_factor * seasonal_factor * np.random.uniform(0.85, 1.15)
            
            # Attendance metrics (days worked out of ~21 working days)
            max_attendance = 21
            
            # Attendance varies by worker type
            if employee['worker_type'] == 'Permanent':
                attendance_factor = np.random.beta(9, 1)  # Skewed toward high attendance
            elif employee['worker_type'] == 'Contract':
                attendance_factor = np.random.beta(8, 2)  # Slightly lower
            else:  # Seasonal
                attendance_factor = np.random.beta(7, 3)  # More variable
                
            days_worked = int(max_attendance * attendance_factor)
            
            # Performance score (1-5 scale)
            # Influenced by productivity and attendance
            performance_base = 3.0  # Average baseline
            productivity_influence = (productivity_level / (base_output * seasonal_factor) - 1) * 1.0  # Normalized influence
            attendance_influence = (days_worked / max_attendance - 0.8) * 2.0  # Normalized influence
            
            performance_score = performance_base + productivity_influence + attendance_influence
            performance_score = max(1.0, min(5.0, performance_score))  # Clamp between 1-5
            
            # Random chance of incidents
            incident_count = 0
            incident_type = None
            
            incident_chance = 0.05  # 5% base chance
            if days_worked < 18:  # Lower attendance correlates with higher incident rate
                incident_chance += 0.05
            if performance_score < 2.5:  # Poor performers have more incidents
                incident_chance += 0.05
                
            if random.random() < incident_chance:
                incident_count = random.randint(1, 2)
                incident_type = np.random.choice(INCIDENT_TYPES)
            
            productivity_data.append({
                'employee_id': employee['employee_id'],
                'month': month_str,
                'productivity_level': productivity_level,
                'days_worked': days_worked,
                'performance_score': round(performance_score, 1),
                'incident_count': incident_count,
                'incident_type': incident_type,
                'department': employee['department'],
                'location': employee['location'],
                'job_title': employee['job_title'],
                'worker_type': employee['worker_type']
            })
    
    return pd.DataFrame(productivity_data)

def generate_survey_data(employees_df: pd.DataFrame, survey_months: int = 6) -> pd.DataFrame:
    """
    Generate employee survey responses for the employee barometer.
    
    Args:
        employees_df: DataFrame with employee data
        survey_months: Number of recent months to generate survey data for
        
    Returns:
        DataFrame with survey responses
    """
    survey_data = []
    
    # Survey questions and possible responses
    questions = {
        'support_feeling': ["Not at all supported", "Rarely supported", "Sometimes supported", "Often supported", "Always supported"],
        'job_satisfaction': ["Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very satisfied"],
        'exhaustion_feeling': ["Never", "Rarely", "Sometimes", "Often", "Always"],
        'work_mood': ["Very unhappy", "Unhappy", "Neutral", "Happy", "Very happy"]
    }
    
    # Common feedback phrases for word cloud generation
    positive_feedback = [
        "Good working environment", "Fair management", "Good team", "Proper equipment provided",
        "Safety is prioritized", "Fair pay", "Respectful treatment", "Clear instructions", 
        "Regular breaks", "Supportive supervisors", "Good communication", "Opportunities to learn"
    ]
    
    negative_feedback = [
        "Long hours", "Too much overtime", "Inadequate breaks", "Poor equipment", 
        "Safety concerns", "Low wages", "Poor communication", "Heavy workload", 
        "Unfair treatment", "Weather challenges", "Transportation issues", "Housing problems"
    ]
    
    # Generate survey months (most recent months)
    today = datetime.now()
    survey_dates = [(today - timedelta(days=30 * i)).strftime('%Y-%m') for i in range(survey_months)]
    
    # Not all employees respond to surveys
    for survey_date in survey_dates:
        # Convert survey date to datetime for comparison
        survey_month_date = datetime.strptime(survey_date + "-01", '%Y-%m-%d')
        
        # Only include employees who were employed at the time of the survey
        active_employees = employees_df[employees_df['join_date'] <= survey_month_date]
        
        # Response rate varies by worker type
        for _, employee in active_employees.iterrows():
            # Determine if employee responds to this survey
            if employee['worker_type'] == 'Permanent':
                response_chance = 0.75  # 75% of permanent workers respond
            elif employee['worker_type'] == 'Contract':
                response_chance = 0.50  # 50% of contract workers respond
            else:  # Seasonal
                response_chance = 0.30  # 30% of seasonal workers respond
                
            if random.random() > response_chance:
                continue  # Skip this employee for this survey
            
            # Base satisfaction level varies by worker type and department
            if employee['worker_type'] == 'Permanent':
                base_satisfaction = 3.5  # Higher baseline
            elif employee['worker_type'] == 'Contract':
                base_satisfaction = 3.0  # Medium baseline
            else:  # Seasonal
                base_satisfaction = 2.7  # Lower baseline
                
            # Adjust based on department
            if employee['department'] in ['Administration', 'Maintenance']:
                dept_factor = 0.3  # These departments tend to have higher satisfaction
            elif employee['department'] in ['Pack House Operations']:
                dept_factor = -0.2  # These tend to have lower satisfaction
            else:
                dept_factor = 0.0
                
            # Adjust for tenure (longer tenure generally correlates with higher satisfaction)
            months_employed = (survey_month_date.year - employee['join_date'].year) * 12 + (survey_month_date.month - employee['join_date'].month)
            tenure_factor = min(months_employed * 0.01, 0.3)  # Up to 0.3 increase for long tenure
            
            # Random factor for individual variation
            random_factor = np.random.normal(0, 0.5)
            
            # Calculate overall satisfaction (1-5 scale)
            satisfaction = base_satisfaction + dept_factor + tenure_factor + random_factor
            satisfaction = max(1.0, min(5.0, satisfaction))  # Clamp between 1-5
            
            # Generate responses based on overall satisfaction
            support_response = questions['support_feeling'][min(int(satisfaction) - 1, 4)]
            satisfaction_response = questions['job_satisfaction'][min(int(satisfaction) - 1, 4)]
            
            # Exhaustion is inversely related to satisfaction (with some randomness)
            exhaustion_level = 6 - min(int(satisfaction + np.random.normal(0, 0.5)), 5)
            exhaustion_response = questions['exhaustion_feeling'][min(exhaustion_level - 1, 4)]
            
            # Mood is correlated with satisfaction (with some randomness)
            mood_level = min(int(satisfaction + np.random.normal(0, 0.7)), 5)
            mood_response = questions['work_mood'][min(mood_level - 1, 4)]
            
            # Generate feedback text
            if satisfaction >= 4:  # Satisfied employees
                feedback_options = random.sample(positive_feedback, 3)
                if random.random() < 0.3:  # Sometimes add a negative point
                    feedback_options.append(random.choice(negative_feedback))
            elif satisfaction >= 3:  # Neutral employees
                feedback_options = random.sample(positive_feedback, 2) + random.sample(negative_feedback, 2)
            else:  # Unsatisfied employees
                feedback_options = random.sample(negative_feedback, 3)
                if random.random() < 0.3:  # Sometimes add a positive point
                    feedback_options.append(random.choice(positive_feedback))
                    
            feedback_text = ". ".join(feedback_options) + "."
            
            survey_data.append({
                'employee_id': employee['employee_id'],
                'survey_month': survey_date,
                'support_feeling': support_response,
                'job_satisfaction': satisfaction_response,
                'exhaustion_feeling': exhaustion_response,
                'work_mood': mood_response,
                'feedback_text': feedback_text,
                'department': employee['department'],
                'location': employee['location'],
                'job_title': employee['job_title'],
                'worker_type': employee['worker_type'],
                'tenure_months': months_employed
            })
    
    return pd.DataFrame(survey_data)

def generate_scenario_analysis_base() -> pd.DataFrame:
    """
    Generate base data for scenario analysis.
    This includes current workforce distribution and costs.
    
    Returns:
        DataFrame with summarized workforce data for scenario planning
    """
    # This will be populated with real data when available
    # For now, return placeholder structure
    data = {
        'location': FARM_LOCATIONS,
        'permanent_count': [120, 80, 100, 90, 110, 50, 45],
        'seasonal_count': [200, 50, 150, 100, 180, 80, 70],
        'contract_count': [20, 10, 15, 12, 18, 8, 7],
        'total_employees': [340, 140, 265, 202, 308, 138, 122],
        'avg_salary_permanent': [8500, 7800, 8200, 7900, 8100, 9200, 9000],
        'avg_salary_seasonal': [3800, 3700, 3900, 3850, 3750, 4200, 4100],
        'avg_salary_contract': [6500, 6200, 6400, 6300, 6350, 7000, 6800],
        'monthly_payroll': [1530000, 712000, 1325000, 1010000, 1386000, 828000, 768000]
    }
    
    return pd.DataFrame(data)

def load_or_generate_data(num_employees: int = 1000) -> Dict[str, pd.DataFrame]:
    """
    Load existing data or generate new dummy data if none exists.
    
    Args:
        num_employees: Number of employees to generate if creating new data
        
    Returns:
        Dictionary of DataFrames containing all required data
    """
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if data already exists
    employees_path = os.path.join(data_dir, 'employees.csv')
    
    if os.path.exists(employees_path):
        # Load existing data
        employees_df = pd.read_csv(employees_path)
        
        # Convert date columns from string to datetime
        date_columns = ['date_of_birth', 'join_date']
        for col in date_columns:
            if col in employees_df.columns:
                employees_df[col] = pd.to_datetime(employees_df[col])
                
        payroll_df = pd.read_csv(os.path.join(data_dir, 'payroll.csv'))
        leave_df = pd.read_csv(os.path.join(data_dir, 'leave.csv'))
        
        # Convert date columns for leave data
        for col in ['start_date', 'end_date']:
            if col in leave_df.columns:
                leave_df[col] = pd.to_datetime(leave_df[col])
                
        productivity_df = pd.read_csv(os.path.join(data_dir, 'productivity.csv'))
        survey_df = pd.read_csv(os.path.join(data_dir, 'survey.csv'))
        scenario_df = pd.read_csv(os.path.join(data_dir, 'scenario_base.csv'))
        
    else:
        # Generate new data
        print("Generating new dummy data...")
        
        # Create employee data first
        employees_df = generate_employee_data(num_employees)
        
        # Calculate start and end dates for historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)  # 2 years of history
        
        # Generate other datasets
        payroll_df = generate_monthly_payroll(employees_df, start_date, months=24)
        leave_df = generate_leave_data(employees_df, start_date, end_date)
        productivity_df = generate_productivity_data(employees_df, start_date, months=24)
        survey_df = generate_survey_data(employees_df, survey_months=6)
        scenario_df = generate_scenario_analysis_base()
        
        # Save generated data
        employees_df.to_csv(employees_path, index=False)
        payroll_df.to_csv(os.path.join(data_dir, 'payroll.csv'), index=False)
        leave_df.to_csv(os.path.join(data_dir, 'leave.csv'), index=False)
        productivity_df.to_csv(os.path.join(data_dir, 'productivity.csv'), index=False)
        survey_df.to_csv(os.path.join(data_dir, 'survey.csv'), index=False)
        scenario_df.to_csv(os.path.join(data_dir, 'scenario_base.csv'), index=False)
        
        print("Data generation complete.")
    
    return {
        'employees': employees_df,
        'payroll': payroll_df,
        'leave': leave_df,
        'productivity': productivity_df,
        'survey': survey_df,
        'scenario_base': scenario_df
    }

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Apply filters to the dataframe based on the filter dictionary.
    """
    filtered_df = df.copy()
    
    # Apply location filter if applicable
    if 'locations' in filters and filters['locations'] and 'location' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['location'].isin(filters['locations'])]
    
    # Apply department filter if applicable
    if 'departments' in filters and filters['departments'] and 'department' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['department'].isin(filters['departments'])]
    
    # Apply worker type filter if applicable
    if 'worker_types' in filters and filters['worker_types'] and 'worker_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['worker_type'].isin(filters['worker_types'])]
    
    # Apply age range filter if applicable
    if 'age_range' in filters and 'date_of_birth' in filtered_df.columns:
        min_age, max_age = filters['age_range']
        today = datetime.now().date()
        min_date = today - timedelta(days=365 * max_age)
        max_date = today - timedelta(days=365 * min_age)
        # Convert datetime64[ns] to date for comparison
        filtered_df = filtered_df[
            (filtered_df['date_of_birth'].dt.date >= min_date) & 
            (filtered_df['date_of_birth'].dt.date <= max_date)
        ]
    
    # Apply tenure range filter if applicable
    if 'tenure_range' in filters and 'join_date' in filtered_df.columns:
        min_tenure, max_tenure = filters['tenure_range']  # in years
        today = datetime.now().date()
        min_date = today - timedelta(days=365 * max_tenure)
        max_date = today - timedelta(days=365 * min_tenure)
        filtered_df = filtered_df[
            (filtered_df['join_date'].dt.date >= min_date) & 
            (filtered_df['join_date'].dt.date <= max_date)
        ]
    
    return filtered_df

def get_unique_values(df: pd.DataFrame, column: str) -> List:
    """
    Get unique values for a given column in a DataFrame.
    
    Args:
        df: Source DataFrame
        column: Column name to get unique values from
        
    Returns:
        List of unique values
    """
    if column in df.columns:
        return sorted(df[column].unique().tolist())
    return []