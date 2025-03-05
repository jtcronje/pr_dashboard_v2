import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import re
from collections import Counter

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.charts import (
    create_kpi_summary,
    create_bar_chart,
    create_stacked_bar_chart,
    create_line_chart,
    create_word_cloud,
    apply_common_layout
)
from utils.data import apply_filters

def render_barometer_page(data_dict, filters):
    """
    Render the employee barometer dashboard page.
    
    Args:
        data_dict: Dictionary containing all data frames
        filters: Dictionary containing selected filter values
    """
    
    # Filter data based on selected filters
    survey_df = apply_filters(data_dict['survey'], filters) if 'survey' in data_dict else pd.DataFrame()
    
    # Check if we have survey data
    if survey_df.empty:
        st.warning("No survey data is available for the selected filters.")
        return
    
    # Display date range context
    start_date, end_date = filters['date_range']
    st.markdown(f"**Showing data from:** {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")
    
    # TOP METRICS SECTION
    # ================
    st.markdown("### Survey Response Metrics")
    
    # Response mapping for numerical conversion
    response_values = {
        # Support feeling
        "Not at all supported": 1,
        "Rarely supported": 2,
        "Sometimes supported": 3,
        "Often supported": 4,
        "Always supported": 5,
        
        # Job satisfaction
        "Very dissatisfied": 1,
        "Dissatisfied": 2,
        "Neutral": 3,
        "Satisfied": 4,
        "Very satisfied": 5,
        
        # Exhaustion feeling (reversed scale - lower is better)
        "Never": 5,
        "Rarely": 4,
        "Sometimes": 3,
        "Often": 2,
        "Always": 1,
        
        # Work mood
        "Very unhappy": 1,
        "Unhappy": 2,
        "Neutral": 3,
        "Happy": 4,
        "Very happy": 5
    }
    
    # Convert responses to numerical values
    survey_with_values = survey_df.copy()
    
    for col in ['support_feeling', 'job_satisfaction', 'exhaustion_feeling', 'work_mood']:
        if col in survey_with_values.columns:
            survey_with_values[f'{col}_value'] = survey_with_values[col].map(response_values)
    
    # Calculate overall satisfaction score (average of all questions)
    satisfaction_columns = [col for col in ['support_feeling_value', 'job_satisfaction_value', 
                                           'exhaustion_feeling_value', 'work_mood_value'] 
                           if col in survey_with_values.columns]
    
    if satisfaction_columns:
        survey_with_values['overall_satisfaction'] = survey_with_values[satisfaction_columns].mean(axis=1)
    
    # Get the most recent survey month
    latest_month = survey_with_values['survey_month'].max()
    latest_survey = survey_with_values[survey_with_values['survey_month'] == latest_month]
    
    # Get the previous month for comparison
    all_months = sorted(survey_with_values['survey_month'].unique())
    if len(all_months) > 1:
        prev_month_idx = all_months.index(latest_month) - 1
        if prev_month_idx >= 0:
            prev_month = all_months[prev_month_idx]
            prev_survey = survey_with_values[survey_with_values['survey_month'] == prev_month]
        else:
            prev_survey = pd.DataFrame()
    else:
        prev_survey = pd.DataFrame()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Response rate (calculated based on employee count if available)
        employee_count = 0
        if 'employees' in data_dict:
            employee_df = apply_filters(data_dict['employees'], filters)
            
            # Filter to only include employees who were employed during the survey month
            if 'join_date' in employee_df.columns:
                # Convert month string to datetime for comparison
                latest_month_dt = datetime.strptime(latest_month + '-01', '%Y-%m-%d')
                employee_df = employee_df[employee_df['join_date'] <= latest_month_dt]
            
            employee_count = len(employee_df)
        
        response_count = len(latest_survey)
        response_rate = (response_count / employee_count * 100) if employee_count > 0 else 0
        
        create_kpi_summary(
            response_count,
            "Total Responses",
            delta=round(response_rate, 1) if employee_count > 0 else None,
            delta_description="response rate" if employee_count > 0 else None,
            format_type="percent" if employee_count > 0 else None
        )
    
    with col2:
        # Average satisfaction score
        avg_satisfaction = latest_survey['overall_satisfaction'].mean() if 'overall_satisfaction' in latest_survey.columns else 0
        
        # Calculate change if previous data is available
        satisfaction_change = None
        if not prev_survey.empty and 'overall_satisfaction' in prev_survey.columns:
            prev_satisfaction = prev_survey['overall_satisfaction'].mean()
            satisfaction_change = avg_satisfaction - prev_satisfaction
        
        create_kpi_summary(
            round(avg_satisfaction, 2),
            "Satisfaction Score",
            delta=round(satisfaction_change, 2) if satisfaction_change is not None else None,
            delta_description="vs previous survey" if satisfaction_change is not None else None
        )
    
    with col3:
        # Trend indicator
        if 'overall_satisfaction' in survey_with_values.columns:
            # Calculate monthly averages
            monthly_avg = survey_with_values.groupby('survey_month')['overall_satisfaction'].mean().reset_index()
            monthly_avg = monthly_avg.sort_values('survey_month')
            
            # Calculate 3-month trend if enough data
            if len(monthly_avg) >= 3:
                recent_months = monthly_avg.tail(3)
                
                # Simple linear regression to get trend
                x = np.arange(len(recent_months))
                y = recent_months['overall_satisfaction'].values
                
                # Calculate slope using numpy's polyfit
                slope = np.polyfit(x, y, 1)[0]
                
                # Determine trend direction
                if slope > 0.05:
                    trend = "Improving"
                    delta_color = "normal"  # Green
                elif slope < -0.05:
                    trend = "Declining"
                    delta_color = "inverse"  # Red
                else:
                    trend = "Stable"
                    delta_color = "off"  # Gray
                
                # Display trend KPI
                st.metric(
                    label="3-Month Trend",
                    value=trend,
                    delta=f"{slope:.2f} points/month",
                    delta_color=delta_color
                )
            else:
                st.metric(
                    label="Trend",
                    value="Insufficient Data",
                    delta=None
                )
        else:
            st.metric(
                label="Trend",
                value="N/A",
                delta=None
            )
    
    # QUESTION ANALYSIS SECTIONS
    # =======================
    
    # Define questions and their options
    questions = {
        'support_feeling': {
            'title': "How supported do you feel by your supervisors or managers?",
            'options': ["Not at all supported", "Rarely supported", "Sometimes supported", "Often supported", "Always supported"]
        },
        'job_satisfaction': {
            'title': "How satisfied are you with your job?",
            'options': ["Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very satisfied"]
        },
        'exhaustion_feeling': {
            'title': "How often do you feel physically or emotionally exhausted from your work?",
            'options': ["Never", "Rarely", "Sometimes", "Often", "Always"]
        },
        'work_mood': {
            'title': "How would you describe your general mood at work?",
            'options': ["Very unhappy", "Unhappy", "Neutral", "Happy", "Very happy"]
        }
    }
    
    # Process each question
    for question_key, question_info in questions.items():
        if question_key in survey_df.columns:
            st.markdown(f"### {question_info['title']}")
            
            # Distribution of responses
            response_counts = survey_df[question_key].value_counts().reset_index()
            response_counts.columns = ['response', 'count']
            
            # Ensure all options are included (even if count is 0)
            all_options = pd.DataFrame({'response': question_info['options']})
            response_counts = pd.merge(all_options, response_counts, on='response', how='left').fillna(0)
            
            # Calculate percentage
            total_responses = response_counts['count'].sum()
            response_counts['percentage'] = (response_counts['count'] / total_responses * 100).round(1)
            
            # Sort by option order
            response_counts['order'] = response_counts['response'].map({opt: i for i, opt in enumerate(question_info['options'])})
            response_counts = response_counts.sort_values('order')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Response distribution chart
                st.plotly_chart(
                    create_bar_chart(
                        response_counts,
                        x_col='response',
                        y_col='percentage',
                        title=f"Distribution of Responses",
                        text_format="percent",
                        category_orders={"response": question_info['options']}
                    ),
                    use_container_width=True
                )
            
            with col2:
                # Department comparison if department data is available
                if 'department' in survey_df.columns:
                    # Get average score by department - ensure the value column exists
                    value_col = f'{question_key}_value'
                    if value_col in survey_with_values.columns:
                        dept_avg = survey_with_values.groupby('department')[value_col].mean().reset_index()
                        dept_avg = dept_avg.sort_values(value_col, ascending=False)
                        
                        st.plotly_chart(
                            create_bar_chart(
                                dept_avg,
                                x_col='department',
                                y_col=value_col,
                                title=f"Average Rating by Department"
                            ),
                            use_container_width=True
                        )
                    else:
                        st.info(f"Value data not available for {question_key}.")
            
            # Trend over time
            if len(survey_df['survey_month'].unique()) > 1:
                value_col = f'{question_key}_value'
                if value_col in survey_with_values.columns:
                    monthly_avg = survey_with_values.groupby('survey_month')[value_col].mean().reset_index()
                    monthly_avg = monthly_avg.sort_values('survey_month')
                    
                    st.plotly_chart(
                        create_line_chart(
                            monthly_avg,
                            x_col='survey_month',
                            y_col=value_col,
                            title=f"Score Trend Over Time",
                            height=400,
                            yaxis_title="Average Score (1-5)",
                            range_y=[1, 5]
                        ),
                        use_container_width=True
                    )
                else:
                    st.info(f"Value data not available for trend analysis of {question_key}.")
    
    # FREE TEXT ANALYSIS
    # ===============
    st.markdown("### Free Text Feedback Analysis")
    
    if 'feedback_text' in survey_df.columns and not survey_df['feedback_text'].isnull().all():
        # Word cloud visualization
        st.markdown("#### Common Themes in Feedback")
        
        # Combine all feedback text
        all_feedback = " ".join(survey_df['feedback_text'].dropna().tolist())
        
        # Generate word cloud
        if all_feedback.strip():
            wordcloud_fig = create_word_cloud(all_feedback, title="Word Cloud of Feedback Themes")
            st.pyplot(wordcloud_fig)
        else:
            st.info("No feedback text available for analysis.")
        
        # Simple sentiment analysis
        st.markdown("#### Sentiment Analysis")
        
        # Define positive and negative keywords
        positive_keywords = [
            'good', 'great', 'excellent', 'happy', 'satisfied', 'helpful', 'supportive',
            'fair', 'respectful', 'clear', 'positive', 'well', 'appreciate', 'thank',
            'enjoy', 'love', 'best', 'improvement', 'better', 'effective', 'proper',
            'opportunities', 'benefits', 'friendly', 'comfortable', 'safe', 'clean'
        ]
        
        negative_keywords = [
            'bad', 'poor', 'inadequate', 'unhappy', 'dissatisfied', 'unhelpful', 'unsupportive',
            'unfair', 'disrespectful', 'unclear', 'negative', 'poorly', 'issues', 'problems',
            'difficult', 'hard', 'worst', 'disappointed', 'frustrating', 'ineffective', 'improper',
            'lack', 'insufficient', 'unfriendly', 'uncomfortable', 'unsafe', 'dirty', 'concerns'
        ]
        
        # Function to analyze sentiment
        def analyze_sentiment(text):
            if not isinstance(text, str) or not text.strip():
                return None
                
            text = text.lower()
            pos_count = sum(1 for word in positive_keywords if re.search(r'\b' + word + r'\b', text))
            neg_count = sum(1 for word in negative_keywords if re.search(r'\b' + word + r'\b', text))
            
            if pos_count > neg_count:
                return "Positive"
            elif neg_count > pos_count:
                return "Negative"
            else:
                return "Neutral"
        
        # Apply sentiment analysis
        survey_df['sentiment'] = survey_df['feedback_text'].apply(analyze_sentiment)
        
        # Count sentiments
        sentiment_counts = survey_df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Remove None values
        sentiment_counts = sentiment_counts[sentiment_counts['Sentiment'].notna()]
        
        if not sentiment_counts.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Create a pie chart of sentiment distribution
                fig = px.pie(
                    sentiment_counts,
                    values='Count',
                    names='Sentiment',
                    title="Feedback Sentiment Distribution",
                    color='Sentiment',
                    color_discrete_map={
                        'Positive': '#4CAF50',
                        'Neutral': '#FFC107',
                        'Negative': '#F44336'
                    }
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Common words by sentiment
                if 'Positive' in sentiment_counts['Sentiment'].values:
                    positive_text = " ".join(survey_df[survey_df['sentiment'] == 'Positive']['feedback_text'].dropna())
                    if positive_text.strip():
                        # Get most common words
                        words = re.findall(r'\b\w+\b', positive_text.lower())
                        word_counts = Counter(words)
                        
                        # Remove stop words (simple list)
                        stop_words = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'it', 'for', 'with', 'as', 'be', 'this', 'on', 'are']
                        for word in stop_words:
                            if word in word_counts:
                                del word_counts[word]
                        
                        # Get top words
                        top_words = pd.DataFrame(word_counts.most_common(10), columns=['Word', 'Count'])
                        
                        st.markdown("##### Top Words in Positive Feedback")
                        st.dataframe(top_words, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data for sentiment analysis.")
        
        # Sample feedback (most recent)
        st.markdown("#### Recent Feedback Examples")
        
        # Get the 5 most recent feedback entries that have text
        recent_feedback = survey_df[survey_df['feedback_text'].notna()]
        recent_feedback = recent_feedback.sort_values('survey_month', ascending=False).head(5)
        
        if not recent_feedback.empty:
            for i, (_, row) in enumerate(recent_feedback.iterrows()):
                month = row['survey_month']
                feedback = row['feedback_text']
                dept = row.get('department', 'Unknown Department')
                sentiment = row.get('sentiment', 'Unknown')
                
                # Color based on sentiment
                sentiment_color = {
                    'Positive': '#4CAF50',
                    'Neutral': '#FFC107',
                    'Negative': '#F44336',
                    'Unknown': '#9E9E9E'
                }
                
                st.markdown(f"""
                <div style="border-left: 5px solid {sentiment_color.get(sentiment, '#9E9E9E')}; padding-left: 10px; margin-bottom: 10px;">
                    <p><strong>Survey Month:</strong> {month} | <strong>Department:</strong> {dept} | <strong>Sentiment:</strong> {sentiment}</p>
                    <p><em>"{feedback}"</em></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No feedback text available to display.")
    else:
        st.info("No free text feedback data available for analysis.")
    
    # IMPROVEMENT TRACKING
    # =================
    st.markdown("### Satisfaction Improvement Tracking")
    
    # Overall satisfaction trend
    if 'overall_satisfaction' in survey_with_values.columns:
        # Calculate monthly averages for overall satisfaction
        monthly_overall = survey_with_values.groupby('survey_month')['overall_satisfaction'].mean().reset_index()
        monthly_overall = monthly_overall.sort_values('survey_month')
        
        if len(monthly_overall) > 1:
            # Overall trend line
            st.plotly_chart(
                create_line_chart(
                    monthly_overall,
                    x_col='survey_month',
                    y_col='overall_satisfaction',
                    title="Overall Satisfaction Trend",
                    height=400,
                    yaxis_title="Overall Satisfaction Score (1-5)",
                    range_y=[1, 5]
                ),
                use_container_width=True
            )
            
            # Question comparison over time
            question_trends = []
            
            for question_key in ['support_feeling_value', 'job_satisfaction_value', 'exhaustion_feeling_value', 'work_mood_value']:
                if question_key in survey_with_values.columns:
                    monthly_avg = survey_with_values.groupby('survey_month')[question_key].mean().reset_index()
                    monthly_avg = monthly_avg.sort_values('survey_month')
                    
                    # Rename for clarity
                    monthly_avg['question'] = question_key.replace('_value', '').title().replace('_', ' ')
                    monthly_avg['score'] = monthly_avg[question_key]
                    
                    question_trends.append(monthly_avg[['survey_month', 'question', 'score']])
            
            if question_trends:
                # Combine all question trends
                all_trends = pd.concat(question_trends)
                
                # Create multi-line chart
                fig = px.line(
                    all_trends,
                    x='survey_month',
                    y='score',
                    color='question',
                    title="Question Scores Over Time",
                    height=400,
                    labels={'score': 'Average Score', 'survey_month': 'Month', 'question': 'Question'},
                    range_y=[1, 5]
                )
                
                # Apply common styling
                fig = apply_common_layout(fig)
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("More data points are needed to track improvement trends.")
    else:
        st.info("Overall satisfaction data is not available for trend analysis.")
    
    # DEPARTMENT COMPARISON OVER TIME
    # ===========================
    if 'department' in survey_df.columns and 'overall_satisfaction' in survey_with_values.columns:
        st.markdown("### Department Satisfaction Comparison")
        
        # Get all available departments
        departments = survey_df['department'].unique()
        
        if len(departments) > 1 and len(survey_df['survey_month'].unique()) > 1:
            # Calculate monthly averages by department
            dept_monthly = survey_with_values.groupby(['survey_month', 'department'])['overall_satisfaction'].mean().reset_index()
            
            # Calculate overall company average for all months
            company_avg = survey_with_values['overall_satisfaction'].mean()
            
            # Get latest month data
            latest_month = survey_with_values['survey_month'].max()
            latest_dept = survey_with_values[survey_with_values['survey_month'] == latest_month]
            
            # Calculate latest month company average
            latest_company_avg = latest_dept['overall_satisfaction'].mean()
            
            # Create multi-line chart
            fig = px.line(
                dept_monthly,
                x='survey_month',
                y='overall_satisfaction',
                color='department',
                title="Department Satisfaction Over Time",
                height=400,
                labels={'overall_satisfaction': 'Satisfaction Score', 'survey_month': 'Month'},
                range_y=[1, 5]
            )
            
            # Apply common styling
            fig = apply_common_layout(fig)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Analyzer for department trends
            st.markdown("#### Trend Analysis")
            st.markdown("*Automated analysis of department satisfaction trends:*")
            
            # Analyze trends for each department
            for department in departments:
                dept_data = dept_monthly[dept_monthly['department'] == department]
                
                if len(dept_data) >= 2:
                    # Sort by survey month to ensure chronological order
                    dept_data = dept_data.sort_values('survey_month')
                    
                    # Calculate slope using linear regression
                    x = np.arange(len(dept_data))
                    y = dept_data['overall_satisfaction'].values
                    slope = np.polyfit(x, y, 1)[0]
                    
                    # Calculate latest vs first month change
                    first_score = dept_data.iloc[0]['overall_satisfaction']
                    latest_score = dept_data.iloc[-1]['overall_satisfaction']
                    total_change = latest_score - first_score
                    
                    # Calculate latest score vs company average
                    vs_avg = latest_score - company_avg
                    
                    # Determine trend direction and concern level
                    if slope > 0.05:
                        trend = "increasing"
                        if latest_score < 3.0:
                            concern = "Still below target despite improvement."
                        elif vs_avg < -0.3:
                            concern = "Below company average. Monitor progress."
                        else:
                            concern = "Positive trajectory. No concerns."
                    elif slope < -0.05:
                        trend = "decreasing"
                        if latest_score < 3.0:
                            concern = "Critical attention needed immediately."
                        elif vs_avg < 0:
                            concern = "Declining and below average. Investigate."
                        else:
                            concern = "Monitor decline despite above-average score."
                    else:
                        trend = "stable"
                        if latest_score < 3.0:
                            concern = "Consistently low. Needs intervention."
                        elif vs_avg < 0:
                            concern = "Below average but stable. Room to improve."
                        else:
                            concern = "Maintaining good performance."
                    
                    # Create brief analysis text (keeping it under 30 words)
                    # Use HTML strong tag instead of markdown ** to avoid potential issues with department names
                    analysis = f"<strong>{department}</strong>: Satisfaction {trend} ({slope:.2f}/month). {concern}"
                    
                    # Display with appropriate color based on slope
                    if slope > 0.05:
                        st.markdown(f"<div style='border-left: 4px solid #4CAF50; padding-left: 10px; margin-bottom: 8px;'>{analysis}</div>", unsafe_allow_html=True)
                    elif slope < -0.05:
                        st.markdown(f"<div style='border-left: 4px solid #F44336; padding-left: 10px; margin-bottom: 8px;'>{analysis}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='border-left: 4px solid #FFC107; padding-left: 10px; margin-bottom: 8px;'>{analysis}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<strong>{department}</strong>: Insufficient data for trend analysis.", unsafe_allow_html=True)
            
            # Latest month department comparison
            dept_avg = latest_dept.groupby('department')['overall_satisfaction'].mean().reset_index()
            dept_avg = dept_avg.sort_values('overall_satisfaction', ascending=False)
            
            # Create bar chart with reference line
            fig = px.bar(
                dept_avg,
                x='department',
                y='overall_satisfaction',
                title=f"Department Satisfaction: {latest_month}",
                height=400,
                labels={'overall_satisfaction': 'Satisfaction Score', 'department': 'Department'}
            )
            
            # Add company average line
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=latest_company_avg,
                x1=len(dept_avg) - 0.5,
                y1=latest_company_avg,
                line=dict(color="red", width=2, dash="dash"),
            )
            
            # Add annotation for company average
            fig.add_annotation(
                x=len(dept_avg) - 0.5,
                y=latest_company_avg,
                text=f"Company Average: {latest_company_avg:.2f}",
                showarrow=False,
                yshift=10,
                font=dict(color="red")
            )
            
            # Apply common styling
            fig = apply_common_layout(fig)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("More data points across departments are needed for comparative analysis.")