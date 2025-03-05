import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# Color palette for consistency across charts - all blues
COLOR_PALETTE = {
    'primary': '#1E88E5',      # Primary blue
    'secondary': '#0D47A1',    # Dark blue
    'accent': '#64B5F6',       # Light blue
    'background': 'transparent', # Transparent background
    'text_primary': '#000000', # Black text
    'text_secondary': '#000000', # Black text
    'success': '#2196F3',      # Blue instead of green
    'warning': '#90CAF9',      # Light blue instead of amber
    'error': '#0D47A1',        # Dark blue instead of red
    'neutral': '#42A5F5',      # Medium blue instead of gray
}

# Sequential blues for heatmaps and continuous scales
BLUE_SCALE = ['#EBF5FB', '#D4E6F1', '#A9CCE3', '#7FB3D5', '#5499C7', '#2980B9', '#2471A3', '#1F618D', '#1A5276', '#154360']

# For categorical charts when more colors are needed - all blues
EXTENDED_PALETTE = [
    '#1E88E5',  # Primary blue
    '#0D47A1',  # Dark blue
    '#64B5F6',  # Light blue
    '#5C6BC0',  # Indigo-blue
    '#7986CB',  # Light indigo-blue
    '#3949AB',  # Dark indigo-blue
    '#42A5F5',  # Lighter blue
    '#1976D2',  # Darker blue
    '#0288D1',  # Another blue shade
    '#0097A7',  # Teal-blue
    '#039BE5',  # Sky blue
]

# Department colors - all blues
DEPARTMENT_COLORS = {
    'Citrus Production': '#1565C0',    # Blue
    'Maize Production': '#1976D2',     # Blue
    'Grape Production': '#1E88E5',     # Blue
    'Cattle Farming': '#2196F3',       # Blue
    'Pack House Operations': '#42A5F5', # Blue
    'Maintenance': '#64B5F6',          # Blue
    'Administration': '#90CAF9',       # Blue
    'Logistics': '#BBDEFB',            # Light blue
    'Management': '#0D47A1'            # Dark blue
}

# Worker type colors - blues
WORKER_TYPE_COLORS = {
    'Permanent': '#0D47A1',  # Dark blue
    'Seasonal': '#1976D2',   # Medium blue
    'Contract': '#42A5F5'    # Light blue
}

# Helper Functions
# ----------------

def format_column_name(column_name):
    """
    Format a column name by replacing underscores with spaces and capitalizing each word.
    
    Args:
        column_name: The column name to format
        
    Returns:
        Formatted column name string
    """
    if not isinstance(column_name, str):
        return column_name
    return ' '.join(word.capitalize() for word in column_name.split('_'))

def format_currency(value, currency=""):
    """Format a value as currency"""
    if pd.isna(value):
        return ""
    return f"{currency} {value:,.0f}".strip()

def format_percent(value, decimal_places=1):
    """Format a value as a percentage"""
    if pd.isna(value):
        return ""
    return f"{value:.{decimal_places}f}%"

def add_data_labels(fig, text_format=None, position="inside"):
    """
    Add data labels to a plotly bar chart.
    
    Args:
        fig: Plotly figure object
        text_format: Format for the labels ('percent', 'currency', or None)
        position: Position of labels ('inside', 'outside', 'auto', or 'none')
        
    Returns:
        Modified Plotly figure object
    """
    if text_format == "percent":
        text_template = '%{text:.1f}%'
    elif text_format == "currency":
        text_template = 'R %{text:,.0f}'
    else:
        text_template = '%{text:,.0f}'
        
    for data in fig.data:
        data.text = data.y
        data.texttemplate = text_template
        data.textposition = position
        
    return fig

def apply_common_layout(fig, title=None, height=None, legend_title=None, show_legend=False):
    """
    Apply common layout settings to all charts for consistency.
    
    Args:
        fig: Plotly figure object
        title: Chart title
        height: Chart height
        legend_title: Custom legend title
        show_legend: Whether to show the legend (default: False)
        
    Returns:
        Updated Plotly figure object
    """
    layout_updates = {
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'margin': dict(l=20, r=20, t=40, b=20),
        'font': dict(family="Arial, sans-serif", size=12, color="#000000"),
        'showlegend': show_legend
    }
    
    if show_legend:
        # Format the legend title if not provided
        if not legend_title and fig.data and hasattr(fig.data[0], 'name'):
            legend_title = format_column_name(fig.data[0].name)
            
        layout_updates['legend'] = dict(
            orientation="v",  # vertical orientation
            yanchor="middle",
            y=0.5,  # center vertically
            xanchor="left",
            x=1.02,  # position just outside the plot area
            font=dict(size=10, color="#000000"),
            bgcolor="rgba(255,255,255,0)"  # Transparent legend background
        )
    
    if title:
        layout_updates['title'] = dict(
            text=title,
            font=dict(size=14, color="#000000"),
            x=0.5,
            xanchor='center'
        )
        
    if height:
        layout_updates['height'] = height
        
    if legend_title and show_legend:
        layout_updates['legend']['title'] = dict(
            text=format_column_name(legend_title),
            font=dict(color="#000000")
        )
        
    fig.update_layout(**layout_updates)
    
    # Update axes with formatted labels
    for axis in fig.layout:
        if axis.startswith('xaxis') or axis.startswith('yaxis'):
            if hasattr(fig.layout[axis], 'title') and hasattr(fig.layout[axis].title, 'text'):
                current_title = fig.layout[axis].title.text
                if current_title:
                    fig.layout[axis].title.text = format_column_name(current_title)
    
    # Format trace names (legend items)
    if show_legend:
        for trace in fig.data:
            if hasattr(trace, 'name'):
                trace.name = format_column_name(trace.name)
    
    return fig

# Reusable Chart Components
# -------------------------

def create_kpi_summary(value, title, delta=None, delta_description="vs previous period", format_type='currency'):
    """
    Create a KPI summary metric with optional delta.
    
    Args:
        value: The main value to display
        title: The title/label for the metric
        delta: The change value to display
        delta_description: Description of what the delta represents
        format_type: How to format the value ('currency', 'percent', or None)
    """
    # Format the main value
    if format_type == 'currency':
        formatted_value = format_currency(value)
    elif format_type == 'percent':
        formatted_value = format_percent(value)
    else:
        formatted_value = f"{value:,}" if isinstance(value, (int, float)) else value
    
    # Handle delta if provided
    if delta is not None:
        # Convert numpy types to Python native types for compatibility
        if hasattr(delta, 'item') and callable(getattr(delta, 'item')):
            delta = delta.item()  # Converts numpy types (like numpy.int64) to Python native types
            
        if format_type == 'currency':
            formatted_delta = format_currency(delta, currency="")
        elif format_type == 'percent':
            formatted_delta = format_percent(delta, decimal_places=1)
        else:
            formatted_delta = f"{delta:,}" if isinstance(delta, (int, float)) else delta
            
        return st.metric(
            label=title,
            value=formatted_value,
            delta=formatted_delta,
            delta_color="normal",
            help=delta_description
        )
    else:
        return st.metric(
            label=title,
            value=formatted_value,
            help=delta_description
        )

def create_pie_chart(df, values_col, names_col, title=None, color_map=None, height=400, show_legend=False):
    """
    Create a pie chart for showing proportions of a whole.
    
    Args:
        df: DataFrame with the data
        values_col: Column for slice values
        names_col: Column for slice names
        title: Chart title
        color_map: Optional dictionary mapping categories to colors
        height: Chart height
        show_legend: Whether to show the legend (default: False)
        
    Returns:
        Plotly figure object
    """
    if len(df) == 0:
        # Return empty chart with message if no data
        fig = go.Figure(go.Pie(labels=["No data available"], values=[1]))
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    if color_map:
        # Get colors for the categories in the data
        colors = [color_map.get(cat, COLOR_PALETTE['neutral']) for cat in df[names_col]]
        fig = px.pie(
            df, 
            values=values_col, 
            names=names_col, 
            title=title,
            color_discrete_sequence=colors,
            height=height
        )
    else:
        fig = px.pie(
            df, 
            values=values_col, 
            names=names_col, 
            title=title,
            color_discrete_sequence=EXTENDED_PALETTE,
            height=height
        )
    
    # Customize layout
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    # Apply common layout with legend control
    apply_common_layout(fig, title=title, height=height, show_legend=show_legend)
    
    return fig

def create_bar_chart(df, x_col, y_col, title=None, color_col=None, color_map=None, orientation='v', 
                     text_format=None, sort_by=None, height=400, show_gridlines=True,
                     category_orders=None, show_legend=False):
    """
    Create a bar chart for categorical comparisons.
    
    Args:
        df: DataFrame with the data
        x_col: Column for x-axis
        y_col: Column for y-axis
        title: Chart title
        color_col: Column for color encoding (optional)
        color_map: Optional dictionary mapping categories to colors
        orientation: 'v' for vertical or 'h' for horizontal bars
        text_format: Format for data labels ('percent', 'currency', or None)
        sort_by: Column to sort by (defaults to y_col if None)
        height: Chart height
        show_gridlines: Whether to show gridlines
        category_orders: Dictionary with category orders for axes
        show_legend: Whether to show the legend (default: False)
        
    Returns:
        Plotly figure object
    """
    if len(df) == 0:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Sort data if requested
    if sort_by:
        df = df.sort_values(by=sort_by)
    
    # Prepare color sequence
    if color_col and color_map:
        color_discrete_map = color_map
    else:
        color_discrete_map = None
    
    # Create the appropriate type of bar chart
    if orientation == 'h':
        fig = px.bar(
            df, 
            y=x_col,  # For horizontal charts, x and y are swapped
            x=y_col, 
            title=title,
            color=color_col if color_col else None,
            orientation='h',
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=EXTENDED_PALETTE if not color_discrete_map else None,
            height=height,
            category_orders=category_orders,
            labels={
                x_col: format_column_name(x_col),
                y_col: format_column_name(y_col)
            }
        )
    else:
        fig = px.bar(
            df, 
            x=x_col, 
            y=y_col, 
            title=title,
            color=color_col if color_col else None,
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=EXTENDED_PALETTE if not color_discrete_map else None,
            height=height,
            category_orders=category_orders,
            labels={
                x_col: format_column_name(x_col),
                y_col: format_column_name(y_col)
            }
        )
    
    # Add data labels if requested
    if text_format:
        fig = add_data_labels(fig, text_format)
    
    # Apply common layout settings
    fig = apply_common_layout(fig, title=title, height=height, show_legend=show_legend)
    
    # Update gridlines based on preference
    if not show_gridlines:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
    
    # Add spacing between bars
    fig.update_layout(
        bargap=0.2,  # Gap between bars of adjacent location coordinates
        bargroupgap=0.1  # Gap between bars of the same location coordinates
    )
    
    # Improve layout for horizontal bars with long labels
    if orientation == 'h' and len(df) > 8:
        # Adjust margins and height to accommodate long labels
        fig.update_layout(margin=dict(l=150, r=20, t=40, b=20))
    
    return fig

def create_stacked_bar_chart(df, x_col, y_col, color_col, title=None, color_map=None, 
                            orientation='v', text_format=None, height=400, category_orders=None, show_legend=False):
    """
    Create a stacked bar chart for categorical comparisons with subcategories.
    
    Args:
        df: DataFrame with the data
        x_col: Column for x-axis (categories)
        y_col: Column for y-axis (values)
        color_col: Column for subcategories (stacking)
        title: Chart title
        color_map: Optional dictionary mapping subcategories to colors
        orientation: 'v' for vertical or 'h' for horizontal bars
        text_format: Format for data labels ('percent', 'currency', or None)
        height: Chart height
        category_orders: Dictionary with category orders for axes
        show_legend: Whether to show the legend (default: False)
        
    Returns:
        Plotly figure object
    """
    if len(df) == 0:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Handle color mapping
    if color_map:
        color_discrete_map = color_map
    else:
        color_discrete_map = None
    
    # Create the stacked bar chart
    if orientation == 'h':
        fig = px.bar(
            df, 
            y=x_col,  # For horizontal charts, x and y are swapped
            x=y_col, 
            color=color_col,
            title=title,
            orientation='h',
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=EXTENDED_PALETTE if not color_discrete_map else None,
            height=height,
            category_orders=category_orders,
            labels={
                x_col: format_column_name(x_col),
                y_col: format_column_name(y_col),
                color_col: format_column_name(color_col)
            }
        )
    else:
        fig = px.bar(
            df, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            title=title,
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=EXTENDED_PALETTE if not color_discrete_map else None,
            height=height,
            category_orders=category_orders,
            labels={
                x_col: format_column_name(x_col),
                y_col: format_column_name(y_col),
                color_col: format_column_name(color_col)
            }
        )
    
    # Apply common layout settings
    fig = apply_common_layout(fig, title=title, height=height, legend_title=color_col, show_legend=show_legend)
    
    # Add spacing between bars
    fig.update_layout(
        bargap=0.2,  # Gap between bars of adjacent location coordinates
        bargroupgap=0.1  # Gap between bars of the same location coordinates
    )
    
    return fig

def create_line_chart(df, x_col, y_col, title=None, color_col=None, color_map=None, 
                     line_shape='linear', height=400, markers=True, yaxis_title=None,
                     hover_data=None, text_format=None, range_y=None,
                     category_orders=None, show_legend=False):
    """
    Create a line chart for time series or trend data.
    
    Args:
        df: DataFrame with the data
        x_col: Column for x-axis (typically dates/times)
        y_col: Column for y-axis (values)
        title: Chart title
        color_col: Column for color encoding (for multiple lines)
        color_map: Optional dictionary mapping categories to colors
        line_shape: Shape of the line ('linear', 'spline', etc.)
        height: Chart height
        markers: Whether to show markers on the line
        yaxis_title: Custom y-axis title
        hover_data: Additional columns to show in hover tooltip
        text_format: Format for data labels ('percent', 'currency', or None)
        range_y: Custom y-axis range as tuple (min, max)
        category_orders: Dictionary with category orders for axes
        show_legend: Whether to show the legend (default: False)
        
    Returns:
        Plotly figure object
    """
    if len(df) == 0:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Handle color mapping
    if color_col and color_map:
        color_discrete_map = color_map
    else:
        color_discrete_map = None
    
    # Create the line chart
    fig = px.line(
        df, 
        x=x_col, 
        y=y_col,
        color=color_col,
        title=title,
        line_shape=line_shape,
        markers=markers,
        color_discrete_map=color_discrete_map,
        color_discrete_sequence=EXTENDED_PALETTE if not color_discrete_map else None,
        height=height,
        hover_data=hover_data,
        category_orders=category_orders
    )
    
    # Apply common layout
    fig = apply_common_layout(fig, title=title, height=height, show_legend=show_legend)
    
    # Custom y-axis title if provided
    if yaxis_title:
        fig.update_yaxes(title_text=yaxis_title)
    
    # Custom y-axis range if provided
    if range_y:
        fig.update_yaxes(range=range_y)
    
    return fig

def create_heatmap(df, x_col, y_col, value_col, title=None, color_scale=None, 
                  height=400, text_format=None, x_title=None, y_title=None, category_orders=None):
    """
    Create a heatmap for visualizing matrix data.
    
    Args:
        df: DataFrame with the data
        x_col: Column for x-axis categories
        y_col: Column for y-axis categories
        value_col: Column for cell values
        title: Chart title
        color_scale: Color scale for the heatmap
        height: Chart height
        text_format: Format for cell text ('percent', 'currency', or None)
        x_title: Custom x-axis title
        y_title: Custom y-axis title
        category_orders: Dictionary defining the ordering of categorical axes
        
    Returns:
        Plotly figure object
    """
    if len(df) == 0:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Pivot data into matrix format if it's not already
    if len(df.columns) == 3:  # If data is in long format (x, y, value)
        matrix_df = df.pivot(index=y_col, columns=x_col, values=value_col)
    else:
        matrix_df = df  # Assume it's already in matrix format
    
    # Use blue scale if no custom scale provided
    if not color_scale:
        color_scale = BLUE_SCALE
        
    # Format cell text based on format_type
    if text_format == 'percent':
        text = [[f"{val:.1f}%" if not pd.isna(val) else "" for val in row] 
                for row in matrix_df.values]
    elif text_format == 'currency':
        text = [[f"R {val:,.0f}" if not pd.isna(val) else "" for val in row] 
                for row in matrix_df.values]
    else:
        text = [[f"{val:,.1f}" if not pd.isna(val) else "" for val in row] 
                for row in matrix_df.values]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix_df.values,
        x=matrix_df.columns,
        y=matrix_df.index,
        colorscale=color_scale,
        text=text,
        texttemplate="%{text}",
        textfont={"size":10},
        hovertemplate='%{y},%{x}: %{text}<extra></extra>'
    ))
    
    # Apply layout
    layout_args = {
        'title': title,
        'height': height,
        'xaxis_title': x_title if x_title else x_col,
        'yaxis_title': y_title if y_title else y_col,
        'xaxis': dict(side='top'),
    }
    
    # Apply category orders if provided
    if category_orders and x_col in category_orders:
        layout_args['xaxis']['categoryorder'] = 'array'
        layout_args['xaxis']['categoryarray'] = category_orders[x_col]
    
    if category_orders and y_col in category_orders:
        layout_args['yaxis']['categoryorder'] = 'array'
        layout_args['yaxis']['categoryarray'] = category_orders[y_col]
    
    fig.update_layout(**layout_args)
    
    return apply_common_layout(fig, title=title, height=height)

def create_scatter_plot(df, x_col, y_col, title=None, color_col=None, size_col=None,
                       color_map=None, size_max=20, opacity=0.7, height=400,
                       hover_data=None, trendline=None, x_title=None, y_title=None):
    """
    Create a scatter plot for correlation analysis.
    
    Args:
        df: DataFrame with the data
        x_col: Column for x-axis
        y_col: Column for y-axis
        title: Chart title
        color_col: Column for color encoding
        size_col: Column for marker size encoding
        color_map: Optional dictionary mapping categories to colors
        size_max: Maximum marker size
        opacity: Marker opacity
        height: Chart height
        hover_data: Additional columns to show in hover tooltip
        trendline: Type of trendline ('ols', 'lowess', or None)
        x_title: Custom x-axis title
        y_title: Custom y-axis title
        
    Returns:
        Plotly figure object
    """
    if len(df) == 0:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Handle color mapping
    if color_col and color_map:
        color_discrete_map = color_map
    else:
        color_discrete_map = None
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=title,
        opacity=opacity,
        size_max=size_max,
        color_discrete_map=color_discrete_map,
        color_discrete_sequence=EXTENDED_PALETTE if not color_discrete_map else None,
        height=height,
        hover_data=hover_data,
        trendline=trendline
    )
    
    # Apply common layout
    fig = apply_common_layout(fig, title=title, height=height)
    
    # Custom axis titles if provided
    if x_title:
        fig.update_xaxes(title_text=x_title)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    
    return fig

def create_word_cloud(text_data, column=None, width=800, height=400, background_color='white', 
                     max_words=200, title=None):
    """
    Create a word cloud visualization from text data.
    
    Args:
        text_data: Either a string, or a DataFrame with text data
        column: If text_data is a DataFrame, the column containing text
        width: Width of the word cloud image
        height: Height of the word cloud image
        background_color: Background color
        max_words: Maximum number of words to include
        title: Title for the visualization
        
    Returns:
        Matplotlib figure with word cloud
    """
    # Process text based on input type
    if isinstance(text_data, pd.DataFrame) and column:
        # Combine all text from the column
        text = ' '.join(text_data[column].dropna().astype(str))
    elif isinstance(text_data, str):
        text = text_data
    else:
        text = "No text data available"
    
    # Create word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        colormap='Blues',
        collocations=False
    ).generate(text)
    
    # Display using matplotlib
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=16)
    
    plt.tight_layout()
    return fig

def create_comparison_chart(before_df, after_df, compare_col, value_col, title=None, 
                           orientation='h', height=400, text_format=None, color_map=None):
    """
    Create a before/after comparison chart for scenario analysis.
    
    Args:
        before_df: DataFrame with 'before' scenario data
        after_df: DataFrame with 'after' scenario data
        compare_col: Column to compare (categories)
        value_col: Column with values to compare
        title: Chart title
        orientation: 'v' for vertical or 'h' for horizontal bars
        height: Chart height
        text_format: Format for data labels ('percent', 'currency', or None)
        color_map: Color mapping for bars
        
    Returns:
        Plotly figure object
    """
    if len(before_df) == 0 or len(after_df) == 0:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Extract comparison data
    before_categories = before_df[compare_col].tolist()
    before_values = before_df[value_col].tolist()
    
    after_categories = after_df[compare_col].tolist()
    after_values = after_df[value_col].tolist()
    
    # Ensure categories match
    all_categories = list(set(before_categories + after_categories))
    
    # Prepare data for plotting
    categories = []
    before_vals = []
    after_vals = []
    
    for cat in all_categories:
        categories.append(cat)
        
        # Get before value
        if cat in before_categories:
            idx = before_categories.index(cat)
            before_vals.append(before_values[idx])
        else:
            before_vals.append(0)
            
        # Get after value
        if cat in after_categories:
            idx = after_categories.index(cat)
            after_vals.append(after_values[idx])
        else:
            after_vals.append(0)
    
    # Create the figure based on orientation
    if orientation == 'h':
        fig = go.Figure()
        
        # Add before scenario bars
        fig.add_trace(go.Bar(
            y=categories,
            x=before_vals,
            name='Current',
            orientation='h',
            marker_color=COLOR_PALETTE['secondary']
        ))
        
        # Add after scenario bars
        fig.add_trace(go.Bar(
            y=categories,
            x=after_vals,
            name='Scenario',
            orientation='h',
            marker_color=COLOR_PALETTE['primary']
        ))
        
    else:
        fig = go.Figure()
        
        # Add before scenario bars
        fig.add_trace(go.Bar(
            x=categories,
            y=before_vals,
            name='Current',
            marker_color=COLOR_PALETTE['secondary']
        ))
        
        # Add after scenario bars
        fig.add_trace(go.Bar(
            x=categories,
            y=after_vals,
            name='Scenario',
            marker_color=COLOR_PALETTE['primary']
        ))
    
    # Set up chart appearance
    fig.update_layout(
        barmode='group',
        title=title
    )
    
    # Apply common layout
    fig = apply_common_layout(fig, title=title, height=height)
    
    return fig

def create_distribution_chart(df, x_col, title=None, color=None, bin_size=None,
                             height=400, x_title=None, y_title="Count"):
    """
    Create a histogram showing the distribution of a numerical variable.
    
    Args:
        df: DataFrame with the data
        x_col: Column containing values to plot
        title: Chart title
        color: Bar color
        bin_size: Size of histogram bins
        height: Chart height
        x_title: Custom x-axis title
        y_title: Custom y-axis title
        
    Returns:
        Plotly figure object
    """
    if len(df) == 0 or x_col not in df.columns:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Set color if not provided
    if not color:
        color = COLOR_PALETTE['primary']
    
    # Create histogram
    fig = px.histogram(
        df,
        x=x_col,
        title=title,
        height=height,
        color_discrete_sequence=[color]
    )
    
    # Apply common layout
    fig = apply_common_layout(fig, title=title, height=height)
    
    # Custom axis titles if provided
    if x_title:
        fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    
    # Add spacing between bars
    fig.update_layout(
        bargap=0.2,  # Gap between bars
        bargroupgap=0.1  # Gap between bar groups
    )
    
    return fig

# Specialized Farm Dashboard Charts
# --------------------------------

def create_farm_map(location_data, value_col=None, title="Farm Locations", height=400, show_legend=False):
    """
    Create a map showing farm locations with optional bubble size for values.
    
    Args:
        location_data: DataFrame with location data including lat, lon columns
        value_col: Optional column for bubble size (e.g., employee_count)
        title: Chart title
        height: Chart height
        show_legend: Whether to show the legend (default: False)
        
    Returns:
        Plotly figure object
    """
    if len(location_data) == 0:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # If value column provided, use it for bar heights
    if value_col and value_col in location_data.columns:
        fig = px.bar(
            location_data,
            x='location',
            y=value_col,
            title=title,
            color='location',
            color_discrete_sequence=EXTENDED_PALETTE,
            height=height
        )
    else:
        # Just count records by location
        location_counts = location_data['location'].value_counts().reset_index()
        location_counts.columns = ['location', 'count']
        
        fig = px.bar(
            location_counts,
            x='location',
            y='count',
            title=title,
            color='location',
            color_discrete_sequence=EXTENDED_PALETTE,
            height=height
        )
    
    # Apply common layout
    fig = apply_common_layout(fig, title=title, height=height, show_legend=show_legend)
    
    # Add data labels
    fig = add_data_labels(fig)
    
    return fig

def create_worker_type_breakdown(df, title="Worker Type Distribution", height=400, show_legend=False):
    """
    Create a pie chart showing the distribution of worker types.
    
    Args:
        df: DataFrame with employee data including worker_type
        title: Chart title
        height: Chart height
        show_legend: Whether to show the legend (default: False)
        
    Returns:
        Plotly figure object
    """
    if len(df) == 0 or 'worker_type' not in df.columns:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Count workers by type
    worker_counts = df['worker_type'].value_counts().reset_index()
    worker_counts.columns = ['worker_type', 'count']
    
    # Create chart
    fig = px.pie(
        worker_counts,
        values='count',
        names='worker_type',
        title=title,
        color_discrete_sequence=EXTENDED_PALETTE,
        height=height
    )
    
    # Apply common layout
    fig = apply_common_layout(fig, title=title, height=height, show_legend=show_legend)
    
    # Update traces for better text display
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    return fig

def create_tenure_distribution(employees_df, title="Employee Tenure Distribution", height=400, show_legend=False):
    """
    Create a histogram showing the distribution of employee tenure.
    
    Args:
        employees_df: DataFrame with employee data including join_date
        title: Chart title
        height: Chart height
        show_legend: Whether to show the legend (default: False)
        
    Returns:
        Plotly figure object
    """
    if len(employees_df) == 0 or 'join_date' not in employees_df.columns:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Calculate tenure in years
    today = datetime.now()
    employees_df = employees_df.copy()
    employees_df['tenure_years'] = employees_df['join_date'].apply(
        lambda x: (today - x).days / 365.25
    )
    
    # Create histogram
    fig = px.histogram(
        employees_df,
        x='tenure_years',
        title=title,
        color='worker_type',
        color_discrete_map=WORKER_TYPE_COLORS,
        height=height,
        nbins=20
    )
    
    # Apply common layout with legend on the right
    fig = apply_common_layout(fig, title=title, height=height, show_legend=True)
    
    # Update legend position to the right
    fig.update_layout(
        legend=dict(
            orientation="v",  # vertical orientation
            yanchor="middle",
            y=0.5,  # center vertically
            xanchor="left",
            x=1.02,  # position just outside the plot
            bgcolor="rgba(255,255,255,0)"  # transparent background
        )
    )
    
    # Update axes
    fig.update_xaxes(title="Tenure (Years)")
    fig.update_yaxes(title="Number of Employees")
    
    # Add spacing between bars
    fig.update_layout(
        bargap=0.2,  # Gap between bars of adjacent location coordinates
        bargroupgap=0.1  # Gap between bars of the same location coordinates
    )
    
    return fig

def create_age_distribution(employees_df, title="Employee Age Distribution", height=400, show_legend=False):
    """
    Create a histogram showing the distribution of employee ages.
    
    Args:
        employees_df: DataFrame with employee data including date_of_birth
        title: Chart title
        height: Chart height
        show_legend: Whether to show the legend (default: False)
        
    Returns:
        Plotly figure object
    """
    if len(employees_df) == 0 or 'date_of_birth' not in employees_df.columns:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Calculate age
    today = datetime.now()
    employees_df = employees_df.copy()
    employees_df['age'] = employees_df['date_of_birth'].apply(
        lambda x: (today - x).days / 365.25
    )
    
    # Create histogram
    fig = px.histogram(
        employees_df,
        x='age',
        title=title,
        color='gender',
        height=height,
        nbins=20,
        labels={
            'age': 'Age (Years)',
            'gender': 'Gender',
            'count': 'Number of Employees'
        }
    )
    
    # Apply common layout with legend
    fig = apply_common_layout(fig, title=title, height=height, show_legend=True)
    
    # Update axes
    fig.update_xaxes(title="Age (Years)")
    fig.update_yaxes(title="Number of Employees")
    
    # Add spacing between bars
    fig.update_layout(
        bargap=0.2,  # Gap between bars of adjacent location coordinates
        bargroupgap=0.1  # Gap between bars of the same location coordinates
    )
    
    return fig

def create_headcount_trend(df, time_col='month', count_col='count', group_col=None, 
                          title="Headcount Trend", height=400, show_legend=False):
    """
    Create a line chart showing headcount trends over time.
    
    Args:
        df: DataFrame with the data
        time_col: Column for time periods (x-axis)
        count_col: Column for headcount values (y-axis)
        group_col: Optional column for grouping (e.g., department, location)
        title: Chart title
        height: Chart height
        show_legend: Whether to show the legend (default: False)
        
    Returns:
        Plotly figure object
    """
    if len(df) == 0:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Sort by time
    df = df.sort_values(by=time_col)
    
    # Create line chart
    if group_col and group_col in df.columns:
        # Group by the specified column
        color_map = None
        if group_col == 'department':
            color_map = DEPARTMENT_COLORS
        elif group_col == 'worker_type':
            color_map = WORKER_TYPE_COLORS
            
        fig = px.line(
            df,
            x=time_col,
            y=count_col,
            color=group_col,
            title=title,
            height=height,
            markers=True,
            color_discrete_map=color_map,
            color_discrete_sequence=EXTENDED_PALETTE if not color_map else None,
            labels={
                time_col: format_column_name(time_col),
                count_col: 'Headcount',
                group_col: format_column_name(group_col)
            }
        )
    else:
        # Single line for total headcount
        fig = px.line(
            df,
            x=time_col,
            y=count_col,
            title=title,
            height=height,
            markers=True,
            color_discrete_sequence=[COLOR_PALETTE['primary']],
            labels={
                time_col: format_column_name(time_col),
                count_col: 'Headcount'
            }
        )
    
    # Apply common layout
    fig = apply_common_layout(fig, title=title, height=height, show_legend=show_legend)
    
    return fig

def create_salary_trend(payroll_df, time_col='month', value_col='base_salary', 
                      group_col=None, agg_func='mean', title="Salary Trend", height=400):
    """
    Create a line chart showing salary trends over time.
    
    Args:
        payroll_df: DataFrame with payroll data
        time_col: Column containing time periods
        value_col: Column containing salary values (base_salary, gross_pay, etc.)
        group_col: Optional column for grouping (e.g., department, job_title)
        agg_func: Aggregation function ('mean', 'median', 'sum')
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly figure object
    """
    if len(payroll_df) == 0:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Aggregate data by time period and optional grouping
    if group_col and group_col in payroll_df.columns:
        # Group by time and the specified column
        if agg_func == 'mean':
            df_agg = payroll_df.groupby([time_col, group_col])[value_col].mean().reset_index()
        elif agg_func == 'median':
            df_agg = payroll_df.groupby([time_col, group_col])[value_col].median().reset_index()
        elif agg_func == 'sum':
            df_agg = payroll_df.groupby([time_col, group_col])[value_col].sum().reset_index()
        else:
            df_agg = payroll_df.groupby([time_col, group_col])[value_col].mean().reset_index()
            
        # Sort by time
        df_agg = df_agg.sort_values(by=time_col)
        
        # Create color map based on grouping column
        color_map = None
        if group_col == 'department':
            color_map = DEPARTMENT_COLORS
        elif group_col == 'worker_type':
            color_map = WORKER_TYPE_COLORS
            
        # Create line chart
        fig = px.line(
            df_agg,
            x=time_col,
            y=value_col,
            color=group_col,
            title=title,
            height=height,
            markers=True,
            color_discrete_map=color_map,
            color_discrete_sequence=EXTENDED_PALETTE if not color_map else None,
        )
        
    else:
        # Aggregate by time only
        if agg_func == 'mean':
            df_agg = payroll_df.groupby(time_col)[value_col].mean().reset_index()
        elif agg_func == 'median':
            df_agg = payroll_df.groupby(time_col)[value_col].median().reset_index()
        elif agg_func == 'sum':
            df_agg = payroll_df.groupby(time_col)[value_col].sum().reset_index()
        else:
            df_agg = payroll_df.groupby(time_col)[value_col].mean().reset_index()
            
        # Sort by time
        df_agg = df_agg.sort_values(by=time_col)
        
        # Create line chart
        fig = px.line(
            df_agg,
            x=time_col,
            y=value_col,
            title=title,
            height=height,
            markers=True,
            color_discrete_sequence=[COLOR_PALETTE['primary']],
        )
    
    # Apply common layout
    fig = apply_common_layout(fig, title=title, height=height)
    
    # Update axes
    fig.update_xaxes(title="Time Period")
    
    # Update y-axis title based on value and aggregation
    value_labels = {
        'base_salary': 'Base Salary',
        'gross_pay': 'Gross Pay',
        'net_pay': 'Net Pay',
        'overtime_pay': 'Overtime Pay'
    }
    
    agg_labels = {
        'mean': 'Average',
        'median': 'Median',
        'sum': 'Total'
    }
    
    value_label = value_labels.get(value_col, value_col.replace('_', ' ').title())
    agg_label = agg_labels.get(agg_func, agg_func.title())
    
    fig.update_yaxes(title=f"{agg_label} {value_label} (R)")
    
    return fig

def create_productivity_chart(productivity_df, x_col, y_col, color_col=None, 
                             title="Productivity Analysis", height=400):
    """
    Create a chart for visualizing productivity metrics.
    
    Args:
        productivity_df: DataFrame with productivity data
        x_col: Column for x-axis
        y_col: Column for y-axis (productivity metric)
        color_col: Optional column for color encoding
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly figure object
    """
    if len(productivity_df) == 0:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Determine chart type based on x_col
    if x_col == 'month' or x_col.startswith('date') or 'time' in x_col.lower():
        # Time-based productivity chart (line chart)
        if color_col and color_col in productivity_df.columns:
            # Group by time and color column
            df_agg = productivity_df.groupby([x_col, color_col])[y_col].mean().reset_index()
            
            # Sort by time
            df_agg = df_agg.sort_values(by=x_col)
            
            # Create color map based on color column
            color_map = None
            if color_col == 'department':
                color_map = DEPARTMENT_COLORS
            elif color_col == 'worker_type':
                color_map = WORKER_TYPE_COLORS
                
            # Create line chart
            fig = px.line(
                df_agg,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title,
                height=height,
                markers=True,
                color_discrete_map=color_map,
                color_discrete_sequence=EXTENDED_PALETTE if not color_map else None,
            )
            
        else:
            # Aggregate by time only
            df_agg = productivity_df.groupby(x_col)[y_col].mean().reset_index()
            
            # Sort by time
            df_agg = df_agg.sort_values(by=x_col)
            
            # Create line chart
            fig = px.line(
                df_agg,
                x=x_col,
                y=y_col,
                title=title,
                height=height,
                markers=True,
                color_discrete_sequence=[COLOR_PALETTE['primary']],
            )
            
    else:
        # Categorical productivity chart (bar chart)
        if color_col and color_col in productivity_df.columns:
            # Group by category and color column
            df_agg = productivity_df.groupby([x_col, color_col])[y_col].mean().reset_index()
            
            # Create color map based on color column
            color_map = None
            if color_col == 'department':
                color_map = DEPARTMENT_COLORS
            elif color_col == 'worker_type':
                color_map = WORKER_TYPE_COLORS
                
            # Create bar chart
            fig = px.bar(
                df_agg,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title,
                height=height,
                color_discrete_map=color_map,
                color_discrete_sequence=EXTENDED_PALETTE if not color_map else None,
                barmode='group'
            )
            
        else:
            # Aggregate by category only
            df_agg = productivity_df.groupby(x_col)[y_col].mean().reset_index()
            
            # Create bar chart
            fig = px.bar(
                df_agg,
                x=x_col,
                y=y_col,
                title=title,
                height=height,
                color_discrete_sequence=[COLOR_PALETTE['primary']],
            )
    
    # Apply common layout
    fig = apply_common_layout(fig, title=title, height=height)
    
    # Format y-axis label
    y_label = y_col.replace('_', ' ').title()
    fig.update_yaxes(title=y_label)
    
    return fig

def create_leave_calendar(leave_df, title="Leave Calendar", height=400):
    """
    Create a heatmap-like calendar showing leave patterns by month and type.
    
    Args:
        leave_df: DataFrame with leave data
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly figure object
    """
    if len(leave_df) == 0:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Extract year and month from start_date
    leave_df = leave_df.copy()
    leave_df['year_month'] = leave_df['start_date'].dt.strftime('%Y-%m')
    leave_df['month'] = leave_df['start_date'].dt.strftime('%b')
    
    # Aggregate by month and leave type
    leave_agg = leave_df.groupby(['year_month', 'leave_type'])['duration_days'].sum().reset_index()
    
    # Pivot the data for the heatmap
    leave_pivot = leave_agg.pivot(index='leave_type', columns='year_month', values='duration_days')
    
    # Fill NaN with zeros
    leave_pivot = leave_pivot.fillna(0)
    
    # Create heatmap
    fig = px.imshow(
        leave_pivot,
        labels=dict(x="Month", y="Leave Type", color="Days"),
        x=leave_pivot.columns,
        y=leave_pivot.index,
        color_continuous_scale=BLUE_SCALE,
        title=title,
        height=height,
        aspect="auto"
    )
    
    # Add text annotations
    for i in range(len(leave_pivot.index)):
        for j in range(len(leave_pivot.columns)):
            value = leave_pivot.iloc[i, j]
            if value > 0:
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{int(value)}",
                    showarrow=False,
                    font=dict(color="white" if value > 20 else "black")
                )
    
    # Apply common layout
    fig = apply_common_layout(fig, title=title, height=height)
    
    # Update coloraxis
    fig.update_layout(coloraxis_colorbar=dict(title="Days"))
    
    return fig

def create_survey_response_chart(survey_df, question_col, time_col='survey_month', 
                                title=None, height=400):
    """
    Create a stacked bar chart showing survey response distributions over time.
    
    Args:
        survey_df: DataFrame with survey data
        question_col: Column containing responses to a specific question
        time_col: Column containing time periods
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly figure object
    """
    if len(survey_df) == 0 or question_col not in survey_df.columns:
        # Return empty chart with message if no data
        fig = go.Figure()
        fig.update_layout(
            title=title if title else "No data available",
            height=height,
            annotations=[dict(text="No data available", showarrow=False, font_size=16)]
        )
        return fig
    
    # Response order mapping for different question types
    response_orders = {
        'support_feeling': ["Not at all supported", "Rarely supported", "Sometimes supported", "Often supported", "Always supported"],
        'job_satisfaction': ["Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very satisfied"],
        'exhaustion_feeling': ["Never", "Rarely", "Sometimes", "Often", "Always"],
        'work_mood': ["Very unhappy", "Unhappy", "Neutral", "Happy", "Very happy"]
    }
    
    # Default title if not provided
    if not title:
        question_labels = {
            'support_feeling': "Feeling Supported by Supervisors",
            'job_satisfaction': "Job Satisfaction",
            'exhaustion_feeling': "Work Exhaustion",
            'work_mood': "General Mood at Work"
        }
        title = question_labels.get(question_col, question_col.replace('_', ' ').title())

    # Count responses by time period and response value
    response_counts = survey_df.groupby([time_col, question_col]).size().reset_index(name='count')
    
    # Create stacked bar chart
    fig = px.bar(
        response_counts,
        x=time_col,
        y='count',
        color=question_col,
        title=title,
        height=height,
        category_orders={question_col: response_orders.get(question_col, None)},
        color_discrete_sequence=BLUE_SCALE
    )
    
    # Apply common layout
    fig = apply_common_layout(fig, title=title, height=height)
    
    # Update axes
    fig.update_xaxes(title="Survey Month")
    fig.update_yaxes(title="Number of Responses")
    
    return fig