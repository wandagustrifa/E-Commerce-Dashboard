import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="E-commerce Data Analysis Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 32px;
        font-weight: bold;
        color: #0D47A1;
        margin-bottom: 25px;
        text-align: center;
    }

    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #0277BD;
        margin-top: 1rem;
    }
    .metric-container {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 25px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: bold;
        color: #27ae60;
        margin-bottom: 5px;
    }

    .metric-label {
        font-size: 14px;
        font-weight: bold;
        color: #7f8c8d;
    }
    
    .insights {
        background-color: #e3f2fd;
        border-left: 5px solid #1E88E5;
        padding: 1rem;
        border-radius: 0 5px 5px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all the datasets needed for the dashboard"""
    try:
        dashboard_data = pd.read_csv('data/dashboard_data.csv')
        
        # Convert date columns to datetime
        date_columns = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
        for col in date_columns:
            dashboard_data[col] = pd.to_datetime(dashboard_data[col])
        
        rfm_data = pd.read_csv('data/rfm_data.csv')
        
        return dashboard_data, rfm_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Load data
dashboard_data, rfm_data = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Overview", "Sales Analysis", "Customer Segmentation", "Geographic Analysis", "Delivery Performance", "Product Analysis"]
)

# Sidebar date filter
if dashboard_data is not None:
    min_date = dashboard_data['order_purchase_timestamp'].min().date()
    max_date = dashboard_data['order_purchase_timestamp'].max().date()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Date Range Filter")
    
    start_date = st.sidebar.date_input(
        "Start date",
        min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End date",
        max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data based on date range
    filtered_data = dashboard_data[
        (dashboard_data['order_purchase_timestamp'].dt.date >= start_date) &
        (dashboard_data['order_purchase_timestamp'].dt.date <= end_date)
    ]
else:
    st.error("Failed to load data. Please check if the CSV files exist in the current directory.")
    st.stop()

# Helper functions
def format_currency_short(value):
    """Format currency in short form, e.g., R$ 3.2M"""
    if value >= 1_000_000:
        return f"R$ {value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"R$ {value/1_000:.1f}K"
    return f"R$ {value:.2f}"

def create_metric_row(metrics_data):
    """Create a row of metric cards"""
    cols = st.columns(len(metrics_data))
    for i, (label, value, prefix, suffix) in enumerate(metrics_data):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{prefix}{value}{suffix}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

#################################################
# OVERVIEW PAGE
#################################################
if page == "Overview":
    st.markdown('<div class="main-header">E-commerce Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Summary metrics
    total_orders = len(filtered_data['order_id'].unique())
    total_revenue = filtered_data['payment_value'].sum()
    total_customers = len(filtered_data['customer_id'].unique())
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    
    metrics_data = [
    ("Total Orders", f"{total_orders:,}", "", ""),
    ("Total Revenue", format_currency_short(total_revenue), "", ""),
    ("Total Customers", f"{total_customers:,}", "", ""),
    ("Average Order Value", format_currency_short(avg_order_value), "", "")
]
    
    create_metric_row(metrics_data)
    
    # Order status distribution
    st.markdown('<div class="section-header">Order Status Distribution</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        status_counts = filtered_data['order_status'].value_counts().reset_index()
        status_counts.columns = ['status', 'count']
        
        fig = px.pie(
            status_counts, 
            values='count', 
            names='status',
            title='Order Status Distribution',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="insights">', unsafe_allow_html=True)
        st.markdown("### Key Insights")
        
        delivered_pct = (filtered_data['order_status'] == 'delivered').mean() * 100
        cancelled_pct = (filtered_data['order_status'] == 'canceled').mean() * 100
        
        st.markdown(f"""
        - **{delivered_pct:.1f}%** of orders were successfully delivered
        - **{cancelled_pct:.1f}%** of orders were canceled
        - Cancelled orders represent a potential revenue recovery opportunity
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Order trend over time
    st.markdown('<div class="section-header">Order Trend Over Time</div>', unsafe_allow_html=True)
    
    # Create a date column for grouping
    filtered_data['order_date'] = filtered_data['order_purchase_timestamp'].dt.date
    orders_by_date = filtered_data.groupby('order_date').agg({
        'order_id': pd.Series.nunique,
        'payment_value': 'sum'
    }).reset_index()
    
    tab1, tab2 = st.tabs(["Orders Count", "Revenue"])
    
    with tab1:
        fig = px.line(
            orders_by_date, 
            x='order_date', 
            y='order_id',
            labels={'order_date': 'Date', 'order_id': 'Number of Orders'},
            title='Daily Order Volume'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.line(
            orders_by_date, 
            x='order_date', 
            y='payment_value',
            labels={'order_date': 'Date', 'payment_value': 'Revenue (R$)'},
            title='Daily Revenue'
        )
        fig.update_layout(yaxis_title='Revenue (R$)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top categories and hourly distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Top 10 Product Categories</div>', unsafe_allow_html=True)
        
        top_categories = filtered_data['product_category_name_english'].value_counts().head(10)
        
        fig = px.bar(
            x=top_categories.values,
            y=top_categories.index,
            orientation='h',
            labels={'x': 'Number of Orders', 'y': 'Category'},
            title='Top 10 Product Categories'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Hourly Order Distribution</div>', unsafe_allow_html=True)
        
        filtered_data['hour'] = filtered_data['order_purchase_timestamp'].dt.hour
        hourly_orders = filtered_data.groupby('hour')['order_id'].nunique().reset_index()
        
        fig = px.bar(
            hourly_orders,
            x='hour',
            y='order_id',
            labels={'hour': 'Hour of Day', 'order_id': 'Number of Orders'},
            title='Orders by Hour of Day'
        )
        fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
        st.plotly_chart(fig, use_container_width=True)

#################################################
# SALES ANALYSIS PAGE
#################################################
elif page == "Sales Analysis":
    st.markdown('<div class="main-header">Sales Analysis</div>', unsafe_allow_html=True)
    
    # Sales metrics
    total_revenue = filtered_data['payment_value'].sum()
    avg_order_value = filtered_data.groupby('order_id')['payment_value'].sum().mean()
    total_items_sold = len(filtered_data)
    
    metrics_data = [
        ("Total Revenue", format_currency_short(total_revenue), "", ""),
        ("Average Order Value", f"{avg_order_value:.2f}", "R$ ", ""),
        ("Total Items Sold", f"{total_items_sold:,}", "", "")
    ]
    
    create_metric_row(metrics_data)
    
    # Sales over time with trend
    st.markdown('<div class="section-header">Sales Trends</div>', unsafe_allow_html=True)
    
    # Aggregate by month for better visualization
    filtered_data['month_year'] = filtered_data['order_purchase_timestamp'].dt.strftime('%Y-%m')
    monthly_sales = filtered_data.groupby('month_year').agg({
        'payment_value': 'sum',
        'order_id': pd.Series.nunique
    }).reset_index()
    
    # Sort by month_year to ensure correct chronological order
    monthly_sales['month_year'] = pd.to_datetime(monthly_sales['month_year'] + '-01')
    monthly_sales = monthly_sales.sort_values('month_year')
    monthly_sales['month_year_str'] = monthly_sales['month_year'].dt.strftime('%b %Y')
    
    tab1, tab2 = st.tabs(["Monthly Revenue", "Monthly Orders"])
    
    with tab1:
        fig = px.line(
            monthly_sales, 
            x='month_year_str', 
            y='payment_value',
            markers=True,
            labels={'month_year_str': 'Month', 'payment_value': 'Revenue (R$)'},
            title='Monthly Revenue Trend'
        )
        fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':monthly_sales['month_year_str'].tolist()})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.line(
            monthly_sales, 
            x='month_year_str', 
            y='order_id',
            markers=True,
            labels={'month_year_str': 'Month', 'order_id': 'Number of Orders'},
            title='Monthly Order Volume Trend'
        )
        fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':monthly_sales['month_year_str'].tolist()})
        st.plotly_chart(fig, use_container_width=True)
    
    # Payment method analysis
    st.markdown('<div class="section-header">Payment Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown('<div class="subsection-header">Payment Method Distribution</div>', unsafe_allow_html=True)
        
        # Use a different approach since we only have the merged data
        # Let's assume payment_type might be in the data based on the analysis code
        if 'payment_type' in filtered_data.columns:
            payment_counts = filtered_data['payment_type'].value_counts().reset_index()
            payment_counts.columns = ['payment_type', 'count']
            
            fig = px.pie(
                payment_counts, 
                values='count', 
                names='payment_type',
                title='Payment Method Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Payment method information is not available in the merged dataset.")
    
    with col2:
        st.markdown('<div class="subsection-header">Order Value Distribution</div>', unsafe_allow_html=True)
        
        # Calculate order values
        order_values = filtered_data.groupby('order_id')['payment_value'].sum().reset_index()
        
        # Create histogram of order values
        fig = px.histogram(
            order_values, 
            x='payment_value',
            nbins=50,
            labels={'payment_value': 'Order Value (R$)', 'count': 'Number of Orders'},
            title='Order Value Distribution'
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

#################################################
# CUSTOMER SEGMENTATION PAGE
#################################################
elif page == "Customer Segmentation":
    st.markdown('<div class="main-header">Customer Segmentation Analysis</div>', unsafe_allow_html=True)
    
    # RFM Analysis Overview
    st.markdown('<div class="section-header">RFM Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        if 'Segment' in rfm_data.columns:
            segment_counts = rfm_data['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            
            # Set segment order
            segment_order = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk Customers', 'Need Attention']
            segment_counts['Segment'] = pd.Categorical(segment_counts['Segment'], categories=segment_order, ordered=True)
            segment_counts = segment_counts.sort_values('Segment')
            
            fig = px.pie(
                segment_counts, 
                values='Count', 
                names='Segment',
                title='Customer Segments Distribution',
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="insights">', unsafe_allow_html=True)
        st.markdown("### RFM Segment Definitions")
        st.markdown("""
        - **Champions**: High spending, frequent buyers who purchased recently
        - **Loyal Customers**: Regular buyers with good spending habits
        - **Potential Loyalists**: Recent buyers with moderate frequency
        - **At Risk Customers**: Formerly regular customers who haven't purchased recently
        - **Need Attention**: Low spending, infrequent customers who aren't very recent
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # RFM Metrics Distribution
    st.markdown('<div class="section-header">RFM Metrics Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Recency", "Frequency", "Monetary"])
    
    with tab1:
        fig = px.box(
            rfm_data, 
            x='Segment', 
            y='recency',
            category_orders={"Segment": segment_order},
            labels={'recency': 'Recency (days)', 'Segment': 'Customer Segment'},
            title='Recency Distribution by Segment'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.box(
            rfm_data, 
            x='Segment', 
            y='frequency',
            category_orders={"Segment": segment_order},
            labels={'frequency': 'Frequency (# of orders)', 'Segment': 'Customer Segment'},
            title='Order Frequency Distribution by Segment'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.box(
            rfm_data, 
            x='Segment', 
            y='monetary',
            category_orders={"Segment": segment_order},
            labels={'monetary': 'Monetary Value (R$)', 'Segment': 'Customer Segment'},
            title='Monetary Value Distribution by Segment'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment characteristics
    st.markdown('<div class="section-header">Segment Characteristics</div>', unsafe_allow_html=True)
    
    # Merge RFM data with dashboard data for additional insights
    customer_segment_mapping = rfm_data[['customer_id', 'Segment']]
    segment_orders = filtered_data.merge(customer_segment_mapping, on='customer_id', how='inner')
    
    metrics_by_segment = segment_orders.groupby('Segment').agg({
        'order_id': pd.Series.nunique,
        'payment_value': 'sum',
        'customer_id': pd.Series.nunique
    }).reset_index()
    
    metrics_by_segment['avg_order_value'] = metrics_by_segment['payment_value'] / metrics_by_segment['order_id']
    metrics_by_segment['orders_per_customer'] = metrics_by_segment['order_id'] / metrics_by_segment['customer_id']
    metrics_by_segment['revenue_per_customer'] = metrics_by_segment['payment_value'] / metrics_by_segment['customer_id']
    
    # Set segment order
    metrics_by_segment['Segment'] = pd.Categorical(
        metrics_by_segment['Segment'], 
        categories=segment_order, 
        ordered=True
    )
    metrics_by_segment = metrics_by_segment.sort_values('Segment')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            metrics_by_segment,
            x='Segment',
            y='avg_order_value',
            labels={'avg_order_value': 'Average Order Value (R$)', 'Segment': 'Customer Segment'},
            title='Average Order Value by Segment',
            color='Segment',
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            metrics_by_segment,
            x='Segment',
            y='revenue_per_customer',
            labels={'revenue_per_customer': 'Revenue per Customer (R$)', 'Segment': 'Customer Segment'},
            title='Revenue per Customer by Segment',
            color='Segment',
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig, use_container_width=True)

#################################################
# GEOGRAPHIC ANALYSIS PAGE
#################################################
elif page == "Geographic Analysis":
    st.markdown('<div class="main-header">Geographic Analysis</div>', unsafe_allow_html=True)
    
    # Regional (State-level) analysis
    st.markdown('<div class="section-header">Regional Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Top States by Sales</div>', unsafe_allow_html=True)
        state_sales = filtered_data.groupby('customer_state')['payment_value'].sum().reset_index()
        state_sales = state_sales.sort_values('payment_value', ascending=False).head(10)
        fig = px.bar(
            state_sales,
            x='payment_value',
            y='customer_state',
            orientation='h',
            labels={'payment_value': 'Sales Revenue (R$)', 'customer_state': 'State'},
            title='Top 10 States by Sales Revenue',
            color='payment_value',
            color_continuous_scale=px.colors.sequential.Blues
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Top States by Order Count</div>', unsafe_allow_html=True)
        state_orders = filtered_data.groupby('customer_state')['order_id'].nunique().reset_index()
        state_orders = state_orders.sort_values('order_id', ascending=False).head(10)
        fig = px.bar(
            state_orders,
            x='order_id',
            y='customer_state',
            orientation='h',
            labels={'order_id': 'Number of Orders', 'customer_state': 'State'},
            title='Top 10 States by Order Count',
            color='order_id',
            color_continuous_scale=px.colors.sequential.Greens
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # City-level analysis
    st.markdown('<div class="section-header">City Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Top Cities by Sales</div>', unsafe_allow_html=True)
        city_sales = filtered_data.groupby('customer_city')['payment_value'].sum().reset_index()
        city_sales = city_sales.sort_values('payment_value', ascending=False).head(10)
        fig = px.bar(
            city_sales,
            x='payment_value',
            y='customer_city',
            orientation='h',
            labels={'payment_value': 'Sales Revenue (R$)', 'customer_city': 'City'},
            title='Top 10 Cities by Sales Revenue',
            color='payment_value',
            color_continuous_scale=px.colors.sequential.Oranges
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Top Cities by Customer Count</div>', unsafe_allow_html=True)
        city_customers = filtered_data.groupby('customer_city')['customer_id'].nunique().reset_index()
        city_customers = city_customers.sort_values('customer_id', ascending=False).head(10)
        fig = px.bar(
            city_customers,
            x='customer_id',
            y='customer_city',
            orientation='h',
            labels={'customer_id': 'Number of Customers', 'customer_city': 'City'},
            title='Top 10 Cities by Customer Count',
            color='customer_id',
            color_continuous_scale=px.colors.sequential.Purples
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sales Geographic Map (full-width di bawah semua grafik)
    st.markdown('<div class="section-header">Sales Geographic Distribution</div>', unsafe_allow_html=True)
    st.info("This map visualizes sales distribution across top cities in Brazil using geolocation data.")
    
    required_columns = ['geolocation_lat', 'geolocation_lng', 'customer_city', 'payment_value']
    if all(col in filtered_data.columns for col in required_columns):
        top_cities_by_sales = (
            filtered_data.groupby(['customer_city', 'geolocation_lat', 'geolocation_lng'])['payment_value']
            .sum()
            .reset_index()
            .rename(columns={'payment_value': 'price'})
            .sort_values('price', ascending=False)
            .head(30)
        )

        m = folium.Map(location=[-15.77972, -47.92972], zoom_start=4, width='100%', height='100%')
        max_price = top_cities_by_sales['price'].max()

        for _, row in top_cities_by_sales.iterrows():
            size = min(50, (row['price'] / max_price) * 100)
            tooltip = f"{row['customer_city']}: R$ {row['price']:,.2f}"
            folium.CircleMarker(
                location=[row['geolocation_lat'], row['geolocation_lng']],
                radius=size / 10,
                popup=tooltip,
                tooltip=tooltip,
                color='crimson',
                fill=True,
                fill_color='crimson',
                fill_opacity=0.6
            ).add_to(m)

        st.markdown('<div class="subsection-header">Sales Distribution Map by City</div>', unsafe_allow_html=True)
        folium_static(m, width=1200, height=600)
    else:
        st.warning("Geolocation data is missing in the dataset.")

#################################################
# DELIVERY PERFORMANCE PAGE
#################################################
elif page == "Delivery Performance":
    st.markdown('<div class="main-header">Delivery Performance Analysis</div>', unsafe_allow_html=True)
    
    # Only consider delivered orders for delivery analysis
    delivered_orders = filtered_data[filtered_data['order_status'] == 'delivered'].copy()
    
    # Calculate delivery metrics if possible
    if 'order_delivered_customer_date' in delivered_orders.columns and 'order_estimated_delivery_date' in delivered_orders.columns:
        delivered_orders['delivery_time'] = (delivered_orders['order_delivered_customer_date'] - delivered_orders['order_purchase_timestamp']).dt.days
        delivered_orders['estimated_delivery_time'] = (delivered_orders['order_estimated_delivery_date'] - delivered_orders['order_purchase_timestamp']).dt.days
        delivered_orders['delivery_vs_estimated'] = delivered_orders['order_delivered_customer_date'] <= delivered_orders['order_estimated_delivery_date']
        
        # Metrics
        avg_delivery_time = delivered_orders['delivery_time'].mean()
        on_time_delivery_rate = delivered_orders['delivery_vs_estimated'].mean() * 100
        median_delivery_time = delivered_orders['delivery_time'].median()
        
        metrics_data = [
            ("Average Delivery Time", f"{avg_delivery_time:.1f}", "", " days"),
            ("On-Time Delivery Rate", f"{on_time_delivery_rate:.1f}", "", "%"),
            ("Median Delivery Time", f"{median_delivery_time:.1f}", "", " days")
        ]
        
        create_metric_row(metrics_data)
        
        # Delivery time distribution
        st.markdown('<div class="section-header">Delivery Time Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="subsection-header">Delivery Time Distribution</div>', unsafe_allow_html=True)
            
            fig = px.histogram(
                delivered_orders,
                x='delivery_time',
                nbins=30,
                labels={'delivery_time': 'Delivery Time (days)'},
                title='Distribution of Delivery Times'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="subsection-header">Actual vs. Estimated Delivery Time</div>', unsafe_allow_html=True)
            
            fig = px.scatter(
                delivered_orders.sample(min(1000, len(delivered_orders))),
                x='estimated_delivery_time',
                y='delivery_time',
                labels={'estimated_delivery_time': 'Estimated Delivery Time (days)', 
                        'delivery_time': 'Actual Delivery Time (days)'},
                title='Actual vs. Estimated Delivery Time'
            )
            
            # Add a diagonal line for reference (y=x)
            fig.add_trace(
                go.Scatter(
                    x=[0, delivered_orders['estimated_delivery_time'].max()],
                    y=[0, delivered_orders['estimated_delivery_time'].max()],
                    mode='lines',
                    name='On Time',
                    line=dict(dash='dash', color='green')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Delivery performance by region
        st.markdown('<div class="section-header">Regional Delivery Performance</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="subsection-header">Average Delivery Time by State</div>', unsafe_allow_html=True)
            
            delivery_by_state = delivered_orders.groupby('customer_state')['delivery_time'].mean().reset_index()
            delivery_by_state = delivery_by_state.sort_values('delivery_time', ascending=False).head(10)
            
            fig = px.bar(
                delivery_by_state,
                x='delivery_time',
                y='customer_state',
                orientation='h',
                labels={'delivery_time': 'Average Delivery Time (days)', 'customer_state': 'State'},
                title='Top 10 States by Average Delivery Time',
                color='delivery_time',
                color_continuous_scale=px.colors.sequential.Reds
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="subsection-header">On-Time Delivery Rate by State</div>', unsafe_allow_html=True)
            
            ontime_by_state = delivered_orders.groupby('customer_state')['delivery_vs_estimated'].mean().reset_index()
            ontime_by_state['on_time_pct'] = ontime_by_state['delivery_vs_estimated'] * 100
            ontime_by_state = ontime_by_state.sort_values('on_time_pct').head(10)
            
            fig = px.bar(
                ontime_by_state,
                x='on_time_pct',
                y='customer_state',
                orientation='h',
                labels={'on_time_pct': 'On-Time Delivery Rate (%)', 'customer_state': 'State'},
                title='10 States with Lowest On-Time Delivery Rates',
                color='on_time_pct',
                color_continuous_scale=px.colors.sequential.Greens
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cross-state vs. same-state delivery analysis
        if 'seller_state' in delivered_orders.columns:
            st.markdown('<div class="section-header">Cross-State vs. Same-State Delivery Analysis</div>', unsafe_allow_html=True)
            
            delivered_orders['cross_state'] = delivered_orders['customer_state'] != delivered_orders['seller_state']
            cross_state_analysis = delivered_orders.groupby('cross_state').agg({
                'delivery_time': 'mean',
                'freight_value': 'mean',
                'delivery_vs_estimated': 'mean',
                'order_id': 'count'
            }).reset_index()
            
            cross_state_analysis['on_time_pct'] = cross_state_analysis['delivery_vs_estimated'] * 100
            cross_state_analysis['delivery_type'] = cross_state_analysis['cross_state'].map({
                True: 'Cross-State Delivery',
                False: 'Same-State Delivery'
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="subsection-header">Delivery Time: Cross-State vs. Same-State</div>', unsafe_allow_html=True)
                
                fig = px.bar(
                    cross_state_analysis,
                    x='delivery_type',
                    y='delivery_time',
                    labels={'delivery_time': 'Average Delivery Time (days)', 'delivery_type': 'Delivery Type'},
                    title='Delivery Time: Cross-State vs. Same-State',
                    color='delivery_type',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown('<div class="subsection-header">Freight Value: Cross-State vs. Same-State</div>', unsafe_allow_html=True)
                
                fig = px.bar(
                    cross_state_analysis,
                    x='delivery_type',
                    y='freight_value',
                    labels={'freight_value': 'Average Freight Value (R$)', 'delivery_type': 'Delivery Type'},
                    title='Freight Value: Cross-State vs. Same-State',
                    color='delivery_type',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Insights on cross-state vs. same-state delivery
            st.markdown('<div class="insights">', unsafe_allow_html=True)
            
            delivery_diff = cross_state_analysis.loc[cross_state_analysis['cross_state'], 'delivery_time'].values[0] - \
                           cross_state_analysis.loc[~cross_state_analysis['cross_state'], 'delivery_time'].values[0]
            
            freight_diff_pct = (cross_state_analysis.loc[cross_state_analysis['cross_state'], 'freight_value'].values[0] / \
                               cross_state_analysis.loc[~cross_state_analysis['cross_state'], 'freight_value'].values[0] - 1) * 100
            
            st.markdown("### Delivery Performance Insights")
            st.markdown(f"""
            - Cross-state deliveries take on average **{delivery_diff:.1f} more days** compared to same-state deliveries
            - Cross-state freight costs are **{freight_diff_pct:.1f}% higher** than same-state deliveries
            - {cross_state_analysis.loc[cross_state_analysis['cross_state'], 'order_id'].values[0] / delivered_orders.shape[0]:.1%} of delivered orders are cross-state deliveries
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Required date columns are missing for delivery analysis.")

#################################################
# PRODUCT ANALYSIS PAGE
#################################################
elif page == "Product Analysis":
    st.markdown('<div class="main-header">Product Analysis</div>', unsafe_allow_html=True)
    
    # Product category metrics
    st.markdown('<div class="section-header">Product Category Performance</div>', unsafe_allow_html=True)
    
    # Calculate metrics by product category
    category_metrics = filtered_data.groupby('product_category_name_english').agg({
        'order_id': pd.Series.nunique,
        'price': 'sum',
        'payment_value': 'sum',
        'customer_id': pd.Series.nunique
    }).reset_index()
    
    category_metrics = category_metrics.rename(columns={
        'order_id': 'orders',
        'price': 'revenue',
        'payment_value': 'payment',
        'customer_id': 'customers'
    })
    
    category_metrics['avg_price'] = category_metrics['revenue'] / category_metrics['orders']
    
    # Sort by revenue for top categories
    top_categories = category_metrics.sort_values('revenue', ascending=False).head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Top 10 Categories by Revenue</div>', unsafe_allow_html=True)
        
        fig = px.bar(
            top_categories,
            x='revenue',
            y='product_category_name_english',
            orientation='h',
            labels={'revenue': 'Revenue (R$)', 'product_category_name_english': 'Category'},
            title='Top 10 Categories by Revenue',
            color='revenue',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Top 10 Categories by Order Count</div>', unsafe_allow_html=True)
        
        top_categories_by_orders = category_metrics.sort_values('orders', ascending=False).head(10)
        
        fig = px.bar(
            top_categories_by_orders,
            x='orders',
            y='product_category_name_english',
            orientation='h',
            labels={'orders': 'Number of Orders', 'product_category_name_english': 'Category'},
            title='Top 10 Categories by Order Count',
            color='orders',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Average price by category
    st.markdown('<div class="section-header">Pricing Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Categories with Highest Average Price</div>', unsafe_allow_html=True)
        
        top_categories_by_price = category_metrics.sort_values('avg_price', ascending=False).head(10)
        
        fig = px.bar(
            top_categories_by_price,
            x='avg_price',
            y='product_category_name_english',
            orientation='h',
            labels={'avg_price': 'Average Price (R$)', 'product_category_name_english': 'Category'},
            title='Categories with Highest Average Price',
            color='avg_price',
            color_continuous_scale=px.colors.sequential.Magenta
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Price Distribution by Category</div>', unsafe_allow_html=True)
        
        # Show price distribution for top 5 categories by revenue
        top5_categories = top_categories['product_category_name_english'].head(5).tolist()
        price_data = filtered_data[filtered_data['product_category_name_english'].isin(top5_categories)]
        
        fig = px.box(
            price_data,
            x='product_category_name_english',
            y='price',
            labels={'price': 'Price (R$)', 'product_category_name_english': 'Category'},
            title='Price Distribution for Top 5 Categories by Revenue',
            color='product_category_name_english',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Product category review performance
    if 'review_score' in filtered_data.columns:
        st.markdown('<div class="section-header">Product Category Review Analysis</div>', unsafe_allow_html=True)
        
        # Calculate average review score by category
        category_reviews = filtered_data.groupby('product_category_name_english')['review_score'].agg(['mean', 'count']).reset_index()
        category_reviews = category_reviews.rename(columns={'mean': 'avg_review', 'count': 'review_count'})
        
        # Filter for categories with significant number of reviews
        min_reviews = 10
        significant_categories = category_reviews[category_reviews['review_count'] >= min_reviews]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="subsection-header">Categories with Highest Average Review</div>', unsafe_allow_html=True)
            
            top_rated = significant_categories.sort_values('avg_review', ascending=False).head(10)
            
            fig = px.bar(
                top_rated,
                x='avg_review',
                y='product_category_name_english',
                orientation='h',
                labels={'avg_review': 'Average Review Score (1-5)', 'product_category_name_english': 'Category'},
                title=f'Top Rated Categories (min {min_reviews} reviews)',
                color='avg_review',
                color_continuous_scale=px.colors.sequential.Greens
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="subsection-header">Categories with Lowest Average Review</div>', unsafe_allow_html=True)
            
            lowest_rated = significant_categories.sort_values('avg_review').head(10)
            
            fig = px.bar(
                lowest_rated,
                x='avg_review',
                y='product_category_name_english',
                orientation='h',
                labels={'avg_review': 'Average Review Score (1-5)', 'product_category_name_english': 'Category'},
                title=f'Lowest Rated Categories (min {min_reviews} reviews)',
                color='avg_review',
                color_continuous_scale=px.colors.sequential.Greens
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # --- Price vs. Review Correlation ---
    if 'review_score' in filtered_data.columns and 'price' in filtered_data.columns:
        st.markdown('<div class="section-header">Price vs. Review Correlation</div>', unsafe_allow_html=True)

        corr_data = filtered_data[['price', 'review_score', 'product_category_name_english']].dropna()
        sample_data = corr_data.sample(min(5000, len(corr_data)), random_state=42)

        # Scatter plot
        fig = px.scatter(
            sample_data,
            x='price',
            y='review_score',
            labels={'price': 'Price (R$)', 'review_score': 'Review Score (1-5)'},
            title='Price vs. Review Score Correlation',
            opacity=0.6,
            color='product_category_name_english',
            color_discrete_sequence=px.colors.qualitative.Plotly,
            trendline='ols'  
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Correlation
        correlation = corr_data['price'].corr(corr_data['review_score'])

        # Show insight
        insight_text = (
            "There is a positive correlation, suggesting higher-priced items tend to receive slightly better reviews."
            if correlation > 0.1 else
            "There is a negative correlation, suggesting higher-priced items tend to receive slightly worse reviews."
            if correlation < -0.1 else
            "There is little to no correlation between price and review score, suggesting that customer satisfaction is not significantly related to price."
        )

        st.markdown(f"""
        <div class="insights">
        <h3>Price vs. Review Insights</h3>
        <ul>
            <li>The correlation coefficient between price and review score is <strong>{correlation:.3f}</strong></li>
            <li>{insight_text}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Data tidak memiliki kolom 'price' dan 'review_score' yang lengkap untuk analisis.")


    # --- Product Category Affinity Analysis ---
    st.markdown('<div class="section-header">Product Category Affinity Analysis</div>', unsafe_allow_html=True)

    st.info("This section contains product affinity analysis (frequently bought together) using Market Basket Analysis.")

    # Allow user to set minimum support
    min_sup = st.slider("Select minimum support", min_value=0.001, max_value=0.05, step=0.001, value=0.01)

    # Prepare data: One-hot encoded basket by product category per order
    basket = dashboard_data.groupby(['order_id', 'product_category_name_english'])['product_category_name_english'].count().unstack().fillna(0)
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Ambil hanya order dengan lebih dari satu kategori
    basket_sets = basket_sets[basket_sets.sum(axis=1) > 1]

    # Jalankan apriori dengan threshold rendah
    frequent_itemsets = apriori(basket_sets, min_support=0.005, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)


    if rules.empty:
        st.warning("No significant association rules found with the chosen parameters.")
    else:
        st.subheader("Top Association Rules")
        top_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)
        top_rules_display = top_rules.copy()
        top_rules_display['antecedents'] = top_rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
        top_rules_display['consequents'] = top_rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
        st.dataframe(top_rules_display)

        # Network graph visualization
        st.subheader("Product Category Relationship Network")
        G = nx.DiGraph()

        for _, row in top_rules.head(15).iterrows():
            for ant in row['antecedents']:
                for cons in row['consequents']:
                    G.add_edge(ant, cons, weight=row['lift'])

        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y, edge_text = [], [], []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_text.append(f"{edge[0]} ➝ {edge[1]}<br>Lift: {edge[2]['weight']:.2f}")

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='text',
            mode='lines'
        )

        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="bottom center",
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color='skyblue',
                size=20,
                line_width=2
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            height=600
                        ))

        st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("""
<footer style="margin-top: 40px; padding: 10px; text-align: center; color: #888; font-size: 14px;">
    &copy; 2025 Your Company. All rights reserved.
</footer>
""", unsafe_allow_html=True)
