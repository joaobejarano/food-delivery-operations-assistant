import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
@st.cache_data
def load_data():
    orders = pd.read_csv('data/orders.csv', parse_dates=['order_time', 'delivery_time'])
    customers = pd.read_csv('data/customers.csv')
    couriers = pd.read_csv('data/couriers.csv')
    incidents = pd.read_csv('data/incidents.csv')
    orders['delivery_duration'] = (orders['delivery_time'] - orders['order_time']).dt.total_seconds() / 60
    return orders, customers, couriers, incidents

orders, customers, couriers, incidents = load_data()

st.title("🚚 Smart Ops Assistant – Dashboard")

# Sidebar filters
st.sidebar.header("🔎 Filters")
regions = st.sidebar.multiselect("Region", options=orders['region'].unique(), default=orders['region'].unique())

min_date = orders['order_time'].min()
max_date = orders['order_time'].max()
date_range = st.sidebar.date_input("Order Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

ratings = st.sidebar.slider("Customer Rating Range", min_value=1, max_value=5, value=(1, 5))

# Apply filters
filtered_orders = orders[
    (orders['region'].isin(regions)) &
    (orders['order_time'].dt.date >= date_range[0]) &
    (orders['order_time'].dt.date <= date_range[1]) &
    (orders['customer_rating'].fillna(0).between(ratings[0], ratings[1]))
]

st.markdown(f"### 📊 Filtered Results – {len(filtered_orders)} Orders")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", len(filtered_orders))
col2.metric("Avg Delivery Time", f"{filtered_orders['delivery_duration'].mean():.1f} min")
col3.metric("Avg Order Value", f"${filtered_orders['total_amount'].mean():.2f}")
col4.metric("Incidents (Total)", len(incidents[incidents['order_id'].isin(filtered_orders['order_id'])]))

# Orders over time
st.subheader("📅 Orders per Day")
orders_by_day = filtered_orders.copy()
orders_by_day['day'] = orders_by_day['order_time'].dt.date
fig_day = px.bar(orders_by_day.groupby('day').size().reset_index(name='orders'),
                 x='day', y='orders', title="Orders per Day")
st.plotly_chart(fig_day, use_container_width=True)

# Orders by Region
st.subheader("📍 Orders by Region")
region_data = filtered_orders['region'].value_counts().reset_index()
region_data.columns = ['Region', 'Orders']
fig_region = px.bar(region_data, x='Region', y='Orders', title="Orders by Region")

st.plotly_chart(fig_region, use_container_width=True)

# Avg Delivery Time by Region
st.subheader("⏱️ Avg Delivery Time by Region")
avg_time = filtered_orders.groupby('region')['delivery_duration'].mean().reset_index()
fig_time = px.bar(avg_time, x='region', y='delivery_duration', labels={'delivery_duration': 'Minutes'})
st.plotly_chart(fig_time, use_container_width=True)

# Customer Ratings
st.subheader("⭐ Customer Rating Distribution")
ratings_count = filtered_orders['customer_rating'].value_counts(dropna=False).sort_index().reset_index()
ratings_count.columns = ['Rating', 'Count']
fig_ratings = px.bar(ratings_count, x='Rating', y='Count', title="Ratings")
st.plotly_chart(fig_ratings, use_container_width=True)

# Incident Types
st.subheader("⚠️ Incident Types")
filtered_incidents = incidents[incidents['order_id'].isin(filtered_orders['order_id'])]
incident_counts = filtered_incidents['incident_type'].value_counts().reset_index()
incident_counts.columns = ['Incident Type', 'Count']
fig_incidents = px.bar(incident_counts, x='Incident Type', y='Count', title="Reported Incidents")
st.plotly_chart(fig_incidents, use_container_width=True)

# Data preview
st.subheader("📄 Sample of Filtered Orders")
st.dataframe(filtered_orders.sample(min(10, len(filtered_orders))))
