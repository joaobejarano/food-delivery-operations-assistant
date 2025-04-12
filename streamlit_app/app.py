import torch
try:
    torch._classes = None
except Exception:
    pass

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import plotly.express as px

from src.forecasting import get_demand_series_by_region, train_prophet_model
from src.rag.qa import answer_question
from src.rag.gpt_insights import generate_operational_insight

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

# Orders per Day
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

# 📈 Demand Forecast
st.subheader("📈 Forecast Demand per Region")

region_to_forecast = st.selectbox("Select Region to Forecast", options=orders['region'].unique())

if st.button("Generate Forecast"):
    st.write(f"🔮 Forecasting demand for: **{region_to_forecast}**")
    
    df_region = get_demand_series_by_region(orders, region_to_forecast)

    # ✅ Verificações de segurança antes de treinar
    if df_region.empty:
        st.warning("⚠️ No data available for this region.")
    elif df_region['y'].isnull().any():
        st.warning("⚠️ Found null values in the series. Please clean the data.")
    elif df_region['y'].nunique() < 2:
        st.warning("⚠️ Not enough variation in data to train a forecast model.")
    elif len(df_region) < 10:
        st.warning("⚠️ Not enough data points to forecast.")
    else:
        try:
            forecast, model = train_prophet_model(df_region, periods=24)
            forecast_to_plot = forecast[['ds', 'yhat']].rename(columns={'ds': 'Datetime', 'yhat': 'Predicted Orders'})
            fig_forecast = px.line(forecast_to_plot, x='Datetime', y='Predicted Orders',
                                title=f"📈 Predicted Demand - Next 24 Hours - {region_to_forecast}")
            st.plotly_chart(fig_forecast, use_container_width=True)

            if model is None:
                st.warning("⚠️ Prophet failed — using simple moving average as fallback.")
        except Exception as e:
            st.error(f"❌ Prophet failed: {e}")

        
# Sample data
st.subheader("📄 Sample of Filtered Orders")
st.dataframe(filtered_orders.sample(min(10, len(filtered_orders))))

st.subheader("🤖 Incident Assistant (RAG + GPT)")

with st.expander("📘 Example questions"):
    st.markdown("""
    - How to deal with rain delays?
    - What to do if the delivery person doesn't show up?
    - What solution to apply when there are many cancellations?
    """)

user_question = st.text_area("Enter your question about logistics operations or incidents:")

if st.button("Query"):
    if user_question.strip() == "":
        st.warning("Enter a question before querying.")
    else:
        with st.spinner("Querying incident database and generating response with AI..."):
            try:            
                answer = answer_question(user_question)
                st.success(answer)
            except Exception as e:
                st.error(f"❌ Error querying the assistant: {e}")


st.subheader("📍 Manager Insight (Powered by GPT-4)")

if not filtered_incidents.empty:
    if st.button("Generate AI Recommendation"):
        with st.spinner("Analyzing incidents and generating recommendation..."):
            insight = generate_operational_insight(filtered_incidents, region=", ".join(regions))
            st.markdown("### 📋 Recommended Action")
            st.markdown(
                f"""
                <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 10px; border: 1px solid #ddd;">
                    {insight}
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("No incidents found in the current filter to generate recommendations.")