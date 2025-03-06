import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data():
    # Adjust the path to your data file
    data = pd.read_csv("data/job_postings.csv", parse_dates=["published_date"])
    data["YearMonth"] = data["published_date"].dt.to_period("M")
    return data

# Load the data
data = load_data()

# Remove timezone information from `published_date`
data["published_date"] = data["published_date"].dt.tz_localize(None)

# Streamlit Dashboard Title
st.title("Job Market Dynamics Dashboard")
st.markdown("Monitor and visualize trends in job postings, roles, and salaries over time.")

# Sidebar Filters
st.sidebar.header("Filter Options")

selected_category = st.sidebar.multiselect(
    "Select Categories",
    options=data["Category"].unique(),
    default=data["Category"].unique()
)

selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=data["country"].unique(),
    default=data["country"].unique()
)

selected_date_range = st.sidebar.date_input(
    "Select Date Range",
    [data["published_date"].min().date(), data["published_date"].max().date()]
)

# Convert `selected_date_range` to datetime format
start_date = pd.to_datetime(selected_date_range[0])
end_date = pd.to_datetime(selected_date_range[1])

# Filter data based on selections
filtered_data = data[
    (data["Category"].isin(selected_category)) &
    (data["country"].isin(selected_countries)) &
    (data["published_date"] >= start_date) &
    (data["published_date"] <= end_date)
]

# Key Insights Section
st.subheader("Key Insights")
total_jobs = len(filtered_data)
average_salary = filtered_data["average_hourly_rate"].mean() if total_jobs > 0 else 0
top_category = (
    filtered_data["Category"].value_counts().idxmax() if total_jobs > 0 else "N/A"
)

st.metric("Total Job Postings", f"{total_jobs:,}")
st.metric("Average Hourly Rate", f"${average_salary:.2f}")
st.metric("Most Popular Category", top_category)

# Trend Analysis Section
st.subheader("Trends Over Time")

# Convert YearMonth to string for JSON serialization
job_trend = filtered_data.groupby("YearMonth").size().reset_index(name="Job Postings")
job_trend["YearMonth"] = job_trend["YearMonth"].astype(str)

fig_job_trend = px.line(job_trend, x="YearMonth", y="Job Postings", title="Job Posting Trends Over Time")
st.plotly_chart(fig_job_trend)

# Convert YearMonth to string for JSON serialization
salary_trend = filtered_data.groupby("YearMonth")["average_hourly_rate"].mean().reset_index()
salary_trend["YearMonth"] = salary_trend["YearMonth"].astype(str)

fig_salary_trend = px.line(salary_trend, x="YearMonth", y="average_hourly_rate", title="Average Hourly Rate Trends")
st.plotly_chart(fig_salary_trend)

# Convert YearMonth to string for JSON serialization
category_trend = filtered_data.groupby(["YearMonth", "Category"]).size().reset_index(name="Job Postings")
category_trend["YearMonth"] = category_trend["YearMonth"].astype(str)

fig_category_trend = px.line(
    category_trend,
    x="YearMonth",
    y="Job Postings",
    color="Category",
    title="Category Demand Over Time"
)
st.plotly_chart(fig_category_trend)

# Geographic Analysis
st.subheader("Geographic Analysis")
geo_avg_salary = filtered_data.groupby("country")["average_hourly_rate"].mean().reset_index()
fig_geo_salary = px.choropleth(
    geo_avg_salary,
    locations="country",
    locationmode="country names",
    color="average_hourly_rate",
    title="Average Hourly Rate by Country",
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig_geo_salary)

# Demand Shifts in Categories
st.subheader("Demand Shifts in Categories")
category_trend = filtered_data.groupby(["YearMonth", "Category"]).size().reset_index(name="Job Postings")
category_trend["YearMonth"] = category_trend["YearMonth"].astype(str)
fig_category_trend = px.line(
    category_trend,
    x="YearMonth",
    y="Job Postings",
    color="Category",
    title="Category Demand Over Time"
)
st.plotly_chart(fig_category_trend)

# User Interaction: Download Filtered Data
st.subheader("Download Filtered Data")
csv = filtered_data.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_job_data.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("Dashboard powered by Streamlit, Plotly, and Seaborn.")
