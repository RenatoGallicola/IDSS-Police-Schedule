import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta

# Mock function to load LSTM model
@st.cache_resource
def load_lstm_model():
    # Replace with actual model loading code
    return "LSTM Model"

# Mock function to make predictions
def make_predictions(model, num_officers, week_number):
    # Replace with actual prediction code
    return np.random.randint(1, num_officers, size=(7, 24, 5))  # Mock data for 7 days, 24 hours, 5 areas

# Mock function for resource allocation
def allocate_resources(predictions):
    # Replace with actual allocation code
    return predictions  # Mock allocation

def get_week_dates(week_number, year=2024):
    # Get the first day of the year
    first_day_of_year = datetime(year, 1, 1)
    # Calculate the first day of the given week
    first_day_of_week = first_day_of_year + timedelta(weeks=week_number - 1)
    # Generate all dates for the week
    week_dates = [first_day_of_week + timedelta(days=i) for i in range(7)]
    return week_dates

def generate_hourly_records_for_week(week_number, year=2024):
    week_dates = get_week_dates(week_number, year)
    records = []
    for date in week_dates:
        for hour in range(24):
            record = {
                'hour': hour,
                'minute': 0,
                'day': date.day,
                'month': date.month,
                'year': date.year
            }
            records.append(record)
    return records

# Main page
st.title("Police Dispatch IDSS")

# Input fields
if 'num_officers' not in st.session_state:
    st.session_state.num_officers = 10
if 'week_number' not in st.session_state:
    st.session_state.week_number = 1

num_officers = st.number_input("Number of officers available for the week", min_value=10, step=1, value=st.session_state.num_officers)
week_number = st.number_input("Week number in the year", min_value=1, max_value=52, step=1, value=st.session_state.week_number)

st.session_state.num_officers = num_officers
st.session_state.week_number = week_number

# Load model and make predictions
model = load_lstm_model()
predictions = make_predictions(model, num_officers, week_number)
allocations = allocate_resources(predictions)

day_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

# Display map
st.subheader("Resource Allocation Map")
day_of_week = st.selectbox("Select day of the week", list(day_mapping.keys()))
hour_of_day = st.slider("Select hour of the day", 0, 23, 12)

# Create DataFrame for map
map_data = pd.DataFrame({
    'lat': [34.0522 + np.random.uniform(-0.01, 0.01) for _ in range(5)],  # Los Angeles latitude
    'lon': [-118.2437 + np.random.uniform(-0.01, 0.01) for _ in range(5)],  # Los Angeles longitude
    'officers_needed': [allocations[day_mapping[day_of_week]][hour_of_day][i] for i in range(5)]
})

# Display map using Streamlit's built-in map function
st.map(map_data)

# Display table - TO BE CHANGED WITH THE CORRECT RESULT
st.subheader("Resource Allocation Table")
selected_day = st.selectbox("Select day for table view", list(day_mapping.keys()))

# Mock area names
area_names = ["Area 1", "Area 2", "Area 3", "Area 4", "Area 5"]

# Create DataFrame with hours as rows and areas as columns
table_data = pd.DataFrame(allocations[day_mapping[selected_day]], columns=area_names)
table_data.index = [f"{i}:00" for i in range(24)]
st.table(table_data)