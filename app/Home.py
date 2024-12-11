from datetime import datetime, timedelta

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


###########################################################
# Mock function to load LSTM model
@st.cache_resource
def load_lstm_model():
    # Replace with actual model loading code
    return "LSTM Model"

# Mock function to make predictions
def make_predictions(model, num_officers, week_number, number_of_area):
    # Replace with actual prediction code
    return np.random.randint(1, num_officers, size=(7, 24, number_of_area))  # Mock data for 7 days, 24 hours, 5 areas

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
predictions = make_predictions(model, num_officers, week_number, 10) # <---- 10 is number of location
###########################################################


# Main page
st.title("Police Dispatch IDSS")

### WARNING this parameters may be changed ??!!
### BACKEND CALL ??
number_of_location = 10
area_names = [f'Area {i+1}' for i in range(number_of_location)]
###

day_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

#####################################  MAP  #####################################

st.subheader("Resource Allocation Map")
day_of_week = st.selectbox("Select day of the week", list(day_mapping.keys()))

time_slot = st.radio(
    "Select time slot",
    options=["0-8", "8-16", "16-24", "Specific"]
)

#### THIS WILL DISEPEAR ####
def load_allocations_data():
    st.session_state.allocations = allocate_resources(predictions)
if 'allocations' not in st.session_state:
    load_allocations_data()
if st.button("Load allocation (button will be delete)"):
    load_allocations_data()
############################


### BACKEND CALL 
### Load the data of each location
def load_location_data():
    st.session_state.location_data = pd.DataFrame({
        'lat': [34.0522 + np.random.uniform(-0.15, 0.15) for _ in range(number_of_location)], 
        'lon': [-118.2437 + np.random.uniform(-0.15, 0.15) for _ in range(number_of_location)], 
    })

### Initialization if the data the first time we load the website
if 'location_data' not in st.session_state:
    load_location_data()

### Button for load the data (can be deleted ???!!!)
if st.button("Load location (button will be delete)"):
    load_location_data()

### BACKEND CALL
### So this function must changes  to make this call
### based on the day and the hour, return a list:
## size: number_of_location
## for each index the number of officiers at this location
def get_officiers_data(day_of_week, hour_of_day):
    allocations = st.session_state.allocations
    return [allocations[day_mapping[day_of_week]][hour_of_day][i] for i in range(number_of_location)]

### Same as last function, but with a range
def get_officiers_data_bigger(day_of_week, start_hour, end_hour):
    res = [0] * number_of_location
    for hour in range(start_hour, end_hour):
        aux = get_officiers_data(day_of_week, hour)
        for i in range(number_of_location) :
            res[i] += aux[i]
    return res

### if the Specific is selected, spawn the slider and call the good function
if time_slot == "Specific":
    hour_of_day = st.slider("Select hour of the day", 0, 23, 12)
    st.session_state.officier_data = get_officiers_data(day_of_week, hour_of_day)
### otherwise, call the other function with the range given
else:
    start_hour, end_hour = map(int, time_slot.split('-'))
    st.session_state.officier_data = get_officiers_data_bigger(day_of_week, start_hour, end_hour)


### Set of the MAP
if st.session_state.location_data is not None and st.session_state.officier_data is not None:

    # init
    location_data = st.session_state.location_data
    officier_data = st.session_state.officier_data
    m = folium.Map(location=[34.0522, -118.2437], zoom_start=10)
    max_officers = max(officier_data)

    # foreach location
    for i in range(number_of_location):

        # init data of location
        lat = location_data['lat'][i]
        lon = location_data['lon'][i]
        off = officier_data[i]

        circle_size = 5 + (off / max_officers) * 25


        # draw the circle
        folium.CircleMarker(
            location=[lat, lon],
            radius=circle_size,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.3
        ).add_to(m)

        # print the texte
        folium.map.Marker(
            [lat, lon],
            tooltip=folium.Tooltip(f"{off} officers are requiered at {area_names[i]}", sticky=True),
            icon=folium.DivIcon(
                html=f"""
                <div style="transform: translate(-50%,-50%); font-size: 2em; font-weight: bold; color: black; text-align: center;">
                    {off}
                </div>
                """
            )
        ).add_to(m)

    st_folium(m, width=700, height=500)
else:
    # This part is only for avoid strange resizing behaviour
    m = folium.Map(location=[34.0522, -118.2437], zoom_start=10)
    st_folium(m, width=700, height=500)

#####################################  TABLE  #####################################

st.subheader("Resource Allocation Table")
selected_day = st.selectbox("Select day for table view", list(day_mapping.keys()))

def generate_table_data(day_of_week):
    table_data = []
    for hour in range(24):
        hourly_data = get_officiers_data(day_of_week, hour)
        table_data.append(hourly_data)
    return pd.DataFrame(table_data, columns=area_names)

# Create DataFrame with hours as rows and areas as columns
table_data = generate_table_data(selected_day)
table_data.index = [f"{i}:00" for i in range(24)]
st.table(table_data)