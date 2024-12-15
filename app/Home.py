import os
import sys
from datetime import datetime, timedelta

import branca.colormap as cm
import folium
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from streamlit_folium import st_folium
import time

module_dir = os.path.dirname(__file__)
module_path = os.path.join(module_dir, '../ResourceAllocation')
sys.path.append(module_path)

from ui_allocation import UIAllocation

module_dir = os.path.dirname(__file__)
module_path = os.path.join(module_dir, '../Models/scripts')
sys.path.append(module_path)

# Import the UIAllocation class

from utils import get_area_center
import threading

###########################################################

def get_next_monday(date):
    while date.weekday() != 0:  # 0 correspond à lundi
        date += timedelta(days=1)
    return date

# Main page
st.title("Police Dispatch IDSS")

num_officers = st.number_input("Number of officers available for the week", min_value=10, step=1, value=1000)

if 'num_officers' not in st.session_state:
    st.session_state.num_officers = None

st.session_state.num_officers = num_officers

csv_file_name = "ResourceAllocation/ui_allocation.csv"

today = datetime.today()
default_date = get_next_monday(today)
selected_date = st.date_input(
    "Choose a date (Only monday are valid):",
    value=default_date
)

if selected_date.weekday() != 0:
        st.error("You need to choose a Monday")
else:
    day = selected_date.day
    month = selected_date.month
    year = selected_date.year

    st.success(f"You chose : {selected_date.strftime('%A %d %B %Y')}.")

tot_shift = 21

def generate_data():
    with st.spinner("Model is loading, please wait..."):
                
                
                shift = 0
                progress_text = f"Processing shift {shift}/{tot_shift}"
                progress_bar = st.progress(0)

                ui_allocation = UIAllocation(num_policemen=num_officers,day=day,month=month,year=year)
                ui_allocation.week_allocation()

                while ui_allocation.get_schedule_completed() == False:
                    shift = ui_allocation.get_shift()
                    progress_text = f"Processing shift {shift}/{tot_shift}" 
                    perc = shift * 100/tot_shift
                    progress_bar.progress(int(perc), text=progress_text)
                    time.sleep(2)
                
                progress_bar.empty()


if st.button("Click here to run the model",use_container_width=True):
    if selected_date.weekday() != 0:
        st.error("Please... monday..")
    else: 
        if os.path.exists(csv_file_name):
                df = pd.read_csv(csv_file_name, sep=',', header=0)
                df.columns = ["badge","name","shift","day","month","year","group","lat","lon","area","time_to_travel","distance"]
                df = df[(df['day'] == day) & (df['month'] == month) & (df['year'] == year)]
                if not df.empty:
                    expected_rows = num_officers * tot_shift
                    if len(df) == expected_rows:
                        st.success("Data already exists for the selected date and is complete. Skipping generation.")
                    else:
                        df = df[(df['day'] != day) | (df['month'] != month) | (df['year'] != year)]
                        df.to_csv(csv_file_name, index=False)
                        st.warning("Data already exists for another amount of officemen. Re-generation.")
                        generate_data()
                    
                else:
                     generate_data()
        else: 
             generate_data()
            
try:
    with st.spinner("Data are generating..."):    
        for floatVal in ["lat","lon","time_to_travel","distance"]:
            df[floatVal] = df[floatVal].astype(float)
        for intVal in ["badge","shift","day","month","year","area"]:
            df[intVal] = df[intVal].astype(int)

        # shift code from 1 to 21 before
        # shift code from 1 to 3 after, and day from 1 to 7
        df['day'] = (df['shift']-1) // 3 + 1
        df['shift'] = df['shift'] % 3
        df['shift'] = df['shift'].replace(0,3)

        st.session_state.big_table = df
except:
    pass

if 'big_table' not in st.session_state:
    st.subheader("Please load the prediction by clicking on the button")
else:

    day_mapping = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
        "Sunday": 7
    }
    shift_mapping = {
        "0-8": 1,
        "8-16": 2,
        "16-24": 3
    }
    ###########################################################

    number_of_location = 11
    area_names = [f'Area {i+1}' for i in range(number_of_location)]

    @st.cache_resource
    def load_location_data():


        df =  pd.DataFrame({
            'lat': [get_area_center(i)[0] for i in range(1,number_of_location+1)], 
            'lon': [get_area_center(i)[1] for i in range(1,number_of_location+1)], 
        })

        return df

    if 'location_data' not in st.session_state:
        st.session_state.location_data = load_location_data() 
    ###########################################################

    def count_policemen_by_area_day_shift(day, shift):  
        local_df = st.session_state.big_table
        filtered_df = local_df[(local_df['day'] == float(day)) & (local_df['shift'] == float(shift))]
        area_counts = filtered_df['area'].value_counts().reindex(range(1,number_of_location+1), fill_value=0)
        return area_counts.tolist()
    ##########################  MAP  ##########################

    df = st.session_state.big_table

    st.title("Police Schedule Lookup")

    # Define colors for groups
    group_colors = [
        "#FF6347", "#32CD32", "#4682B4", "#FFD700"
    ]

    # Define shift details
    shift_details = {
        1: "Night Shift",
        2: "Morning Shift", 
        3: "Evening Shift"
    }

    # Define days of week
    days_of_week = {
        1: "Mo", 
        2: "Tu", 
        3: "We", 
        4: "Th", 
        5: "Fr", 
        6: "Sa", 
        7: "Su"
    }

    # Create schedule DataFrame
    schedule_data = []
    for shift_num, shift_name in shift_details.items():
        shift_row = {"Shift": shift_name}
        
        for day_num, day_name in days_of_week.items():
            # Check if this shift exists for this day
            day_shift = df[
                (df['day'] == day_num) & 
                (df['shift'] == shift_num)
            ]
            
            # If shift exists, mark with group, otherwise mark as empty
            if day_shift.empty:
                shift_row[day_name] = ""
            else:  
                group_num = day_shift["group"].iloc[0]
                shift_row[day_name] = f"G {group_num + 1}"
                #shift_row[day_name] = day_shift["group"].iloc[0]
        
        schedule_data.append(shift_row)

    # Create DataFrame
    schedule_df = pd.DataFrame(schedule_data)

    # Set index to Shift column
    schedule_df.set_index('Shift', inplace=True)

    # # Display title
    # st.subheader("Weekly Schedule")

    # Custom CSS to make cells larger
    st.markdown("""
    <style>
    .dataframe {
        font-size: 16px;
        text-align: center;
    }
    .dataframe th, .dataframe td {
        min-width: 100px;
        height: 60px;
        vertical-align: middle !important;
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

        # Apply colors to groups
    def color_groups(val):
        if isinstance(val, str) and val.startswith("G "):
            group_num = int(val.split(" ")[1].strip()) - 1
            color = group_colors[group_num]
            return f'background-color: {color}99; color: white;'  # Adding '99' for more opacity
        return ''

    # Display styled schedule table
    st.dataframe(schedule_df.style.map(color_groups), use_container_width=True)

    st.subheader(f"Resource Allocation Map from : {selected_date.strftime('%d %B %Y')}.")
    day_of_week = st.selectbox("Select day of the week", list(day_mapping.keys()))

    shift = st.selectbox("Select shift", list(shift_mapping.keys()))
    st.session_state.officier_data = count_policemen_by_area_day_shift(day_mapping[day_of_week],shift_mapping[shift])

    ### Set of the MAP
    if st.session_state.location_data is not None and st.session_state.officier_data is not None:

        # init
        location_data = st.session_state.location_data
        officier_data = st.session_state.officier_data
        m = folium.Map(location=[34.0522, -118.2437], zoom_start=10)
        max_officers = max(officier_data)
        min_officers = min(officier_data)

        # Create a color map
        colormap = cm.LinearColormap(colors=['green', 'yellow', 'red'], vmin=min_officers, vmax=max_officers)

        # foreach location
        for i in range(number_of_location):

            # init data of location
            lat = location_data['lat'][i]
            lon = location_data['lon'][i]
            off = officier_data[i]

            circle_size = 10 + (off / max_officers) * 25
            color = colormap(off)

            # draw the circle
            folium.CircleMarker(
                location=[lat, lon],
                radius=circle_size,
                tooltip=folium.Tooltip(f"{off} officers are requiered at {area_names[i]}", sticky=True),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)

            # print the text
            folium.map.Marker(
                [lat, lon],
                tooltip=folium.Tooltip(f"{off} officers are requiered at {area_names[i]}", sticky=True),
                icon=folium.DivIcon(
                    icon_size=(70,36),
                    icon_anchor=(0,0),
                    html=f"""
                    <div style="font-size: 2em; font-weight: bold; color: black; text-align: center; transform: translate(-50%,-50%);">
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
