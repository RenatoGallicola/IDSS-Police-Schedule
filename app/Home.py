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

if st.button("Click here to run the model",use_container_width=True):
    if selected_date.weekday() != 0:
        st.error("Please... monday..")
    else:    
        with st.spinner("Model is loading, please wait..."):
            
            tot_shift = 4
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


csv_file_name = "ResourceAllocation/ui_allocation.csv"
try:
    with st.spinner("Data are generating..."):    
        df = df = pd.read_csv(csv_file_name, sep=',', header=0)
        df.columns = ["badge","name","shift","day","month","year","group","lat","lon","area","time_to_travel","distance"]
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
    ###########################################################

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

    st.subheader("Resource Allocation Map")
    day_of_week = st.selectbox("Select day of the week", list(day_mapping.keys()))

    shift = st.selectbox("Select day of the week", list(shift_mapping.keys()))
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

            #circle_size = 5 + (off / max_officers) * 25
            circle_size = 30
            color = colormap(off)

            # draw the circle
            folium.CircleMarker(
                location=[lat, lon],
                radius=circle_size,
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
                    icon_size=(150,36),
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
