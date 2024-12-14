import numpy as np
import pandas as pd
import streamlit as st

number_of_location = 11
area_names = [f'Area {i+1}' for i in range(number_of_location)]


def color_shifts(val):
    """
    Color coding for shifts with darker, more distinct colors
    - Dark grey for no shift
    - Dark green for night shift
    - Dark blue for morning shift
    - Dark orange for evening shift
    """
    if val == '':
        return 'background-color: #808080; color: white'  # Dark Grey
    else:
        return 'background-color: #006400; color: white'  # Dark Green

def get_shift_name(shift_num):
    """Convert shift number to shift name"""
    shift_details = {
        1: "Night Shift",
        2: "Morning Shift", 
        3: "Evening Shift"
    }
    return shift_details.get(shift_num, "Unknown Shift")

def get_day_name(day_num):
    """Convert day number to day name"""
    days_of_week = {
        1: "Monday", 
        2: "Tuesday", 
        3: "Wednesday", 
        4: "Thursday", 
        5: "Friday", 
        6: "Saturday", 
        7: "Sunday"
    }
    return days_of_week.get(day_num, "Unknown Day")

if 'big_table' not in st.session_state:
    st.title("Go home first to load data")
else:
    df = st.session_state.big_table

    st.title("Police Schedule Lookup")

    badge_number = st.number_input("Badge number", min_value=10000, max_value=100000, step=1, value=24960)

    officer_schedule = df[df['badge'] == badge_number]

    if not officer_schedule.empty:
        # Define shift details
        shift_details = {
            1: "Night Shift",
            2: "Morning Shift", 
            3: "Evening Shift"
        }

        # Define days of week
        days_of_week = {
            1: "Monday", 
            2: "Tuesday", 
            3: "Wednesday", 
            4: "Thursday", 
            5: "Friday", 
            6: "Saturday", 
            7: "Sunday"
        }

        # Create schedule DataFrame
        schedule_data = []
        for shift_num, shift_name in shift_details.items():
            shift_row = {"Shift": shift_name}
            
            for day_num, day_name in days_of_week.items():
                # Check if this shift exists for this day
                day_shift = officer_schedule[
                    (officer_schedule['day'] == day_num) & 
                    (officer_schedule['shift'] == shift_num)
                ]
                
                # If shift exists, mark with shift name, otherwise mark as empty
                if day_shift.empty:
                    shift_row[day_name] = ""
                else:    
                    shift_row[day_name] = area_names[int(day_shift["area"].iloc[0])-1]
            
            schedule_data.append(shift_row)

        # Create DataFrame
        schedule_df = pd.DataFrame(schedule_data)
        
        # Set index to Shift column
        schedule_df.set_index('Shift', inplace=True)

        # Display title
        st.subheader(f"Schedule for Badge Number {badge_number}")
        
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
        
        # Display styled schedule table
        st.dataframe(
            schedule_df.style.applymap(color_shifts),
            use_container_width=True
        )

        # Create detailed shift information table
        st.subheader("Shift Details")
        
        # Prepare shift details data
        shift_details_data = []
        for _, row in officer_schedule.iterrows():
            shift_detail = {
                "Day": get_day_name(row['day']),
                "Shift": get_shift_name(row['shift']),
                "Time to Travel": row.get('time_to_travel', 'N/A'),
                "Distance": row.get('distance', 'N/A'),
                "Area": area_names[row.get('area', 'N/A')-1]
            }
            shift_details_data.append(shift_detail)
        
        # Create DataFrame for shift details
        shift_details_df = pd.DataFrame(shift_details_data)
        
        # Display shift details table
        st.dataframe(shift_details_df, use_container_width=True)

    else:
        st.subheader("This badge number doesn't exist...")