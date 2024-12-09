import streamlit as st
import pandas as pd

# Add a warning for now that this is a mockup that only have Monday and Tuesday 
st.warning("This is a mockup that only has data for Monday and Tuesday.")

# Mock data for demonstration purposes
data = {
    'badge_number': [101, 102, 103, 104, 105],
    'Monday': [
        ['Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4'],
        ['Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5'],
        ['Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1'],
        ['Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2'],
        ['Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3']
    ],
    'Tuesday': [
        ['Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5'],
        ['Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1'],
        ['Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2'],
        ['Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3'],
        ['Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 1', 'Area 2', 'Area 3', 'Area 4']
    ],
    # Add similar data for other days of the week
}

# Convert the mock data to a DataFrame
schedule_df = pd.DataFrame(data)

# Streamlit app
st.title("Police Schedule Lookup")

# Input for badge number with autocomplete
badge_number = st.selectbox("Enter Badge Number", schedule_df['badge_number'])

# Input for day of the week
day_of_week = st.selectbox("Select Day of the Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Display the schedule for the selected badge number and day
if badge_number and day_of_week:
    schedule = schedule_df[schedule_df['badge_number'] == badge_number][day_of_week].values[0]
    schedule_df = pd.DataFrame(schedule, columns=['Area'], index=[f"{hour}:00" for hour in range(24)])
    st.table(schedule_df)