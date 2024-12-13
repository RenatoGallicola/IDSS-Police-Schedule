import numpy as np
import pandas as pd
import streamlit as st

if 'big_table' not in st.session_state:
    st.title("Go home first !!!")
    df = None
else:
    df = st.session_state.big_table

st.title("Police Schedule Lookup")

badge_number = num_officers = st.number_input("Badge number", step=1, value=101)

officer_schedule = df[df['badge'] == badge_number]

