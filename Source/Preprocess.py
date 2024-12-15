import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load Data
dtype_useful = {
    'DATE OCC': 'string', # Date occurred : Useful for analysis
    'TIME OCC': 'Int64', # Time occurred : Useful for analysis
    'AREA': 'Int64', # Area ID : Useful for analysis
    'AREA NAME': 'category', # Area Name, not useful for analysis but we will keep it for now
    'Part 1-2': 'Int64', # Part 1 or Part 2 crime : Maybe useful for analysis
    'Crm Cd': 'Int64', # Crime Code : Useful for analysis
    'Crm Cd Desc': 'category', # Crime Description : Not useful for analysis but we will keep it for now
    'Vict Age': 'float64', # Victim Age : Useful for analysis (demographics)
    'Vict Sex': 'category', # Victim Gender : Useful for analysis (demographics)
    'Vict Descent': 'category', # Victim Descent (ethic / racial background) : Useful for analysis (demographics)
    'Premis Cd': 'Int64', # Premise Code : Useful for analysis
    'Premis Desc': 'category', # Premise Description : Not useful for analysis but we will keep it for now
    'LAT': 'float64', # Latitude : Useful for analysis
    'LON': 'float64' # Longitude : Useful for analysis
}

df = pd.read_csv('la_crime_data.csv', dtype=dtype_useful, parse_dates=['DATE OCC'])

# 2. Handle Missing Data
df.fillna({
    'Vict Age': df['Vict Age'].median(), 
    'TIME OCC': df['TIME OCC'].mode()[0],
    'AREA': df['AREA'].mode()[0],
    'Part 1-2': df['Part 1-2'].mode()[0],
    'Crm Cd': df['Crm Cd'].mode()[0],
    'Premis Cd': df['Premis Cd'].mode()[0]
}, inplace=True)

# Convert integer columns back to integers after filling missing values
int_columns = ['TIME OCC', 'AREA', 'Part 1-2', 'Crm Cd', 'Premis Cd']
df[int_columns] = df[int_columns].astype('int64')

# 3. Feature Engineering 
# From TIME OCC (HHMM), extract hour and minute.
df['Hour'] = df['TIME OCC'] // 100
df['Minute'] = df['TIME OCC'] % 100

# Extract day of the week from DATE OCC
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
df['Day of Week'] = df['DATE OCC'].dt.day_name()

# 5. Group Areas into Larger Regions
area_mapping = {
    1: 'Central',
    2: 'Central',
    3: 'Southwest',
    4: 'Southwest',
    5: 'West',
    6: 'West',
    7: 'West',
    8: 'West',
    9: 'Valley',
    10: 'Valley',
    11: 'Valley',
    12: 'South',
    13: 'South',
    14: 'South',
    15: 'North',
    16: 'North',
    17: 'North',
    18: 'East',
    19: 'East',
    20: 'East',
    21: 'East'
}
df['Large Area'] = df['AREA'].map(area_mapping).fillna('Other')

# Remove all the THEFT OF IDENTITY rows from the data
df = df[df['Crm Cd Desc'] != 'THEFT OF IDENTITY']

# Print the count of each Crime Description
crime_desc_counts = df['Crm Cd Desc'].value_counts()
print("Crime Description Counts:")
for desc, count in crime_desc_counts.items():
    print(f"{desc}: {count}")
# Keep only the useful columns
useful_columns = list(dtype_useful.keys()) + ['Hour', 'Minute', 'Day of Week', 'Large Area']
df = df[useful_columns]

# Save the cleaned data
df.to_csv('cleaned_la_crime_data.csv', index=False)