import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import datetime
from tqdm import tqdm
import os
from utils import lat_min, lat_max, lon_min, lon_max, mapLosAngeles, lat, lon, num_lat_intervals, num_lon_intervals

# Check if the processed dataset already exists
output_file_path = '../../datasets/processed_crime_data_FULL.csv'
if os.path.exists(output_file_path):
    print(f"Processed dataset already exists at {output_file_path}. Skipping processing.")
else:
# Load the dataset
    # Make sure to change the path to the correct one for your file
    print("Loading dataset...")
    file_path = '../../datasets/crime_data.csv'
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")

    # 1. Dataset Preprocessing
    # ---------------------------------------------------------
    print("Starting dataset preprocessing...")
    # Remove unnecessary columns for our problem
    columns_to_drop = ['DR_NO', 'Date Rptd', 'Rpt Dist No', 'Part 1-2', 'Crm Cd', 'Crm Cd 1', 'Crm Cd 2', 'Crm Cd 3',
                       'Crm Cd 4',
                       'Premis Cd', 'Weapon Used Cd', 'Weapon Desc', 'Status', 'Status Desc', 'Cross Street',
                       'LOCATION', 'Mocodes']
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    print("Unnecessary columns removed.")

    # Rename columns for consistency
    column_rename_map = {
        'DATE OCC': 'date_occ',
        'TIME OCC': 'time_occ',
        'AREA': 'area',
        'AREA NAME': 'area_name',
        'Vict Age': 'vict_age',
        'Vict Sex': 'vict_sex',
        'Vict Descent': 'vict_descent',
        'Premis Desc': 'premis_desc',
        'LAT': 'lat',
        'LON': 'lon'
    }
    df.rename(columns=column_rename_map, inplace=True)
    print("Columns renamed for consistency.")

    # Convert 'date_occ' column to datetime and create temporal columns, then drop 'date_occ'
    print("Converting 'date_occ' column to datetime format and creating temporal columns...")
    df['date_occ'] = pd.to_datetime(df['date_occ'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df.dropna(subset=['date_occ', 'lat', 'lon'], inplace=True)  # Drop rows with essential missing values

    # Filter rows based on latitude and longitude range
    df = df[(df['lat'] >= lat_min) & (df['lat'] <= lat_max) & (df['lon'] >= lon_min) & (df['lon'] <= lon_max)]
    print("Filtered rows based on latitude and longitude range.")

    # Create temporal columns and remove 'date_occ'
    df['hour'] = df['time_occ'].apply(lambda x: int(str(x).zfill(4)[:2]))  # Extract hour from time_occ
    df['day_of_week'] = df['date_occ'].dt.day_name()  # Day of the week (e.g., Monday)
    df['day'] = df['date_occ'].dt.day  # Day of the month
    df['month'] = df['date_occ'].dt.month  # Month of the year
    df['year'] = df['date_occ'].dt.year  # Year
    print("New temporal columns created.")
    df.drop(columns=['date_occ', 'time_occ'], inplace=True)
    print("'date_occ' and 'time_occ' columns dropped.")

    # Add a column to indicate crime occurrence (value 1 for crimes)
    df['crime_occurrence'] = 1
    print("'crime_occurrence' column added.")

    # Spatial Feature Engineering
    # ---------------------------------------------------------
    
    print("Processing spatial features...")
    lat_array = np.array(lat)
    lon_array = np.array(lon)
    df['fix_lat'] = df['lat'].apply(lambda x: lat_array[np.argmin(np.abs(lat_array - x))])
    df['fix_lon'] = df['lon'].apply(lambda x: lon_array[np.argmin(np.abs(lon_array - x))])
    print("Spatial features processed into discrete ranges: 'fix_lat' and 'fix_lon'.")
    df.drop(columns=['lat', 'lon'], inplace=True)
    print("Dropped original latitude and longitude columns.")

    # 2. Generation of Negative Data with Balanced Smoothing (Optimized Version)
    # ---------------------------------------------------------
    

     # Reorder columns
    df = df[['hour', 'day', 'month', 'year', 'day_of_week', 'fix_lat', 'fix_lon', 'crime_occurrence']]
    print("Columns reordered.")

    print("Generating balanced negative data (optimized version)...")
    negative_data = []

    # Iterate through each unique combination of area, day, month, year, and hour
    grouped = df.groupby(['hour', 'day', 'month', 'year'])
    all_lat_lon_pairs = set((lat_i, lon_i) for lat_i in lat for lon_i in lon)
    for (hour, day, month, year), group in tqdm(grouped, desc="Generating negative data"):
        
        existing_lat_lon_pairs = set(zip(group['fix_lat'], group['fix_lon']))
        possible_lat_lon_pairs = all_lat_lon_pairs - existing_lat_lon_pairs

        for fix_lat, fix_lon in possible_lat_lon_pairs:
            lat_index = np.argmin(np.abs(lat_array - fix_lat))
            lon_index = np.argmin(np.abs(lon_array - fix_lon))
            
            
            if mapLosAngeles[lat_index, lon_index] == 1:
                # Check if the location is valid in mapLosAngeles
                negative_data.append({
                    'hour': hour,
                    'day': day,
                    'month': month,
                    'year': year,
                    'day_of_week': group['day_of_week'].iloc[0],
                    'fix_lat': fix_lat,
                    'fix_lon': fix_lon,
                    'crime_occurrence': 0  # Indicate no crime occurrence
                })

    print("Negative examples generated (optimized version).")

    # Verifica la generazione dei dati negativi
    print(f"Number of negative examples generated: {len(negative_data)}")

    # Verifica la concatenazione dei dati
    print(f"Original dataset size: {df.shape[0]}")
    negative_df = pd.DataFrame(negative_data)
    df = pd.concat([df, negative_df], ignore_index=True)
    print(f"Dataset size after concatenation: {df.shape[0]}")

    # Verifica la distribuzione delle classi
    print(df['crime_occurrence'].value_counts())


    # 3. Keep Only Required Columns
    # ---------------------------------------------------------
    print("Filtering final columns...")
    columns_to_keep = ['hour', 'day', 'month', 'year', 'day_of_week', 'fix_lat', 'fix_lon', 'crime_occurrence']
    df = df[columns_to_keep]
    print("Final columns selected.")

    # Save the processed dataset
    df.to_csv(output_file_path, index=False)
    print(f"Processed dataset saved to {output_file_path}.")



df = pd.read_csv(output_file_path)
print("Dataset processed_crime_data_with_negative_optimized loaded successfully.")





from sklearn.preprocessing import StandardScaler

# 5. Encoding delle Variabili Categoriali
# ---------------------------------------------------------
print("Encoding categorical variables...")
category_columns = ['day_of_week']
for column in category_columns:
    df[column] = df[column].astype('category').cat.codes
print("Categorical variables encoded.")

# 6. Standardizzazione delle Variabili Numeriche
# ---------------------------------------------------------
print("Standardizing numerical features...")
scaler = StandardScaler()

# Seleziona le colonne numeriche per la standardizzazione
numerical_columns = ['fix_lat', 'fix_lon']

# Applica lo scaler e aggiorna il dataframe
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
print("Numerical features standardized.")

print("Splitting dataset into training, validation, and test sets based on temporal criteria...")


# Create the training set (data before 2023)
training_set = df[df['year'] < 2024]

# Create the test set (data of 2023)
test_set = df[(df['year'] == 2024) & (df['month'] <= 6)]

# Create the validation set (data of 2024)
validation_set = df[(df['year'] == 2024) & (df['month'] > 6)]

print(f"Training set: {training_set['year'].min()} - {training_set['year'].max()} ({len(training_set)} records)")
print(f"Test set: {test_set['year'].min()} - {test_set['year'].max()} ({len(test_set)} records)")
print(f"Validation set: {validation_set['year'].min()} - {validation_set['year'].max()} ({len(validation_set)} records)")

# Salva i dataset suddivisi
training_set.to_csv('training_set_full.csv', index=False)
validation_set.to_csv('validation_set_full.csv', index=False)
test_set.to_csv('test_set_full.csv', index=False)

print("Datasets saved: training_set_full.csv, validation_set_full.csv, test_set_full.csv")
