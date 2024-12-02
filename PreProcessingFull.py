import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import datetime
from tqdm import tqdm
import os

# Check if the processed dataset already exists
output_file_path = 'processed_crime_data_FULL.csv'
if os.path.exists(output_file_path):
    print(f"Processed dataset already exists at {output_file_path}. Skipping processing.")
else:
# Load the dataset
    # Make sure to change the path to the correct one for your file
    print("Loading dataset...")
    file_path = 'crime_data.csv'
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

    # Create temporal columns and remove 'date_occ'
    df['hour'] = df['time_occ'].apply(lambda x: int(str(x).zfill(4)[:2]))  # Extract hour from time_occ
    df['minute'] = df['time_occ'].apply(lambda x: int(str(x).zfill(4)[2:]))  # Extract minute from time_occ
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

    # Define the number of intervals for lat and lon
    num_lat_intervals = 20  # You can adjust this value based on the distribution
    num_lon_intervals = 40  # You can adjust this value based on the distribution

    # Create latitude and longitude ranges
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()

    # Create bins for latitude and longitude
    lat_bins = np.linspace(lat_min, lat_max, num_lat_intervals + 1)
    lon_bins = np.linspace(lon_min, lon_max, num_lon_intervals + 1)

    print(lat_bins);
    print(lon_bins);

    # Digitize latitude and longitude to create discrete intervals
    df['fix_lat'] = np.digitize(df['lat'], bins=lat_bins) - 1
    df['fix_lon'] = np.digitize(df['lon'], bins=lon_bins) - 1

    print("Spatial features processed into discrete ranges: 'fix_lat' and 'fix_lon'.")

    # Drop the original latitude and longitude columns
    df.drop(columns=['lat', 'lon'], inplace=True)
    print("Dropped original latitude and longitude columns.")

    # 2. Generation of Negative Data with Balanced Smoothing (Optimized Version)
    # ---------------------------------------------------------
    print("Generating balanced negative data (optimized version)...")
    negative_data = []

    # Iterate through each unique combination of area, day, month, year, and hour
    grouped = df.groupby(['hour', 'day', 'month', 'year' ])
    for (hour, day, month, year), group in tqdm(grouped, desc="Generating negative data"):
        num_crimes = len(group)

        # Generate negative cases for all lat/lon pairs not covered by crimes
        all_possible_lat = np.arange(num_lat_intervals)
        all_possible_lon = np.arange(num_lon_intervals)

        # Remove existing lat/lon pairs to avoid exact overlap
        existing_lat_lon_pairs = set(zip(group['fix_lat'], group['fix_lon']))
        possible_lat_lon_pairs = np.array([(lat, lon) for lat in all_possible_lat for lon in all_possible_lon
                                           if (lat, lon) not in existing_lat_lon_pairs])

        # Use all available lat/lon pairs for negative examples
        selected_pairs = possible_lat_lon_pairs

        negative_data.extend([
            {
                'hour': hour,
                'minute': np.random.choice(range(60)),
                'day_of_week': group['day_of_week'].iloc[0],
                'day': day,
                'month': month,
                'year': year,
                'area_name': 'null',
                'fix_lat': fix_lat,
                'fix_lon': fix_lon,
                'crime_occurrence': 0  # Indicate no crime occurrence
            }
            for fix_lat, fix_lon in selected_pairs
        ])

    print("Negative examples generated (optimized version).")

    # Convert negative examples to DataFrame and concatenate
    negative_df = pd.DataFrame(negative_data)
    df = pd.concat([df, negative_df], ignore_index=True)
    print("Negative examples added to the dataset.")
    print(f"Dataset size after concatenation: {df.shape[0]}")


    # 3. Keep Only Required Columns
    # ---------------------------------------------------------
    print("Filtering final columns...")
    columns_to_keep = ['hour', 'day', 'month', 'year', 'day_of_week', 'fix_lat', 'fix_lon','minute',
                       'crime_occurrence']
    df = df[columns_to_keep]
    print("Final columns selected.")

    # Save the processed dataset
    df.to_csv(output_file_path, index=False)
    print(f"Processed dataset saved to {output_file_path}.")

    # 4. Visualize Class Distribution
    # ---------------------------------------------------------
    print("Visualizing class distribution...")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='crime_occurrence', data=df)
    plt.xlabel('Crime Occurrence (0 = No Crime, 1 = Crime)')
    plt.ylabel('Count')
    plt.title('Class Distribution: Crime Occurrence')
    plt.xticks(ticks=[0, 1], labels=['No Crime', 'Crime'])
    plt.tight_layout()
    plt.show()

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

# Calcola la suddivisione temporale utilizzando mesi anziché anni
df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']].astype(str).agg('-'.join, axis=1),
                            format='%Y-%m-%d-%H-%M')

# Trova la data di inizio e di fine del dataset
min_date = df['date'].min()
max_date = df['date'].max()

# Calcola il numero totale di mesi tra la data di inizio e di fine
total_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)

# Calcola la suddivisione temporale in base al 70-15-15
train_months = int(total_months * 0.7)
validation_months = int(total_months * 0.15)

# Crea i limiti di tempo per ciascun set
train_end_date = min_date + pd.DateOffset(months=train_months)
validation_end_date = train_end_date + pd.DateOffset(months=validation_months)

# Crea il training set (70%)
training_set = df[df['date'] <= train_end_date]

# Crea il validation set (15%)
validation_set = df[(df['date'] > train_end_date) & (df['date'] <= validation_end_date)]

# Crea il test set (15%)
test_set = df[df['date'] > validation_end_date]

# Stampa informazioni sulla suddivisione
print(f"Training set: {training_set['date'].min()} - {training_set['date'].max()} ({len(training_set)} records)")
print(f"Validation set: {validation_set['date'].min()} - {validation_set['date'].max()} ({len(validation_set)} records)")
print(f"Test set: {test_set['date'].min()} - {test_set['date'].max()} ({len(test_set)} records)")

# Salva i dataset suddivisi
training_set.to_csv('training_set_full.csv', index=False)
validation_set.to_csv('validation_set_full.csv', index=False)
test_set.to_csv('test_set_full.csv', index=False)

print("Datasets saved: training_set_full.csv, validation_set_full.csv, test_set_full.csv")
