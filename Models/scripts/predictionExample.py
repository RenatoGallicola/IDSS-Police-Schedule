import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
from utils import mapLosAngeles, lat, lon, lat_scaled, lon_scaled, lon_min, lon_max, lat_min, lat_max, class_weights, \
    add_cyclic_features, columns_periods


# Define input values

# Function to generate dataset
def generate_records(hour, minute, day, month, year):
    records = []
    day_of_week = datetime(year, month, day, hour, minute).strftime('%A')

    for longitude in lon_scaled:
        for latitude  in lat_scaled:
            record = {
                'hour': hour,
                'day': day,
                'month': month,
                'year': year,
                'day_of_week': day_of_week,
                'fix_lat': latitude,
                'fix_lon': longitude,
                # 'minute': minute,
                'crime_occurrence': 1  # Assuming all are positive cases initially
            }
            records.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(records)
    return df


# Example usage
hour = 21
minute = 0
day = 2
month = 9
year = 2024
dataset = generate_records(hour, minute, day, month, year)

# Encode and standardize features to match the model's training process
encoder = LabelEncoder()
dataset['day_of_week'] = encoder.fit_transform(dataset['day_of_week'])

scaler = StandardScaler()
# numerical_columns = ['hour', 'minute', 'day', 'month', 'year', 'fix_lat', 'fix_lon', 'day_of_week']
# dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

numerical_columns = ['year', 'fix_lat', 'fix_lon']
dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

columns_periods = {
    'hour': 24,  # 24 ore in un giorno
    'day': 31,  # 31 giorni in un mese (massimo possibile)
    'month': 12,  # 12 mesi in un anno
    'day_of_week': 7  # 7 giorni in una settimana
}

dataset = add_cyclic_features(dataset, columns_periods)

# Ensure all data is of numeric type
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset.fillna(0, inplace=True)
dataset = dataset.astype(float)
# Identifica le colonne temporali e spaziali
# Separa le variabili temporali
spatial_columns = ['fix_lat', 'fix_lon']
temporal_columns = ['hour', 'day', 'month', 'year', 'day_of_week',
                    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                    'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
# Input temporale (14 feature)
X_temporal = dataset[temporal_columns].values.reshape((len(dataset), 1, len(temporal_columns)))

# Input spaziale (2 feature)
X_spatial = dataset[spatial_columns].values.reshape((len(dataset), 1, len(spatial_columns)))

print("Shape of X_temporal:", X_temporal.shape)  # (961, 1, 14)
print("Shape of X_spatial:", X_spatial.shape)    # (961, 1, 2)


# Specifica la funzione personalizzata per il caricamento
model = load_model('../crime_prediction_lstm_model_Divided_30_EPOCH.h5')

model.summary()


# Esegui le predizioni
predictions = model.predict([X_temporal, X_spatial])


# Initialize prediction matrix and fill it with predictions
pred_matrix = predictions.reshape((len(lon_scaled), len(lat_scaled)))

# Set values to 0 where mapLosAngeles has 0
pred_matrix[mapLosAngeles == 0] = 0

# Save the predicted percentages matrix with lat and lon as rows and columns
predicted_matrix_with_lat_lon = pd.DataFrame(pred_matrix, index=lon_scaled, columns=lat_scaled)
predicted_matrix_with_lat_lon.to_csv('predicted_crime_percentages_matrix.csv')

# Show the predicted percentages matrix
print(pred_matrix)

lat_values = predicted_matrix_with_lat_lon.index.values
lon_values = predicted_matrix_with_lat_lon.columns.values

# Plot the predicted percentages matrix using the saved matrix
plt.figure(figsize=(12, 12))
lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
c = plt.pcolormesh(lon_grid, lat_grid, predicted_matrix_with_lat_lon.values, shading='auto', cmap='viridis')
plt.colorbar(c, label='Predicted Crime Occurrence Percentage')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Predicted Crime Occurrence Percentages for {hour} : {day}/{month}/{year}')
plt.show()

# Saving dataset to CSV for further use
dataset.to_csv('crime_dataset.csv', index=False)

# Load validation dataset
validation_set_path = '../../datasets/validation_set_full.csv'
validation_df = pd.read_csv(validation_set_path)

# Filter validation set for the specific date
validation_filtered = validation_df[(validation_df['hour'] == hour) &
                                    (validation_df['day'] == day) &
                                    (validation_df['month'] == month) &
                                    (validation_df['year'] == year)]

# Drop 'date' column if it exists
if 'date' in validation_filtered.columns:
    validation_filtered = validation_filtered.drop(['date'], axis=1)

# Print the filtered validation matrix
print(f'Validation Matrix for date {day}/{month}/{year}:')

print(validation_filtered)

# Initialize validation matrix based on actual crime occurrences using the same mapping as prediction matrix
validation_matrix = np.zeros((len(lon), len(lat)))
for index, row in validation_filtered.iterrows():
    closest_lat_idx = np.argmin(np.abs(lat_scaled - row['fix_lat']))
    closest_lon_idx = np.argmin(np.abs(lon_scaled - row['fix_lon']))
    if row['crime_occurrence'] > 0 and mapLosAngeles[closest_lon_idx, closest_lat_idx] == 1:  # Ensure crime_occurrence is positive and mapLosAngeles has value 1
        validation_matrix[closest_lon_idx, closest_lat_idx] = row['crime_occurrence']

# Save the validation matrix
validation_matrix_with_lat_lon = pd.DataFrame(validation_matrix, index=lon, columns=lat)

# Plot the map of Los Angeles with the validation matrix overlay
plt.figure(figsize=(12, 12))
lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)

# Plot the base map of Los Angeles
plt.pcolormesh(lon_grid, lat_grid, mapLosAngeles, shading='auto', cmap='gray')

# Overlay the validation matrix with yellow color
plt.pcolormesh(lon_grid, lat_grid, validation_matrix_with_lat_lon.values, shading='auto', cmap='YlOrBr', alpha=0.6)

plt.colorbar(label='Actual Crime Occurrence Percentage (Validation)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Los Angeles Map with Validation Matrix Overlay for {hour} : {day}/{month}/{year}')
plt.show()

#
# print("Latitudine scalata (lat_scaled):", lat_scaled[:5], "...", lat_scaled[-5:])
# print("Longitudine scalata (lon_scaled):", lon_scaled[:5], "...", lon_scaled[-5:])
# print("Matrice della mappa (mapLosAngeles):", mapLosAngeles.shape)
