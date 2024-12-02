import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model

# Define input values
lat = [33.7037, 33.72482, 33.74594, 33.76706, 33.78818, 33.8093, 33.83042, 33.85154,
       33.87266, 33.89378, 33.9149, 33.93602, 33.95714, 33.97826, 33.99938, 34.0205,
       34.04162, 34.06274, 34.08386, 34.10498, 34.1261, 34.14722, 34.16834, 34.18946,
       34.21058, 34.2317, 34.25282, 34.27394, 34.29506, 34.31618, 34.3373]

lon = [-118.6682, -118.65110333, -118.63400667, -118.61691, -118.59981333,
       -118.58271667, -118.56562, -118.54852333, -118.53142667, -118.51433,
       -118.49723333, -118.48013667, -118.46304, -118.44594333, -118.42884667,
       -118.41175, -118.39465333, -118.37755667, -118.36046, -118.34336333,
       -118.32626667, -118.30917, -118.29207333, -118.27497667, -118.25788,
       -118.24078333, -118.22368667, -118.20659, -118.18949333, -118.17239667, -118.1553]
    # Scale latitude and longitude values

scaler = StandardScaler()
lat_scaled = scaler.fit_transform(np.array(lat).reshape(-1, 1)).flatten()
lon_scaled = scaler.fit_transform(np.array(lon).reshape(-1, 1)).flatten()

# Save a version of lat and lon as multidimensional arrays
lat_multidimensional = np.column_stack((lat, lat_scaled))
lon_multidimensional = np.column_stack((lon, lon_scaled))

# Save the multidimensional arrays to CSV files
np.savetxt('lat_multidimensional.csv', lat_multidimensional, delimiter=',', header='lat,lat_scaled', comments='')
np.savetxt('lon_multidimensional.csv', lon_multidimensional, delimiter=',', header='lon,lon_scaled', comments='')


# Function to generate dataset
def generate_records(hour, minute, day, month, year):
    records = []
    day_of_week = datetime(year, month, day, hour, minute).strftime('%A')

    for latitude in lat:
        for longitude in lon:
            record = {
                'hour': hour,
                'day': day,
                'month': month,
                'year': year,
                'day_of_week': day_of_week,
                'fix_lat': latitude,
                'fix_lon': longitude,
                'minute': minute,
                'crime_occurrence': 1  # Assuming all are positive cases initially
            }
            records.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(records)
    return df


# Example usage
hour = 16
minute = 0
day = 25
month = 4
year = 2024
dataset = generate_records(hour, minute, day, month, year)

# Encode and standardize features to match the model's training process
encoder = LabelEncoder()
dataset['day_of_week'] = encoder.fit_transform(dataset['day_of_week'])

scaler = StandardScaler()
numerical_columns = ['hour', 'minute', 'day', 'month', 'year', 'fix_lat', 'fix_lon', 'day_of_week']
dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

# Ensure all data is of numeric type
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset.fillna(0, inplace=True)
dataset = dataset.astype(float)

# Load pre-trained LSTM model
model = load_model('crime_prediction_lstm_model_FULL_1_EPOCH.h5')

# Predict crime occurrence for all records in the dataset
X = dataset.drop(['crime_occurrence'], axis=1).values.reshape((len(dataset), 1, len(numerical_columns)))
predictions = model.predict(X)

# Initialize prediction matrix and fill it with predictions
pred_matrix = predictions.reshape((len(lat), len(lon)))

# Save the predicted percentages matrix with lat and lon as rows and columns
predicted_matrix_with_lat_lon = pd.DataFrame(pred_matrix, index=lat, columns=lon)
predicted_matrix_with_lat_lon.to_csv('predicted_crime_percentages_matrix.csv')

# Show the predicted percentages matrix
print(pred_matrix)

# Plot the predicted percentages matrix using the saved matrix
plt.figure(figsize=(12, 8))
lat_values = predicted_matrix_with_lat_lon.index.values
lon_values = predicted_matrix_with_lat_lon.columns.values
lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
c = plt.pcolormesh(lon_grid, lat_grid, predicted_matrix_with_lat_lon.values, shading='auto', cmap='viridis')
plt.colorbar(c, label='Predicted Crime Occurrence Percentage')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Predicted Crime Occurrence Percentages (from CSV)')
plt.show()

# Saving dataset to CSV for further use
dataset.to_csv('crime_dataset.csv', index=False)

# Mapping lat and lon values to indices for visualization
lat_mapping = {i: lat[i] for i in range(len(lat))}
lon_mapping = {i: lon[i] for i in range(len(lon))}

print("Latitude Mapping:")
print(lat_mapping)
print("Longitude Mapping:")
print(lon_mapping)

# Load validation dataset
validation_set_path = 'validation_set_full.csv'
validation_df = pd.read_csv(validation_set_path)

# Filter validation set for the specific date
validation_filtered = validation_df[(validation_df['hour'] == hour) &
                                    (validation_df['day'] == day) &
                                    (validation_df['month'] == month) &
                                    (validation_df['year'] == year) ]

# Drop 'date' column if it exists
if 'date' in validation_filtered.columns:
    validation_filtered = validation_filtered.drop(['date'], axis=1)

# Print the filtered validation matrix
print("Validation Matrix for date 11, 17, 6, 2023, day_of_week 2:")
print(validation_filtered)

# Initialize validation matrix based on actual crime occurrences using the same mapping as prediction matrix
validation_matrix = np.zeros((len(lat_scaled), len(lon_scaled)))
for index, row in validation_filtered.iterrows():
    closest_lat_idx = np.argmin(np.abs(lat_scaled - row['fix_lat']))
    closest_lon_idx = np.argmin(np.abs(lon_scaled - row['fix_lon']))
    if row['crime_occurrence'] > 0:  # Ensure crime_occurrence is positive
        validation_matrix[closest_lat_idx, closest_lon_idx] = row['crime_occurrence']

# Save the validation matrix
validation_matrix_with_lat_lon = pd.DataFrame(validation_matrix, index=lat_scaled, columns=lon_scaled)
validation_matrix_with_lat_lon.to_csv('actual_validation_crime_percentages_matrix.csv')

# Show the validation matrix
print(validation_matrix)

# Plot the validation percentages matrix
plt.figure(figsize=(12, 8))
lat_values = validation_matrix_with_lat_lon.index.values
lon_values = validation_matrix_with_lat_lon.columns.values
lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
c = plt.pcolormesh(lon_grid, lat_grid, validation_matrix_with_lat_lon.values, shading='auto', cmap='viridis')
plt.colorbar(c, label='Actual Crime Occurrence Percentage (Validation)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Actual Crime Occurrence Percentages (Validation)')
plt.show()
