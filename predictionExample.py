import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model

# Define input values
lat = [0.0, 1.716715, 3.43343, 5.150145, 6.86686, 8.583575, 10.30029, 12.017005, 13.73372, 15.450435, 17.16715, 18.883865, 20.60058, 22.317295, 24.03401, 25.750725, 27.46744, 29.184155, 30.90087, 32.617585, 34.3343]
lon = [-118.6676, -115.70091, -112.73422, -109.76753, -106.80084, -103.83415, -100.86746, -97.90077, -94.93408, -91.96739, -89.0007, -86.03401, -83.06732, -80.10063, -77.13394, -74.16725, -71.20056, -68.23387, -65.26718, -62.30049, -59.3338, -56.36711, -53.40042, -50.43373, -47.46704, -44.50035, -41.53366, -38.56697, -35.60028, -32.63359, -29.6669, -26.70021, -23.73352, -20.76683, -17.80014, -14.83345, -11.86676, -8.90007, -5.93338, -2.96669, 0.0]

# Function to generate dataset
def generate_records(hour, minute, day, month, year):
    records = []
    dt = datetime(year, month, day, hour, minute)
    day_of_week = dt.strftime('%A')

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
                'crime_occurrence': 1,  # Assuming all are positive cases initially
                'date': dt.strftime('%Y-%m-%d %H:%M:%S')
            }
            records.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(records)
    return df

# Example usage
hour = 14
minute = 0
day = 1
month = 12
year = 2023
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

# Keep 'date' column for LSTM input to ensure 9 features
X = dataset.drop(columns=['crime_occurrence']).values.reshape((len(dataset), 1, len(numerical_columns) + 1))

# Load pre-trained LSTM model
model = load_model('crime_prediction_lstm_model_FULL_not_working.h5')

# Predict crime occurrence percentages
predictions = model.predict(X)

# Reshape predictions into a matrix with latitudes as rows and longitudes as columns
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
