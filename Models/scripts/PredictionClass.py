import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import mapLosAngeles, lat, lon, lat_scaled, lon_scaled, lon_min, lon_max, lat_min, lat_max, class_weights, \
    add_cyclic_features, columns_periods, spatial_columns, temporal_columns


class PredictionClass:

    def __init__(self):
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = load_model('../crime_prediction_lstm_model_Divided_30_EPOCH.h5')


    

    def generate_records(self, hour, day, month, year):
        records = []
        day_of_week = datetime(year, month, day, hour).strftime('%A')

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
                    'crime_occurrence': 1  # Assuming all are positive cases initially
                }
                records.append(record)

        # Convert to DataFrame
        df = pd.DataFrame(records)
        return df

    def normalize_records(self, df):
        df['day_of_week'] = self.encoder.fit_transform(df['day_of_week'])

        numerical_columns = ['year', 'fix_lat', 'fix_lon']
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])

        df = add_cyclic_features(df, columns_periods)

                # Ensure all data is of numeric type
        df = df.apply(pd.to_numeric, errors='coerce')
        df.fillna(0, inplace=True)
        df = df.astype(float)

        return df
    
    def splitting_data(self, df):
        # Input temporale (14 feature)
        X_temporal = df[temporal_columns].values.reshape((len(df), 1, len(temporal_columns)))

        # Input spaziale (2 feature)
        X_spatial = df[spatial_columns].values.reshape((len(df), 1, len(spatial_columns)))

        return X_temporal, X_spatial


    def predict(self,hour, day, month, year):
        df = self.generate_records(hour, day, month, year)
        df = self.normalize_records(df)
        X_temporal, X_spatial = self.splitting_data(df)
        # Esegui le predizioni
        predictions = self.model.predict([X_temporal, X_spatial])


        # Initialize prediction matrix and fill it with predictions
        pred_matrix = predictions.reshape((len(lon_scaled), len(lat_scaled)))

        # Set values to 0 where mapLosAngeles has 0
        pred_matrix[mapLosAngeles == 0] = 0
        
        return pd.DataFrame(pred_matrix, index=lon_scaled, columns=lat_scaled)


    def plot_heatmap(self, df):
        lat_values = df.index.values
        lon_values = df.columns.values

        # Plot the predicted percentages matrix using the saved matrix
        plt.figure(figsize=(12, 12))
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
        c = plt.pcolormesh(lon_grid, lat_grid, df.values, shading='auto', cmap='viridis')
        plt.colorbar(c, label='Predicted Crime Occurrence Percentage')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Predicted Crime Occurrence Percentages')
        plt.show()

        
