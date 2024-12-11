import pandas as pd
import folium
from folium.plugins import MarkerCluster

# Load the dataset
dataset_path = '../../datasets/crime_data.csv'
try:
    df = pd.read_csv(dataset_path)
except Exception as e:
    print("Failed to load data. Error:", e)
    exit()

# Preprocess data
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Extract necessary data for visualization
unique_areas = df[['area_name', 'lat', 'lon']].drop_duplicates()

# Create a map centered around Los Angeles
crime_map = folium.Map(location=[34.05, -118.25], zoom_start=10)

# Add a marker cluster
marker_cluster = MarkerCluster().add_to(crime_map)

# Add markers for each unique area to the cluster
for _, row in unique_areas.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=row['area_name'],
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(marker_cluster)

# Save the map as an HTML file
map_path = '../../crime_areas_map.html'
crime_map.save(map_path)

print(f'Map saved as {map_path}')
