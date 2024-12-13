import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
import tensorflow as tf

mapLosAngeles = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

num_lat_intervals = 30
num_lon_intervals = 30

# Latitude range: 33.7059 to 34.3343
# Longitude range: -118.6676 to -118.1554

lat_min, lat_max = 33.7037, 34.3373
lon_min, lon_max = -118.6682, -118.1553

# Create bins for latitude and longitude
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

# Scale latitude and longitude
scaler = StandardScaler()
lat_scaled = scaler.fit_transform(np.array(lat).reshape(-1, 1)).flatten()
lon_scaled = scaler.fit_transform(np.array(lon).reshape(-1, 1)).flatten()
#

# Create dictionaries for normal to scaled and scaled to normal conversions
lat_to_scaled = {lat[i]: lat_scaled[i] for i in range(len(lat))}
lon_to_scaled = {lon[i]: lon_scaled[i] for i in range(len(lon))}

scaled_to_lat = {lat_scaled[i]: lat[i] for i in range(len(lat_scaled))}
scaled_to_lon = {lon_scaled[i]: lon[i] for i in range(len(lon_scaled))}

class_weights = {0: 0.5296199547898351, 1: 8.940255961693563}


def add_cyclic_features(df, columns_periods):
    """
    Aggiunge le trasformazioni cicliche (sin e cos) per variabili temporali.

    Args:
        df (pd.DataFrame): Il dataframe che contiene le variabili temporali.
        columns_periods (dict): Dizionario con colonne e periodicità.
                                Esempio: {'shift': 3, 'day_of_week': 7, 'month': 12}

    Returns:
        pd.DataFrame: DataFrame originale con colonne aggiuntive (sin e cos per ogni variabile ciclica).
    """
    for col, period in columns_periods.items():
        if col in df.columns:
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    return df


columns_periods = {
    'shift': 3,  # 24 ore in un giorno
    'day': 31,  # 31 giorni in un mese (massimo possibile)
    'month': 12,  # 12 mesi in un anno
    'day_of_week': 7  # 7 giorni in una settimana
}

spatial_columns = ['fix_lat', 'fix_lon']
temporal_columns = ['shift', 'day', 'month', 'year', 'day_of_week',
                    'shift_sin', 'shift_cos', 'day_sin', 'day_cos',
                    'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']

class_weight_dict = {0: 0.7788253176824149, 1: 1.3966187219940793}


def weighted_binary_crossentropy(y_true, y_pred):
    """
    Weighted binary crossentropy loss function for Keras.
    """
    class_weights = tf.constant([class_weight_dict[0], class_weight_dict[1]])
    weights = tf.gather(class_weights, tf.cast(y_true, tf.int32))
    bce = binary_crossentropy(y_true, y_pred)
    weighted_bce = weights * bce
    return K.mean(weighted_bce)


areas = {
    1: [(0, 7), (5, 16)],
    2: [(0, 0), (10, 7)],
    3: [(5, 7), (10, 16)],
    4: [(0, 16), (10, 25)],
    5: [(10, 5), (15, 11)],
    6: [(10, 11), (15, 20)],
    7: [(10, 20), (15, 25)],
    8: [(9, 25), (15, 30)],
    9: [(15, 8), (20, 17)],
    10: [(15, 17), (20, 27)],
    11: [(20, 18), (30, 27)],
}

area_centers = {
    1: (3, 13),
    2: (5, 5),
    3: (7, 13),
    4: (5, 20),
    5: (13, 8),
    6: (13, 15),
    7: (13, 22),
    8: (13, 27),
    9: (18, 13),
    10: (18, 22),
    11: (25, 23),
}


def get_area_number(x, y):
    """
    This function takes two indices (x, y) and returns the area number
    based on the defined area boundaries.

    Args:
    x (int): X-coordinate (0 to 30)
    y (int): Y-coordinate (0 to 30)

    Returns:
    int: The area number (1 to 11) or 0 if not in any defined area
    """

    # Check which area the point belongs to
    for area_number, ((x_min, y_min), (x_max, y_max)) in areas.items():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return area_number

    # If the point is not in any area, return 0
    return 0


def get_area_center(area_number):
    """
    This function takes an area number and returns the center coordinates (x, y).

    Args:
    area_number (int): The area number (1 to 11)

    Returns:
    tuple: The center coordinates (x, y) or None if the area number is invalid
    """

    # Return the center coordinates separately for the given area number
    center = area_centers.get(area_number, (None, None))
    return lat[center[0]], lon[center[1]]