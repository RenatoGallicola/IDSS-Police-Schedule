import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
from tqdm import tqdm

# Enable GPU acceleration
print("Checking for GPU availability...")

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"GPUs found: {len(physical_devices)}. Using GPU for training.")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No GPU found. Using CPU for training.")

# Load the training, validation, and test datasets
print("Loading datasets...")
training_set_path = 'training_set_bal_70.csv'
validation_set_path = 'validation_set_bal_70.csv'
test_set_path = 'test_set_bal_70.csv'
updated_test_set_path = 'updated_test_set_bal_70.csv'

train_df = pd.read_csv(training_set_path)
val_df = pd.read_csv(validation_set_path)
test_df = pd.read_csv(test_set_path)
print("Datasets loaded successfully.")

# 1. Preprocessing the Datasets for LSTM
# ---------------------------------------------------------
print("Preprocessing the datasets for LSTM...")

# Check if the expected categorical columns are present
required_columns = ['day_of_week', 'area_name', 'hour', 'minute', 'day', 'month', 'year', 'fix_lat', 'fix_lon', 'crime_occurrence']
missing_columns = [col for col in required_columns if col not in train_df.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing from the dataset: {missing_columns}")

# Concatenate train, validation, and test to perform consistent encoding and scaling
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Encoding categorical variables
print("Encoding categorical variables...")
categorical_columns = ['day_of_week', 'area_name']
encoder = LabelEncoder()
for col in categorical_columns:
    if col in full_df.columns:
        full_df[col] = encoder.fit_transform(full_df[col])
print("Categorical variables encoded.")

# Standardizing numerical features
print("Standardizing numerical features...")
numerical_columns = ['hour', 'minute', 'day', 'month', 'year', 'fix_lat', 'fix_lon']
scaler = StandardScaler()
full_df[numerical_columns] = scaler.fit_transform(full_df[numerical_columns])
print("Numerical features standardized.")

# Ensure all data is of numeric type
print("Ensuring all columns are numeric...")
full_df = full_df.apply(pd.to_numeric, errors='coerce')
full_df.fillna(0, inplace=True)
full_df = full_df.astype(float)
print("All columns converted to numeric types.")

# Split back into train, validation, and test sets
train_df = full_df.iloc[:len(train_df)]
val_df = full_df.iloc[len(train_df):len(train_df) + len(val_df)]
test_df = full_df.iloc[len(train_df) + len(val_df):]

# Splitting datasets into features and target
X_train = train_df.drop(columns=['crime_occurrence']).values.reshape((train_df.shape[0], 1, train_df.shape[1] - 1))
y_train = train_df['crime_occurrence'].values

X_val = val_df.drop(columns=['crime_occurrence']).values.reshape((val_df.shape[0], 1, val_df.shape[1] - 1))
y_val = val_df['crime_occurrence'].values

X_test = test_df.drop(columns=['crime_occurrence']).values.reshape((test_df.shape[0], 1, test_df.shape[1] - 1))
y_test = test_df['crime_occurrence'].values

# 2. Building the LSTM Model
# ---------------------------------------------------------
model_path = 'crime_prediction_lstm_model.h5'
if not os.path.exists(model_path):
    print("Building the LSTM model...")
    model = Sequential()
    model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print("LSTM model built successfully.")

    # 3. Training the LSTM Model
    # ---------------------------------------------------------
    print("Training the LSTM model...")
    history = model.fit(X_train, y_train, epochs=30, batch_size=128, validation_data=(X_val, y_val), verbose=1)
    print("LSTM model training completed.")

    # Save the model
    model.save(model_path)
    print(f"Model saved to '{model_path}'.")
else:
    print(f"Model already exists at '{model_path}'. Skipping training.")
    model = tf.keras.models.load_model(model_path)

# 4. Generate Negative Data for the Test Set After Training
# ---------------------------------------------------------
if os.path.exists(updated_test_set_path):
    print(f"Updated test set already exists at '{updated_test_set_path}'. Loading it...")
    test_df = pd.read_csv(updated_test_set_path)
else:
    print("Removing existing negative data from the test set...")
    test_df = test_df[test_df['crime_occurrence'] == 1]

    print("Generating balanced negative data for the test set after training...")
    negative_data = []

    # Define the number of intervals for lat and lon
    num_lat_intervals = 20  # Adjust based on distribution
    num_lon_intervals = 40  # Adjust based on distribution

    # Create bins for latitude and longitude
    lat_min, lat_max = test_df['fix_lat'].min(), test_df['fix_lat'].max()
    lon_min, lon_max = test_df['fix_lon'].min(), test_df['fix_lon'].max()

    lat_bins = np.linspace(lat_min, lat_max, num_lat_intervals + 1)
    lon_bins = np.linspace(lon_min, lon_max, num_lon_intervals + 1)

    # Iterate through each unique combination of area, day, month, year, and hour in the test set
    grouped = test_df.groupby(['area_name', 'day', 'month', 'year', 'hour'])
    for (area, day, month, year, hour), group in tqdm(grouped, desc="Generating negative data for test set"):
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
                'area_name': area,
                'fix_lat': fix_lat,
                'fix_lon': fix_lon,
                'crime_occurrence': 0  # Indicate no crime occurrence
            }
            for fix_lat, fix_lon in selected_pairs
        ])

    # Convert negative examples to DataFrame and concatenate with test_df
    negative_df = pd.DataFrame(negative_data)
    test_df = pd.concat([test_df, negative_df], ignore_index=True)
    print(f"Negative examples added to the test set. Test set now has {test_df.shape[0]} records.")

    # Save the updated test dataset
    print("Saving the updated test set...")
    test_df.to_csv(updated_test_set_path, index=False)
    print(f"Updated test set saved to '{updated_test_set_path}'.")

# 5. Evaluating the Model Again After Adding Negative Data
# ---------------------------------------------------------
print("Evaluating the model on the updated test set...")
X_test = test_df.drop(columns=['crime_occurrence']).values.reshape((test_df.shape[0], 1, test_df.shape[1] - 1))
y_test = test_df['crime_occurrence'].values

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Updated Test Accuracy: {accuracy * 100:.2f}%")
