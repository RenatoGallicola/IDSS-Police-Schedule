import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf
import os

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
training_set_path = 'training_set_full.csv'
validation_set_path = 'validation_set_full.csv'
test_set_path = 'test_set_full.csv'

train_df = pd.read_csv(training_set_path).drop(columns=['date'])
val_df = pd.read_csv(validation_set_path).drop(columns=['date'])
test_df = pd.read_csv(test_set_path).drop(columns=['date'])
print("Datasets loaded successfully.")

# 1. Preprocessing the Datasets for LSTM
# ---------------------------------------------------------
print("Preprocessing the datasets for LSTM...")

# Check if the expected categorical columns are present
required_columns = ['day_of_week', 'hour', 'minute', 'day', 'month', 'year', 'fix_lat', 'fix_lon', 'crime_occurrence']
missing_columns = [col for col in required_columns if col not in train_df.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing from the dataset: {missing_columns}")

# Concatenate train, validation, and test to perform consistent encoding and scaling
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# # Encoding categorical variables
print("Encoding categorical variables...")
categorical_columns = ['day_of_week']
encoder = LabelEncoder()
for col in categorical_columns:
    if col in full_df.columns:
        full_df[col] = encoder.fit_transform(full_df[col])
print("Categorical variables encoded.")

# # Standardizing numerical features
print("Standardizing numerical features...")
numerical_columns = ['hour', 'minute', 'day', 'month', 'year', 'fix_lat', 'fix_lon']
scaler = StandardScaler()
full_df[numerical_columns] = scaler.fit_transform(full_df[numerical_columns])
print("Numerical features standardized.")

# # Ensure all data is of numeric type
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

# Calculate class weights based on the distribution of classes in the training set
print("Calculating class weights...")
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# 2. Building the LSTM Model
# ---------------------------------------------------------
model_path = 'crime_prediction_lstm_model_FULL.h5'
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

    # Compile the model with additional metrics for imbalanced data
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
    print("LSTM model built successfully.")

    # 3. Training the LSTM Model
    # ---------------------------------------------------------
    print("Training the LSTM model...")
    history = model.fit(X_train, y_train, epochs=30, batch_size=128, validation_data=(X_val, y_val), class_weight=class_weight_dict, verbose=1)
    print("LSTM model training completed.")

    # Save the model
    model.save(model_path)
    print(f"Model saved to '{model_path}'.")
else:
    print(f"Model already exists at '{model_path}'. Skipping training.")
    model = tf.keras.models.load_model(model_path)

# 5. Evaluating the Model
# ---------------------------------------------------------
print("Evaluating the model on the test set...")
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Precision: {precision:.2f}")
print(f"Test Recall: {recall:.2f}")
