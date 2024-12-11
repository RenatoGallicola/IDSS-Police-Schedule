import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Input, Attention, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
import tensorflow as tf
import os
from tensorflow.python.keras.callbacks import EarlyStopping, LearningRateScheduler
from utils import add_cyclic_features, columns_periods

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
training_set_path = '../../datasets/training_set_full.csv'
validation_set_path = '../../datasets/validation_set_full.csv'
test_set_path = '../../datasets/test_set_full.csv'

train_df = pd.read_csv(training_set_path)
val_df = pd.read_csv(validation_set_path)
test_df = pd.read_csv(test_set_path)
print("Datasets loaded successfully.")

# 1. Preprocessing the Datasets for LSTM
# ---------------------------------------------------------
print("Preprocessing the datasets for LSTM...")

# Check if the expected categorical columns are present
required_columns = ['day_of_week', 'hour', 'day', 'month', 'year', 'fix_lat', 'fix_lon', 'crime_occurrence']
missing_columns = [col for col in required_columns if col not in train_df.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing from the dataset: {missing_columns}")

# Concatenate train, validation, and test to perform consistent encoding and scaling
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# # Standardizing numerical features
print("Standardizing numerical features...")


full_df = add_cyclic_features(full_df, columns_periods)

numerical_columns = ['year', 'fix_lat', 'fix_lon']
scaler = StandardScaler()
full_df[numerical_columns] = scaler.fit_transform(full_df[numerical_columns])
print("Numerical features standardized.")

# # Ensure all data is of numeric type
print("Ensuring all columns are numeric...")
full_df = full_df.apply(pd.to_numeric, errors='coerce')
full_df.fillna(0, inplace=True)
full_df = full_df.astype(float)
print("All columns converted to numeric types.")

print(full_df.shape)
print(full_df.columns)

# Split back into train, validation, and test sets
train_df = full_df.iloc[:len(train_df)]
val_df = full_df.iloc[len(train_df):len(train_df) + len(val_df)]
test_df = full_df.iloc[len(train_df) + len(val_df):]

# Splitting datasets into features and target

# Separa le variabili temporali
spatial_columns = ['fix_lat', 'fix_lon']
temporal_columns = ['hour', 'day', 'month', 'year', 'day_of_week',
                    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                    'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']

# Prepara i dati
X_train_temporal = train_df[temporal_columns].values.reshape((train_df.shape[0], 1, len(temporal_columns)))
X_train_spatial = train_df[spatial_columns].values.reshape((train_df.shape[0], 1, len(spatial_columns)))

X_val_temporal = val_df[temporal_columns].values.reshape((val_df.shape[0], 1, len(temporal_columns)))
X_val_spatial = val_df[spatial_columns].values.reshape((val_df.shape[0], 1, len(spatial_columns)))

X_test_temporal = test_df[temporal_columns].values.reshape((test_df.shape[0], 1, len(temporal_columns)))
X_test_spatial = test_df[spatial_columns].values.reshape((test_df.shape[0], 1, len(spatial_columns)))

y_train = train_df['crime_occurrence'].values

y_val = val_df['crime_occurrence'].values

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

model_path = '../crime_prediction_lstm_model_Divided_30_EPOCH.h5'
if not os.path.exists(model_path):
    print("Building the LSTM model...")

    # Input
    temporal_input = Input(shape=(1, len(temporal_columns)), name="Temporal_Input")
    spatial_input = Input(shape=(1, len(spatial_columns)), name="Spatial_Input")

    # Ramo temporale con attenzione
    temporal_lstm = LSTM(128, activation='relu', return_sequences=True)(temporal_input)
    temporal_attention = Attention()([temporal_lstm, temporal_lstm])
    temporal_dense = Dense(64, activation='relu')(temporal_attention)
    temporal_flatten = Flatten()(temporal_dense)  # Appiattisci il tensore

    # Ramo spaziale
    spatial_lstm = LSTM(128, activation='relu', return_sequences=False)(spatial_input)
    spatial_dense = Dense(64, activation='relu')(spatial_lstm)

    # Merge
    merged = Concatenate()([temporal_flatten, spatial_dense])
    final_dense = Dense(128, activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(final_dense)

    model = tf.keras.models.Model(inputs=[temporal_input, spatial_input], outputs=output)
    # Creazione del modello

    # Compila il modello
    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall(), AUC()])

    # Callback
    early_stopping = EarlyStopping(monitor='val_recall', patience=5, restore_best_weights=True)

    def scheduler(epoch, lr):
        return lr if epoch < 10 else lr * tf.math.exp(-0.1)

    lr_scheduler = LearningRateScheduler(scheduler)

    # Addestramento
    history = model.fit(
        [X_train_temporal, X_train_spatial], y_train,
        validation_data=([X_val_temporal, X_val_spatial], y_val),
        epochs=30,
        batch_size=128,
        callbacks=[early_stopping],
        class_weight=class_weight_dict,
        verbose=1
    )
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

# y_pred = (model.predict(X_test) > 0.5).astype("int32")
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(conf_matrix)

results = model.evaluate([X_test_temporal, X_test_spatial], y_test, verbose=1)

loss, accuracy, precision, recall = results[0], results[1], results[2], results[3]

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Precision: {precision:.2f}")
print(f"Test Recall: {recall:.2f}")
# kappa = cohen_kappa_score(y_test, y_pred)
# print(f"Cohen's Kappa: {kappa:.2f}")
