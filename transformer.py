import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Input, MultiHeadAttention, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import random
import keras_tuner as kt

# Load dataset
df = pd.read_csv(r"/Users/emekcapoglu/Desktop/EmekUtku_Capoglu/Datasets/Final Database.csv") 

############# PREPROCESSING ##################

# Keep the first column and define the features and target
features = df.drop(columns=['RUL'])
target = df['RUL']
# Scaling features between 0 and 1
scaler = MinMaxScaler()
SCALED_F = scaler.fit_transform(features)

# Sequence length
sequence_length = 15

# Function for sequences
def create_sequences(data, target, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(target[i + seq_length])
    return np.array(sequences), np.array(targets)

# Create sequences
X, y = create_sequences(SCALED_F, target, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Define TransformerEncoderLayer
class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            Dense(d_model)  # (batch_size, seq_len, d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x)  # (batch_size, seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # (batch_size, seq_len, d_model)

# Define the hypermodel for Bayesian optimization
class TransformerHyperModel(kt.HyperModel):
    def build(self, hp):
        num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
        dff = hp.Int('dff', min_value=128, max_value=512, step=128)
        dropout_rate = hp.Float('dropout_rate', min_value=0.01, max_value=0.3, step=0.01)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        
        inputs = Input(shape=(sequence_length, X_train.shape[2]))
        transformer_encoder = TransformerEncoderLayer(d_model=inputs.shape[-1], num_heads=num_heads, dff=dff, rate=dropout_rate)
        x = transformer_encoder(inputs, training=True)
        x = Flatten()(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
        return model

# Instantiate the tuner and perform Bayesian optimization
tuner = kt.BayesianOptimization(
    TransformerHyperModel(),
    objective='val_loss',
    max_trials=10,
    directory='bayesian_opt',
    project_name='rul_predictionTransformer',
    overwrite=False  # Ensure that the tuner starts fresh each time
)

# Create training dataset
batch_size = 32
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

# Run the hyperparameter search
tuner.search(train_ds, validation_data=test_ds, epochs=50, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# Print the best hyperparameters used for final training
print("\nHyperparameters used for final training with 150 epochs:")
print(f"Number of Heads: {best_hps.get('num_heads')}")
print(f"Feed Forward Network Dimensionality (dff): {best_hps.get('dff')}")
print(f"Dropout Rate: {best_hps.get('dropout_rate')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")

# Build the model with the best hyperparameters and train it for 150 epochs
best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(train_ds, validation_data=test_ds, epochs=200, batch_size=batch_size, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)])

# Plot training history with limited range for loss
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.ylim([40, 100])  
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.ylim([0, 10])  
plt.title('Mean Absolute Error over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.show()

# Evaluate the best model
test_loss, test_mae = best_model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Predict on 10 random instances from the test set
random_indices = random.sample(range(X_test.shape[0]), 10)
X_sample = X_test[random_indices]
y_sample = y_test[random_indices]

predictions = best_model.predict(X_sample)
print("Predictions vs Actual RUL:")
for i in range(10):
    print(f"Prediction: {predictions[i][0]:.2f}, Actual: {y_sample[i]:.2f}")

# Plot predicted vs actual RUL as bar graphs
plt.figure(figsize=(10, 6))
bar_width = 0.35
indices = np.arange(10)

plt.bar(indices, y_sample, bar_width, label='Actual RUL', alpha=0.6)
plt.bar(indices + bar_width, predictions.flatten(), bar_width, label='Predicted RUL', alpha=0.6)

plt.title('Predicted vs Actual RUL')
plt.xlabel('Sample Index')
plt.ylabel('RUL')
plt.xticks(indices + bar_width / 2, [str(i) for i in range(10)])
plt.legend()

plt.show()

# Calculate  metrics
predictions_all = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions_all)
mae = mean_absolute_error(y_test, predictions_all)
r2 = r2_score(y_test, predictions_all)

# Display the metrics in a DataFrame
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'MAE', 'R²'],
    'Value': [mse, mae, r2]
})

print(metrics_df)
