import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers, Sequential, Input
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Flatten, MultiHeadAttention, Layer
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras_tuner import BayesianOptimization
import tkinter as tk
from tkinter import ttk

# Load dataset
df = pd.read_csv(r"C:\Users\emek\Desktop\emek\RUL(MS Thesis)\Final Database.csv")

############# PREPROCESSING ##################

# Drop rows with missing RUL values
df = df.dropna(subset=['RUL'])

# Define the features and target
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

############# DATA AUGMENTATION ##################

# Data augmentation by adding noise
def add_noise(data, noise_level=0.0001):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

# Augment dataset with noise 
augmented_X_train = np.array([add_noise(x) for x in X_train])

############# SIMSIAM MODEL DEFINITION ##################

# Define TransformerEncoderLayer with L2 Regularization
class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, l2_strength=1e-4):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),  # (batch_size, seq_len, dff)
            Dense(d_model, kernel_regularizer=regularizers.l2(l2_strength))  # (batch_size, seq_len, d_model)
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

# Define the encoder
def get_encoder(input_shape, num_heads, dff, rate=0.1, l2_strength=1e-6):
    inputs = layers.Input(shape=input_shape)
    transformer_encoder = TransformerEncoderLayer(d_model=input_shape[-1], num_heads=num_heads, dff=dff, rate=rate, l2_strength=l2_strength)
    x = transformer_encoder(inputs, training=True)
    x = Flatten(name='backbone_pool')(x)
    outputs =  layers.Dense(2, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(x)
    model = Model(inputs,outputs, name="encoder")
    return model

# Define the predictor
def get_predictor():
    model = Sequential([
        Input((2,)),
        Dense(2, activation='relu', kernel_regularizer=regularizers.l2(1e-6)),  # L2 regularization 1e-4
        layers.BatchNormalization(),
        layers.ReLU(),
        Dense(2),
    ], name="predictor")
    return model

# Loss function for SimSiam
def simsiam_loss(p, z):
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    return -tf.reduce_mean(tf.reduce_sum(p * z, axis=1))

# SimSiam model
class SimSiam(Model):
    def __init__(self, encoder, predictor):
        super(SimSiam, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        (x1, x2) = data

        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(x1, training=True), self.encoder(x2, training=True)
            p1, p2 = self.predictor(z1, training=True), self.predictor(z2, training=True)
            loss = simsiam_loss(p1, z2) / 2 + simsiam_loss(p2, z1) / 2

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

# Example usage
input_shape = (sequence_length, X_train.shape[2])

# Create datasets for SimSiam training
ssl_ds_one = tf.data.Dataset.from_tensor_slices(augmented_X_train).shuffle(1024).batch(128)
ssl_ds_two = tf.data.Dataset.from_tensor_slices(augmented_X_train).shuffle(1024).batch(128)
ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))

# Training the SimSiam model
encoder = get_encoder(input_shape=input_shape, num_heads= 4, dff=512, rate=0.02)
predictor = get_predictor()
simsiam = SimSiam(encoder, predictor)
simsiam.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.045, momentum=0.9))

simsiam.fit(ssl_ds, epochs=50)

# Extract the backbone model from SimSiam encoder
backbone = Model(simsiam.encoder.input, simsiam.encoder.get_layer("backbone_pool").output)

#backbone.trainable = False
backbone.trainable = True


inputs = Input(shape=input_shape)
#x = backbone(inputs, training=False)
x = backbone(inputs, training=True) 
outputs = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(1e-6))(x)
transformer_model = Model(inputs, outputs,name= 'transformer_model')
transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.045), loss='mse', metrics=['mae'])



test_ds = (X_test, y_test)

history = transformer_model.fit(
    X_train, y_train,
    validation_data=test_ds,
    epochs=200,
    callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=30)]
)

# Plot training history with limited range for loss
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.ylim([20,500])  
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.ylim([5, 20])  
plt.title('Mean Absolute Error over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.show()

# Evaluate the model
test_loss, test_mae = transformer_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Predict on test set
test_predictions = transformer_model.predict(X_test)


r2 = r2_score(y_test, test_predictions)
mse = mean_squared_error(y_test, test_predictions)
mae = mean_absolute_error(y_test, test_predictions)

# Display the metrics in a DataFrame
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'MAE', 'R²'],
    'Value': [mse, mae, r2]
})

print(metrics_df)

# Function to display metrics in a new tab
def display_metrics(metrics_df):
    root = tk.Tk()
    root.title("Model Performance Metrics")

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    tree = ttk.Treeview(frame, columns=("Metric", "Value"), show="headings")
    tree.heading("Metric", text="Metric")
    tree.heading("Value", text="Value")

    for _, row in metrics_df.iterrows():
        tree.insert("", tk.END, values=(row['Metric'], row['Value']))

    tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    root.mainloop()

# Call the function to display metrics
display_metrics(metrics_df)

# Predict on 10 random instances from the test set
random_indices = random.sample(range(X_test.shape[0]), 10)
X_sample = X_test[random_indices]
y_sample = y_test[random_indices]

predictions = transformer_model.predict(X_sample)
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