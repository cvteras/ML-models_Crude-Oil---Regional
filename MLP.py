
"""

@author: cgarn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the data
df = pd.read_excel(r'C:\Users\_')

# Define features and target
features = ['CL=F', 'USO', 'GSCI index']
target = 'Weekly U.S. Ending Stocks excluding SPR of Crude Oil  (Thousand Barrels)'

# Extract features and target
X = df[features].values
y = df[target].values

# Use MinMaxScaler to normalize the features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y.reshape(-1, 1))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=101)

# Build the MLP model
model = Sequential()
model.add(Dense(120, input_dim=3, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu')) 
model.add(Dense(1, activation='linear'))

# Use Adam optimizer with a lower learning rate
optimizer = Adam(learning_rate=0.01)

model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

# Train the model
history = model.fit(X_train, y_train, epochs=117, batch_size=40, validation_split=0.15, verbose=2)



# Plot training and validation metrics
training_metric = history.history['mse']
validation_metric = history.history['val_mse']

epochs = range(1, len(training_metric) + 1)
plt.plot(epochs, training_metric, 'gray', label='Training MSE')
plt.plot(epochs, validation_metric, 'black', label='Validation MSE')
plt.title('Training and Validation Loss Curves', fontsize=10)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('MSE', fontsize=10)
plt.legend()
plt.show()

# Make predictions on the test set
predictions = model.predict(X_test)

# Inverse transform the scaled predictions to the original scale
predictions_original_scale = scaler_y.inverse_transform(predictions)

# Print the first few predicted values
print("Predicted Values:")
print(predictions_original_scale[:10])

# Calculate additional metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
msle = np.mean(np.log1p(predictions_original_scale + 1) - np.log1p(scaler_y.inverse_transform(y_test) + 1))

mse = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on Test Set: {mse[0]}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')
print(f'Mean Squared Logarithmic Error: {msle}')

# Calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate MAPE
mape = mean_absolute_percentage_error(scaler_y.inverse_transform(y_test), predictions_original_scale)

# Print MAPE
print(f'Mean Absolute Percentage Error (MAPE): {mape:.4f}%')
