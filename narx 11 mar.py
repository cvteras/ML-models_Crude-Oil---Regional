# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:03:41 2024

@author: cgarn
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load data from Excel file
file_path = r'C:\Users\cgarn\Desktop\oil trading\paper 1\data\compiled db.xlsx'
df = pd.read_excel(file_path)

# Extract features and target variable
X = df[['CL=F', 'USO', 'GSCI index']].values.astype(np.float32)
Y = df['Weekly U.S. Ending Stocks excluding SPR of Crude Oil  (Thousand Barrels)'].values.reshape(-1, 1).astype(np.float32)

# Normalize data using MinMaxScaler
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_normalized = scaler_X.fit_transform(X)
Y_normalized = scaler_Y.fit_transform(Y)

# Convert data to PyTorch tensors
X_tensor = torch.from_numpy(X_normalized).float()
Y_tensor = torch.from_numpy(Y_normalized).float()

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=42)

# Define the neural network architecture
class NARX(nn.Module):
    def __init__(self, input_size, hidden_size, extra_hidden_size, output_size):
        super(NARX, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.extra_hidden = nn.Linear(hidden_size, extra_hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(extra_hidden_size, output_size)

    def forward(self, x):
        hidden_output = self.relu(self.hidden(x))
        extra_hidden_output = self.relu(self.extra_hidden(hidden_output))
        output = self.output(extra_hidden_output)
        return output

# Define the grid of hyperparameters to search
learning_rates = [0.001, 0.01, 0.1]
num_epochs_list = [50, 100, 150]

best_mse = float('inf')
best_lr = None
best_epochs = None
best_model = None

# Perform grid search
for lr in learning_rates:
    for num_epochs in num_epochs_list:
        # Initialize the model
        narx_model = NARX(input_size=X_train.shape[1], hidden_size=10, extra_hidden_size=9, output_size=1)

        # Define loss function and optimizer (Adam)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(narx_model.parameters(), lr=lr)

        # Training loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            Y_pred = narx_model(X_train)
            loss = criterion(Y_pred, Y_train)
            loss.backward()
            optimizer.step()

        # Evaluate on the validation set
        with torch.no_grad():
            Y_val_pred = narx_model(X_val)
            val_mse = mean_squared_error(Y_val.numpy(), Y_val_pred.numpy())

        # Update the best hyperparameters if the validation MSE is lower
        if val_mse < best_mse:
            best_mse = val_mse
            best_lr = lr
            best_epochs = num_epochs
            best_model = narx_model

print(f'Best learning rate: {best_lr}')
print(f'Best number of epochs: {best_epochs}')
print(f'Best validation MSE: {best_mse}')

# Print true and predicted values
with torch.no_grad():
    Y_val_pred = best_model(X_val)
    Y_val_pred_np = scaler_Y.inverse_transform(Y_val_pred.numpy())
    Y_val_np = scaler_Y.inverse_transform(Y_val.numpy())
    print("True values:", Y_val_np.flatten())
    print("Predicted values:", Y_val_pred_np.flatten())

# Calculate and print MAPE
mape = np.mean(np.abs((Y_val_np - Y_val_pred_np) / Y_val_np)) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape:.4f}%')


# Convert tensors to numpy arrays and inverse transform to the original scale
Y_val_np = scaler_Y.inverse_transform(Y_val.numpy())
Y_val_pred_np = scaler_Y.inverse_transform(Y_val_pred.numpy())

# Plot true vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(Y_val_np, color='gray', label='True')  # Plot true values in red
plt.plot(Y_val_pred_np, color='black', label='Predicted')  # Plot predicted values in blue
plt.title('True vs Predicted Values', fontsize=14)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Crude Oil Inventory (Thousand Barrels)', fontsize=12)
plt.legend()
plt.show()
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score

# Calculate and print additional evaluation metrics
mae = mean_absolute_error(Y_val_np, Y_val_pred_np)
r2 = r2_score(Y_val_np, Y_val_pred_np)
evs = explained_variance_score(Y_val_np, Y_val_pred_np)

print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R-squared (R2) Score: {r2:.4f}')
print(f'Explained Variance Score (EVS): {evs:.4f}')

