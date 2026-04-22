import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import os

from models import MLP, CustomGRU
from utils import create_windows

# ================================
# PARAMETERS (ROLL NO BASED)
# ================================
window_size = 17
prediction_horizon = 2
hidden_size = 14

print("Window Size:", window_size)
print("Horizon:", prediction_horizon)
print("Hidden Size:", hidden_size)

# ================================
# AUTO LOAD DATASET (KAGGLE SAFE)
# ================================
file_path = None

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('.csv') or filename.endswith('.txt'):
            file_path = os.path.join(dirname, filename)

print("Using dataset:", file_path)

# Load dataset
if file_path.endswith('.txt'):
    df = pd.read_csv(file_path, sep=';', low_memory=False)
else:
    df = pd.read_csv(file_path)

# Select column
data = df.iloc[:, 1].dropna().values

# Normalize
data = (data - data.mean()) / data.std()

# ================================
# WINDOWING
# ================================
X, y = create_windows(data, window_size, prediction_horizon)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32)

# ================================
# TRAIN TEST SPLIT
# ================================
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ================================
# MODELS
# ================================
mlp = MLP(window_size, prediction_horizon)
gru = CustomGRU(1, hidden_size, prediction_horizon)

# ================================
# TRAIN FUNCTION
# ================================
def train(model, X, y, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        output = model(X)

        # WHY: regression problem → MSE
        loss = loss_fn(output, y)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return losses, model

# ================================
# TRAIN MODELS
# ================================
mlp_losses, mlp = train(mlp, X_train, y_train)
gru_losses, gru = train(gru, X_train, y_train)

# ================================
# PLOT LOSS
# ================================
plt.figure()
plt.plot(mlp_losses, label="MLP")
plt.plot(gru_losses, label="GRU")
plt.legend()
plt.title("Training Loss")
plt.savefig("results/loss.png")
plt.show()

# ================================
# PREDICTIONS
# ================================
pred = gru(X_test).detach().numpy()
actual = y_test.numpy()

# ================================
# PLOT PREDICTION
# ================================
plt.figure()
plt.plot(actual[:100, 0], label="Actual")
plt.plot(pred[:100, 0], label="Predicted")
plt.legend()
plt.title("Prediction vs Actual")
plt.savefig("results/prediction.png")
plt.show()

# ================================
# METRICS
# ================================
mse = ((pred - actual)**2).mean()
mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mse)

print("\n===== RESULTS =====")
print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)

# ================================
# ABLATION STUDY
# ================================
with open("results/ablation.txt", "w") as f:
    for ws in [8, 17, 34]:
        X_ab, y_ab = create_windows(data, ws, prediction_horizon)

        X_ab = torch.tensor(X_ab, dtype=torch.float32).unsqueeze(-1)
        y_ab = torch.tensor(y_ab, dtype=torch.float32)

        split = int(0.8 * len(X_ab))
        X_tr, X_te = X_ab[:split], X_ab[split:]
        y_tr, y_te = y_ab[:split], y_ab[split:]

        model = CustomGRU(1, hidden_size, prediction_horizon)
        _, model = train(model, X_tr, y_tr, epochs=20)

        pred_ab = model(X_te).detach().numpy()
        actual_ab = y_te.numpy()

        mse_ab = ((pred_ab - actual_ab)**2).mean()

        result = f"Window Size = {ws}, MSE = {mse_ab}\n"
        print(result)
        f.write(result)
