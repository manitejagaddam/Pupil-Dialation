import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load CSV
df = pd.read_csv("../recordings/pupil_hr_log.csv")

# Smooth the signal (optional)
df["pupil_radius"] = df["pupil_radius"].rolling(window=3, min_periods=1).mean()

# Create time series windows
def create_sequences(data, target, window_size=30):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(target[i+window_size])
    return np.array(X), np.array(y)

radius = df["pupil_radius"].values
hr = df["heart_rate"].values

# Normalize radius
scaler = MinMaxScaler()
radius = scaler.fit_transform(radius.reshape(-1, 1)).flatten()

X, y = create_sequences(radius, hr, window_size=30)
X = X[..., np.newaxis]  # Add channel dimension for CNN

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
