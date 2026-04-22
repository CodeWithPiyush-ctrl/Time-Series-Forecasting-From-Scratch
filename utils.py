import numpy as np

# ================================
# WINDOWING FUNCTION
# ================================
def create_windows(data, window_size, horizon):
    X, y = [], []

    for i in range(len(data) - window_size - horizon):
        # WHY: input is past sequence
        X.append(data[i:i+window_size])

        # WHY: output is future prediction
        y.append(data[i+window_size:i+window_size+horizon])

    return np.array(X), np.array(y)
