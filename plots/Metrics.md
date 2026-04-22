# 📊 Model Evaluation Metrics

## 🔢 Parameters Used
- Window Size: 17
- Prediction Horizon: 2
- Hidden Size: 14

---

## 📈 Final Model Performance (Custom GRU)

- MSE: 0.3199  
- MAE: 0.4763  
- RMSE: 0.5657  

---

## 🔁 Ablation Study Results

### Window Size = 8
- MSE: 0.4131  

### Window Size = 17 (Original)
- MSE: 0.4019  

### Window Size = 34
- MSE: 0.4258  

---

## 📊 Observations

- Smaller window size (8):
  - Model lacks sufficient past context
  - Higher error due to limited temporal information

- Medium window size (17):
  - Balanced performance
  - Captures enough sequence information without overfitting

- Larger window size (34):
  - Slight increase in error
  - Model struggles with longer dependencies and noise

---

## ❌ Failure Analysis

- Model struggles with:
  - Sudden spikes in data
  - Rapid fluctuations in trend

- Predictions tend to:
  - Lag behind actual values
  - Smooth out sharp peaks

---

## 🧠 Key Insights

- GRU performs better than MLP due to memory capability
- Window size significantly impacts performance
- Too small → underfitting  
- Too large → harder optimization  

---
