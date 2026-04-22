# 📊 Model Evaluation Metrics

## 🔢 Parameters
- Window Size: 17  
- Prediction Horizon: 2  
- Hidden Size: 14  

---

## 📈 Electricity Dataset Results

- MSE: 0.3383  
- MAE: 0.4865  
- RMSE: 0.5816  

---

## 📈 Stock Dataset Results

- MSE: 0.0040  
- MAE: 0.0532  
- RMSE: 0.0631  

---

## 🔍 Observations

- Electricity dataset:
  - Higher error due to noisy and irregular consumption patterns  
  - Model struggles with sudden fluctuations  

- Stock dataset:
  - Much lower error  
  - Data is smoother and easier to predict in short windows  

---

## ❌ Failure Analysis

- Model struggles with:
  - sudden spikes in electricity data  
  - rapid changes in trend  

- Predictions tend to:
  - lag behind actual values  
  - smooth out sharp peaks  

---

## 🧠 Key Insights

- GRU performs well due to its memory mechanism  
- Performance varies significantly across datasets  
- Data characteristics (noise, volatility) strongly affect results  
