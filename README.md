# Time Series Forecasting using Custom GRU

This project implements a time-series forecasting pipeline from scratch.

## Parameters (Roll No: 102303307)
- Window Size: 17
- Prediction Horizon: 2
- Hidden Size: 14

## Models
- MLP (baseline)
- Custom GRU (from scratch)

## 📊 Datasets Used

1. Electricity Consumption Dataset (Kaggle - Household Electric Power Consumption)
2. Stock Price Dataset (Yahoo Finance / Kaggle)

Both datasets were processed using the same pipeline, model, and parameters.

## Key Idea
Convert time-series into input-output windows and train models to predict future values.

## 📈 Results

### Electricity Dataset
<img width="640" height="480" alt="Electricity_loss" src="https://github.com/user-attachments/assets/7823c794-4275-4b1b-9a15-2a9f4e18c6ac" />
<img width="640" height="480" alt="Electricity_prediction" src="https://github.com/user-attachments/assets/ede137ed-39a2-47e8-9d48-f0bf14814e8b" />



### Stock Dataset
<img width="640" height="480" alt="Stock_loss" src="https://github.com/user-attachments/assets/5db88f07-b2ea-467e-b9b4-917f5019a2b5" />
<img width="640" height="480" alt="Stock_prediction" src="https://github.com/user-attachments/assets/8fc5cb45-9e5c-44c0-bc4c-7f8eb728ba35" />


## Run
pip install -r requirements.txt  
python main.py
