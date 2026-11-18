# HHS Unaccompanied Alien Children (UAC) Predictive Operations & Capacity Intelligence Suite

An industry-grade, end-to-end **Time-Series Forecasting & Prescriptive Decision Intelligence** web application built with Python and Streamlit. Designed to transform the U.S. Department of Health and Human Services (HHS) UAC daily operations reporting from reactive descriptive tracking into proactive shelter capacity, medical staffing, and sponsor placement planning.

## 1. Project Folder Structure

```
HHS_UAC_Forecasting_Project/
├── app.py                                      # Main Streamlit web application & CLI verification suite
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation & benchmark leaderboard
├── AGENTS.md                                   # Google Antigravity IDE agent configuration
├── .streamlit/
│   └── config.toml                             # Streamlit server & UI configuration
├── data/
│   └── HHS_Unaccompanied_Alien_Children_Program.csv  # Daily HHS/CBP UAC time-series dataset
├── notebooks/
│   ├── 3.0-HHS_UAC_Model_Training.ipynb        # Primary Optimum ML Model Training Notebook (Random Forest CV)
│   ├── Train_01_Baseline_Naive_and_Moving_Average.ipynb
│   ├── Train_02_Statistical_ARIMA_SARIMA_and_Exponential_Smoothing.ipynb
│   ├── Train_03_Machine_Learning_Random_Forest_Regressor.ipynb
│   ├── Train_04_Machine_Learning_Gradient_Boosting_and_Ensemble.ipynb
│   ├── 01_Time_Series_EDA_and_Feature_Engineering.ipynb
│   ├── 02_Baseline_and_Statistical_Models_ARIMA_SARIMA_ETS.ipynb
│   ├── 03_Machine_Learning_Models_RF_GBR_and_Ensemble.ipynb
│   └── 04_Master_End_to_End_All_Models_Benchmark_and_KPIs.ipynb
├── models/
│   ├── random_forest_regressor.pkl             # Serialized Random Forest model artifact
│   ├── mass_balance_ensemble.pkl               # Serialized Mass-Balance Reconciled Ensemble artifact
│   ├── sarima_model.pkl                        # Serialized ARIMA/SARIMA artifact
│   ├── gradient_boosting_regressor.pkl         # Serialized Gradient Boosting artifact
│   ├── exponential_smoothing_model.pkl         # Serialized Exponential Smoothing artifact
│   ├── baseline_naive_model.pkl                # Serialized Naive Persistence artifact
│   └── baseline_ma7_model.pkl                  # Serialized 7-Day Moving Average artifact
└── src/
    ├── __init__.py
    ├── data_pipeline.py                        # Time-series cleaning, interpolation & 72-feature engineering
    └── evaluate_models.py                      # 180-day walk-forward validation benchmark script

```

## 2. Quick Start

```
# 1. Navigate into the project folder
cd HHS_UAC_Forecasting_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit web application
streamlit run app.py

```

You can also run the built-in CLI verification suite directly:

```
python3 app.py

```

## 3. Walk-Forward Validation Leaderboard (180-Day Out-of-Sample Test)

### A. Active HHS Care Load (`Children in HHS Care`)

| Model | Family | MAE | RMSE | MAPE (%) | Forecast Accuracy (%) | Horizon 7d MAE | Horizon 14d MAE | Horizon 30d MAE | Forecast Stability Index | Model Robustness | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Random Forest Regressor** | Machine Learning | 4.61 | 5.94 | 0.21% | 99.79% | 5.85 | 6.44 | 7.34 | 98.7 | 99.8 | 
| **ARIMA / SARIMA** | Statistical | 4.82 | 6.21 | 0.22% | 99.78% | 6.59 | 7.43 | 8.71 | 95.5 | 99.8 | 
| **Mass-Balance Reconciled Ensemble** | Hybrid Reconciled | 5.09 | 6.53 | 0.23% | 99.77% | 6.46 | 7.11 | 8.11 | 99.4 | 99.8 | 
| **Gradient Boosting Regressor** | Machine Learning | 5.35 | 6.86 | 0.24% | 99.76% | 6.79 | 7.47 | 8.52 | 99.3 | 99.7 | 
| **Exponential Smoothing** | Statistical | 5.51 | 7.12 | 0.25% | 99.75% | 7.53 | 8.49 | 9.96 | 100.0 | 99.8 | 
| **Naive Persistence** | Baseline | 6.93 | 8.56 | 0.31% | 99.69% | 10.66 | 12.43 | 15.14 | 99.8 | 99.6 | 
| **Moving Average (7d)** | Baseline | 23.23 | 28.21 | 1.04% | 98.96% | 35.75 | 41.66 | 50.75 | 96.8 | 98.2 | 

### B. Daily Placement / Discharge Demand (`Children discharged from HHS Care`)

| Model | Family | MAE | RMSE | sMAPE (%) | Forecast Accuracy (%) | Horizon 7d MAE | Horizon 14d MAE | Horizon 30d MAE | Forecast Stability Index | Model Robustness | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Mass-Balance Reconciled Ensemble** | Hybrid Reconciled | 3.42 | 4.55 | 27.04% | 72.96% | 4.34 | 4.78 | 5.45 | 93.9 | 80.6 | 
| **Random Forest Regressor** | Machine Learning | 3.53 | 4.69 | 27.09% | 72.91% | 4.48 | 4.93 | 5.62 | 98.0 | 76.5 | 
| **ARIMA / SARIMA** | Statistical | 3.60 | 4.83 | 28.62% | 71.38% | 4.92 | 5.55 | 6.51 | 90.5 | 81.6 | 
| **Gradient Boosting Regressor** | Machine Learning | 3.75 | 4.99 | 29.61% | 70.39% | 4.76 | 5.24 | 5.97 | 99.4 | 76.1 | 
| **Naive Persistence** | Baseline | 3.96 | 5.25 | 31.43% | 68.57% | 6.09 | 7.10 | 8.65 | 99.9 | 74.9 | 
| **Moving Average (7d)** | Baseline | 4.05 | 5.32 | 31.51% | 68.49% | 6.23 | 7.26 | 8.85 | 85.0 | 80.2 | 
| **Exponential Smoothing** | Statistical | 4.10 | 5.39 | 32.31% | 67.69% | 5.61 | 6.32 | 7.41 | 91.1 | 78.2 | 
