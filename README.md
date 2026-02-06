# Time-Series-Forecasting-ARIMA-
Time Series Forecasting of the Exchange Rate Using ARIMA
# FX Time Series Analysis with ARIMA

## Overview
This project presents a rigorous time series analysis of foreign exchange (FX) data using ARIMA models.  
The objective is to evaluate the predictability of FX prices and assess whether linear time series models provide forecasting power beyond a naive random-walk benchmark.

## Methodology
- Descriptive analysis of FX prices
- Stationarity testing using the Augmented Dickey-Fuller (ADF) test
- Automatic differencing selection
- Model identification via ACF and PACF
- ARIMA model estimation and selection (AIC/BIC)
- Residual diagnostics (Ljung–Box test)
- Rolling-origin forecast evaluation
- Forecast accuracy metrics (RMSE, MAE)

## Key Results
- The series is non-stationary in levels and stationary after first differencing (I(1))
- ACF and PACF of the differenced series show no significant autocorrelation
- ARIMA(0,1,0) is selected as the most parsimonious model
- Forecast evaluation yields RMSE ≈ 0.05 and MAE ≈ 0.04
- Results are consistent with weak-form market efficiency in FX markets


