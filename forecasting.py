import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

CSV_PATH = "forex_data_2020-2025.csv"
COLUMN = None   
NLAGS = 40
MAX_D = 2      


def _find_date_col(df):
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            return c
    first = df.columns[0]
    if pd.to_datetime(df[first], errors="coerce").notna().mean() > 0.6:
        return first
    return None

def load_series(path=CSV_PATH, column=COLUMN):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    date_col = _find_date_col(df)
    if date_col:
        df = pd.read_csv(path, parse_dates=[date_col], low_memory=False).set_index(date_col).sort_index()
    nums = df.select_dtypes(include=["number"])
    if column:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        return df[column].dropna()
    if nums.shape[1] == 0:
        raise ValueError("No numeric columns found.")
    return nums.iloc[:, 0].dropna()

def run_adf(series, autolag="AIC"):
    res = adfuller(series, autolag=autolag)
    stat, pval, nlags, nobs, crit = res[0], res[1], res[2], res[3], res[4]
    print(f"\nADF on '{series.name or 'series'}' (n={len(series)})")
    print(f" Test statistic : {stat:.6f}")
    print(f" p-value        : {pval:.6f}")
    print(f" # lags used    : {nlags}")
    print(f" # obs          : {nobs}")
    for k, v in crit.items():
        print(f"  critical {k} : {v:.6f}")
    print("=>", "STATIONARY" if pval < 0.05 else "NON-STATIONARY")
    mean_y = series.mean()
    print("Mean:", mean_y)
    return {"stat": stat, "pval": pval, "nlags": nlags, "nobs": nobs, "crit": crit}


def find_min_d(series, max_d=MAX_D):
    s = series.copy()
    for d in range(0, max_d+1):
        info = run_adf(s)
        if info["pval"] < 0.05:
            return d, s
        s = s.diff().dropna()
    print(f"\nADF did not indicate stationarity up to d={max_d}; proceeding with d={max_d}.")
    run_adf(s)
    return max_d, s

def plot_acf_pacf(series, nlags=NLAGS):
    nlags = min(nlags, max(1, len(series)-1))
    plt.figure(figsize=(9,4)); plot_acf(series, lags=nlags, zero=True); plt.title(f"ACF (d-series) n={len(series)}"); plt.tight_layout(); plt.show()
    try:
        plt.figure(figsize=(9,4)); plot_pacf(series, lags=nlags, zero=True, method='ywm'); plt.title(f"PACF (d-series) n={len(series)}"); plt.tight_layout(); plt.show()
    except Exception:
        from statsmodels.tsa.stattools import pacf
        vals = pacf(series, nlags=nlags, method='ywm')
        ci = 1.96/np.sqrt(len(series)); lags = np.arange(len(vals))
        plt.figure(figsize=(9,4)); plt.bar(lags, vals, width=0.4); plt.axhline(0, linestyle='--'); plt.axhline(ci, linestyle='--'); plt.axhline(-ci, linestyle='--')
        plt.title("PACF (fallback)"); plt.tight_layout(); plt.show()

def fit_compare_and_diag(series, d, orders=[(0,1,0),(0,1,1),(1,1,0)]):
    adapted = [(p,d,q) for (p,_,q) in orders]
    fits = {}
    for order in adapted:
        try:
            m = ARIMA(series if d==0 else original_series, order=order)
            fit = m.fit()
            fits[order] = fit
            print(f"\nFitted ARIMA{order} — AIC: {fit.aic:.3f}, BIC: {fit.bic:.3f}")
            print(" params:")
            print(fit.params)
        except Exception as e:
            print(f"ARIMA{order} failed: {e}")
    if not fits:
        return None
    best = min(fits.keys(), key=lambda k: fits[k].aic)
    print(f"\nBest by AIC: ARIMA{best} (AIC={fits[best].aic:.3f})\n")
    best_fit = fits[best]
    resid = best_fit.resid.dropna()
    print("Ljung-Box results for residuals (lags 10 and 20):")
    lb = acorr_ljungbox(resid, lags=[10,20], return_df=True)
    print(lb)
    plt.figure(figsize=(9,3)); plt.plot(resid); plt.title(f"Residuals — ARIMA{best}"); plt.tight_layout(); plt.show()
    plt.figure(figsize=(9,4)); plot_acf(resid, lags=min(NLAGS, len(resid)-1), zero=False); plt.title("Residual ACF"); plt.tight_layout(); plt.show()
    return {"fits": fits, "best": best, "best_fit": best_fit}

# ---- forecasting & evaluation ---
def rolling_forecast_eval(series, order, test_size=None):
    n = len(series)
    if test_size is None:
        test_size = max(10, int(0.2 * n))
    train_end = n - test_size
    forecasts = []
    actuals = []
    index_test = series.index[train_end:]
    for i in range(test_size):
        train = series.iloc[:train_end + i]
        try:
            model = ARIMA(train, order=order).fit()
            fc = model.get_forecast(steps=1)
            pred = fc.predicted_mean.iloc[0]
        except Exception:
            pred = train.iloc[-1]  # fallback 
        forecasts.append(pred)
        actuals.append(series.iloc[train_end + i])
    forecasts = np.array(forecasts)
    actuals = np.array(actuals)
    rmse = np.sqrt(np.mean((forecasts - actuals)**2))
    mae = np.mean(np.abs(forecasts - actuals))
    return {"forecast": forecasts, "actual": actuals, "index": index_test, "rmse": rmse, "mae": mae}

if __name__ == "__main__":
    original_series = load_series(CSV_PATH, COLUMN)
    d, d_series = find_min_d(original_series, MAX_D)
    print(f"\nSelected differencing order d = {d}\n")
    plot_acf_pacf(d_series, NLAGS)
    out = fit_compare_and_diag(d_series if d==0 else original_series, d)

    # Forecasting using ARIMA(0,d,0)
    selected_order = (0, d, 0)
    print(f"\nPerforming rolling-origin forecast evaluation with ARIMA{selected_order} ...")
    res = rolling_forecast_eval(original_series, order=selected_order, test_size=max(20, int(0.2 * len(original_series))))
    print(f"RMSE = {res['rmse']:.6g}, MAE = {res['mae']:.6g}")

    # plot Forecast vs Actual
    plt.figure(figsize=(10,4))
    plt.plot(res["index"], res["actual"], label="Actual")
    plt.plot(res["index"], res["forecast"], label="Forecast", linestyle="--")
    plt.title(f"Forecast vs Actual — ARIMA{selected_order}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot RMSE as a single bar
    plt.figure(figsize=(4,3))
    plt.bar(["RMSE"], [res["rmse"]])
    plt.title("RMSE (rolling forecast)")
    plt.tight_layout()
    plt.show()
