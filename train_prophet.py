#!/usr/bin/env python3
# train_prophet.py

import pandas as pd
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')  # 防止在非GUI環境報錯
import matplotlib.pyplot as plt

def main():
    # 1) 讀取原始資料
    file_path = '資料集_new.csv'
    data = pd.read_csv(file_path)

    # 必要的資料清洗、欄位改名
    data.columns = data.columns.str.strip()
    data.rename(columns={
        '日期': 'ds',
        'CPC': 'y',
    }, inplace=True)
    data['ds'] = pd.to_datetime(data['ds'])
    data.sort_values('ds', inplace=True)

    # 2) 建立 Prophet 模型，並訓練
    model = Prophet(
        changepoint_prior_scale=0.5,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive'
    )
    model.fit(data[['ds','y']])

    # 3) 產生未來 7 天 (或更多) 預測
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    # 4) 輸出預測結果到 CSV
    forecast.to_csv('latest_forecast.csv', index=False)

    # 5) 繪圖 - (A) 全部歷史 + 未來
    fig1 = model.plot(forecast)
    plt.title("Prophet Forecast (All history + Future)")
    plt.savefig("./static/plot_full.png")  # 儲存圖檔
    plt.close(fig1)

    # (B) 只畫最近 15 天 + 未來 7 天
    last_date = data['ds'].max()
    start_15days_ago = last_date - pd.Timedelta(days=14)
    forecast_range = forecast[forecast['ds'] >= start_15days_ago].copy()
    data_recent15 = data[(data['ds'] >= start_15days_ago) & (data['ds'] <= last_date)]

    plt.figure(figsize=(10,6))
    # 畫最近15天真實值
    plt.plot(data_recent15['ds'], data_recent15['y'], 'ko-', label='Recent 15d Actual')
    # 畫未來預測線
    future_part = forecast_range[forecast_range['ds'] > last_date]
    plt.plot(future_part['ds'], future_part['yhat'], 'b--', label='Forecast Next 7d')
    plt.fill_between(future_part['ds'], future_part['yhat_lower'], future_part['yhat_upper'],
                     color='blue', alpha=0.2, label='Conf. Interval')

    plt.title("Prophet - Recent 15 Days + Next 7 Days")
    plt.xlabel("Date")
    plt.ylabel("CPC")
    plt.legend()
    plt.savefig("./static/plot_recent_future.png")
    plt.close()

    print("Prophet training and forecast done. Output: latest_forecast.csv, plot_full.png, plot_recent_future.png")

if __name__ == "__main__":
    main()
