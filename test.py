from joblib import load
import pandas as pd


# 從檔案中載入模型
loaded_model = load('xgb_model.pkl')
print("Model loaded successfully.")

file_path = './資料集.csv'
data = pd.read_csv(file_path)

# 去除列名空白
data.columns = data.columns.str.strip()

# 假設欄位為這些，若有需要請自行 rename
data.rename(columns={
    '日期': 'ds',
    'CPC': 'y',
    '日本': 'japan',
    '南韓': 'korea',
    '香港': 'hongkong',
    '新加坡': 'singapore',
    '上海': 'shanghai',
    '舟山': 'zhoushan'
}, inplace=True)

data['ds'] = pd.to_datetime(data['ds'])
data.sort_values(by='ds', inplace=True)  # 依日期排序

# 如果有缺失值，先補一下
data[['y','japan','korea','hongkong','singapore','shanghai','zhoushan']] = \
    data[['y','japan','korea','hongkong','singapore','shanghai','zhoushan']].ffill().bfill()

N_LAGS = 3

for i in range(1, N_LAGS+1):
    data[f'y_lag_{i}'] = data['y'].shift(i)

# 去掉有 NaN（因為 shift 後前幾筆沒值）
data.dropna(inplace=True)

print(data.tail(10))
# 測試預測
# predictions = loaded_model.predict(X[:5])
# print("Predictions:", predictions)
last_row = data.iloc[-1]  # 取最後一天記錄
X_future = pd.DataFrame([{
    'japan': last_row['japan'],  # 當天 or 預估的外生變數
    'korea': last_row['korea'],
    'hongkong': last_row['hongkong'],
    'singapore': last_row['singapore'],
    'shanghai': last_row['shanghai'],
    'zhoushan': last_row['zhoushan'],
    # 其他外生欄位...
    'y_lag_1': last_row['y'],       # 最後一天本身的 y 當 lag_1
    'y_lag_2': last_row['y_lag_1'], # 以此類推
    'y_lag_3': last_row['y_lag_2']
}])

# 用模型預測未來一天 (下一天) 的 CPC
next_day_pred = loaded_model.predict(X_future)[0]
print(X_future)

print(f"Predicted CPC for next day: {next_day_pred:.2f}")