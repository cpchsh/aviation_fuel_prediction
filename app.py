from flask import Flask, render_template, redirect, url_for, request
import joblib
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = "anyrandomsecret"  # 若要使用flash或session需設

#======================#
#   1) 載入 XGB 模型   #
#======================#
# 假設你已在其他地方用 joblib.dump(xgb_model, "xgb_model.pkl")
# 並把檔案放在專案根目錄
xgb_model_path = "./xgb_model.pkl"
xgb_model = joblib.load(xgb_model_path)


#======================#
#   2) 預測下一天CPC (XGB) #
#======================#
def predict_next_day_xgb(xgb_model):
    """
    用XGB預測下一天
    具體做法:
     - 從資料庫/檔案中抓最新特徵(含lag等)
     - xgb_model.predict([features]) -> y_pred
    """
    # 假設 features = [ .. ]，維度要跟訓練時一致
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

    #print(data.tail(10))
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
    y_pred = xgb_model.predict(X_future)[0]
    return y_pred


#======================#
#   3) 首頁(顯示預測)   #
#======================#
@app.route("/")
def index():
    # 1) 用 XGB 預測下一天
    next_day_pred = predict_next_day_xgb(xgb_model)
    next_day_pred_str = f"{next_day_pred:.2f}"  # 格式化
    
    # 1) 讀取預測結果 CSV (latest_forecast.csv)
    if os.path.exists("./latest_forecast.csv"):
        forecast = pd.read_csv("latest_forecast.csv")
        # 取最後 7 天的預測
        future_part = forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        #print("future_part",future_part)
        value = future_part.iloc[0,1]
        f_value = f"{value:.2f}"
        f_day = future_part.iloc[0,0]
        future_table_html = future_part.to_html(index=False)
    else:
        f_day = "N/A"
        f_value="N/A"
        future_table_html = "<p>No forecast data found. Please check if the offline script has run.</p>"
    
    # 2)圖檔檢查
    plot_full_url = "static/plot_full.png" if os.path.exists("./static/plot_full.png") else ""
    plot_recent_url = "static/plot_recent_future.png" if os.path.exists("./static/plot_recent_future.png") else ""
    
    return render_template("index.html",
                           next_day=f_day,
                           xgb_pred=next_day_pred_str,
                           prophet_fvalue=f_value,
                           future_table_html=future_table_html,
                           plot_full_url=plot_full_url,
                           plot_recent_url=plot_recent_url)
    
#======================#
#   4) 重新訓練Prophet  #
#======================#
@app.route("/train", methods=["POST"])
def train_prophet():
    """
    用 subprocess 執行 train_prophet.py
    成功後 redirect 回首頁
    """
    import subprocess  # ← 用於在Python程式中執行外部指令，如 train_prophet.py
    import sys
    # 執行外部指令: python train_prophet.py
    subprocess.run([sys.executable, "train_prophet.py"], check=True)
    # 執行完後回到首頁
    return redirect(url_for('index'))

#==========================#
#  /append -> append資料  #
#==========================#
@app.route("/append", methods=["POST"])
def append_data():
    """
    1) 讀取 collectedata/tempdata.csv
    2) 與 資料集.csv 做比對：
       - 若 tempdata.csv 第一筆 ds > 資料集.csv最後一筆 ds => append
       - 否則不動
    3) 給一個提示頁(3秒)表示成功or無更新，然後回到 "/"
    """
    # 1) 讀檔
    temp_path = "./collectedata/tempdata.csv" # path
    if not os.path.exists(temp_path):
        return render_template("append_result.html",
                               message="tempdata.csv not found",
                               sec=3)
    temp_df = pd.read_csv(temp_path, header=None)
    # 假設temp_df結構: 2024/11/4, 597.0,595.0,596.0,581.0,593.0,590.0,585.0
    # => columns=[0,1,2,3,4,5,6,7], 0=日期, 1=日本, 2=韓國, 3=香港, ...
    temp_df.columns = ['日期','日本','南韓','香港','新加坡','上海','舟山','CPC']

    # 2) 讀取原資料集
    data_path = './資料集.csv'
    if not os.path.exists(data_path):
        return render_template("append_result.html",
                               message="資料集.csv not found!",
                               sec=3)
    data_df = pd.read_csv(data_path)
    data_df.columns = data_df.columns.str.strip()

    # 轉datetime
    temp_df['日期'] = pd.to_datetime(temp_df['日期'])
    data_df['日期'] = pd.to_datetime(data_df['日期'])
    temp_df.sort_values('日期', inplace=True)
    data_df.sort_values('日期', inplace=True)

    last_date = data_df['日期'].max() #資料集中最新日期
    first_temp_date = temp_df.iloc[0]['日期'] # tempdata第一筆日期

    if first_temp_date > last_date:
        # 代表有新的 => append
        # 這裡就直接把temp_df整個加進去
        appended_df = pd.concat([data_df, temp_df], ignore_index = True)
        appended_df.sort_values('日期', inplace=True)
        # 重新輸出
        appended_df.to_csv(data_path, index=False)
        msg = f"New data appended successfully! (First day={first_temp_date.date()})"
    else:
        # 沒更新
        msg = f"No update. (tempdata first day={first_temp_date.date()} not > last data day={last_date.date()})"
    
    # 顯示提示頁，3秒後回首頁
    return render_template("append_result.html",
                           message=msg,
                           sec=3)

#==========================#
#   6) 自訂 XGB 輸入 form   #
#==========================#
@app.route("/xgb_form", methods=["GET"])
def xgb_input_form():
    """
    顯示一個表單，讓使用者輸入:
      - japan, korea, hongkong, singapore, shanghai, zhoushan
    """
    return render_template("xgb_form.html")

@app.route("/xgb_predict", methods = ["POST"])
def xgb_predict_custom():
    """
    接收表單數值:
    'japan','korea','hongkong','singapore','shanghai','zhoushan'
    y_lag1, y_lag2, y_lag3以"簡易以japan的值+ -5處理"
    然後做 xgb_model.predict
    """
    # 從表單獲得輸入
    japan=float(request.form.get("japan", 0))
    korea=float(request.form.get("korea", 0))
    hongkong=float(request.form.get("hongkong", 0))
    singapore=float(request.form.get("singapore", 0))
    shanghai=float(request.form.get("shanghai", 0))
    zhoushan=float(request.form.get("zhoushan", 0))

    # simple rule to deal y_lag1 ~y_lag3
    y_lag_1 = japan + 5
    y_lag_2 = japan
    y_lag_3 = japan - 5

    # feature
    X_custom = pd.DataFrame([{
        'japan':     japan,
        'korea':     korea,
        'hongkong':  hongkong,
        'singapore': singapore,
        'shanghai':  shanghai,
        'zhoushan':  zhoushan,
        'y_lag_1':   y_lag_1,
        'y_lag_2':   y_lag_2,
        'y_lag_3':   y_lag_3
    }])

    # 預測
    pred_val = xgb_model.predict(X_custom)[0]
    result_str = f"{pred_val:.2f}"

    # 顯示結果頁 or redirect
    return render_template("xgb_result.html", 
                           result_val=result_str,
                           j=japan,k=korea,h=hongkong,
                           s=singapore,sh=shanghai,zh=zhoushan)


if __name__ == "__main__":
    app.run(debug=True)  # 除錯用
    #app.run(host="0.0.0.0", port=5000)
