import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima_model import ARIMA
import math
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
import os

os.makedirs('outputs/', exist_ok=True)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submission = pd.read_csv('data/submission.csv')


df = test.copy()
df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
test_groupby_date_store = df.groupby(['date', 'store_id'])['amount','holyday'].sum()
test_groupby_date_store = test_groupby_date_store.reset_index()

test_groupby_date_store = test_groupby_date_store.set_index('date')
store_list = test_groupby_date_store.store_id.unique()

store_list.sort()


p = d = q = range(0, 3)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
warnings.filterwarnings("ignore") # specify to ignore warning messages


def get_optimal_params(y):
# Define the p, d and q parameters to take any value between 0 and 2
  
    param_dict = {}
    for param in pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                           )
            results = mod.fit()
            model = ARIMA(y, order=param)
            results_ARIMA = model.fit(disp=-1)
            param_dict[results.aic] = param
        except:
            continue

    min_aic = min(param_dict.keys())
    optimal_params = param_dict[min_aic]
#     print("ARIMA{} - AIC: {}".format(optimal_params, min_aic))
    return optimal_params


    #submission_copy_backup 

for store_i in store_list[:]:
    test_df = test_groupby_date_store[test_groupby_date_store.store_id == store_i]
    test_df = test_df.resample('D').sum()
    try:
        test_df = test_df[len(test_df)%28:].resample('28D').sum()
        ts_log = np.log(test_df.amount)
        optimal_params = get_optimal_params(ts_log)

        model = ARIMA(ts_log, order=optimal_params)
        results_ARIMA = model.fit(disp=-1)
        predic_len = 3

        ts = max(test_df.index) #+ timedelta(days=28)
    #         ts = max(test_df.index) + 1
        ts_end = ts + timedelta(days=predic_len*28)

    #    predictions_ARIMA_log = results_ARIMA.predict(start=ts, end=ts_end)#.map(lambda x: np.exp(x))# * math.e
        fcst = results_ARIMA.forecast(3)
        fcst_i = sum(map(lambda x: np.exp(x) if np.exp(x) > 0 else 0, fcst[0]))
        min_conf = (list(map(lambda x: np.exp(x)  if np.exp(x) > 0 else 0, fcst[2][0]))[0] 
                    +list(map(lambda x: np.exp(x) if np.exp(x) > 0 else 0, fcst[2][1]))[0]
                    +list(map(lambda x: np.exp(x) if np.exp(x) > 0 else 0, fcst[2][2]))[0])

        prediction_i = min_conf * 0.7
    except Exception as e:
        print(e)
        df_len = len(test_df)
        if df_len >= 28:
            temp_df = test_df.iloc[-28:]
            prediction_i = sum(temp_df.amount) / 28 * 100 * 0.5
        else:
             prediction_i = sum(test_df.amount) / df_len * 100 * 0.3
#     print(prediction_i)
    submission.loc[submission['store_id'] == store_i, 'total_sales'] = prediction_i


submission.to_csv('outputs/submission_v2.csv', index=False)