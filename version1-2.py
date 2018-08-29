import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima_model import ARIMA
import math
import sys
import os

os.makedirs('outputs/', exist_ok=True)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submission = pd.read_csv('data/submission.csv')

# Best: 1.465
pct = 1.465 #v1-3

df = test.copy()
df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
test_groupby_date_store = df.groupby(['date', 'store_id'])['amount','holyday'].sum()
test_groupby_date_store = test_groupby_date_store.reset_index()

test_groupby_date_store = test_groupby_date_store.set_index('date')
store_list = test_groupby_date_store.store_id.unique()

store_list.sort()


for store_i in store_list[:]:
    test_df = test_groupby_date_store[test_groupby_date_store.store_id == store_i]
    test_df = test_df.resample('D').sum()
    test_df.holyday = test_df.holyday.map(lambda x: x if x <= 1 else 1)
    test_df = test_df.resample('M').sum()
    prediction_i = sum(test_df.amount) / len(test_df.amount) * pct
    submission.loc[submission['store_id'] == store_i, 'total_sales'] = prediction_i

submission.to_csv('outputs/submission_{}.csv'.format(pct), index=False)