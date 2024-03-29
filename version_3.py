import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import math
import seaborn as sns
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
import scipy.stats as st
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")  # specify to ignore warning messages

sns.set(color_codes=True)

test = pd.read_csv('data/test.csv')
submission = pd.read_csv('data/submission.csv')


df_copy = test.copy()
df_copy.date = pd.to_datetime(df_copy.date)

# df_copy.date = pd.to_datetime(df_copy.date + " " + df_copy.time, format='%d/%m/%y %H:%M:%S')

df_copy.date = pd.to_datetime(df_copy.date.astype(str) + " " + df_copy.time, format='%Y-%m-%d %H:%M:%S')

# Remove negative values from the data set.


def reduce_noise_by_removing_neg_vals(df_copy):
    df_pos = df_copy[df_copy.amount > 0]
    df_neg = df_copy[df_copy.amount < 0]

    # exact_match = []
    # larger_match = []
    # no_match = []

    start = datetime.now()

    for nega_i in df_neg.to_records()[:]:
        store_i = nega_i[1]
        date_i = nega_i[2]
        card_i = nega_i[4]
        amt_i = nega_i[5]
        row_i = df_pos[df_pos.store_id == store_i]
        row_i = row_i[row_i.card_id == card_i]
        row_i = row_i[row_i.amount >= abs(amt_i)]
        row_i = row_i[row_i.date <= date_i]
        if len(row_i[row_i.amount == abs(amt_i)]) > 0:
            row_i = row_i[row_i.amount == abs(amt_i)]
            matched_row = row_i[row_i.date == max(row_i.date)]
            # df_pos.loc[matched_row.index, 'amount'] = 0
            df_pos = df_pos.loc[~df_pos.index.isin(matched_row.index), :]
        elif len(row_i[row_i.amount > abs(amt_i)]) > 0:
            matched_row = row_i[row_i.date == max(row_i.date)]
            df_pos.loc[matched_row.index, 'amount'] = matched_row.amount + amt_i
        # else:
        #     pass
            # no_match.append(nega_i)
    end = datetime.now()
    time_took = (end - start).seconds / 60

    print(round(time_took, 2))
    return df_pos

df_pos = reduce_noise_by_removing_neg_vals(df_copy)


def adf_test(y):
    # perform Augmented Dickey Fuller test
    print('Results of Augmented Dickey-Fuller test:')
    dftest = adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value ({})'.format(key)] = value
    print(dfoutput)


def ts_diagnostics(y, lags=None, title='', filename=''):
    '''
    Calculate acf, pacf, qq plot and Augmented Dickey Fuller test for a given time series
    '''
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # weekly moving averages (5 day window because of workdays)
    rolling_mean = pd.Series.rolling(y, window=2).mean()
    rolling_std = pd.Series.rolling(y, window=2).std()

    fig = plt.figure(figsize=(14, 12))
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    hist_ax = plt.subplot2grid(layout, (2, 1))

    # time series plot
    y.plot(ax=ts_ax)
    rolling_mean.plot(ax=ts_ax, color='crimson')
    rolling_std.plot(ax=ts_ax, color='darkslateblue')
    plt.legend(loc='best')
    ts_ax.set_title(title, fontsize=24)

    # acf and pacf
    plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)

    # qq plot
    sm.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('QQ Plot')

    # hist plot
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    plt.tight_layout()
    plt.show()

    # perform Augmented Dickey Fuller test
    print('Results of Dickey-Fuller test:')
    dftest = adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    return


df = df_pos.copy()
test_groupby_date_store = df.groupby(['date', 'store_id'])['amount', 'holyday'].sum()
test_groupby_date_store = test_groupby_date_store.reset_index()

test_groupby_date_store = test_groupby_date_store.set_index('date')
store_list = test_groupby_date_store.store_id.unique()

store_list.sort()


def get_optimal_params(y):
    # Define the p, d and q parameters to take any value between 0 and 1

    param_dict = {}
    for param in pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            )
            results = mod.fit()
            model = ARIMA(y, order=param)
            results_ARIMA = model.fit(disp=-1)
            results_ARIMA.summary()
            param_dict[results.aic] = param
        except:
            continue

    min_aic = min(param_dict.keys())
    optimal_params = param_dict[min_aic]
    return optimal_params


# Seperate Prams (28-14-7 model)
# 49.42492
#     expected_return_pct_lending = 0.13 * (100 + 16 + 6.8) / 365
#     expected_return_pct_lending = 0.13 * (100 + 16 + 12) / 365


# 49.44151
#     expected_return_pct_lending = 0.13 * (100 + 16 + 6.8) / 365
#     expected_return_pct_lending = 0.13 * (100 + 16 + 8) / 365

# 49.44993
#     expected_return_pct_lending = 0.13 * (100 + 16 + 6.8) / 365
#     expected_return_pct_lending = 0.13 * (100 + 16 + 6) / 365

# 49.47571
#     expected_return_pct_lending = 0.13 * (100 + 16 + 6.8) / 365
#     expected_return_pct_lending = 0.13 * (100 + 16) / 365


# 49.50232
#     expected_return_pct_lending = 0.13 * (100 + 16 + 6.8) / 365
#     expected_return_pct_lending = 0.13 * (100 + 10) / 365

# 49.52056
# expected_return_pct_lending = 0.13 * (100 + 16 + 6.8) / 365
# expected_return_pct_lending = 0.13 * (100 + 6) / 365

#49.53453
# 28: expected_return_pct_lending = 0.13 * (100 + 16 + 6.8) / 365 &
# 14: expected_return_pct_lending = 0.13 * (100 + 3) / 365 == 49.78721 + FCST for 7 period

# 49.53924
# 28: expected_return_pct_lending = 0.13 * (100 + 16 + 6.8) / 365 &
# 14: expected_return_pct_lending = 0.13 * (100 + 2) / 365 == 49.78721 + FCST for 7 period

# 49.78685
# 28: expected_return_pct_lending = 0.13 * (100 + 16 + 6.8) / 365 &
# 14: expected_return_pct_lending = 0.13 * (100 + 16 + 6.9) / 365 == 49.78721

# Best Param (28-14-7 model)
# 49.80907
# sampling_p = 28
# mean_period = 2 * 3 #14 * 2*3
#
# predic_len = math.floor(100 / sampling_p)
#
# expected_return_pct_lending = 0.13 * (100 + 16 + 6) / 365
# expected_loss_pct_lending = 1.00
# min_period = 6

# expected_return_pct_lending = 0.13 * (100 + 16 + 8) / 365 == 54.6889
# expected_return_pct_lending = 0.13 * (100 + 16 + 7) / 365 == 54.61

# expected_return_pct_lending = 0.13 * (100 + 16 + 6.9) / 365 == 54.60209
# expected_return_pct_lending = 0.13 * (100 + 16 + 6.85) / 365 == 54.59813

# expected_return_pct_lending = 0.13 * (100 + 16 + 6.8) / 365 == 49.78721
# expected_return_pct_lending = 0.13 * (100 + 16 + 6.6) / 365 == 49.79594
# expected_return_pct_lending = 0.13 * (100 + 16 + 6.5) / 365 == 49.80032
# expected_return_pct_lending = 0.13 * (100 + 16 + 6.3) / 365 == 49.80907
# expected_return_pct_lending = 0.13 * (100 + 16 + 6.2) / 365 == 49.81345
# expected_return_pct_lending = 0.13 * (100 + 16 + 6) / 365 == 49.82221
# expected_return_pct_lending = 0.13 * (100 + 16 + 5) / 365 == 49.86618


# 49.83673 (28-21-14-7 model)
# sampling_p = 28
# mean_period = 2 * 3 #14 * 2*3
#
# predic_len = math.floor(100 / sampling_p)
# expected_return_pct_lending = 0.13 * (100 + 16 + 7) / 365 == 54.62457
#expected_return_pct_lending = 0.13 * (100 + 16 + 6.8) / 365 == 49.80176
# expected_return_pct_lending = 0.13 * (100 + 16 + 6) / 365 == 49.83673
# expected_loss_pct_lending = 1.00
# min_period = 6


sampling_p = 28
mean_period = 2 * 3 #14 * 2*3

predic_len = math.floor(100 / sampling_p)

expected_return_pct_lending = 0.13 * (100 + 16 + 6.8) / 365
expected_loss_pct_lending = 1.00
optimal_prob = expected_loss_pct_lending / (expected_loss_pct_lending + expected_return_pct_lending)
optimal_z_score = st.norm.ppf(optimal_prob)

min_period = 6


max_pdq = 2
p = d = q = range(0, max_pdq)
pdq = list(itertools.product(p, d, q))


pdqs = dict()
print(optimal_prob)
print(optimal_z_score)
output_file_name_fmt = 'outputs/py_4arima_pos_sep_{optimal_p}-{sampling_period}_no_sales_prob&no mean{mean_period}&min_period {min_period}_pdq{max_pdq}.csv'
output_file_name = output_file_name_fmt.format(optimal_p=round(optimal_prob, 4),
                                               sampling_period=sampling_p,
                                               mean_period=mean_period,
                                               min_period=min_period,
                                               max_pdq=max_pdq)

submission_copy = submission.copy()


def arima_main(input_df, sampling_period_days, fcst_period):
    input_df = input_df[len(input_df) % sampling_period_days:].resample(str(sampling_period_days) + 'D').sum()
    prob_of_no_sales = len(input_df[(input_df.amount == 0) | (input_df.amount.isna())]) / len(input_df)
    ts_log = np.log(input_df.amount)
    ts_log = ts_log[~ts_log.isin([np.nan, np.inf, -np.inf])]

    if len(ts_log) < min_period:
        return None
    if sampling_period_days >= 28:
        expected_return_pct_lending = 0.13 * (100 + 16 + 6.8) / 365
    elif sampling_period_days >= 14:
        expected_return_pct_lending = 0.13 * (100 + 16 + 14) / 365
    else:
        expected_return_pct_lending = 0.13 * (100 + 16 + 6.8) / 365

    expected_loss_pct_lending = 1.00
    optimal_prob = expected_loss_pct_lending / (expected_loss_pct_lending + expected_return_pct_lending)
    optimal_z_score = st.norm.ppf(optimal_prob)

    optimal_params = get_optimal_params(ts_log)
    pdqs[store_i] = optimal_params

    model = ARIMA(ts_log, order=optimal_params)
    results_ARIMA = model.fit(disp=-1)
    fcst = results_ARIMA.forecast(fcst_period)

    fcst_means = fcst[0]
    fcst_stds = fcst[1]
    fcst_i = fcst_means - (fcst_stds * optimal_z_score)
    fcst_i = sum(map(lambda x: np.exp(x) if np.exp(x) > 0 else 0, fcst_i))
    prediction_i = fcst_i * (1 - prob_of_no_sales)
    return prediction_i


for store_i in store_list[:]:
    prediction_i = None
    test_df = test_groupby_date_store[test_groupby_date_store.store_id == store_i]
    test_df_daily = test_df.resample('D').sum()
    prediction_i = arima_main(test_df_daily, sampling_period_days=28, fcst_period=3)
    # if prediction_i is None:
    #     prediction_i = arima_main(test_df_daily, sampling_period_days=21, fcst_period=4)
    if prediction_i is None:
        prediction_i = arima_main(test_df_daily, sampling_period_days=14, fcst_period=7)
    if prediction_i is None:
        prediction_i = arima_main(test_df_daily, sampling_period_days=7, fcst_period=12)
    if prediction_i is None:
        test_df = test_df_daily[len(test_df_daily) % 14:].resample('14D').sum()

        prob_of_no_sales = len(test_df[(test_df.amount == 0) | (test_df.amount.isna())]) / len(test_df)
        ts_log = ts_log[~ts_log.isin([np.nan, np.inf, -np.inf])]
        ts_log_wkly = np.log(test_df.amount)

        estimated_amt = np.exp(ts_log_wkly.mean() - ts_log_wkly.std() * optimal_z_score) * (1 - prob_of_no_sales)
        prediction_i = estimated_amt * mean_period

    submission_copy.loc[submission_copy['store_id'] == store_i, 'total_sales'] = prediction_i

submission_copy.to_csv(output_file_name, index=False)

print(output_file_name)
