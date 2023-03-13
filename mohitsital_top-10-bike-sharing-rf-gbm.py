# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import matplotlib.pyplot as plt
dfs = {}

for name in ['train','test']:

    df = pd.read_csv('/kaggle/input/bike-sharing-demand/%s.csv' % name)

    df['_data'] = name

    dfs[name] = df
# combine train and test data into one df

df = dfs['train'].append(dfs['test'])



# lowercase column names

df.columns = map(str.lower, df.columns)
# logarithmic transformation of dependent cols

# (adding 1 first so that 0 values don't become -inf)

for col in ['casual', 'registered', 'count']:

    df['%s_log' % col] = np.log(df[col] + 1)
# parse datetime colum & add new time related columns

dt = pd.DatetimeIndex(df['datetime'])

df.set_index(dt, inplace=True)



df['date'] = dt.date

df['day'] = dt.day

df['month'] = dt.month

df['year'] = dt.year

df['hour'] = dt.hour

df['dow'] = dt.dayofweek

df['woy'] = dt.weekofyear
#def get_day(day_start):

#    day_end = day_start + pd.offsets.DateOffset(hours=23)

#    return pd.date_range(day_start, day_end, freq="H")



# tax day

#df.loc[get_day(pd.datetime(2011, 4, 15)), "workingday"] = 1

#df.loc[get_day(pd.datetime(2012, 4, 16)), "workingday"] = 1

# thanksgiving friday

#df.loc[get_day(pd.datetime(2011, 11, 25)), "workingday"] = 0

#df.loc[get_day(pd.datetime(2012, 11, 23)), "workingday"] = 0

# tax day

#df.loc[get_day(pd.datetime(2011, 4, 15)), "holiday"] = 0

#df.loc[get_day(pd.datetime(2012, 4, 16)), "holiday"] = 0



# thanksgiving friday

#df.loc[get_day(pd.datetime(2011, 11, 25)), "holiday"] = 1

#df.loc[get_day(pd.datetime(2012, 11, 23)), "holiday"] = 1



#storms

#df.loc[get_day(pd.datetime(2012, 5, 21)), "holiday"] = 1

#tornado

#df.loc[get_day(pd.datetime(2012, 6, 1)), "holiday"] = 1
df['peak'] = df[['hour', 'workingday']].apply(lambda x: (0, 1)[(x['workingday'] == 1 and  ( x['hour'] == 8 or 17 <= x['hour'] <= 18 or 12 <= x['hour'] <= 13)) or (x['workingday'] == 0 and  10 <= x['hour'] <= 19)], axis = 1)
#sandy

df['holiday'] = df[['month', 'day', 'holiday', 'year']].apply(lambda x: (x['holiday'], 1)[x['year'] == 2012 and x['month'] == 10 and (x['day'] in [30])], axis = 1)



#christmas day and others

df['holiday'] = df[['month', 'day', 'holiday']].apply(lambda x: (x['holiday'], 1)[x['month'] == 12 and (x['day'] in [24, 26, 31])], axis = 1)

df['workingday'] = df[['month', 'day', 'workingday']].apply(lambda x: (x['workingday'], 0)[x['month'] == 12 and x['day'] in [24, 31]], axis = 1)

df['ideal'] = df[['temp', 'windspeed']].apply(lambda x: (0, 1)[x['temp'] > 27 and x['windspeed'] < 30], axis = 1)

df['sticky'] = df[['humidity', 'workingday']].apply(lambda x: (0, 1)[x['workingday'] == 1 and x['humidity'] >= 60], axis = 1)
# #



def get_rmsle(y_pred, y_actual):

    diff = np.log(y_pred + 1) - np.log(y_actual + 1)

    mean_error = np.square(diff).mean()

    return np.sqrt(mean_error)





def get_data():

    data = df[df['_data'] == 'train'].copy()

    return data





def custom_train_test_split(data, cutoff_day=15):

    train = data[data['day'] <= cutoff_day]

    test = data[data['day'] > cutoff_day]



    return train, test





def prep_data(data, input_cols):

    X = data[input_cols]

    y_r = data['registered_log']

    y_c = data['casual_log']



    return X, y_r, y_c





def predict_on_validation_set(model, input_cols):

    data = get_data()



    train, test = custom_train_test_split(data)



    X_train, y_train_r, y_train_c = prep_data(train, input_cols)

    X_test, y_test_r, y_test_c = prep_data(test, input_cols)



    model_r = model.fit(X_train, y_train_r)

    y_pred_r = np.exp(model_r.predict(X_test)) - 1



    model_c = model.fit(X_train, y_train_c)

    y_pred_c = np.exp(model_c.predict(X_test)) - 1



    y_pred_comb = np.round(y_pred_r + y_pred_c)

    y_pred_comb[y_pred_comb < 0] = 0



    y_test_comb = np.exp(y_test_r) + np.exp(y_test_c) - 2



    score = get_rmsle(y_pred_comb, y_test_comb)

    return (y_pred_comb, y_test_comb, score)



df_test = df[df['_data'] == 'test'].copy()



# predict on test set & transform output back from log scale

def predict_on_test_set(model, x_cols):

    # prepare training set

    df_train = df[df['_data'] == 'train'].copy()

    X_train = df_train[x_cols]

    y_train_cas = df_train['casual_log']

    y_train_reg = df_train['registered_log']



    # prepare test set

    X_test = df_test[x_cols]



    casual_model = model.fit(X_train, y_train_cas)

    y_pred_cas = casual_model.predict(X_test)

    y_pred_cas = np.exp(y_pred_cas) - 1

    registered_model = model.fit(X_train, y_train_reg)

    y_pred_reg = registered_model.predict(X_test)

    y_pred_reg = np.exp(y_pred_reg) - 1

    # add casual & registered predictions together

    return y_pred_cas + y_pred_reg
# random forest model

params = {'n_estimators': 1000, 'max_depth': 15, 'random_state': 0, 'min_samples_split' : 5, 'n_jobs': -1}

rf_model = RandomForestRegressor(**params)

rf_cols = [

    'weather', 'temp', 'atemp', 'windspeed',

    'workingday', 'season', 'holiday', 'sticky',

    'hour', 'dow', 'woy', 'peak',

]

rf_p, rf_t, rf_score = predict_on_validation_set(rf_model, rf_cols)

print(rf_score)

# GBM model

params = {'n_estimators': 150, 'max_depth': 5, 'random_state': 0, 'min_samples_leaf' : 10, 'learning_rate': 0.1, 'subsample': 0.7, 'loss': 'ls'}

gbm_model = GradientBoostingRegressor(**params)

gbm_cols = [

    'weather', 'temp', 'atemp', 'humidity', 'windspeed',

    'holiday', 'workingday', 'season',

    'hour', 'dow', 'year', 'ideal'

    ]





(gbm_p, gbm_t, gbm_score) = predict_on_validation_set(gbm_model, gbm_cols)

print(gbm_score)



# the blend gives a better score on the leaderboard, even though it does not on the validation set

y_p = np.round(.2*rf_p + .8*gbm_p)

print(get_rmsle(y_p, rf_t))

# predctions on test dataset

rf_pred = predict_on_test_set(rf_model, rf_cols)

gbm_pred = predict_on_test_set(gbm_model, gbm_cols)

# taking weighted average of output from two models

y_pred = np.round(.20*rf_pred + .80*gbm_pred)

# output predictions for submission

df_test['count'] = y_pred

final_df = df_test[['datetime', 'count']].copy()

final_df.to_csv('output5.csv', index=False)