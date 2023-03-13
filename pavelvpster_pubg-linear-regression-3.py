import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore')

from copy import deepcopy

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import gc, sys
gc.enable()

import os
print(os.listdir("../input"))
# Thanks to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
#        else:
#            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
def take_part_of_data(df, part):
    
    match_ids = df['matchId'].unique()
    match_ids_part = np.random.choice(match_ids, int(part * len(match_ids)))
    
    df = df[df['matchId'].isin(match_ids_part)]
    
    del match_ids
    del match_ids_part
def add_new_features_1(df):
    
    # calculate total distance
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    
    # sum heals and boosts
    df['healsAndBoosts'] = df['heals'] + df['boosts']
    
    # headshot rate
    df['headshotKillsOverKills'] = df['headshotKills'] / df['kills']
    df['headshotKillsOverKills'].fillna(0, inplace=True)
    
    # kill streake rate
    df['killStreaksOverKills'] = df['killStreaks'] / df['kills']
    df['killStreaksOverKills'].fillna(0, inplace=True)
    
    # kills and assists
    df['killsAndAssists'] = df['kills'] + df['assists']
    
    # teamwork
    df['assistsAndRevives'] = df['assists'] + df['revives']
def add_new_features_2(df):
    
    # number of players joined
    df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
    
    # normalize features by number of players joined
    df['killsAndAssistsOverPlayersJoined'] = df['killsAndAssists'] * ((100 - df['playersJoined']) / 100 + 1)
    df['matchDurationOverPlayersJoined'] = df['matchDuration'] * ((100 - df['playersJoined']) / 100 + 1)
    df['damageDealtOverPlayersJoined'] = df['damageDealt'] * ((100 - df['playersJoined']) / 100 + 1)
def add_new_features_3(df):
    
    # total distance over kills and assists
    df['totalDistanceOverKillsAndAssists'] = df['totalDistance'] / df['killsAndAssists']
    df['totalDistanceOverKillsAndAssists'].fillna(0, inplace=True)
    df['totalDistanceOverKillsAndAssists'].replace(np.inf, 0, inplace=True)
    
    # total distance over heals and boosts
    df['totalDistanceOverHealsAndBoosts'] = df['totalDistance'] / df['healsAndBoosts']
    df['totalDistanceOverHealsAndBoosts'].fillna(0, inplace=True)
    df['totalDistanceOverHealsAndBoosts'].replace(np.inf, 0, inplace=True)
def add_new_features_4(df):
    
    df['headshotRate'] = df['kills'] / df['headshotKills']
    df['killStreakRate'] = df['killStreaks'] / df['kills']
    df['healsAndBoosts'] = df['heals'] + df['boosts']
    df['totalDistance'] = df['rideDistance'] + df['walkDistance'] + df['swimDistance']
    df['killPlaceOverMaxPlace'] = df['killPlace'] / df['maxPlace']
    df['headshotKillsOverKills'] = df['headshotKills'] / df['kills']
    df['distanceOverWeapons'] = df['totalDistance'] / df['weaponsAcquired']
    df['walkDistanceOverHeals'] = df['walkDistance'] / df['heals']
    df['walkDistanceOverKills'] = df['walkDistance'] / df['kills']
    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']
    df["skill"] = df['headshotKills'] + df['roadKills']
    
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    
    df.fillna(0, inplace=True)
def feature_engineering(df, is_train=True):
    
    # fix rank points
    df['rankPoints'] = np.where(df['rankPoints'] <= 0, 0, df['rankPoints'])
    
    features = list(df.columns)
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchDuration")
    features.remove("matchType")
    if 'winPlacePerc' in features:
        features.remove('winPlacePerc')
    
    y = None
    
    # average y for training dataset
    if is_train:
        y = df.groupby(['matchId','groupId'])['winPlacePerc'].agg('mean')
    elif 'winPlacePerc' in df.columns:
        y = df['winPlacePerc']
    
    # mean by match and group
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    if is_train:
        df_out = agg.reset_index()[['matchId','groupId']]
    else:
        df_out = df[['matchId','groupId']]
    
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    
    # max by match and group
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    # max by match and group
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    # number of players in group
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    
    # mean by match
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    # number of groups in match
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    # drop match id and group id
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)
    
    del agg, agg_rank
    
    return df_out, y
class Estimator(object):
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        raise NotImplementedException
    
    def predict(self, x):
        raise NotImplementedException
class ScikitLearnEstimator(Estimator):
    
    def __init__(self, estimator):
        self.estimator = estimator
    
    def fit(self, x_train, y_train, x_valid, y_valid):
        self.estimator.fit(x_train, y_train)
    
    def predict(self, x):
        return self.estimator.predict(x)
def fit_predict_step(estimator, x_train, y_train, train_idx, valid_idx, x_test, oof):
    
    # prepare train and validation data
    x_train_train = x_train[train_idx]
    y_train_train = y_train[train_idx]
    x_train_valid = x_train[valid_idx]
    y_train_valid = y_train[valid_idx]
    
    # fit estimator
    estimator.fit(x_train_train, y_train_train, x_train_valid, y_train_valid)
    
    # collect OOF
    oof_part = estimator.predict(x_train_valid)
    
    print('MAE:', mean_absolute_error(y_train_valid, oof_part))
    
    oof[valid_idx] = oof_part
    
    # make predictions for test data
    y_part = estimator.predict(x_test)
    
    return y_part
def fit_predict(estimator, x_train, y_train, x_test):
    
    oof = np.zeros(x_train.shape[0])
    
    y = np.zeros(x_test.shape[0])
    
    kf = KFold(n_splits=5, random_state=42)
    
    for train_idx, valid_idx in kf.split(x_train):
        
        y_part = fit_predict_step(estimator, x_train, y_train, train_idx, valid_idx, x_test, oof)
        
        # average predictions for test data
        y += y_part / kf.n_splits
    
    print('Final MAE:', mean_absolute_error(y_train, oof))
    
    return oof, y
def fit_step(estimator, x_train, y_train, train_idx, valid_idx, oof):
    
    # prepare train and validation data
    x_train_train = x_train[train_idx]
    y_train_train = y_train[train_idx]
    x_train_valid = x_train[valid_idx]
    y_train_valid = y_train[valid_idx]
    
    # fit estimator
    estimator.fit(x_train_train, y_train_train, x_train_valid, y_train_valid)
    
    # collect OOF
    oof_part = estimator.predict(x_train_valid)
    
    mae = mean_absolute_error(y_train_valid, oof_part)
    print('MAE:', mae)
    
    oof[valid_idx] = oof_part
    
    return estimator, mae
def fit(estimator, x_train, y_train):
    
    oof = np.zeros(x_train.shape[0])
    
    kf = KFold(n_splits=5, random_state=42)
    
    trained_estimators = []
    
    for train_idx, valid_idx in kf.split(x_train):
        
        e, mae = fit_step(estimator, x_train, y_train, train_idx, valid_idx, oof)
        
        trained_estimators.append(deepcopy(e))
    
    print('Final MAE:', mean_absolute_error(y_train, oof))
    
    return oof, trained_estimators
def predict(trained_estimators, x_test):
    
    y = np.zeros(x_test.shape[0])
    
    for estimator in trained_estimators:
        
        y_part = estimator.predict(x_test)
        
        # average predictions for test data
        y += y_part / len(trained_estimators)
    
    return y
def pipeline_fit(estimator, df_train, scaler=None):
    
    # add new features
    add_new_features_1(df_train)
    add_new_features_2(df_train)
    add_new_features_3(df_train)
    add_new_features_4(df_train)
    
    # feature engineering
    x_train, y_train = feature_engineering(df_train, is_train=True)
    x_train = reduce_mem_usage(x_train)
    gc.collect()
    
    # scale
    if not (scaler is None):
        scaler.fit(x_train)
        scaled_x_train = scaler.transform(x_train)
    else:
        scaled_x_train = x_train.values
    
    # fit
    oof, trained_estimators = fit(estimator, scaled_x_train, y_train.values)
    
    del x_train
    del scaled_x_train
    del y_train
    gc.collect()
    
    return oof, trained_estimators
def pipeline_predict(trained_estimators, df_test, scaler=None):
    
    # add new features
    add_new_features_1(df_test)
    add_new_features_2(df_test)
    add_new_features_3(df_test)
    add_new_features_4(df_test)
    
    # feature engineering
    x_test, _ = feature_engineering(df_test, is_train=False)
    x_test = reduce_mem_usage(x_test)
    gc.collect()
    
    # scale
    if not (scaler is None):
        scaled_x_test = scaler.transform(x_test)
    else:
        scaled_x_test = x_test.values
    
    # predict
    y = predict(trained_estimators, scaled_x_test)
    
    del x_test
    del scaled_x_test
    gc.collect()
    
    return y
df_train = pd.read_csv('../input/train_V2.csv', index_col='Id')
df_train.shape
df_train = reduce_mem_usage(df_train)
df_train.head().T
gc.collect()
df_train.drop(df_train[df_train['winPlacePerc'].isnull()].index, inplace=True)
from sklearn.linear_model import LinearRegression
matchTypes = df_train['matchType'].unique()
matchTypes

df_oof_group = df_train.groupby(['matchId', 'groupId']).first().reset_index()[['matchId','groupId']]
df_oof = pd.DataFrame(index=pd.MultiIndex.from_arrays(df_oof_group.values.T, names=['matchId', 'groupId']))
df_oof['oof'] = 0
del df_oof_group
gc.collect()

trained_estimators = {}
trained_scalers = {}

for matchType in matchTypes:
    print("----------", matchType, "----------")
    
    df_train_part = df_train[df_train['matchType'] == matchType]
    
    part_trained_scaler = StandardScaler()
    oof_part, part_trained_estimators = pipeline_fit(ScikitLearnEstimator(LinearRegression()), df_train_part, part_trained_scaler)
    
    df_oof_part_group = df_train_part.groupby(['matchId', 'groupId']).first().reset_index()[['matchId','groupId']]
    df_oof_part = pd.DataFrame(index=pd.MultiIndex.from_arrays(df_oof_part_group.values.T, names=['matchId', 'groupId']))
    df_oof_part['oof'] = oof_part
    df_oof.update(df_oof_part)
    del df_oof_part_group
    
    trained_estimators[matchType] = part_trained_estimators
    trained_scalers[matchType] = part_trained_scaler

oof = df_oof['oof'].values
del df_train
del df_oof

gc.collect()
df_test = pd.read_csv('../input/test_V2.csv', index_col = 'Id')
df_test.shape
df_test = reduce_mem_usage(df_test)
df_test_id = pd.DataFrame(index=df_test.index)
gc.collect()

df_y = pd.DataFrame(index=df_test.index)
df_y['y'] = 0

for matchType in matchTypes:
    print("----------", matchType, "----------")
    
    df_test_part = df_test[df_test['matchType'] == matchType]
    
    y_part = pipeline_predict(trained_estimators[matchType], df_test_part, trained_scalers[matchType])
    
    df_y_part = pd.DataFrame(index=df_test_part.index)
    df_y_part['y'] = y_part
    df_y.update(df_y_part)

y = df_y['y'].values
del df_test
del df_y

gc.collect()
df_oof = pd.DataFrame()
df_oof['linear_oof'] = oof
df_oof.to_csv('linear_oof.csv', index_label='id')
df_submission = pd.DataFrame(index=df_test_id.index)
df_submission['winPlacePerc'] = y
df_submission.to_csv('linear_raw.csv', index_label='Id')
df_test = pd.read_csv('../input/test_V2.csv')
df_test.shape
df_submission = df_submission.merge(df_test[['Id', 'matchId', 'groupId', 'maxPlace', 'numGroups']], on='Id', how='left')
df_submission.head()
df_submission_group = df_submission.groupby(['matchId', 'groupId']).first().reset_index()

df_submission_group['rank'] = df_submission_group.groupby(['matchId'])['winPlacePerc'].rank()

df_submission_group = df_submission_group.merge(df_submission_group.groupby('matchId')['rank'].max().to_frame('max_rank').reset_index(), on='matchId', how='left')

df_submission_group['adjusted_perc'] = (df_submission_group['rank'] - 1) / (df_submission_group['numGroups'] - 1)

df_submission = df_submission.merge(df_submission_group[['adjusted_perc', 'matchId', 'groupId']], on=['matchId', 'groupId'], how='left')

df_submission['winPlacePerc'] = df_submission['adjusted_perc']

df_submission.head()
df_submission.loc[df_submission.maxPlace == 0, 'winPlacePerc'] = 0
df_submission.loc[df_submission.maxPlace == 1, 'winPlacePerc'] = 1
# Thanks to https://www.kaggle.com/anycode/simple-nn-baseline-4
t = df_submission.loc[df_submission.maxPlace > 1]
gap = 1.0 / (t.maxPlace.values - 1)
fixed_perc = np.around(t.winPlacePerc.values / gap) * gap
df_submission.loc[df_submission.maxPlace > 1, 'winPlacePerc'] = fixed_perc
df_submission.loc[(df_submission.maxPlace > 1) & (df_submission.numGroups == 1), 'winPlacePerc'] = 0

assert df_submission['winPlacePerc'].isnull().sum() == 0
df_submission[['Id', 'winPlacePerc']].to_csv('linear_adjusted.csv', index=False)