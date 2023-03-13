# Parameters

N_FOLDS = 3

MAX_BOOST_ROUNDS = 700

LEARNING_RATE = .0022

import numpy as np

import pandas as pd

import lightgbm as lgb

import gc
# Load data

train = pd.read_csv('../input/train_2016_v2.csv')

prop = pd.read_csv('../input/properties_2016.csv')
# Process data



for c, dtype in zip(prop.columns, prop.dtypes):	

    if dtype == np.float64:		

        prop[c] = prop[c].astype(np.float32)



df_train = train.merge(prop, how='left', on='parcelid')



x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)

y_train = df_train['logerror'].values

print(x_train.shape, y_train.shape)



train_columns = x_train.columns



for c in x_train.dtypes[x_train.dtypes == object].index.values:

    x_train[c] = (x_train[c] == True)



del df_train; gc.collect()

# Prepare for LightGBM



x_train = x_train.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)



params = {}

params['max_bin'] = 10

params['learning_rate'] = LEARNING_RATE # shrinkage_rate

params['boosting_type'] = 'gbdt'

params['objective'] = 'regression'

params['metric'] = 'l1'          # or 'mae'

params['sub_feature'] = 0.50      # feature_fraction 

params['bagging_fraction'] = 0.85 # sub_row

params['bagging_freq'] = 40

params['num_leaves'] = 512        # num_leaf

params['min_data'] = 500         # min_data_in_leaf

params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

params['verbose'] = 0

# Cross-validate

cv_results = lgb.cv(params, d_train, num_boost_round=MAX_BOOST_ROUNDS, nfold=N_FOLDS, 

                    verbose_eval=20, early_stopping_rounds=40)
# Display results

print('Current parameters:\n', params)

print('\nBest num_boost_round:', len(cv_results['l1-mean']))

print('Best CV score:', cv_results['l1-mean'][-1])