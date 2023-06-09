import pandas as pd

import numpy as np


import seaborn as sns

import matplotlib.pyplot as plt

import os

from sklearn.preprocessing import LabelEncoder

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import StratifiedKFold

from sklearn.metrics import log_loss
datadir = '../input'

gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),

                      index_col='device_id')

gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),

                     index_col = 'device_id')

phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))

# Get rid of duplicate device ids in phone

phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')

events = pd.read_csv(os.path.join(datadir,'events.csv'),

                     parse_dates=['timestamp'], index_col='event_id')

appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), 

                        usecols=['event_id','app_id','is_active'],

                        dtype={'is_active':bool})

applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))
gatrain['trainrow'] = np.arange(gatrain.shape[0])

gatest['testrow'] = np.arange(gatest.shape[0])
brandencoder = LabelEncoder().fit(phone.phone_brand)

phone['brand'] = brandencoder.transform(phone['phone_brand'])

gatrain['brand'] = phone['brand']

gatest['brand'] = phone['brand']

Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 

                       (gatrain.trainrow, gatrain.brand)))

Xte_brand = csr_matrix((np.ones(gatest.shape[0]), 

                       (gatest.testrow, gatest.brand)))

print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))
m = phone.phone_brand.str.cat(phone.device_model)

modelencoder = LabelEncoder().fit(m)

phone['model'] = modelencoder.transform(m)

gatrain['model'] = phone['model']

gatest['model'] = phone['model']

Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), 

                       (gatrain.trainrow, gatrain.model)))

Xte_model = csr_matrix((np.ones(gatest.shape[0]), 

                       (gatest.testrow, gatest.model)))

print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))
appencoder = LabelEncoder().fit(appevents.app_id)

appevents['app'] = appencoder.transform(appevents.app_id)

napps = len(appencoder.classes_)

deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)

                       .groupby(['device_id','app'])['app'].agg(['size'])

                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)

                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)

                       .reset_index())

deviceapps.head()
d = deviceapps.dropna(subset=['trainrow'])

Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), 

                      shape=(gatrain.shape[0],napps))

d = deviceapps.dropna(subset=['testrow'])

Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)), 

                      shape=(gatest.shape[0],napps))

print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))
applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]

applabels['app'] = appencoder.transform(applabels.app_id)

labelencoder = LabelEncoder().fit(applabels.label_id)

applabels['label'] = labelencoder.transform(applabels.label_id)

nlabels = len(labelencoder.classes_)
devicelabels = (deviceapps[['device_id','app']]

                .merge(applabels[['app','label']])

                .groupby(['device_id','label'])['app'].agg(['size'])

                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)

                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)

                .reset_index())

devicelabels.head()
d = devicelabels.dropna(subset=['trainrow'])

Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), 

                      shape=(gatrain.shape[0],nlabels))

d = devicelabels.dropna(subset=['testrow'])

Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)), 

                      shape=(gatest.shape[0],nlabels))

print('Labels data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))
Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label), format='csr')

Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label), format='csr')

print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))
Xtrain
targetencoder = LabelEncoder().fit(gatrain.group)

y = targetencoder.transform(gatrain.group)

nclasses = len(targetencoder.classes_)
import xgboost as xgb
dtrain = xgb.DMatrix(Xtrain, y)
params = {

        "eta": 0.1,

        "booster": "gblinear",

        "objective": "multi:softprob",

        "alpha": 4,

        "lambda": 0,

        "silent": 1,

        "seed": 1233,

        "num_class": 12,

        "eval_metric": "mlogloss"

    }
xgb.cv(params, dtrain, 

       num_boost_round=50, 

       #early_stopping_rounds = 5, 

       maximize = False)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

#model.fit(Xtrain, y) hmm this is going to take too long.
model = xgb.train(params, dtrain, num_boost_round=25)
dtest = xgb.DMatrix(Xtest)
model.predict(dtest)

pred = pd.DataFrame(model.predict(dtest), index = gatest.index, columns=targetencoder.classes_)
pred.head()
pred.to_csv('xgb_subm.csv',index=True) 