# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from keras import regularizers

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

from datetime import tzinfo, timedelta, datetime

from sklearn import metrics

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler



from keras.layers.advanced_activations import PReLU

from keras.layers.core import Dense, Dropout, Activation

from keras.layers.normalization import BatchNormalization

from keras.models import Sequential

from keras.utils import np_utils

import csv



for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import zipfile

z1 = zipfile.ZipFile('../input/sf-crime/train.csv.zip')

z2 = zipfile.ZipFile('../input/sf-crime/test.csv.zip')

z1.extractall()

z2.extractall()

train_data = pd.read_csv(z1.open('train.csv'))

test_data = pd.read_csv(z2.open('test.csv'))
train_data.head()
def get_data(fn):

    data = []

    with open(fn) as f:

        reader = csv.DictReader(f)

        data = [row for row in reader]

    return data





def get_fields(data, fields):

    extracted = []

    for row in data:

        extract = []

        for field, f in sorted(fields.items()):

            info = f(row[field])

            if type(info) == list:

                extract.extend(info)

            else:

                extract.append(info)

        extracted.append(np.array(extract, dtype=np.float32))

    return extracted





def shuffle(X, y, seed=1337):

    np.random.seed(seed)

    shuffle = np.arange(len(y))

    np.random.shuffle(shuffle)

    X = X[shuffle]

    y = y[shuffle]

    return X, y





def preprocess_data(X, scaler=None):

    if not scaler:

        scaler = StandardScaler()

        scaler.fit(X)

    X = scaler.transform(X)

    return X, scaler





def dating(x):

    date, time = x.split(' ')

    y, m, d = map(int, date.split('-'))

    time = time.split(':')[:2]

    time = int(time[0]) * 60 + int(time[1])

    return [y, m, d, time]



days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

districts = ['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']

labels = 'ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS'.split(',')

data_fields = {

    'X': lambda x: float(x),

    'Y': lambda x: float(x),

    'DayOfWeek': lambda x: days.index(x) / float(len(days)),

    'Address': lambda x: [1 if 'block' in x.lower() else 0],

    'PdDistrict': lambda x: [1 if x == d else 0 for d in districts],

    'Dates': dating,

}
for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

label_fields = {'Category': lambda x: labels.index(x.replace(',', ''))}

print('Loading training data...')

raw_train = get_data('../working/train.csv')

print('Creating training data...')

X = np.array(get_fields(raw_train, data_fields), dtype=np.float32)

print('Creating training labels...')

y = np.array(get_fields(raw_train, label_fields))

del raw_train



X, y = shuffle(X, y)

X, scaler = preprocess_data(X)

Y = np_utils.to_categorical(y)



input_dim = X.shape[1]

output_dim = len(labels)

print('Input dimensions: {}'.format(input_dim))
def build_model(input_dim, output_dim, hn=64, dp=0.5, layers=3):

    model = Sequential()

    model.add(Dense(output_dim, input_shape=(input_dim,)))

    model.add(PReLU())

    

    model.add(Dense(output_dim, kernel_regularizer=regularizers.l2(0.01),

                activity_regularizer=regularizers.l1(0.01)))

    model.add(PReLU())



    model.add(Dense(output_dim, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='nadam')

    return model
from sklearn.model_selection import KFold

EPOCHS = 1

BATCHES = 128

HN = 64

RUN_FOLDS = True

nb_folds = 4

kfolds = KFold(nb_folds, random_state=None,  shuffle=False)

av_ll = 0.

f = 0

if RUN_FOLDS:

    for train, valid in kfolds.split(X):

        print('---' * 20)

        print('Fold', f)

        print('---' * 20)

        f += 1

        X_train = X[train]

        X_valid = X[valid]

        Y_train = Y[train]

        Y_valid = Y[valid]

        y_valid = y[valid]



        print("Building model...")

        model = build_model(input_dim, output_dim, HN)



        print("Training model...")



        model.fit(X_train, Y_train, nb_epoch=EPOCHS, batch_size=BATCHES, validation_data=(X_valid, Y_valid), verbose=0)

        valid_preds = model.predict_proba(X_valid)

        ll = metrics.log_loss(y_valid, valid_preds)

        print("LL:", ll)

        av_ll += ll

print('Average LL:', av_ll / nb_folds)
print("Generating submission...")



model = Sequential()

model.add(Dense(100, input_shape=(18,)))

model.add(PReLU())

    

model.add(Dense(100, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))

model.add(PReLU())



model.add(Dense(output_dim=39, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='nadam')



model.fit(X, Y, nb_epoch=EPOCHS, batch_size=BATCHES, verbose=0)



print('Loading testing data...')

raw_test = get_data('../working/test.csv')

print('Creating testing data...')

X_test = np.array(get_fields(raw_test, data_fields), dtype=np.float32)

del raw_test

X_test, _ = preprocess_data(X_test, scaler)



print('Predicting over testing data...')

preds = model.predict_proba(X_test, verbose=0)

valid_pred = model.predict(X, verbose=0)

predicted = []

for row in valid_pred:

    predicted.append(np.argmax(row))

validY = []

for row in Y:

    validY.append(np.argmax(row))
from sklearn import metrics

from sklearn.metrics import f1_score,confusion_matrix

#valid_preds = valid_preds[:, 0]

f1 = f1_score(validY, predicted, average='weighted')

print('F1 score = %f' % f1)
c = confusion_matrix(validY, predicted)

reverse_c = list(zip(*np.array(c)))

for i in range(len(c[1])):

    print(labels[i])

    fn = sum(c[i]) - c[i][i]

    fp = sum(reverse_c[i]) - c[i][i]

    print("Правильных результатов: " + str(c[i][i]))

    print("Ошибки первого рода: "+ str(fn))

    print("Ошибки второго рода: " + str(fp))