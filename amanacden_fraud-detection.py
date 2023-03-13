# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_identity_data = pd.read_csv('../input/train_identity.csv')

train_transaction_data = pd.read_csv('../input/train_transaction.csv')
pd.set_option('display.max_columns', 500)

train_transaction_data.head(5)
train_identity_data.head(5)
# https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature

train_transaction_data['Transaction_dow'] = np.floor((train_transaction_data['TransactionDT'] / (3600 * 24) - 1) % 7)

train_transaction_data['Transaction_hour'] = np.floor(train_transaction_data['TransactionDT'] / 3600) % 24
na_columns = train_identity_data.isna().sum()

print(na_columns[na_columns==0])
import seaborn as sns

import matplotlib.pyplot as plt

sns.distplot(train_identity_data['id_01'])
# number of nan values in each column

pd.set_option('display.max_rows', 500)

na_columns = train_transaction_data.isna().sum()

print(na_columns[na_columns==0])

transaction_data_columns = train_transaction_data.columns

numericCols = train_transaction_data._get_numeric_data().columns
categoricalCols = list(set(transaction_data_columns) - set(numericCols))

print('The categorical columns in transaction data are: ',categoricalCols)
train_transaction_data[categoricalCols] = train_transaction_data[categoricalCols].replace({ np.nan:'missing'})

train_transaction_data[numericCols] = train_transaction_data[numericCols].replace({ np.nan:-1})
sns.countplot(train_transaction_data['isFraud'])
sns.countplot(train_transaction_data['card6'])
cardTypes = ['credit','debit','debit or credit','charge card']

for i,i_card in enumerate(cardTypes):

    cardData = eval('train_transaction_data.loc[train_transaction_data["card6"]=="'+i_card+'"]')

    plt.figure(i)

    sns.countplot(cardData['isFraud']).set_title(i_card)
sns.countplot(train_transaction_data['card4'])
cardTypes = ['discover','mastercard','visa','american express','missing']

for i,i_card in enumerate(cardTypes):

    cardData = eval('train_transaction_data.loc[train_transaction_data["card4"]=="'+i_card+'"]')

    plt.figure(i)

    sns.countplot(cardData['isFraud']).set_title(i_card)
print('The average transaction amount for non fraudulent transactions is: ', 

      np.mean(train_transaction_data.loc[train_transaction_data["isFraud"]==0]['TransactionAmt']))

print('The average transaction amount for fraudulent transactions is: ', 

      np.mean(train_transaction_data.loc[train_transaction_data["isFraud"]==1]['TransactionAmt']))
print('The Maximum transaction amount for non fraudulent transactions is: ', 

      np.max(train_transaction_data.loc[train_transaction_data["isFraud"]==0]['TransactionAmt']))

print('The Maximum transaction amount for fraudulent transactions is: ', 

      np.max(train_transaction_data.loc[train_transaction_data["isFraud"]==1]['TransactionAmt']))
print('The Minimum transaction amount for non fraudulent transactions is: ', 

      np.min(train_transaction_data.loc[train_transaction_data["isFraud"]==0]['TransactionAmt']))

print('The Minimum transaction amount for fraudulent transactions is: ', 

      np.min(train_transaction_data.loc[train_transaction_data["isFraud"]==1]['TransactionAmt']))
maxCardData = {}

minCardData = {}

meanCardData = {}

for i,i_card in enumerate(cardTypes):

    cardData = eval('train_transaction_data.loc[train_transaction_data["card4"]=="'+i_card+'"]')

    maxCardData[i_card] = np.max(cardData['TransactionAmt'])

    minCardData[i_card] = np.min(cardData['TransactionAmt'])

    meanCardData[i_card] = np.mean(cardData['TransactionAmt'])
print('The maximum transactions by card are:', maxCardData)

print('The minimum transactions by card are:', minCardData)

print('The average transactions by card are:', meanCardData)
normalDataTransaction = train_transaction_data.loc[train_transaction_data["isFraud"]==0]

normalDataTransaction.TransactionAmt.describe()
fraudDataTransaction = train_transaction_data.loc[train_transaction_data["isFraud"]==1]

fraudDataTransaction.TransactionAmt.describe()
prodTypes = train_transaction_data['ProductCD'].unique()

for i,i_prod in enumerate(prodTypes):

    productData = eval('train_transaction_data.loc[train_transaction_data["ProductCD"]=="'+i_prod+'"]')

    plt.figure(i)

    ax = sns.barplot(x="isFraud", y="isFraud", data=productData, estimator=lambda x: len(x) / len(productData) * 100)

    ax.set(ylabel="Percent")

    ax.set_title(i_prod)
del fraudDataTransaction,normalDataTransaction,productData,cardData
print(train_identity_data.columns)
identity_data_columns = train_identity_data.columns

numericCols = train_identity_data._get_numeric_data().columns

categoricalCols = list(set(identity_data_columns) - set(numericCols))

print('The categorical columns in identity data are: ',categoricalCols)

train_identity_data[categoricalCols] = train_identity_data[categoricalCols].replace({ np.nan:'missing'})

train_identity_data[numericCols] = train_identity_data[numericCols].replace({ np.nan:-1})
np.unique(train_identity_data['id_30'])
#collapse os types - id_30

train_identity_data.loc[train_identity_data['id_30'].str.contains('Mac', na=False), 'id_30'] = 'mac'

train_identity_data.loc[train_identity_data['id_30'].str.contains('iOS', na=False), 'id_30'] = 'iOS'

train_identity_data.loc[train_identity_data['id_30'].str.contains('Android', na=False), 'id_30'] = 'android'

train_identity_data.loc[train_identity_data['id_30'].str.contains('Windows', na=False), 'id_30'] = 'Windows'

train_identity_data.loc[train_identity_data['id_30'].str.contains('Linux', na=False), 'id_30'] = 'Linux'
train_identity_data['device_name'] = train_identity_data['DeviceInfo'].str.split('/', expand=True)[0]



train_identity_data.loc[train_identity_data['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

train_identity_data.loc[train_identity_data['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

train_identity_data.loc[train_identity_data['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

train_identity_data.loc[train_identity_data['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

train_identity_data.loc[train_identity_data['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

train_identity_data.loc[train_identity_data['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

train_identity_data.loc[train_identity_data['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

train_identity_data.loc[train_identity_data['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

train_identity_data.loc[train_identity_data['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

train_identity_data.loc[train_identity_data['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

train_identity_data.loc[train_identity_data['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

train_identity_data.loc[train_identity_data['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

train_identity_data.loc[train_identity_data['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

train_identity_data.loc[train_identity_data['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

train_identity_data.loc[train_identity_data['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

train_identity_data.loc[train_identity_data['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

train_identity_data.loc[train_identity_data['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'



train_identity_data.loc[train_identity_data.device_name.isin(train_identity_data.device_name.value_counts()[train_identity_data.device_name.value_counts() < 200].index), 'device_name'] = "Others"
np.unique(train_identity_data['id_30'])
print(np.unique(train_identity_data['id_31']))

train_identity_data['id_31'] = train_identity_data['id_31'].str.replace('\d+', '')

print(np.unique(train_identity_data['id_31']))
sns.countplot(train_identity_data['DeviceType'])
raw_train_data = pd.merge(train_transaction_data, train_identity_data, on='TransactionID', how='left')
del train_transaction_data,train_identity_data
raw_train_data_columns = raw_train_data.columns

numericCols = raw_train_data._get_numeric_data().columns

categoricalCols = list(set(raw_train_data_columns) - set(numericCols))

print('The categorical columns in training data are: ',categoricalCols)

raw_train_data[categoricalCols] = raw_train_data[categoricalCols].replace({ np.nan:'missing'})

raw_train_data[numericCols] = raw_train_data[numericCols].replace({ np.nan:-1})
print(raw_train_data.head(10))
deviceTypes = raw_train_data['DeviceType'].unique()

for i,i_device in enumerate(deviceTypes):

    deviceData = eval('raw_train_data.loc[raw_train_data["DeviceType"]=="'+i_device+'"]')

    plt.figure(i)

    sns.countplot(deviceData['isFraud']).set_title(i_device)
variables = list(numericCols)

variables.remove('isFraud')
correlationMatrix = raw_train_data.loc[:, variables].corr().abs()

#correlationMatrix = raw_train_data.loc[:, variables].corr()
plt.figure(figsize=(20,20))

heat = sns.heatmap(data=correlationMatrix)

plt.title('Heatmap of Correlation')
na_vals = np.sum(raw_train_data.loc[:,variables]==-1)/raw_train_data.shape[0]

goodNumericVars = []

for i_var in variables:    

    if na_vals[i_var] < 0.85:        

        goodNumericVars.append(i_var)

goodNumericVars.remove('TransactionDT')

goodNumericVars.remove('TransactionID')
corrThresh = 0.9

# Select upper triangle of correlation matrix

upper = correlationMatrix.where(np.triu(np.ones(correlationMatrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > corrThresh)]
to_drop.remove('TransactionDT')
for i_var in to_drop:

    if i_var in goodNumericVars:

        goodNumericVars.remove(i_var)
del to_drop,corrThresh,upper,correlationMatrix,na_vals
variables = list(categoricalCols)

na_vals = np.sum(raw_train_data.loc[:,variables]=='missing')/raw_train_data.shape[0]

goodCategoricalVars = []

for i_var in variables:    

    if na_vals[i_var] < 0.85:        

        goodCategoricalVars.append(i_var)
featuresToUse = goodNumericVars+goodCategoricalVars

train_data = raw_train_data.loc[:,featuresToUse]

target_data = raw_train_data['isFraud']
featuresToUse
#https://www.kaggle.com/artgor/eda-and-models#Feature-engineering

train_data['TransactionAmt_to_mean_card1'] = train_data['TransactionAmt'] / train_data.groupby(['card1'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_card4'] = train_data['TransactionAmt'] / train_data.groupby(['card4'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_card5'] = train_data['TransactionAmt'] / train_data.groupby(['card5'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_addr1'] = train_data['TransactionAmt'] / train_data.groupby(['addr1'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_id31'] = train_data['TransactionAmt'] / train_data.groupby(['id_31'])['TransactionAmt'].transform('mean')

train_data['card1_card2'] = train_data['card1'].astype(str) + '_' + train_data['card2'].astype(str)

train_data['addr1_dist1'] = train_data['addr1'].astype(str) + '_' + train_data['dist1'].astype(str)

train_data['card1_addr1'] = train_data['card1'].astype(str) + '_' + train_data['addr1'].astype(str)

train_data['card1_addr2'] = train_data['card1'].astype(str) + '_' + train_data['addr2'].astype(str)

train_data['card2_addr1'] = train_data['card2'].astype(str) + '_' + train_data['addr1'].astype(str)

train_data['card2_addr2'] = train_data['card2'].astype(str) + '_' + train_data['addr2'].astype(str)

train_data['card4_addr1'] = train_data['card4'].astype(str) + '_' + train_data['addr1'].astype(str)

train_data['card4_addr2'] = train_data['card4'].astype(str) + '_' + train_data['addr2'].astype(str)

train_data['DeviceInfo_P_emaildomain'] = train_data['DeviceInfo'].astype(str) + '_' + train_data['P_emaildomain'].astype(str)

train_data['P_emaildomain_addr1'] = train_data['P_emaildomain'].astype(str) + '_' + train_data['addr1'].astype(str)

train_data['id01_addr1'] = train_data['id_01'].astype(str) + '_' + train_data['addr1'].astype(str)

train_data['TransactionAmt_to_std_card1'] = train_data['TransactionAmt'] / train_data.groupby(['card1'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_card4'] = train_data['TransactionAmt'] / train_data.groupby(['card4'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_card5'] = train_data['TransactionAmt'] / train_data.groupby(['card5'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_addr1'] = train_data['TransactionAmt'] / train_data.groupby(['addr1'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_id31'] = train_data['TransactionAmt'] / train_data.groupby(['id_31'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_decimal'] = ((train_data['TransactionAmt'] - train_data['TransactionAmt'].astype(int)) * 1000).astype(int)
scale_pos_weight = np.sqrt(len(target_data.loc[target_data==0])/len(target_data.loc[target_data==1]))
del raw_train_data
test_identity_data = pd.read_csv('../input/test_identity.csv')

test_transaction_data = pd.read_csv('../input/test_transaction.csv')
# https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature

test_transaction_data['Transaction_dow'] = np.floor((test_transaction_data['TransactionDT'] / (3600 * 24) - 1) % 7)

test_transaction_data['Transaction_hour'] = np.floor(test_transaction_data['TransactionDT'] / 3600) % 24

transaction_data_columns = test_transaction_data.columns

numericCols = test_transaction_data._get_numeric_data().columns

categoricalCols = list(set(transaction_data_columns) - set(numericCols))

test_transaction_data[categoricalCols] = test_transaction_data[categoricalCols].replace({ np.nan:'missing'})

test_transaction_data[numericCols] = test_transaction_data[numericCols].replace({ np.nan:-1})
identity_data_columns = test_identity_data.columns

numericCols = test_identity_data._get_numeric_data().columns

categoricalCols = list(set(identity_data_columns) - set(numericCols))

test_identity_data[categoricalCols] = test_identity_data[categoricalCols].replace({ np.nan:'missing'})

test_identity_data[numericCols] = test_identity_data[numericCols].replace({ np.nan:-1})

test_identity_data['id_31'] = test_identity_data['id_31'].str.replace('\d+', '')



test_identity_data.loc[test_identity_data['id_30'].str.contains('Mac', na=False), 'id_30'] = 'mac'

test_identity_data.loc[test_identity_data['id_30'].str.contains('iOS', na=False), 'id_30'] = 'iOS'

test_identity_data.loc[test_identity_data['id_30'].str.contains('Android', na=False), 'id_30'] = 'android'

test_identity_data.loc[test_identity_data['id_30'].str.contains('Windows', na=False), 'id_30'] = 'Windows'

test_identity_data.loc[test_identity_data['id_30'].str.contains('Linux', na=False), 'id_30'] = 'Linux'
test_identity_data['device_name'] = test_identity_data['DeviceInfo'].str.split('/', expand=True)[0]



test_identity_data.loc[test_identity_data['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

test_identity_data.loc[test_identity_data['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

test_identity_data.loc[test_identity_data['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

test_identity_data.loc[test_identity_data['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

test_identity_data.loc[test_identity_data['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

test_identity_data.loc[test_identity_data['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

test_identity_data.loc[test_identity_data['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

test_identity_data.loc[test_identity_data['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

test_identity_data.loc[test_identity_data['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

test_identity_data.loc[test_identity_data['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

test_identity_data.loc[test_identity_data['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

test_identity_data.loc[test_identity_data['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

test_identity_data.loc[test_identity_data['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

test_identity_data.loc[test_identity_data['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

test_identity_data.loc[test_identity_data['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

test_identity_data.loc[test_identity_data['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

test_identity_data.loc[test_identity_data['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'



test_identity_data.loc[test_identity_data.device_name.isin(test_identity_data.device_name.value_counts()[test_identity_data.device_name.value_counts() < 200].index), 'device_name'] = "Others"
raw_test_data = pd.merge(test_transaction_data, test_identity_data, on='TransactionID', how='left')

transactionID = raw_test_data.loc[:,'TransactionID']

del test_identity_data,test_transaction_data
raw_test_data_columns = raw_test_data.columns

numericCols = raw_test_data._get_numeric_data().columns

categoricalCols = list(set(raw_test_data_columns) - set(numericCols))

print('The categorical columns in training data are: ',categoricalCols)

raw_test_data[categoricalCols] = raw_test_data[categoricalCols].replace({ np.nan:'missing'})

raw_test_data[numericCols] = raw_test_data[numericCols].replace({ np.nan:-1})
test_data = raw_test_data.loc[:,featuresToUse]

#https://www.kaggle.com/artgor/eda-and-models#Feature-engineering

test_data['TransactionAmt_to_mean_card1'] = test_data['TransactionAmt'] / test_data.groupby(['card1'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_card4'] = test_data['TransactionAmt'] / test_data.groupby(['card4'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_card5'] = test_data['TransactionAmt'] / test_data.groupby(['card5'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_addr1'] = test_data['TransactionAmt'] / test_data.groupby(['addr1'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_id31'] = test_data['TransactionAmt'] / test_data.groupby(['id_31'])['TransactionAmt'].transform('mean')

test_data['card1_card2'] = test_data['card1'].astype(str) + '_' + test_data['card2'].astype(str)

test_data['addr1_dist1'] = test_data['addr1'].astype(str) + '_' + test_data['dist1'].astype(str)

test_data['card1_addr1'] = test_data['card1'].astype(str) + '_' + test_data['addr1'].astype(str)

test_data['card1_addr2'] = test_data['card1'].astype(str) + '_' + test_data['addr2'].astype(str)

test_data['card2_addr1'] = test_data['card2'].astype(str) + '_' + test_data['addr1'].astype(str)

test_data['card2_addr2'] = test_data['card2'].astype(str) + '_' + test_data['addr2'].astype(str)

test_data['card4_addr1'] = test_data['card4'].astype(str) + '_' + test_data['addr1'].astype(str)

test_data['card4_addr2'] = test_data['card4'].astype(str) + '_' + test_data['addr2'].astype(str)

test_data['DeviceInfo_P_emaildomain'] = test_data['DeviceInfo'].astype(str) + '_' + test_data['P_emaildomain'].astype(str)

test_data['P_emaildomain_addr1'] = test_data['P_emaildomain'].astype(str) + '_' + test_data['addr1'].astype(str)

test_data['id01_addr1'] = test_data['id_01'].astype(str) + '_' + test_data['addr1'].astype(str)

test_data['TransactionAmt_to_std_card1'] = test_data['TransactionAmt'] / test_data.groupby(['card1'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_card4'] = test_data['TransactionAmt'] / test_data.groupby(['card4'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_card5'] = test_data['TransactionAmt'] / test_data.groupby(['card5'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_addr1'] = test_data['TransactionAmt'] / test_data.groupby(['addr1'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_id31'] = test_data['TransactionAmt'] / test_data.groupby(['id_31'])['TransactionAmt'].transform('std')

# New feature - decimal part of the transaction amount.

test_data['TransactionAmt_decimal'] = ((test_data['TransactionAmt'] - test_data['TransactionAmt'].astype(int)) * 1000).astype(int)
goodCategoricalVars.append('card1_addr1')

goodCategoricalVars.append('card1_addr2')

goodCategoricalVars.append('card2_addr1')

goodCategoricalVars.append('card2_addr2')

goodCategoricalVars.append('card4_addr1')

goodCategoricalVars.append('card4_addr2')

goodCategoricalVars.append('DeviceInfo_P_emaildomain')

goodCategoricalVars.append('P_emaildomain_addr1')

goodCategoricalVars.append('id01_addr1')

goodCategoricalVars.append('card1_card2')

goodCategoricalVars.append('device_name')

goodCategoricalVars.append('addr1_dist1')

from sklearn.preprocessing import LabelEncoder

for i_cat in goodCategoricalVars:

    le = LabelEncoder()

    curData = pd.concat([train_data.loc[:,i_cat],test_data.loc[:,i_cat]],axis = 0)

    le.fit(curData)

    train_data.loc[:,i_cat] = le.transform(train_data.loc[:,i_cat])   

    test_data.loc[:,i_cat] = le.transform(test_data.loc[:,i_cat])
del raw_test_data
from sklearn.model_selection import StratifiedKFold,train_test_split

from sklearn.metrics import roc_curve, auc

from lightgbm import LGBMClassifier

import lightgbm as lgb



cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
def compute_roc_auc(clf,index):

    y_predict = clf.predict_proba(train_data.iloc[index])[:,1]

    fpr, tpr, thresholds = roc_curve(target_data.iloc[index], y_predict)

    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score
params = {'bagging_fraction': 0.7982116702024386,

          'feature_fraction': 0.1785051643813966,

          'max_depth': int(49.17611603427576),

          'min_child_weight': 3.2852905549011155,

          'min_data_in_leaf': int(31.03480802715621),

          'n_estimators': int(1491.3676131788188),

          'num_leaves': int(52.851307790411965),

          'reg_alpha': 0.45963319421692145,

          'reg_lambda': 0.6591286807489907,

          'metric':'auc',

          'boosting_type': 'dart',

          'colsample_bytree':.8,

          'subsample':.9,

          'min_split_gain':.01,

          'max_bin':127,

          'bagging_freq':5,

          'learning_rate':0.1  ,

          'early_stopping_rounds':500

}
fprs_lgb, tprs_lgb, scores_lgb = [], [], []

feature_importances = pd.DataFrame()

feature_importances['feature'] = train_data.columns     

predictions = np.zeros(len(test_data))

for (train, test), i in zip(cv.split(train_data, target_data), range(5)):

    lgb_best = LGBMClassifier(boosting = params['boosting_type'],n_estimators =  params['n_estimators'],

                     learning_rate =  params['learning_rate'],num_leaves =  params['num_leaves'],

                     colsample_bytree = params['colsample_bytree'],subsample =  params['subsample'],

                     max_depth =  params['max_depth'],reg_alpha =  params['reg_alpha'],

                     reg_lambda =  params['reg_lambda'],min_split_gain =  params['min_split_gain'],

                     min_child_weight =  params['min_child_weight'],max_bin =  params['max_bin'],

                     bagging_freq =  params['bagging_freq'],feature_fraction =  params['feature_fraction'],

                     bagging_fraction =  params['bagging_fraction'],min_data_in_leaf = params['min_data_in_leaf'])

    lgb_best.fit(train_data.iloc[train,:], target_data.iloc[train],eval_metric='auc')

    feature_importances['fold_{}'.format(i + 1)] = lgb_best.feature_importances_

    _, _, auc_score_train = compute_roc_auc(lgb_best,train)

    fpr, tpr, auc_score = compute_roc_auc(lgb_best,test)

    scores_lgb.append((auc_score_train, auc_score))

    fprs_lgb.append(fpr)

    tprs_lgb.append(tpr) 

    predictions += lgb_best.predict_proba(test_data)[:,1]/cv.n_splits
plt.figure()

lw = 2

colors = ['black','red','blue','darkorange','green']

for i in range(0,5):

    plt.plot(fprs_lgb[i], tprs_lgb[i], color=colors[i],

             lw=lw, label='ROC curve (area = %0.5f)' % scores_lgb[i][1])

    

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.show()
scores_lgb
print('Mean AUC:', np.mean(scores_lgb,axis = 1))
feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(cv.n_splits)]].mean(axis=1)

plt.figure(figsize=(16, 16))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

plt.title('50 TOP feature importance over {} folds average'.format(cv.n_splits));
del train_data,target_data
#y_test = lgb_best.predict_proba(test_data)[:,1]
data = {'TransactionID':transactionID,'isFraud':predictions}

submissionDF = pd.DataFrame(data)

submissionDF.to_csv('sample_submission.csv',index=False)