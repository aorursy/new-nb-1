import numpy as np, pandas as pd, os

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import roc_auc_score

from tqdm import tqdm



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
from scipy.stats import multivariate_normal



def get_mv(x,y):

    ones = (y==1).astype(bool)

    x2 = x[ones]

    cov1 = np.cov(x2.T)

    m1 = np.mean(x2, axis = 0)

    

    zeros = (y==0).astype(bool)

    x2b = x[zeros]

    cov2 = np.cov(x2b.T)

    m2 = np.mean(x2b, axis = 0)

    

    mv1 = multivariate_normal(mean=m1, cov=cov1)

    mv2 = multivariate_normal(mean=m2, cov=cov2)

    

    return mv1, mv2



def calc_prob(x,mv1,mv2):

    y_pred2 = np.zeros((len(x),))

    for i in range(len(x)):

        a = mv1.pdf(x[i])

        b = mv2.pdf(x[i])

        y_pred2[i] = a/(a+b)

    return y_pred2
# INITIALIZE VARIABLES

cols = [c for c in train.columns if c not in ['id', 'target']]

cols.remove('wheezy-copper-turtle-magic')

oof = np.zeros(len(train))

preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for i in tqdm(range(512)):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])

    

    # STRATIFIED K-FOLD

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3, train2['target']):

        

        # MODEL AND PREDICT WITH QDA

        #clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

        

        

        mv1, mv2 = get_mv(train3[train_index,:],train2.loc[train_index]['target'].values)

        oof[idx1[test_index]] = calc_prob(train3[test_index,:],mv1,mv2)

        preds[idx2] += calc_prob(test3,mv1,mv2)/ skf.n_splits

       

    #if i%64==0: print(i)

        

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)

print('QDA scores CV =',round(auc,5))
# INITIALIZE VARIABLES

test['target'] = preds

oof = np.zeros(len(train))

preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for k in tqdm(range(512)):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==k] 

    train2p = train2.copy(); idx1 = train2.index 

    test2 = test[test['wheezy-copper-turtle-magic']==k]

    

    # ADD PSEUDO LABELED DATA

    test2p = test2[ (test2['target']<=0.01) | (test2['target']>=0.99) ].copy()

    test2p.loc[ test2p['target']>=0.5, 'target' ] = 1

    test2p.loc[ test2p['target']<0.5, 'target' ] = 0 

    train2p = pd.concat([train2p,test2p],axis=0)

    train2p.reset_index(drop=True,inplace=True)

    

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     

    train3p = sel.transform(train2p[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])

        

    # STRATIFIED K FOLD

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3p, train2p['target']):

        test_index3 = test_index[ test_index<len(train3) ] # ignore pseudo in oof

        

        mv1, mv2 = get_mv(train3p[train_index,:],train2p.loc[train_index]['target'].values)

        oof[idx1[test_index3]] = calc_prob(train3[test_index3,:],mv1,mv2)

        preds[test2.index] += calc_prob(test3,mv1,mv2) / skf.n_splits



       

    #if k%64==0: print(k)

        

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)

print('Pseudo Labeled QDA scores CV =',round(auc,5))
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv',index=False)



import matplotlib.pyplot as plt

plt.hist(preds,bins=100)

plt.title('Final Test.csv predictions')

plt.show()