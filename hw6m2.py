import numpy as np
import pandas as pd
import os
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import gc
from sklearn import neighbors
from sklearn import metrics, preprocessing

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

data = train.iloc[:,257]
y = train.iloc[:,257]

skf = StratifiedKFold(n_splits=5, random_state=69)
cols = [c for c in train.columns if c not in ['id', 'target']]
prediction = np.zeros(len(test))
cols.remove('wheezy-copper-turtle-magic')
lent = np.zeros(len(train))

for model in range(512):

    x = train[train['wheezy-copper-turtle-magic']==model]
    y = Y[train['wheezy-copper-turtle-magic']==model]
    idx = x.index
    
    x.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)
    
    clf = CatBoostRegressor()
    clf.fit(x[cols],y)
    
    important_features = [i for i in range(len(cols)) if clf.feature_importances_[i] > 0] 
    cols_important = [cols[i] for i in important_features]
    skf = StratifiedKFold(n_splits=10, random_state=42)
    
    for train_index, valid_index in skf.split(x.iloc[:,1:-1], y):
        clf = neighbors.KNeighborsClassifier(6)
        clf.fit(x.loc[train_index][cols_important], y[train_index])
        lent[idx[valid_index]] = clf.predict_proba(x.loc[valid_index][cols_important])[:,1]
        
print(roc_auc_score)