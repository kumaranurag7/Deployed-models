import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt

import urllib
import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import RFE

from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import xgboost as xgb

from gensim.models import Word2Vec
import re

import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
import nltk
from string import punctuation

train = pd.read_csv('Train.csv')

#Remove unneccasary columns
train.drop(['id', 'host_id'], axis = 1, inplace = True)
train.drop(['host_name'], axis = 1, inplace = True)
# Drop reviews_per_month due to collinearity and also last_review
train.drop(['reviews_per_month','last_review'],axis = 1, inplace = True)

print('Shape of training data:',train.shape)

# Fill missing values
train.name.fillna('no name', inplace = True) # as listing name cannot be imputed

# Treat skewness
skew_treat = train.skew().index[2:]
for i in skew_treat:
    if train[i].skew() > 0.5:
        train[i] = train[i].apply(lambda x: np.log1p(x + 1))

#Outlier Treatment
catgs = train.room_type.unique() #['Private room', 'Entire home/apt', 'Shared room']
for i in catgs:
    # Winsorization, limit outlier at certain percentile value, no upper limit, lower limit = 10%, based on room_type
    upper,lower = train[train.room_type == i].price.quantile(1), train[train.room_type == i].price.quantile(0.1)
    
    for j in range(train.shape[0]):
        if train.iloc[j, -8] == i:
            if train.iloc[j,-7] > upper:
                train.iloc[j,-7] = upper
            elif train.iloc[j,-7] < lower:
                train.iloc[j,-7] = lower
            else:
                pass
        else:
            pass

#combine train & test before encoding as there are a few values that are not present in either train or test data.
# One hot encoding
def l_encoding(df):
    le1 = LabelEncoder()
    df.neighbourhood = le1.fit_transform(df.neighbourhood)

    #neighbourhood_group
    le2 = LabelEncoder()
    df.neighbourhood_group = le2.fit_transform(df.neighbourhood_group)

    #room_type
    le3 = LabelEncoder()
    df.room_type = le3.fit_transform(df.room_type)
    
    return df

train = l_encoding(train)

def name_process(df):  
    custom_set_of_stopwords = set(stopwords.words('english')+list(punctuation))
    #df = train
    clean_txt = []
    for w in range(len(df.name)):
        desc = df['name'][w].lower()

        #remove punctuation
        desc = re.sub('[^a-zA-Z]', ' ', desc)

        #remove tags
        desc=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)

        #remove special chars, digits
        desc=re.sub("(\\d|\\W)+"," ",desc)
        clean_txt.append(desc)
    df['cleaned_name'] = clean_txt
    df.head()

    corpus = []
    for col in df.cleaned_name:
        word_list = col.split(" ")
        corpus.append(word_list)

    for i in range(len(corpus)):
        corpus[i] = [x for x in corpus[i] if x not in custom_set_of_stopwords]

    #generate vectors
    model = Word2Vec(corpus, min_count=1, size = 30)
    embeds = []
    for i in range(df.shape[0]):
        divider = 0
        summer = np.zeros(30,dtype = 'int')
        tester = np.zeros(30,dtype = 'int')
        for j in range(len(df.iloc[i,-1].split(' '))):
            w_f = df.iloc[i,-1].split(' ')[j]
            if (len(w_f) > 2) & (w_f not in custom_set_of_stopwords):
                divider +=1
                summer = np.add(model[df.iloc[i,-1].split(' ')[j]],summer)
        all_zero_flag = 0 
        for i in range(len(summer)):
            if summer[i] == 0:
                all_zero_flag += 1

        if all_zero_flag == 0:
            embeds.append(summer/divider)
        else:
            embeds.append(summer)

    t_embeds = embeds.copy()    
    df_embeds = pd.DataFrame(t_embeds, columns = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v15','v16',
                                        'v17','v18','v19','v20','v21','v22','v23','v24','v25','v26','v27','v28','v29','v30'])

    df22 = pd.concat([df,df_embeds], axis = 1)
    df22.drop(['name','cleaned_name'],axis=1,inplace = True)
    return df22

train = name_process(train)


X = train.drop('price', axis =1)
y = train['price'].copy()

# Feature Scaling
ss = StandardScaler()
X = ss.fit_transform(X)


# split Data
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.2,random_state = 14) 


#Ridge
en = Ridge()
ypred_en = en.fit(xtrain,ytrain).predict(xtest)

y_true_scale = []
pred_true_scale = []
for i in range(6846):
    y_true_scale.append((np.expm1(ytest.iloc[i])))
    pred_true_scale.append((np.expm1(ypred_en[i])))
    
en_rmse = np.sqrt(mean_squared_error(y_true_scale, pred_true_scale))
print('test error: ', round(en_rmse,3))
print('r2_  score: ', round(r2_score(y_true_scale, pred_true_scale),3))

# XGBoost
xgb = XGBRegressor()
ypred_xgb = xgb.fit(xtrain,ytrain).predict(xtest)

y_true_scale = []
pred_true_scale = []
for i in range(6846):
    y_true_scale.append((np.expm1(ytest.iloc[i])))
    pred_true_scale.append((np.expm1(ypred_xgb[i])))
    
xgb_rmse = np.sqrt(mean_squared_error(y_true_scale, pred_true_scale))
print('true scale test error: ', round(xgb_rmse,3))
print('true scale r2_score: ', round(r2_score(y_true_scale, pred_true_scale),3))


# Hyper parameter tuning needs to be performed every time, after finding wordr2vec vectors, as it changes every time!
# XGBoost Hyperparameter tuning without cross-validation
grid = {'max_depth':[6],
        'gamma':[0],
        'subsample':[0.9],
        'colsample_bytree':[1],
        'reg_alpha':[0],
        'learning_rate' : [0.3],
        'min_child_weight':[1],
        'n_estimators': [160],
}

best_score = 500
for g in ParameterGrid(grid):
    xgb=XGBRegressor()
    xgb.set_params(**g)
    xgb.fit(xtrain,ytrain)
    ypred3_2 = xgb.predict(xtest)

    # for inv log1p
    ytrue = []
    ypred = []
    for i in range(6846):
        ytrue.append((np.expm1(ytest.iloc[i])))
        ypred.append((np.expm1(ypred3_2[i])))

    e = np.sqrt(mean_squared_error(ytrue, ypred))
    
    xgb_rmse = np.sqrt(mean_squared_error(ytrue, ypred))
    print('true scale test error: ', round(xgb_rmse,3))
    print('true scale r2_score: ', round(r2_score(ytrue, ypred),3))
    
    if xgb_rmse < best_score:
        best_score = xgb_rmse
        #print(xgb_rmse)
        #print(g)
    

pickle.dump(lr, open('up.pkl', 'wb'))













