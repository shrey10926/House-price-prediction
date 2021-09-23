# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:37:25 2021

@author: Shrey
"""

import os
import optuna
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.datasets import load_boston
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

boston = load_boston()

data = pd.DataFrame(boston.data, columns = boston.feature_names)
data

data1 = data.copy(deep = True)
data1

data1['medv'] = boston.target

data1.drop_duplicates(keep = 'first', inplace = True)

r_list = ['RAD', 'ZN', 'CHAS', 'INDUS', 'PTRATIO']
data2 = data1.drop(r_list, axis = 1)

y = data2['medv']
x = data2.drop(['medv'], axis = 1)


num_col = [cname for cname in x.columns if x[cname].dtype in ['int64', 'float64']]
cat_col = [cname for cname in x.columns if x[cname].dtype == 'object']

num_trans = SimpleImputer(strategy = 'mean')
cat_trans = Pipeline(steps = [('impute', SimpleImputer(strategy = 'most_frequent')), 
                              ('onehotencode', OneHotEncoder(handle_unknown = 'ignore'))])

preproc = ColumnTransformer(transformers = [('num', num_trans, num_col),
                                            ('cat', cat_trans, cat_col)])

lire_model = LinearRegression(n_jobs = -1)

lire_pipe = Pipeline(steps = [('lire_preproc', preproc), ('lire_model', lire_model)])

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 69)


# OPTUNA LIRE
def objective(trial):
    
    lire_model__fit_intercept = trial.suggest_categorical('lire_model__fit_intercept', [True, False])
    lire_model__normalize = trial.suggest_categorical('lire_model__normalize', [True, False])
    
    params = {'lire_model__fit_intercept' : lire_model__fit_intercept, 
              'lire_model__normalize' : lire_model__normalize}
    
    lire_pipe.set_params(**params)
    
    return np.mean(-1 * cross_val_score(lire_pipe, train_x, train_y,
                                     cv = 5, n_jobs = -1, scoring = 'neg_mean_absolute_error'))
    

lire_study = optuna.create_study(direction = 'minimize')
lire_study.optimize(objective, n_trials = 10)

lire_pipe.set_params(**lire_study.best_params)
lire_pipe.fit(train_x, train_y)


pickle.dump(lire_pipe, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))