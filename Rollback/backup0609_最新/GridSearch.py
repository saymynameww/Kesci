# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:20:55 2018

@author: Administrator
"""

import numpy as np  
import pandas as pd  
import scipy as sp  
import copy,os,sys,psutil  
import lightgbm as lgb  
import warnings
warnings.filterwarnings("ignore")
from lightgbm.sklearn import LGBMClassifier 
from sklearn.model_selection import GridSearchCV,train_test_split 
from sklearn.datasets import dump_svmlight_file  
  
from sklearn import  metrics   #Additional scklearn functions  
from datetime import datetime
 
  
  
def print_best_score(gsearch,param_test):  
     # 输出best score  
    print("Best score: %0.6f" % gsearch.best_score_)  
    print("Best parameters set:")  
    # 输出最佳的分类器到底使用了怎样的参数  
    best_parameters = gsearch.best_estimator_.get_params()  
    for param_name in sorted(param_test.keys()):  
        print("\t%s: %r" % (param_name, best_parameters[param_name]))  
  
def lightGBM_CV():  
    time_start = datetime.now()
    print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S')) 
    print ('获取内存占用率： '+(str)(psutil.virtual_memory().percent)+'%')  
    print('Loading data...')
    train_path = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/train.csv')
    test_path = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/test.csv')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print('loading all the features and label...')
    train_feature = train.drop(['user_id','label'],axis=1)
    train_label = train['label']
    test_feature = test.drop(['user_id'],axis=1)
    
    X_train, X_test, Y_train, Y_test = train_test_split(train_feature, train_label, test_size=0.2,random_state=1017)
    param_test = {  
        'max_depth': [x for x in range(3,9,2)],  
        'num_leaves': [x for x in range(10,40,5)], 
        'max_bin': [x for x in range(5,260,5)], 
    }  
    estimator = LGBMClassifier(  
        num_leaves = 30, # cv调节50是最优值  
        max_depth = 6,  
        learning_rate =0.1,   
        n_estimators = 1000,   
#        objective = 'regression',   
#        min_child_weight = 1,   
#        subsample = 0.8,  
#        colsample_bytree=0.8,  
#        nthread = 7,  
    )  
    gsearch = GridSearchCV( estimator , param_grid = param_test, scoring='f1',iid=False, cv=5 )  
    gsearch.fit( X_train, Y_train )  
    print_best_score(gsearch,param_test)  
    
    time_end = datetime.now()
    print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
    print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')
  
  
if __name__ == '__main__':  
    lightGBM_CV()  