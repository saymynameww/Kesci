# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:33:52 2018

@author: NURBS
"""

import os
import sys
import pandas as pd
from sklearn import model_selection
import numpy as np
import sklearn.metrics
from datetime import datetime
import lightgbm as lgb
from save_log_test import Logger

time_start = datetime.now()
print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))

print('Loading data...')
train_path = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/train.csv')
test_path = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/test.csv')
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print('loading important features and label...')
feature_list = train.columns.values.tolist()
feature_list.remove('user_id')
feature_list.remove('label')
feature_list = np.array(feature_list)
feature_importance = [34, 57, 73, 46, 48, 9, 10, 31, 15, 1, 8, 19, 62, 108, 0, 2, 1, 1, 0, 4, 1, 1, 12, 9, 2, 0, 23, 5, 1, 17, 13, 5, 5, 1, 26, 55, 9, 11, 14, 2, 1, 24, 19, 26, 14, 3, 23, 3, 6, 1, 0, 0, 21, 9, 21, 13, 0, 2, 7, 11, 14, 8, 10, 8, 3, 10, 11, 
12, 10, 16, 7, 4, 11, 18, 11, 9, 8, 10, 4, 5, 21, 16, 3, 11, 10, 3, 5, 4, 3, 2, 6, 8, 0, 9, 6, 9, 8, 9, 2, 10, 8, 7, 2, 2, 2, 0, 4, 0, 5, 16, 9, 8, 12, 4, 5, 18, 5, 3, 4, 4, 2, 3, 3, 4, 4, 5, 3, 2, 3, 0, 0, 0, 1, 3, 0, 0, 1, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 2, 34, 6, 9]
feature_importance = np.array(feature_importance)
feature_importance_check = np.vstack((feature_list,feature_importance)).T
feature_importance_check = pd.DataFrame(feature_importance_check)
feature_importance_check[1] = feature_importance_check[1].astype(int)
importance_threshold = 3
used_feature = feature_list[feature_importance>=importance_threshold]
train_feature = train[used_feature]
train_label = train['label']
test_feature = test[used_feature]

#print('loading all the features and label...')
#importance_threshold = 0
#train_feature = train.drop(['user_id','label'],axis=1)
#train_label = train['label']
#test_feature = test.drop(['user_id'],axis=1)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, train_label, test_size=0.2,random_state=1017)
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

def F1_score(params):
    max_boost_rounds = 1000
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=max_boost_rounds,
                    early_stopping_rounds=20,
                    verbose_eval=False,
                    )
    prediction = gbm.predict(X_test)
    threshold = 0.4
    prediction[prediction>=threshold]=1
    prediction[prediction<threshold]=0
    F1 = sklearn.metrics.f1_score(Y_test, prediction)
    return F1

def pretrain():
    params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'binary_error',
          }
    max_boost_rounds = 1000
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=max_boost_rounds,
                    early_stopping_rounds=20,
                    verbose_eval=False,
                    )

def train_tune(mode):
    if mode == 1:
        print('Tuning params...')
        params = {
                  'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': 'binary_error',
                  }
        max_F1 = F1_score(params)
        print('best F1 updated:',max_F1)
        best_params = {}
        
        print("调参1：学习率")
        for learning_rate in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]: 
            print('============================',learning_rate)
            params['learning_rate'] = learning_rate
            F1 = F1_score(params)
            if F1 > max_F1:
                max_F1 = F1
                print('best F1 updated:',max_F1)
                best_params['learning_rate'] = learning_rate
        if 'learning_rate' in best_params:
            params['learning_rate'] = best_params['learning_rate']
        #else:
        #    del params['learning_rate']
        
        print("调参2：提高准确率")
        for num_leaves in range(20,100,5): #(20,100,5)
            for max_depth in range(3,9,1): #(3,9,1)
                print('============================',num_leaves,max_depth)
                params['num_leaves'] = num_leaves
                params['max_depth'] = max_depth
                F1 = F1_score(params)
                if F1 > max_F1:
                    max_F1 = F1
                    print('best F1 updated:',max_F1)
                    best_params['num_leaves'] = num_leaves
                    best_params['max_depth'] = max_depth
        if 'num_leaves' in best_params:
            params['num_leaves'] = best_params['num_leaves']
            params['max_depth'] = best_params['max_depth']
        #else:
        #    del params['num_leaves'],params['num_leaves']
            
        
        print("调参3：降低过拟合")
        for min_data_in_leaf in range(10,100,5): #(10,200,5)
            print('============================',min_data_in_leaf)
            params['min_data_in_leaf'] = min_data_in_leaf
            F1 = F1_score(params)
            if F1 > max_F1:
                max_F1 = F1
                print('best F1 updated:',max_F1)
                best_params['min_data_in_leaf'] = min_data_in_leaf
        if 'min_data_in_leaf' in best_params:
            params['min_data_in_leaf'] = best_params['min_data_in_leaf']
        #else:
        #    del params['min_data_in_leaf']
        
        print("调参4：采样")
        for feature_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            for bagging_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                for bagging_freq in range(0,50,5): #
                    print('============================',feature_fraction,bagging_fraction,bagging_freq)
                    params['feature_fraction'] = feature_fraction
                    params['bagging_fraction'] = bagging_fraction
                    params['bagging_freq'] = bagging_freq
                    F1 = F1_score(params)
                    if F1 > max_F1:
                        max_F1 = F1
                        print('best F1 updated:',max_F1)
                        best_params['feature_fraction'] = feature_fraction
                        best_params['bagging_fraction'] = bagging_fraction
                        best_params['bagging_freq'] = bagging_freq
        if 'feature_fraction' in best_params:
            params['feature_fraction'] = best_params['feature_fraction']
            params['bagging_fraction'] = best_params['bagging_fraction']
            params['bagging_freq'] = best_params['bagging_freq']
        #else:
        #    del params['feature_fraction'],params['bagging_fraction'],params['bagging_freq']
        
        
        print("调参5：正则化")
        for lambda_l1 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            for lambda_l2 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0,0.2,0.4,0.6,0.8,1.0]
                for min_split_gain in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                    print('============================',lambda_l1,lambda_l2,min_split_gain)
                    params['lambda_l1'] = lambda_l1
                    params['lambda_l2'] = lambda_l2
                    params['min_split_gain'] = min_split_gain
                    F1 = F1_score(params)
                    if F1 > max_F1:
                        max_F1 = F1
                        print('best F1 updated:',max_F1)
                        best_params['lambda_l1'] = lambda_l1
                        best_params['lambda_l2'] = lambda_l2
                        best_params['min_split_gain'] = min_split_gain
        if 'lambda_l1' in best_params:
            params['lambda_l1'] = best_params['lambda_l1']
            params['lambda_l2'] = best_params['lambda_l2']
            params['min_split_gain'] = best_params['min_split_gain']
        #else:
        #    del params['lambda_l1'],params['lambda_l2'],params['min_split_gain']
        print('Tuning params DONE, best_params:',best_params)
        print('ALL params:',params)
        return params
    elif mode == 2:
        params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'binary_error', 'verbose': 1, 'learning_rate': 0.04, 'num_leaves': 30, 'max_depth': 7, 'min_data_in_leaf': 95, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 25, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_split_gain': 1.0}
        print('ALL params:',params)
        return params
    else:
        print('mode error!')


def train_func(params):
    print('Training...')
    cv_results = lgb.cv(train_set=lgb_train,
                         params=params,
                         nfold=5,
                         num_boost_round=6000,
                         early_stopping_rounds=50,
                         verbose_eval=100,
                         metrics=['auc'])
    optimum_boost_rounds = np.argmax(cv_results['auc-mean'])
    print('Optimum boost rounds = {}'.format(optimum_boost_rounds))
    print('Best CV result = {}'.format(np.max(cv_results['auc-mean'])))
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=optimum_boost_rounds,
                    verbose_eval=100,
                    valid_sets=lgb_eval
                    )
    gbm.save_model(os.path.join(os.pardir,os.pardir,'model/lgb_model.txt'))
    return gbm


def predict_func(gbm,threshold):
    print('Validating...')
    validation = gbm.predict(X_test)
    validation[validation>=threshold]=1
    validation[validation<threshold]=0
    print('二分阈值：'+str(threshold))
    print('f1_score：' + str(sklearn.metrics.f1_score(Y_test, validation)))
    print('特征阈值：'+str(importance_threshold)+' 特征数：'+ str(X_test.columns.size))
    print('所用特征重要性：'+ str(list(gbm.feature_importance())))
    
    ########################## 保存结果 ############################
    print('Save result...')
    prediction = gbm.predict(test_feature)
    df_result = pd.DataFrame()
    df_result['user_id'] = test['user_id']
    df_result['result'] = prediction
    df_result.to_csv(os.path.join(os.pardir,os.pardir,'result/lgb_result.csv'), index=False)
    prediction[prediction >= threshold]=1
    prediction[prediction < threshold]=0
    prediction = list(map(int,prediction))
    print('为1的个数：' + str(len(np.where(np.array(prediction)==1)[0])))
    print('为0的个数：' + str(len(np.where(np.array(prediction)==0)[0])))
    
#params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'binary_error', 'learning_rate': 0.01, 'verbose': 1, 'num_leaves': 20, 'max_depth': 4, 'min_data_in_leaf': 80, 'feature_fraction': 0.5, 'bagging_fraction': 0.9, 'bagging_freq': 25, 'lambda_l1': 1, 'lambda_l2': 1, 'min_split_gain': 1}
#params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'binary_error', 'learning_rate': 0.05, 'verbose': 1, 'num_leaves': 30, 'max_depth': 6, 'min_data_in_leaf': 40, 'feature_fraction': 0.7, 'bagging_fraction': 0.9, 'bagging_freq': 25, 'lambda_l1': 1.0, 'lambda_l2': 1.0, 'min_split_gain': 1.0}
params = train_tune(mode=2)
stdout_backup = sys.stdout
#with open("train_info.log", "w+") as log_file:
#    sys.stdout = log_file
#    gbm = train_func(params) 
#    predict_func(gbm,0.4)
#    sys.stdout = stdout_backup
#    print(log_file.readlines())
sys.stdout = Logger("train_info.txt")
gbm = train_func(params) 
predict_func(gbm,0.4)    
sys.stdout = stdout_backup
time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')

    








