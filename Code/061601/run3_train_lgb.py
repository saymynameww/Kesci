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
from save_log import Logger
import matplotlib.pyplot as plt

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
feature_importance = [144, 175, 284, 216, 168, 74, 50, 100, 83, 58, 42, 72, 185, 167, 17, 3, 3, 5, 5, 8, 12, 2, 64, 42, 10, 18, 92, 20, 21, 40, 56, 33, 18, 7, 160, 211, 90, 75, 92, 56, 7, 64, 101, 92, 92, 40, 144, 42, 21, 9, 0, 0, 97, 83, 84, 47, 0, 14, 31, 82, 113, 68, 75, 66, 50, 56, 92, 61, 81, 83, 63, 73, 52, 98, 111, 75, 75, 45, 26, 45, 84, 79, 35, 49, 43, 22, 49, 50, 57, 36, 58, 51, 34, 52, 44, 75, 51, 57, 39, 32, 39, 88, 17, 20, 17, 5, 16, 4, 22, 112, 74, 88, 38, 58, 73, 103, 50, 24, 50, 33, 3, 19, 31, 11, 31, 27, 18, 7, 23, 16, 8, 1, 13, 11, 1, 7, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 98, 23, 115, 17, 55]
feature_importance = np.array(feature_importance)
feature_importance_check = np.vstack((feature_list,feature_importance)).T
feature_importance_check = pd.DataFrame(feature_importance_check)
feature_importance_check[1] = feature_importance_check[1].astype(int)
importance_threshold = 1
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
#    threshold = 0.4
#    prediction[prediction>=threshold]=1
#    prediction[prediction<threshold]=0
    F1 = sklearn.metrics.roc_auc_score(Y_test, prediction)
    return F1

def pretrain():
    params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
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
                  'metric': 'auc',
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
        else:
            del params['learning_rate']
        
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
        else:
            del params['num_leaves'],params['max_depth']
            
        
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
        else:
            del params['min_data_in_leaf']
        
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
        else:
            del params['feature_fraction'],params['bagging_fraction'],params['bagging_freq']
#        params['feature_fraction'] = 0.8
#        params['bagging_fraction'] = 0.8
#        params['bagging_freq'] = 10
        
        
#        print("调参5：正则化")
#        for lambda_l1 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#            for lambda_l2 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0,0.2,0.4,0.6,0.8,1.0]
#                for min_split_gain in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#                    print('============================',lambda_l1,lambda_l2,min_split_gain)
#                    params['lambda_l1'] = lambda_l1
#                    params['lambda_l2'] = lambda_l2
#                    params['min_split_gain'] = min_split_gain
#                    F1 = F1_score(params)
#                    if F1 > max_F1:
#                        max_F1 = F1
#                        print('best F1 updated:',max_F1)
#                        best_params['lambda_l1'] = lambda_l1
#                        best_params['lambda_l2'] = lambda_l2
#                        best_params['min_split_gain'] = min_split_gain
#        if 'lambda_l1' in best_params:
#            params['lambda_l1'] = best_params['lambda_l1']
#            params['lambda_l2'] = best_params['lambda_l2']
#            params['min_split_gain'] = best_params['min_split_gain']
#        else:
#            del params['lambda_l1'],params['lambda_l2'],params['min_split_gain']
        params['lambda_l1'] = 1
        params['lambda_l2'] = 1
        params['min_split_gain'] = 1
        
        print('Tuning params DONE, best_params:',best_params)
        return params
    elif mode == 2:
        params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'verbose': 1, 'learning_rate': 0.04, 'num_leaves': 30, 'max_depth': 5, 'min_data_in_leaf': 40, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 10, 'lambda_l1': 1, 'lambda_l2': 1, 'min_split_gain': 1}
        return params
    else:
        print('mode error!')


def train_func(params):
    print('Training...')
    print('All params:',params)
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
    fig, ax = plt.subplots(1, 1, figsize=[16, 40])
    lgb.plot_importance(gbm, ax=ax, max_num_features=200)
    plt.savefig('feature_importance.png')
    
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
    
params = train_tune(mode=2)
stdout_backup = sys.stdout
sys.stdout = Logger("train_info.txt")
gbm = train_func(params) 
predict_func(gbm,0.4)    
sys.stdout = stdout_backup
time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')

    








