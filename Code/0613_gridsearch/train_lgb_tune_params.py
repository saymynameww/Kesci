# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:33:52 2018

@author: NURBS
"""

import os
import pandas as pd
from sklearn import model_selection
import numpy as np
import sklearn.metrics
from datetime import datetime

time_start = datetime.now()
print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))

print('Loading data...')
train_path_1 = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/train_1.csv')
train_path_2 = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/train_2.csv')
train_path = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/train.csv')
test_path = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/test.csv')
train_1 = pd.read_csv(train_path_1)
train_2 = pd.read_csv(train_path_2)
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

#print('loading important features and label...')
#feature_list = train.columns.values.tolist()
#feature_list.remove('user_id')
#feature_list.remove('label')
#feature_list = np.array(feature_list)
#feature_importance = [219, 419, 620, 386, 487, 140, 54, 250, 155, 54, 76, 138, 408, 952, 25, 7, 2, 15, 21, 23, 22, 9, 96, 
#                      69, 16, 16, 224, 51, 17, 95, 98, 46, 56, 39, 304, 442, 157, 145, 118, 74, 20, 186, 168, 169, 135, 62, 
#                      272, 67, 47, 35, 0, 1, 157, 156, 172, 96, 0, 37, 74, 142, 116, 103, 116, 76, 55, 78, 164, 143, 85, 162, 
#                      76, 95, 79, 166, 109, 104, 90, 79, 52, 92, 127, 113, 36, 90, 87, 46, 67, 72, 70, 79, 53, 75, 33, 33, 68, 
#                      73, 105, 96, 66, 63, 82, 78, 20, 50, 26, 11, 27, 9, 37, 98, 99, 85, 75, 55, 90, 165, 58, 63, 72, 66, 28, 
#                      45, 61, 27, 40, 64, 28, 27, 32, 34, 10, 12, 14, 38, 6, 9, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 182, 
#                      32, 294, 54, 116]
#feature_importance = np.array(feature_importance)
#feature_importance_check = np.vstack((feature_list,feature_importance)).T
#feature_importance_check = pd.DataFrame(feature_importance_check)
#feature_importance_check[1] = feature_importance_check[1].astype(int)
#importance_threshold = 80
#used_feature = feature_list[feature_importance>=importance_threshold]
#train_feature = train[used_feature]
#train_label = train['label']
#test_feature = test[used_feature]

print('loading all the features and label...')
train_feature = train.drop(['user_id','label'],axis=1)
train_label = train['label']
test_feature = test.drop(['user_id'],axis=1)
#X_train = train_1.drop(['user_id','label'],axis=1)
#Y_train = train_1['label']
#X_test = train_2.drop(['user_id','label'],axis=1)
#Y_test = train_2['label']
#test_feature = test.drop(['user_id'],axis=1)
#X_train_all = train.drop(['user_id','label'],axis=1)
#Y_train_all = train['label']

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, train_label, test_size=0.2,random_state=1017)


###################### lgb ##########################
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

def F1_score(gbm,threshold):
    temp = gbm.predict(X_test)
    temp[temp>=threshold]=1
    temp[temp<threshold]=0
    F1 = sklearn.metrics.f1_score(Y_test, temp)
    print('f1_score：' + str(F1))
    return F1

def train_tune(params):
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=1000,
                    early_stopping_rounds=20,
                    verbose_eval=False,
                    )
    F1 = F1_score(gbm,0.4)
    return F1

def train_func(params):
    print('Training...')
    params['learning_rate']=0.01
    cv_results = lgb.cv(train_set=lgb_train,
#                         train_set=lgb.Dataset(X_train_all, Y_train_all),
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
#                    train_set = lgb.Dataset(X_train_all, Y_train_all),
                    valid_sets=lgb_eval,
                    num_boost_round=optimum_boost_rounds,
                    early_stopping_rounds=50,
                    verbose_eval=100,
                    )
    gbm.save_model(os.path.join(os.pardir,os.pardir,'Model/lgb_model.txt'))
    return gbm


def predict_func(threshold):
    print('Validating...')
    temp = gbm.predict(X_test)
    temp[temp>=threshold]=1
    temp[temp<threshold]=0
    print('二分阈值：'+str(threshold))
    print('f1_score：' + str(sklearn.metrics.f1_score(Y_test, temp)))
#    print('特征阈值：'+str(importance_threshold)+' 特征数：'+ str(X_test.columns.size))
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

#params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': ['auc'], 
#          'num_leaves': 40, 'max_depth': 6, 'verbose': 1, 'max_bin': 249, 'min_data_in_leaf': 36, 
#          'feature_fraction': 0.85, 'bagging_fraction': 0.9, 'bagging_freq': 5, 
#          'lambda_l1': 1.0, 'lambda_l2': 0.7, 'min_split_gain': 0.0, 
#          'learning_rate': 0.01}
#params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': ['binary_error'],'num_leaves': 52, 'max_depth': 8, 'feature_fraction': 0.1, 'bagging_fraction': 0.1, 'bagging_freq': 5, 'lambda_l1': 0.3, 'lambda_l2': 0, 'min_split_gain': 0.7}
gbm = train_func(params) 
#gbm = lgb.booster(model_file='model/lgb_model.txt')
predict_func(0.4)
    
time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')
    
    








