# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:33:52 2018

@author: Administrator
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
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

time_start = datetime.now()
print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))
print('Loading data...')
train_path_a = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/train_a.csv')
train_path_b = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/train_b.csv')
test_path = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/test.csv')
train_a = pd.read_csv(train_path_a)
train_b = pd.read_csv(train_path_b)
train = pd.concat([train_a,train_b],axis=0)
#train = train_a
test = pd.read_csv(test_path)

def feature_selection(feature_mode,R_threshold):
    if feature_mode == 1:
        print('Loading all the features and label...')
        importance_threshold = 0
        train_feature = train.drop(['user_id','label'],axis=1)
        train_label = train['label']
        test_feature = test.drop(['user_id'],axis=1)
    elif feature_mode == 2:
        print('Loading Pearson important features and label...')
        feature_list = train.columns.values.tolist()
        feature_list.remove('user_id')
        feature_list.remove('label')
        feature_list.remove('device_type')
        feature_list.remove('register_type')
        pearson = []
        for name in feature_list:
            pearson.append(pearsonr(train.label, train[name]))
        pearson = pd.DataFrame(pearson).rename({0:'R_value',1:'P_value'},axis=1)
        pearson['Feature_name'] = feature_list
        P_threshold = 0.05
#        R_threshold = 0.05
        importance_threshold = R_threshold
        used_feature = pearson[(pearson.P_value<=P_threshold) & ((pearson.R_value>=R_threshold)|(pearson.R_value<=-R_threshold))]
        used_feature = ['device_type','register_type']+used_feature.Feature_name.tolist()
        train_feature = train[used_feature]
        train_label = train['label']
        test_feature = test[used_feature]        
    elif feature_mode == 3:
        print('Loading result-based important features and label...')
        feature_list = train.columns.values.tolist()
        feature_list.remove('user_id')
        feature_list.remove('label')
        feature_list = np.array(feature_list)
        feature_importance = [11, 100, 127, 40, 44, 26, 31, 30, 56, 40, 52, 4, 1, 35, 12, 1, 4, 4, 31, 27, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 15, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 4, 0, 12, 23, 5, 8, 0, 26, 4, 27, 17, 13, 8, 3, 0, 10, 16, 17, 8, 2, 27, 2, 3, 2, 7, 5, 13, 10, 3, 10, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 2, 39, 8, 6, 10, 3, 0, 8, 7, 9, 9, 0, 18, 0, 0, 0, 18, 2, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 11, 1, 3, 2, 1, 0, 6, 5, 0, 1, 0, 5, 1, 1, 1, 13, 8, 10, 4, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 6, 2, 1, 11, 2, 2, 1, 2, 12, 0, 0, 0, 5, 3, 5, 2, 1, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 9, 7, 10, 6, 8, 0, 3, 6, 3, 8, 2, 3, 0, 6, 0, 9, 3, 7, 4, 2, 8, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 3, 2, 2, 1, 10, 1, 1, 7, 2, 10, 0, 5, 0, 8, 6, 5, 1, 3, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 8, 13, 7, 5, 7, 1, 1, 6, 3, 2, 0, 13, 2, 2, 0, 5, 8, 7, 4, 2, 14, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 6, 0, 11, 6, 7, 8, 4, 3, 10, 4, 14, 12, 10, 7, 7, 22, 6, 3, 6, 6, 1, 2, 10, 9, 1, 7, 7, 3, 2, 5, 6, 4, 5, 5, 0, 2, 2, 4, 7, 9, 2, 6, 1, 2, 2, 3, 5, 0, 2, 0, 4, 5, 20, 5, 8, 7, 6, 9, 5, 5, 7, 3, 2, 2, 1, 2, 2, 0, 1, 0, 2, 1, 1, 0, 0, 1, 0, 0, 0, 11, 0, 57, 5, 18, 0, 1, 0, 0, 0, 2, 1]
        feature_importance = np.array(feature_importance)
        feature_importance_check = np.vstack((feature_list,feature_importance)).T
        feature_importance_check = pd.DataFrame(feature_importance_check)
        feature_importance_check[1] = feature_importance_check[1].astype(int)
        importance_threshold = 1
        used_feature = feature_list[feature_importance>=importance_threshold]
        train_feature = train[used_feature]
        train_label = train['label']
        test_feature = test[used_feature]
    elif feature_mode == 4:
        print('Loading Pearson important features and label...')
        feature_list = train.columns.values.tolist()
        feature_list.remove('user_id')
        feature_list.remove('label')
        pearson = []
        for name in feature_list:
            pearson.append(pearsonr(train.label, train[name]))
        pearson = pd.DataFrame(pearson).rename({0:'R_value',1:'P_value'},axis=1)
        pearson['Feature_name'] = feature_list
        P_threshold = 0.05
#        R_threshold = 0.05
        importance_threshold = R_threshold
        used_feature = pearson[(pearson.P_value<=P_threshold) & ((pearson.R_value>=R_threshold)|(pearson.R_value<=-R_threshold))]
        used_feature = used_feature.Feature_name.tolist()
        train_feature = train[used_feature]
        train_label = train['label']
        test_feature = test[used_feature]
        
    return train_feature,train_label,test_feature,importance_threshold

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

def train_tune(train_mode):
    if train_mode == 1:
        print('Tuning params...')
        params = {
                  'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': 'auc',
                  }
        max_F1 = F1_score(params)
        print('best F1 updated:',max_F1)
        best_params = {}
        
#        print("调参1：学习率")
#        for learning_rate in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]: 
#            print('============================',learning_rate)
#            params['learning_rate'] = learning_rate
#            F1 = F1_score(params)
#            if F1 > max_F1:
#                max_F1 = F1
#                print('best F1 updated:',max_F1)
#                best_params['learning_rate'] = learning_rate
#        if 'learning_rate' in best_params:
#            params['learning_rate'] = best_params['learning_rate']
#        else:
#            del params['learning_rate']
        params['learning_rate'] = 0.08
        
#        print("调参2：提高准确率")
#        for num_leaves in range(15,45,1): #(20,200,10)
#            for max_depth in range(3,10,1): #(3,9,1)
#                print('============================',num_leaves,max_depth)
#                params['num_leaves'] = num_leaves
#                params['max_depth'] = max_depth
#                F1 = F1_score(params)
#                if F1 > max_F1:
#                    max_F1 = F1
#                    print('best F1 updated:',max_F1)
#                    best_params['num_leaves'] = num_leaves
#                    best_params['max_depth'] = max_depth
#        if 'num_leaves' in best_params:
#            params['num_leaves'] = best_params['num_leaves']
#            params['max_depth'] = best_params['max_depth']
#        else:
#            del params['num_leaves'],params['max_depth']
        params['num_leaves'] = 23
        params['max_depth'] = 9
        
        print("调参3：降低过拟合")
        for min_data_in_leaf in range(10,800,10): #(10,200,5)
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
        params['min_data_in_leaf'] = 220
        
#        print("调参4：采样")
#        for feature_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#            for bagging_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#                for bagging_freq in range(0,50,5): #
#                    print('============================',feature_fraction,bagging_fraction,bagging_freq)
#                    params['feature_fraction'] = feature_fraction
#                    params['bagging_fraction'] = bagging_fraction
#                    params['bagging_freq'] = bagging_freq
#                    F1 = F1_score(params)
#                    if F1 > max_F1:
#                        max_F1 = F1
#                        print('best F1 updated:',max_F1)
#                        best_params['feature_fraction'] = feature_fraction
#                        best_params['bagging_fraction'] = bagging_fraction
#                        best_params['bagging_freq'] = bagging_freq
#        if 'feature_fraction' in best_params:
#            params['feature_fraction'] = best_params['feature_fraction']
#            params['bagging_fraction'] = best_params['bagging_fraction']
#            params['bagging_freq'] = best_params['bagging_freq']
#        else:
#            del params['feature_fraction'],params['bagging_fraction'],params['bagging_freq']
        params['feature_fraction'] = 0.9
        params['bagging_fraction'] = 0.8
        params['bagging_freq'] = 5
        
        
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
    elif train_mode == 2:
        params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'verbose': 1, 'learning_rate': 0.07, 'num_leaves': 32, 'max_depth': 5, 'min_data_in_leaf': 20, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'lambda_l1': 1, 'lambda_l2': 1}#, 'min_split_gain': 1
#        params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'verbose': 1, 'learning_rate': 0.04, 'num_leaves': 18, 'max_depth': 8, 'min_data_in_leaf': 20, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'lambda_l1': 1, 'lambda_l2': 1}#, 'min_split_gain': 1
        return params
    else:
        print('mode error!')


def train_func(params):
    print('Training...')
    print('All params:',params)
    cv_results = lgb.cv(train_set=lgb_train,
                         params=params,
                         nfold=5,
                         num_boost_round=1000,
                         early_stopping_rounds=5,
                         verbose_eval=False,
                         metrics=['auc'])
    optimum_boost_rounds = np.argmax(cv_results['auc-mean'])
    print('Optimum boost rounds = {}'.format(optimum_boost_rounds))
    print('Best cv auc = {}'.format(np.max(cv_results['auc-mean'])))
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=optimum_boost_rounds,
                    verbose_eval=False,
                    valid_sets=lgb_eval
                    )
    gbm.save_model(os.path.join(os.pardir,os.pardir,'model/lgb_model.txt'))
    return gbm

def test_threshold(gbm,threshold,f):
    validation = gbm.predict(X_test)
    validation[validation>=threshold]=1
    validation[validation<threshold]=0
    print('二分阈值：'+str(threshold)+' 验证集F1_score：' + str(sklearn.metrics.f1_score(Y_test, validation)),file=f)
    prediction = gbm.predict(test_feature)
    prediction[prediction >= threshold]=1
    prediction[prediction < threshold]=0
    prediction = list(map(int,prediction))
    print('测试集为1的个数：' + str(len(np.where(np.array(prediction)==1)[0])),file=f)
#    print('测试集为0的个数：' + str(len(np.where(np.array(prediction)==0)[0])),file=f)
    
def predict_func(gbm,show_importance):
    print('特征阈值：'+str(importance_threshold)+' 特征数：'+ str(X_test.columns.size))
    if show_importance==1:
        print('所用特征重要性：'+ str(list(gbm.feature_importance())))
    fig, ax = plt.subplots(1, 1, figsize=[16, 40])
    lgb.plot_importance(gbm, ax=ax, max_num_features=400)
    plt.savefig('feature_importance.png')
    
    ########################## 保存结果 ############################
    prediction = gbm.predict(test_feature)
    df_result = pd.DataFrame()
    df_result['user_id'] = test['user_id']
    df_result['result'] = prediction
    df_result.to_csv(os.path.join(os.pardir,os.pardir,'result/lgb_result.csv'), index=False)
    
    
feature_mode = 4
R_threshold = 0.05
train_mode = 2
show_importance = 0
stdout_backup = sys.stdout
sys.stdout = Logger("train_info.txt")
print('\n')
train_feature,train_label,test_feature,importance_threshold = feature_selection(feature_mode,R_threshold)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, train_label, test_size=0.2,random_state=1017)
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)
params = train_tune(train_mode)
gbm = train_func(params) 
predict_func(gbm,show_importance) 
sys.stdout = stdout_backup
time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')

with open('threshold_info.txt', 'w+') as f:
    for i in range(36,41):#30,51
        test_threshold(gbm,i/100,f)    







