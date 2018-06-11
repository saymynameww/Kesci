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
train_path = os.path.join(os.pardir, 'Kesci-data-dealt/train_and_test/train.csv')
test_path = os.path.join(os.pardir, 'Kesci-data-dealt/train_and_test/test.csv')
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# =============================================================================
# print('loading important features and label...')
# feature_list = train.columns.values.tolist()
# feature_list.remove('user_id')
# feature_list.remove('label')
# feature_list = np.array(feature_list)
# feature_importance = [219, 419, 620, 386, 487, 140, 54, 250, 155, 54, 76, 138, 408, 952, 25, 7, 2, 15, 21, 23, 22, 9, 96, 
#                       69, 16, 16, 224, 51, 17, 95, 98, 46, 56, 39, 304, 442, 157, 145, 118, 74, 20, 186, 168, 169, 135, 62, 
#                       272, 67, 47, 35, 0, 1, 157, 156, 172, 96, 0, 37, 74, 142, 116, 103, 116, 76, 55, 78, 164, 143, 85, 162, 
#                       76, 95, 79, 166, 109, 104, 90, 79, 52, 92, 127, 113, 36, 90, 87, 46, 67, 72, 70, 79, 53, 75, 33, 33, 68, 
#                       73, 105, 96, 66, 63, 82, 78, 20, 50, 26, 11, 27, 9, 37, 98, 99, 85, 75, 55, 90, 165, 58, 63, 72, 66, 28, 
#                       45, 61, 27, 40, 64, 28, 27, 32, 34, 10, 12, 14, 38, 6, 9, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 182, 
#                       32, 294, 54, 116]
# feature_importance = np.array(feature_importance)
# feature_importance_check = np.vstack((feature_list,feature_importance)).T
# feature_importance_check = pd.DataFrame(feature_importance_check)
# feature_importance_check[1] = feature_importance_check[1].astype(int)
# importance_threshold = 80
# used_feature = feature_list[feature_importance>=importance_threshold]
# train_feature = train[used_feature]
# train_label = train['label']
# test_feature = test[used_feature]
# =============================================================================

print('loading all the features and label...')
train_feature = train.drop(['user_id','label'],axis=1)
train_label = train['label']
test_feature = test.drop(['user_id'],axis=1)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, train_label, test_size=0.2,random_state=1017)


###################### lgb ##########################
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

# =============================================================================
# print('Tuning params...')
# params = {
#           'boosting_type': 'gbdt',
#           'objective': 'binary',
#           'metric': 'binary_logloss',
#           }
# 
# ### 交叉验证(调参)
# cv_results = lgb.cv(
#                     params,
#                     lgb_train,
#                     seed=1017,
#                     nfold=5,
#                     metrics=['binary_error'],
#                     early_stopping_rounds=10,
#                     verbose_eval=False
#                     )
# min_merror = pd.Series(cv_results['binary_error-mean']).min()
# #min_merror = float('Inf')
# best_params = {}
# 
# print("调参1：提高准确率")
# for num_leaves in range(20,100,5): #(20,100,5)
#     for max_depth in range(3,9,1): #(3,9,1)
#         print('============================',num_leaves,max_depth)
#         params['num_leaves'] = num_leaves
#         params['max_depth'] = max_depth
# 
#         cv_results = lgb.cv(
#                             params,
#                             lgb_train,
#                             seed=1017,
#                             nfold=5,
#                             metrics=['binary_error'],
#                             early_stopping_rounds=10,
#                             verbose_eval=False
#                             )
#             
#         mean_merror = pd.Series(cv_results['binary_error-mean']).min()
#         boost_rounds = pd.Series(cv_results['binary_error-mean']).idxmin()
#             
#         if mean_merror < min_merror:
#             min_merror = mean_merror
#             best_params['num_leaves'] = num_leaves
#             best_params['max_depth'] = max_depth
#             
# if 'num_leaves' in best_params:
#     params['num_leaves'] = best_params['num_leaves']
# if 'max_depth' in best_params:
#     params['max_depth'] = best_params['max_depth']
# 
# print("调参2：降低过拟合")
# #min_merror = float('Inf')
# for max_bin in range(5,260,5): #(5,260,5)
#     for min_data_in_leaf in range(10,50,5): #(10,200,5)
#             print('============================',max_bin,min_data_in_leaf)
#             params['max_bin'] = max_bin
#             params['min_data_in_leaf'] = min_data_in_leaf
#             
#             cv_results = lgb.cv(
#                                 params,
#                                 lgb_train,
#                                 seed=1017,
#                                 nfold=5,
#                                 metrics=['binary_error'],
#                                 early_stopping_rounds=10,
#                                 verbose_eval=False
#                                 )
#                     
#             mean_merror = pd.Series(cv_results['binary_error-mean']).min()
#             boost_rounds = pd.Series(cv_results['binary_error-mean']).idxmin()
# 
#             if mean_merror < min_merror:
#                 min_merror = mean_merror
#                 best_params['max_bin']= max_bin
#                 best_params['min_data_in_leaf'] = min_data_in_leaf
# 
# if 'min_data_in_leaf' in best_params:
#     params['min_data_in_leaf'] = best_params['min_data_in_leaf']
# if 'max_bin' in best_params:
#     params['max_bin'] = best_params['max_bin']
# 
# print("调参3：降低过拟合")
# #min_merror = float('Inf')
# for feature_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#     for bagging_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#         for bagging_freq in range(0,50,5): #
#             print('============================',feature_fraction,bagging_fraction,bagging_freq)
#             params['feature_fraction'] = feature_fraction
#             params['bagging_fraction'] = bagging_fraction
#             params['bagging_freq'] = bagging_freq
#             
#             cv_results = lgb.cv(
#                                 params,
#                                 lgb_train,
#                                 seed=1017,
#                                 nfold=5,
#                                 metrics=['binary_error'],
#                                 early_stopping_rounds=10,
#                                 verbose_eval=False
#                                 )
#                     
#             mean_merror = pd.Series(cv_results['binary_error-mean']).min()
#             boost_rounds = pd.Series(cv_results['binary_error-mean']).idxmin()
# 
#             if mean_merror < min_merror:
#                 min_merror = mean_merror
#                 best_params['feature_fraction'] = feature_fraction
#                 best_params['bagging_fraction'] = bagging_fraction
#                 best_params['bagging_freq'] = bagging_freq
# 
# if 'feature_fraction' in best_params:
#     params['feature_fraction'] = best_params['feature_fraction']
# if 'bagging_fraction' in best_params:
#     params['bagging_fraction'] = best_params['bagging_fraction']
# if 'bagging_freq' in best_params:
#     params['bagging_freq'] = best_params['bagging_freq']
# 
# 
# print("调参4：降低过拟合")
# #min_merror = float('Inf')
# for lambda_l1 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#     for lambda_l2 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0,0.2,0.4,0.6,0.8,1.0]
#         for min_split_gain in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#             print('============================',lambda_l1,lambda_l2,min_split_gain)
#             params['lambda_l1'] = lambda_l1
#             params['lambda_l2'] = lambda_l2
#             params['min_split_gain'] = min_split_gain
#             
#             cv_results = lgb.cv(
#                                 params,
#                                 lgb_train,
#                                 seed=1017,
#                                 nfold=5,
#                                 metrics=['binary_error'],
#                                 early_stopping_rounds=10,
#                                 verbose_eval=False
#                                 )
#                     
#             mean_merror = pd.Series(cv_results['binary_error-mean']).min()
#             boost_rounds = pd.Series(cv_results['binary_error-mean']).idxmin()
# 
#             if mean_merror < min_merror:
#                 min_merror = mean_merror
#                 best_params['lambda_l1'] = lambda_l1
#                 best_params['lambda_l2'] = lambda_l2
#                 best_params['min_split_gain'] = min_split_gain
# 
# if 'lambda_l1' in best_params:
#     params['lambda_l1'] = best_params['lambda_l1']
# if 'lambda_l2' in best_params:
#     params['lambda_l2'] = best_params['lambda_l2']
# if 'min_split_gain' in best_params:
#     params['min_split_gain'] = best_params['min_split_gain']
# 
# print('Tuning params DONE, best_params:',best_params)
# =============================================================================


def train_func(params):
    print('Training...')
    params['learning_rate']=0.01
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
    gbm.save_model('model/lgb_model.txt')
    return gbm


def predict_func():
    print('Validating...')
    temp = gbm.predict(X_test)
    valid_result = pd.DataFrame()
    valid_result['user_id'] = X_test.reset_index()['index']
    valid_result['result'] = temp
    threshold = valid_result.sort_values(by='result', axis=0, ascending=False).iloc[np.sum(Y_test)-1,1]
    threshold = 0.4
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
    df_result.to_csv('result/lgb_result.csv', index=False)
    prediction[prediction >= threshold]=1
    prediction[prediction < threshold]=0
    prediction = list(map(int,prediction))
    print('为1的个数：' + str(len(np.where(np.array(prediction)==1)[0])))
    print('为0的个数：' + str(len(np.where(np.array(prediction)==0)[0])))

#params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': ['auc'], 
#          'num_leaves': 21, 'max_depth': 7, 'verbose': 1, 'max_bin': 249, 'min_data_in_leaf': 102, 
#          'feature_fraction': 0.85, 'bagging_fraction': 0.9, 'bagging_freq': 5, 
#          'lambda_l1': 1.0, 'lambda_l2': 0.7, 'min_split_gain': 0.0, 
#          'learning_rate': 0.01}
params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': ['auc'], 'num_leaves': 40, 'max_depth': 6, 'verbose': 1, 'max_bin': 249, 'min_data_in_leaf': 36, 'feature_fraction': 0.85, 'bagging_fraction': 0.9, 'bagging_freq': 5, 'lambda_l1': 1.0, 'lambda_l2': 0.7, 'min_split_gain': 0.0, 'learning_rate': 0.01}
gbm = train_func(params) 
#gbm = lgb.booster(model_file='model/lgb_model.txt')
predict_func()
    
time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')
    
    








