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

print('loading important features and label...')
feature_list = train.columns.values.tolist()
feature_list.remove('user_id')
feature_list.remove('label')
feature_list = np.array(feature_list)
feature_importance = [45, 0, 36, 26, 10, 11, 0, 0, 21, 8, 8, 0, 0, 15, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 24, 10, 0, 0, 12, 3, 0, 4, 0, 0, 0, 20, 0, 0, 0, 0, 0, 8, 0, 0, 0, 7, 0, 4, 0, 0, 0, 0, 0, 5, 0, 0, 14, 26, 0, 9, 0, 9, 7, 0, 19, 11, 15, 0, 18, 9, 13, 0, 0, 0, 10, 0, 0, 0, 4, 0, 0, 0, 12, 0, 7, 6, 0, 3, 0, 6, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 15, 28, 0, 0, 3, 4, 0, 9, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 22, 0, 0]
feature_importance = np.array(feature_importance)
feature_importance_check = np.vstack((feature_list,feature_importance)).T
feature_importance_check = pd.DataFrame(feature_importance_check)
feature_importance_check[1] = feature_importance_check[1].astype(int)
importance_threshold = 1
used_feature = feature_list[feature_importance>=importance_threshold]
#train_feature = train[used_feature]
#train_label = train['label']
#test_feature = test[used_feature]
X_train = train_1[used_feature]
Y_train = train_1['label']
X_test = train_2[used_feature]
Y_test = train_2['label']
test_feature = test.drop(['user_id'],axis=1)
X_train_all = train[used_feature]
Y_train_all = train['label']

print('loading all the features and label...')
#train_feature = train.drop(['user_id','label'],axis=1)
#train_label = train['label']
#test_feature = test.drop(['user_id'],axis=1)

#X_train = train_1.drop(['user_id','label'],axis=1)
#Y_train = train_1['label']
#X_test = train_2.drop(['user_id','label'],axis=1)
#Y_test = train_2['label']
#test_feature = test.drop(['user_id'],axis=1)
#X_train_all = train.drop(['user_id','label'],axis=1)
#Y_train_all = train['label']

#X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature, train_label, test_size=0.2,random_state=1017)


###################### lgb ##########################
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

def train_tune(params):
    evals_results = {}
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=1000,
                    early_stopping_rounds=20,
                    verbose_eval=False,
                    evals_result = evals_results
                    )
    optimum_boost_rounds = np.argmin(evals_results['valid_0']['binary_error'])
    mean_binary_error = np.mean(evals_results['valid_0']['binary_error'][:optimum_boost_rounds])
#    mean_binary_error = evals_results['valid_0']['binary_error'][optimum_boost_rounds]
    return mean_binary_error

# =============================================================================
# print('Tuning params...')
# 
# params = {
#           'boosting_type': 'gbdt',
#           'objective': 'binary',
#           'metric': 'binary_error',
#           'learning_rate':0.01
#           }
# min_mean_binary_error = train_tune(params)
# print('min_mean_binary_error updated:',min_mean_binary_error)
# best_params = {}
# 
# print("调参1：提高准确率")
# for num_leaves in range(20,100,2): #(20,100,5)
#     print('============================',num_leaves)
#     params['num_leaves'] = num_leaves
#         
#     mean_binary_error = train_tune(params)
#         
#     if mean_binary_error < min_mean_binary_error:
#         min_mean_binary_error = mean_binary_error
#         print('min_mean_binary_error updated:',min_mean_binary_error)
#         best_params['num_leaves'] = num_leaves
#             
# if 'num_leaves' in best_params:
#     params['num_leaves'] = best_params['num_leaves']
# else:
#     del params['num_leaves']
# 
# print("调参2：降低过拟合")
# for min_data_in_leaf in range(10,100,5): #(10,200,5)
#         print('============================',min_data_in_leaf)
#         params['min_data_in_leaf'] = min_data_in_leaf
#         
#         mean_binary_error = train_tune(params)
# 
#         if mean_binary_error < min_mean_binary_error:
#             min_mean_binary_error = mean_binary_error
#             print('min_mean_binary_error updated:',min_mean_binary_error)
#             best_params['min_data_in_leaf'] = min_data_in_leaf
# 
# if 'min_data_in_leaf' in best_params:
#     params['min_data_in_leaf'] = best_params['min_data_in_leaf']
# else:
#     del params['min_data_in_leaf']
# 
# print("调参3：降低过拟合")
# for feature_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#     print('============================',feature_fraction)
#     params['feature_fraction'] = feature_fraction
#     
#     mean_binary_error = train_tune(params)
#             
#     if mean_binary_error < min_mean_binary_error:
#         min_mean_binary_error = mean_binary_error
#         print('min_mean_binary_error updated:',min_mean_binary_error)
#         best_params['feature_fraction'] = feature_fraction
#         
# if 'feature_fraction' in best_params:
#     params['feature_fraction'] = best_params['feature_fraction']
# else:
#     del params['feature_fraction']
# 
# print("调参4：降低过拟合")    
# for bagging_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#     for bagging_freq in range(0,30,5): #
#         print('============================',bagging_fraction,bagging_freq)
#         params['bagging_fraction'] = bagging_fraction
#         params['bagging_freq'] = bagging_freq
#         
#         mean_binary_error = train_tune(params)
#         
#         if mean_binary_error < min_mean_binary_error:
#             min_mean_binary_error = mean_binary_error
#             print('min_mean_binary_error updated:',min_mean_binary_error)
#             best_params['bagging_fraction'] = bagging_fraction
#             best_params['bagging_freq'] = bagging_freq
# 
# if 'bagging_fraction' in best_params:
#     params['bagging_fraction'] = best_params['bagging_fraction']
# else:
#     del params['bagging_fraction']
# if 'bagging_freq' in best_params:
#     params['bagging_freq'] = best_params['bagging_freq']
# else:
#     del params['bagging_freq']
# 
# print("调参5：降低过拟合")
# #min_merror = float('Inf')
# for lambda_l1 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#     for lambda_l2 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0,0.2,0.4,0.6,0.8,1.0]
#         print('============================',lambda_l1,lambda_l2)
#         params['lambda_l1'] = lambda_l1
#         params['lambda_l2'] = lambda_l2
#         
#         mean_binary_error = train_tune(params)
#         
#         if mean_binary_error < min_mean_binary_error:
#             min_mean_binary_error = mean_binary_error
#             print('min_mean_binary_error updated:',min_mean_binary_error)
#             best_params['lambda_l1'] = lambda_l1
#             best_params['lambda_l2'] = lambda_l2
# 
# if 'lambda_l1' in best_params:
#     params['lambda_l1'] = best_params['lambda_l1']
# else:
#     del params['lambda_l1']
# if 'lambda_l2' in best_params:
#     params['lambda_l2'] = best_params['lambda_l2']
# else:
#     del params['lambda_l2']
# 
# print("调参6：降低过拟合")
# for min_split_gain in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#     print('============================',min_split_gain)
#     params['min_split_gain'] = min_split_gain
#     
#     mean_binary_error = train_tune(params)
#         
#     if mean_binary_error < min_mean_binary_error:
#         min_mean_binary_error = mean_binary_error
#         print('min_mean_binary_error updated:',min_mean_binary_error)
#         best_params['min_split_gain'] = min_split_gain
# 
# if 'min_split_gain' in best_params:
#     params['min_split_gain'] = best_params['min_split_gain']
# else:
#     del params['min_split_gain']
# 
# print('Tuning params DONE, best_params:',best_params)
# print('all params:',params)
# =============================================================================

def train_func(params):
    print('Training...')
    evals_results = {}
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=1000,
                    early_stopping_rounds=20,
                    verbose_eval=10,
                    evals_result = evals_results
                    )
    optimum_boost_rounds = np.argmin(evals_results['valid_0']['binary_error'])
    print('Optimum boost rounds = {}'.format(optimum_boost_rounds))
    print('Min mean_binary_error = {}'.format(np.min(evals_results['valid_0']['binary_error'])))
    ## 提交用
    gbm = lgb.train(params=params,
                    train_set = lgb.Dataset(X_train_all, Y_train_all),
                    num_boost_round=optimum_boost_rounds,
                    verbose_eval=100,
                    )
    gbm.save_model(os.path.join(os.pardir,os.pardir,'Model/lgb_model.txt'))
    return gbm


def predict_func(gbm,threshold):
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
#params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': ['binary_error'],'num_leaves': 84, 'max_depth': 8, 'feature_fraction': 1, 'bagging_fraction': 1, 'bagging_freq': 5, 'lambda_l1': 0.3, 'lambda_l2': 0, 'min_split_gain': 0.7}
params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'binary_error', 'learning_rate': 0.01, 'verbose': 1, 'num_leaves': 98, 'feature_fraction': 0.1, 'bagging_fraction': 0.6, 'bagging_freq': 10, 'lambda_l1': 0, 'lambda_l2': 0.3, 'min_split_gain': 0.3}
gbm = train_func(params) 
#gbm = lgb.booster(model_file='model/lgb_model.txt')
predict_func(gbm,0.5)
    
time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')
    
    








