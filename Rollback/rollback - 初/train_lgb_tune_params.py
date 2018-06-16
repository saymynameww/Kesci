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
train_path = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/train.csv')
test_path = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/test.csv')
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# =============================================================================
# used_feature = ['create_count', 'create_day_diff_mean', 'create_day_diff_std', 'create_day_diff_max',
#                 'create_day_diff_min', 'create_mode', 'last_day_cut_max_day',
#                 'register_type', 'device_type', 'register_day_cut_max_day',
#                 'launch_count', 'launch_day_diff_mean', 'launch_day_diff_std',
#                 'launch_day_diff_max', 'launch_day_diff_min', 'launch_day_diff_kur',
#                 'launch_day_diff_ske', 'launch_day_diff_last', 'launch_day_cut_max_day',
#                 'activity_count',
#                 'activity_day_diff_mean',
#                 'activity_day_diff_std',
#                 'activity_day_diff_max', 'activity_day_diff_min', 'activity_day_diff_kur',
#                 'activity_day_diff_ske',
#                 'activity_day_diff_last',
#                 '0_page_count', '1_page_count', '2_page_count', '3_page_count', '4_page_count',
#                 '0_page_count_div_sum', '1_page_count_div_sum', '2_page_count_div_sum',
#                 '3_page_count_div_sum', '4_page_count_div_sum',
#                 '0_action_count',
#                 '1_action_count', '2_action_count', '3_action_count', '4_action_count',
#                 '5_action_count', '0_action_count_div_sum', '1_action_count_div_sum',
#                 '2_action_count_div_sum', '3_action_count_div_sum',
#                 '4_action_count_div_sum', '5_action_count_div_sum',
#                 'video_id_mode', 'author_id_mode', 'activity_count_mean',
#                 'activity_count_std', 'activity_count_max', 'activity_count_min',
#                 'activity_count_kur', 'activity_count_ske', 'activity_count_last',
#                 'activity_diff_count_mean', 'activity_diff_count_std', 'activity_diff_count_max',
#                 'activity_diff_count_min', 'activity_diff_count_kur', 'activity_diff_count_ske',
#                 'activity_diff_count_last', 'activity_page0_mean', 'activity_page0_std',
#                 'activity_page0_max', 'activity_page0_min', 'activity_page0_kur', 'activity_page0_ske',
#                 'activity_page0_last', 'activity_page1_mean', 'activity_page1_std', 'activity_page1_max',
#                 'activity_page1_min', 'activity_page1_kur', 'activity_page1_ske', 'activity_page1_last',
#                 'activity_page2_mean', 'activity_page2_std', 'activity_page2_max', 'activity_page2_min',
#                 'activity_page2_kur', 'activity_page2_ske', 'activity_page2_last', 'activity_page3_mean',
#                 'activity_page3_std', 'activity_page3_max', 'activity_page3_min', 'activity_page3_kur',
#                 'activity_page3_ske', 'activity_page3_last', 'activity_page4_mean', 'activity_page4_std',
#                 'activity_page4_max', 'activity_page4_min', 'activity_page4_kur', 'activity_page4_ske',
#                 'activity_page4_last', 'activity_type0_mean', 'activity_type0_std', 'activity_type0_max',
#                 'activity_type0_min', 'activity_type0_kur', 'activity_type0_ske', 'activity_type0_last',
#                 'activity_type1_mean', 'activity_type1_std', 'activity_type1_max', 'activity_type1_min',
#                 'activity_type1_kur', 'activity_type1_ske', 'activity_type1_last', 'activity_type2_mean',
#                 'activity_type2_std', 'activity_type2_max', 'activity_type2_min', 'activity_type2_kur',
#                 'activity_type2_ske', 'activity_type2_last', 'activity_type3_mean', 'activity_type3_std',
#                 'activity_type3_max', 'activity_type3_min', 'activity_type3_kur', 'activity_type3_ske',
#                 'activity_type3_last', 'activity_type4_mean', 'activity_type4_std', 'activity_type4_max',
#                 'activity_type4_min', 'activity_type4_kur', 'activity_type4_ske', 'activity_type4_last',
#                 'activity_type5_mean', 'activity_type5_std', 'activity_type5_max', 'activity_type5_min',
#                 'activity_type5_kur', 'activity_type5_ske', 'activity_type5_last', 'activity_day_cut_max_day',
#                 'max_activity_day',
#                  'create_sub_register', 'activity_sub_register', 'launch_sub_register',
#                 ]
# used_feature = np.array(used_feature)
# print(used_feature)
# importance_feature = [21,60,54,71,106,44,50,27,33,58,43,11,19,35,64,32,45,9,37,143,142,10,7,18,8]
# used_feature = used_feature[np.array(importance_feature)]
# print(used_feature)
# train_feature = train[used_feature]
# test_feature = test[used_feature]
# label = train['label']
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


print('Tuning params...')
# =============================================================================
# ### 设置初始参数--不含交叉验证参数
# print('设置参数')
# params = {
#           'boosting_type': 'gbdt',
#           'objective': 'binary',
#           'metric': 'binary_logloss',
#           }
# 
# ### 交叉验证(调参)
# print('交叉验证')
# min_merror = float('Inf')
# best_params = {}
# 
# # 准确率
# print("调参1：提高准确率")
# for num_leaves in range(39,42,1):
#     for max_depth in range(5,8,1):
#         print('============================',num_leaves,max_depth)
#         params['num_leaves'] = num_leaves
#         params['max_depth'] = max_depth
# 
#         cv_results = lgb.cv(
#                             params,
#                             lgb_train,
#                             seed=1017,
#                             nfold=3,
#                             metrics=['binary_error'],
#                             early_stopping_rounds=10,
#                             verbose_eval=10
#                             )
#             
#         mean_merror = pd.Series(cv_results['binary_error-mean']).min()
#         boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()
#             
#         if mean_merror < min_merror:
#             min_merror = mean_merror
#             best_params['num_leaves'] = num_leaves
#             best_params['max_depth'] = max_depth
#             
# params['num_leaves'] = best_params['num_leaves']
# params['max_depth'] = best_params['max_depth']
# 
# # 过拟合
# min_merror = float('Inf')
# print("调参2：降低过拟合")
# for max_bin in range(1,3,1):
#     for min_data_in_leaf in range(33,38,1):
#             print('============================',max_bin,min_data_in_leaf)
#             params['max_bin'] = max_bin
#             params['min_data_in_leaf'] = min_data_in_leaf
#             
#             cv_results = lgb.cv(
#                                 params,
#                                 lgb_train,
#                                 seed=1017,
#                                 nfold=3,
#                                 metrics=['binary_error'],
#                                 early_stopping_rounds=3,
#                                 verbose_eval=10
#                                 )
#                     
#             mean_merror = pd.Series(cv_results['binary_error-mean']).min()
#             boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()
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
# min_merror = float('Inf')
# for feature_fraction in [0.85,0.9,0.95]:
#     for bagging_fraction in [0.85,0.9,0.95]:
#         for bagging_freq in range(4,7,1):
#             print('============================',feature_fraction,bagging_fraction,bagging_freq)
#             params['feature_fraction'] = feature_fraction
#             params['bagging_fraction'] = bagging_fraction
#             params['bagging_freq'] = bagging_freq
#             
#             cv_results = lgb.cv(
#                                 params,
#                                 lgb_train,
#                                 seed=1017,
#                                 nfold=3,
#                                 metrics=['binary_error'],
#                                 early_stopping_rounds=3,
#                                 verbose_eval=10
#                                 )
#                     
#             mean_merror = pd.Series(cv_results['binary_error-mean']).min()
#             boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()
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
# min_merror = float('Inf')
# for lambda_l1 in [0.9,1.0]:
#     for lambda_l2 in [0.6,0.7,0.8,0.9]:
#         for min_split_gain in [0.0,0.1]:
#             print('============================',lambda_l1,lambda_l2,min_split_gain)
#             params['lambda_l1'] = lambda_l1
#             params['lambda_l2'] = lambda_l2
#             params['min_split_gain'] = min_split_gain
#             
#             cv_results = lgb.cv(
#                                 params,
#                                 lgb_train,
#                                 seed=1017,
#                                 nfold=3,
#                                 metrics=['binary_error'],
#                                 early_stopping_rounds=3,
#                                 verbose_eval=10
#                                 )
#                     
#             mean_merror = pd.Series(cv_results['binary_error-mean']).min()
#             boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()
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
# print('best_params:',best_params)
# =============================================================================

params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': ['auc'], 'num_leaves': 40, 'max_depth': 6, 'verbose': 1, 'max_bin': 249, 'min_data_in_leaf': 36, 'feature_fraction': 0.85, 'bagging_fraction': 0.9, 'bagging_freq': 5, 'lambda_l1': 1.0, 'lambda_l2': 0.7, 'min_split_gain': 0.0, 'learning_rate': 0.01}
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
    gbm.save_model(os.path.join(os.pardir,os.pardir,'Model/lgb_model.txt'))
    return gbm


def predict_func(gbm):
    print('Validating...')
    temp = gbm.predict(X_test)
    valid_result = pd.DataFrame()
    valid_result['user_id'] = X_test.reset_index()['index']
    valid_result['result'] = temp
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
    df_result.to_csv(os.path.join(os.pardir,os.pardir,'result/lgb_result.csv'), index=False)
    prediction[prediction >= threshold]=1
    prediction[prediction < threshold]=0
    prediction = list(map(int,prediction))
    print('为1的个数：' + str(len(np.where(np.array(prediction)==1)[0])))
    print('为0的个数：' + str(len(np.where(np.array(prediction)==0)[0])))

gbm = train_func(params) 
predict_func(gbm)
    
time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')





