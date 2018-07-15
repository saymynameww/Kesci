# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:33:52 2018

@author: Administrator
"""

import os
import sys
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime
import lightgbm as lgb
from save_log import Logger
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")


def feature_selection(feature_mode,importance_threshold,R_threshold):
    train_path_a = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/train_a.csv')
    train_path_b = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/train_b.csv')
    test_path = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/test.csv')
    train_a = pd.read_csv(train_path_a)
    train_b = pd.read_csv(train_path_b)
    train = pd.concat([train_a,train_b],axis=0)
    test = pd.read_csv(test_path)
    test_userid = test.pop('user_id')
    if feature_mode == 1:
        print('Loading all the features and label...')
        train_feature = train.drop(['user_id','label'],axis=1)
        train_label = train['label']
        online_test_feature = test
        print('特征数：'+ str(train_feature.columns.size))
    elif feature_mode == 2:
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
        used_feature = feature_list[feature_importance>=importance_threshold]
        train_feature = train[used_feature]
        train_label = train['label']
        online_test_feature = test[used_feature]
        print('importance_threshold:'+str(importance_threshold)+' 特征数：'+ str(train_feature.columns.size))
    elif feature_mode == 3:
        print('Loading Pearson important features and label...')
        feature_list = train.columns.values.tolist()
        feature_list.remove('user_id')
        feature_list.remove('label')
        pearson = []
        for name in feature_list:
            pearson.append(pearsonr(train['label'], train[name]))
        pearson = pd.DataFrame(pearson).rename({0:'R_value',1:'P_value'},axis=1)
        pearson['Feature_name'] = feature_list
        P_threshold = 0.05
        used_feature = pearson[(pearson.P_value<=P_threshold) & ((pearson.R_value>=R_threshold)|(pearson.R_value<=-R_threshold))]
        used_feature = used_feature.Feature_name.tolist()
        train_feature = train[used_feature]
        train_label = train['label']
        online_test_feature = test[used_feature]
        print('R_threshold:'+str(R_threshold)+' 特征数：'+ str(train_feature.columns.size))
    train_feature, offline_test_feature, train_label, offline_test_label = train_test_split(train_feature, train_label, test_size=0.1,random_state=624)
    return train_feature,train_label,online_test_feature,test_userid,offline_test_feature,offline_test_label

def auc_score(params):
    lgb_train = lgb.Dataset(train_feature, train_label)
    lgb_eval = lgb.Dataset(offline_test_feature, offline_test_label, reference=lgb_train)
    
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=lgb_eval,
                    verbose_eval=False,
                    early_stopping_rounds=20)

    y_pred = gbm.predict(offline_test_feature, num_iteration=gbm.best_iteration)
    auc = roc_auc_score(offline_test_label,y_pred)
    return auc
        
def param_tune():
    print('Tuning params...')
    params = {
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric': 'auc',
              }
    max_auc = auc_score(params)
    print('best auc updated:',max_auc)
    best_params = {}
    
#    print("调参1：学习率")
#    for learning_rate in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]: 
#        print('============================',learning_rate)
#        params['learning_rate'] = learning_rate
#        auc = auc_score(params)
#        if auc > max_auc:
#            max_auc = auc
#            print('best auc updated:',max_auc)
#            best_params['learning_rate'] = learning_rate
#    if 'learning_rate' in best_params:
#        params['learning_rate'] = best_params['learning_rate']
#    else:
#        del params['learning_rate']
    params['learning_rate'] = 0.01
    
#    print("调参2：提高准确率")
#    for num_leaves in range(15,45,1): #(20,200,10)
#        for max_depth in range(3,10,1): #(3,9,1)
#            print('============================',num_leaves,max_depth)
#            params['num_leaves'] = num_leaves
#            params['max_depth'] = max_depth
#            auc = auc_score(params)
#            if auc > max_auc:
#                max_auc = auc
#                print('best auc updated:',max_auc)
#                best_params['num_leaves'] = num_leaves
#                best_params['max_depth'] = max_depth
#    if 'num_leaves' in best_params:
#        params['num_leaves'] = best_params['num_leaves']
#        params['max_depth'] = best_params['max_depth']
#    else:
#        del params['num_leaves'],params['max_depth']
    params['num_leaves'] = 23
    params['max_depth'] = 9
    
    print("调参3：降低过拟合")
    for min_data_in_leaf in range(10,800,10): #(10,200,5)
        print('============================',min_data_in_leaf)
        params['min_data_in_leaf'] = min_data_in_leaf
        auc = auc_score(params)
        if auc > max_auc:
            max_auc = auc
            print('best auc updated:',max_auc)
            best_params['min_data_in_leaf'] = min_data_in_leaf
    if 'min_data_in_leaf' in best_params:
        params['min_data_in_leaf'] = best_params['min_data_in_leaf']
    else:
        del params['min_data_in_leaf']
    params['min_data_in_leaf'] = 220
    
#    print("调参4：采样")
#    for feature_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#        for bagging_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#            for bagging_freq in range(0,50,5): #
#                print('============================',feature_fraction,bagging_fraction,bagging_freq)
#                params['feature_fraction'] = feature_fraction
#                params['bagging_fraction'] = bagging_fraction
#                params['bagging_freq'] = bagging_freq
#                auc = auc_score(params)
#                if auc > max_auc:
#                    max_auc = auc
#                    print('best auc updated:',max_auc)
#                    best_params['feature_fraction'] = feature_fraction
#                    best_params['bagging_fraction'] = bagging_fraction
#                    best_params['bagging_freq'] = bagging_freq
#    if 'feature_fraction' in best_params:
#        params['feature_fraction'] = best_params['feature_fraction']
#        params['bagging_fraction'] = best_params['bagging_fraction']
#        params['bagging_freq'] = best_params['bagging_freq']
#    else:
#        del params['feature_fraction'],params['bagging_fraction'],params['bagging_freq']
    params['feature_fraction'] = 0.9
    params['bagging_fraction'] = 0.8
    params['bagging_freq'] = 5
    
    
#    print("调参5：正则化")
#    for lambda_l1 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#        for lambda_l2 in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]: #[0,0.2,0.4,0.6,0.8,1.0]
#            for min_split_gain in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#                print('============================',lambda_l1,lambda_l2,min_split_gain)
#                params['lambda_l1'] = lambda_l1
#                params['lambda_l2'] = lambda_l2
#                params['min_split_gain'] = min_split_gain
#                auc = auc_score(params)
#                if auc > max_auc:
#                    max_auc = auc
#                    print('best auc updated:',max_auc)
#                    best_params['lambda_l1'] = lambda_l1
#                    best_params['lambda_l2'] = lambda_l2
#                    best_params['min_split_gain'] = min_split_gain
#    if 'lambda_l1' in best_params:
#        params['lambda_l1'] = best_params['lambda_l1']
#        params['lambda_l2'] = best_params['lambda_l2']
#        params['min_split_gain'] = best_params['min_split_gain']
#    else:
#        del params['lambda_l1'],params['lambda_l2'],params['min_split_gain']
    params['lambda_l1'] = 1
    params['lambda_l2'] = 1
    params['min_split_gain'] = 1
    
    print('Tuning params DONE, best_params:',best_params)
    return params

def train_func(params):
    cv_auc = []
    offline_auc = []
    cv_prediction = []
    model_i = 0
    print('All params:',params)
    skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=624)
    for train_in,test_in in skf.split(train_feature,train_label):
        if type(train_feature)==pd.core.frame.DataFrame:
            X_train,X_test,y_train,y_test = train_feature.iloc[train_in],train_feature.iloc[test_in],train_label.iloc[train_in],train_label.iloc[test_in]
        elif type(train_feature)==np.ndarray:
            X_train,X_test,y_train,y_test = train_feature[train_in],train_feature[test_in],train_label[train_in],train_label[test_in]
    
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_sets=lgb_eval,
                        verbose_eval=False,
                        early_stopping_rounds=50)
        
        model_path = os.path.join(os.pardir,os.pardir, 'Model/')
        gbm.save_model(model_path+'model_'+str(model_i)+'.txt')
        model_i += 1
    
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        cv_auc.append(roc_auc_score(y_test,y_pred))
        offline_y_pred = gbm.predict(offline_test_feature, num_iteration=gbm.best_iteration)
        offline_auc.append(roc_auc_score(offline_test_label,offline_y_pred))
        cv_prediction.append(gbm.predict(online_test_feature, num_iteration=gbm.best_iteration))
    return cv_auc,offline_auc,cv_prediction
        
def predict_func():
    mean_cv_auc = np.sum(cv_auc)/N
    mean_offline_auc = np.sum(offline_auc)/N
    mean_cv_prediction = np.sum(cv_prediction,axis=0)/N
    print('mean_cv_auc:',mean_cv_auc)
    print('mean_offline_auc:',mean_offline_auc)
    if is_save_result:
        result = pd.DataFrame()
        result['userid'] = list(test_userid.values)
        result['probability'] = list(mean_cv_prediction)
        time_date = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
        submit_file_name = '%s_%s.csv'%(str(time_date),str(mean_cv_auc).split('.')[1])
        result.to_csv(submit_file_name,index=False,sep=',')
        print(submit_file_name+' 线上:{}')
    
    model_path = os.path.join(os.pardir,os.pardir, 'Model/')
    gbm = lgb.Booster(model_file=model_path+'model_'+str(1)+'.txt')
    if show_importance==1:
        print('所用特征重要性：'+ str(list(gbm.feature_importance())))
    fig, ax = plt.subplots(1, 1, figsize=[16, 80])
    lgb.plot_importance(gbm, ax=ax, max_num_features=400)
    plt.savefig('feature_importance.png')
        
if __name__ == "__main__":
    time_start = datetime.now()
    print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))
    stdout_backup = sys.stdout
#    sys.stdout = Logger("train_info.txt")
    print('\n')
    
    N = 5
    show_importance = False
    is_save_result = False
    param_mode = 2
    train_feature,train_label,online_test_feature,test_userid,offline_test_feature,offline_test_label = \
    feature_selection(feature_mode=1,importance_threshold=1,R_threshold=0.05) # 1:all 2:importance 3:pearson
    if param_mode == 1:
        params = param_tune()
    elif param_mode == 2:
        params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'verbose': 1, 'learning_rate': 0.07, 'num_leaves': 32, 'max_depth': 5, 'min_data_in_leaf': 140, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'lambda_l1': 1, 'lambda_l2': 1}#, 'min_split_gain': 1
    cv_auc,offline_auc,cv_prediction = train_func(params)
    predict_func()
    
    sys.stdout = stdout_backup
    time_end = datetime.now()
    print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
    print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')








