# -*- coding: utf-8 -*-
"""
Created on Thu May 31 20:09:01 2018

@author: ASUS
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

from xgboost.sklearn import XGBClassifier  
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt  
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

input_dir = os.path.join(os.pardir, 'Kesci-data')
print('Input files:\n{}'.format(os.listdir(input_dir)))
print('Loading data sets...')
register_df = pickle.load(open(os.path.join(input_dir, 'user_register.pkl'),"rb"))
launch_df = pickle.load(open(os.path.join(input_dir, 'app_launch.pkl'),"rb"))
video_df = pickle.load(open(os.path.join(input_dir, 'video_create.pkl'),"rb"))
activity_df = pickle.load(open(os.path.join(input_dir, 'user_activity.pkl'),"rb"))


def data_during(start_day,end_day):
    launch_df_selected = launch_df.loc[(launch_df['day'] >= start_day) & (launch_df['day'] <= end_day)]
    video_df_selected = video_df.loc[(video_df['day'] >= start_day) & (video_df['day'] <= end_day)]
    activity_df_selected = activity_df.loc[(activity_df['day'] >= start_day) & (activity_df['day'] <= end_day)]
    launch_freq = launch_df_selected.groupby('user_id').agg({'user_id': 'mean', 'day': 'count'})
    launch_freq.columns = ['user_id', 'launch_count']
    video_freq = video_df_selected.groupby('user_id').agg({'user_id': 'mean', 'day': 'count'})
    video_freq.columns = ['user_id', 'video_count']
    activity_freq = activity_df_selected.groupby('user_id').agg({'user_id': 'mean', 'day': 'count'})
    activity_freq.columns = ['user_id', 'activity_count']
    merged_df = launch_freq.merge(video_freq,how='outer',on='user_id')
    merged_df = merged_df.merge(activity_freq,how='outer',on='user_id')
    merged_df = merged_df.fillna(0)
    merged_df['total_count'] = np.sum(merged_df[['launch_count','video_count','activity_count']],axis = 1)
    return merged_df


def prepare_set(start_day,end_day):
    user_info = register_df.loc[(register_df['register_day'] >= start_day) & (register_df['register_day'] <= end_day)]
    x_raw = user_info.merge(data_during(end_day-6,end_day-0),how='left',on='user_id').fillna(0)
    x_raw = x_raw.merge(data_during(end_day-13,end_day-7),how='left',on='user_id').fillna(0)
    x_raw = x_raw.merge(data_during(end_day-22,end_day-14),how='left',on='user_id').fillna(0)
    x_raw = x_raw.merge(data_during(end_day-0,end_day-0),how='left',on='user_id').fillna(0)
    
    activity_df_selected = activity_df.loc[(activity_df['day'] >= start_day) & (activity_df['day'] <= end_day)]#训练集和测试集时间区间是否应保持一致？
    author_count = pd.DataFrame(activity_df_selected['author_id'].value_counts())
    author_count['index'] = author_count.index
    author_count.columns = ['author_count','author_id']
    x_raw = x_raw.merge(author_count,how='left',left_on='user_id',right_on='author_id').fillna(0)
    x_raw = x_raw.drop(['author_id'],axis=1)#改列名
    
    for i in range(6):
        action_freq = activity_df_selected[['user_id','action_type']].loc[activity_df_selected['action_type']==i]
        action_freq = action_freq.groupby('user_id').agg({'user_id': 'mean', 'action_type': 'count'})
        x_raw = x_raw.merge(action_freq,how='left',on='user_id').fillna(0)#改列名
    
    label_data = data_during(end_day+1,end_day+7)
    label_data['total_count'] = np.sum(label_data[['launch_count','video_count','activity_count']],axis = 1)
    label_data.loc[label_data['total_count'] > 1, 'total_count'] = 1
    label_data = label_data[['user_id','total_count']]
    xy_set = x_raw.merge(label_data,how='left',on='user_id').fillna(0)
    x = xy_set.drop(['user_id','total_count'],axis=1).values
    y = xy_set['total_count'].values
    return x,y

train_x,train_y = prepare_set(1,23)
test_x,test_y = prepare_set(1,30)


## 待修改参数：xgb.cv(metrics='auc',show_progress=False)
#clf.fit(eval_metric='auc')      xgb1 = XGBClassifier(scale_pos_weight=1)   gsearch1 = GridSearchCV(scoring='roc_auc',n_jobs=4)

# 这里原blog指定了xy_set和predictors(特征名)
def modelFit(clf,train_x,train_y,isCv=True,cv_folds=5,early_stopping_rounds=50):  
    if isCv:  
        xgb_param = clf.get_xgb_params()  
        xgtrain = xgb.DMatrix(train_x,label=train_y)  
        cvresult = xgb.cv(xgb_param,xgtrain,num_boost_round=clf.get_params()['n_estimators'],nfold=cv_folds,  
                          metrics='auc',early_stopping_rounds=early_stopping_rounds)#是否显示目前几颗树  
        clf.set_params(n_estimators=cvresult.shape[0]) 
    
    #训练  
    clf.fit(train_x,train_y,eval_metric='auc')  
  
    #预测  
    train_predictions = clf.predict(train_x)  
    train_predprob = clf.predict_proba(train_x)[:,1]#1的概率  
  
    #打印  
    print("\nModel Report")  
    print("Accuracy : %.4g" % metrics.accuracy_score(train_y, train_predictions))  
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, train_predprob))  
  
    feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)  
    feat_imp.plot(kind='bar',title='Feature importance')  
    plt.ylabel('Feature Importance Score')  

# 类别平衡的问题：min_child_weight=2,scale_pos_weight=0,
def tun_parameters(train_x,train_y):  
    xgb1 = XGBClassifier(learning_rate=0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,  
                         colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)  
    modelFit(xgb1,train_x,train_y)

## tuning max_depth and min_child_weight
# n_estimators根据上面结果改,n_jobs可以设置并行CPU核数
param_test1 = {
        'max_depth':range(3,10,2),
        'min_child_weight':range(1,6,2)
}  
gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=5, 
                                                min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,  
                                                objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27),
                        param_grid=param_test1,scoring='roc_auc',iid=False,cv=5)
gsearch1.fit(train_x,train_y)  
print(gsearch1.grid_scores_,gsearch1.best_params_,gsearch1.best_score_)
#若最佳参数在参数范围边缘，则扩大范围，也可以跟下个参数一起调节

param_test2 = {
        'max_depth':[4,5,6],
        'min_child_weight':[4,5,6]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27), 
                        param_grid=param_test2,scoring='roc_auc',iid=False,cv=5)
gsearch2.fit(train_x,train_y)
print(gsearch2.grid_scores_,gsearch2.best_params_,gsearch2.best_score_)

param_test2b = {
        'min_child_weight':[6,8,10,12]
}
gsearch2b = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=4,
                                                   min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                   objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                        param_grid = param_test2b, scoring='roc_auc',iid=False, cv=5)
gsearch2b.fit(train_x,train_y)
print(gsearch2b.grid_scores_,gsearch2b.best_params_,gsearch2b.best_score_)
#modelfit(gsearch2b.best_estimator_, train_x, train_y)

# tuning gamma
# max_depth and min_child_weight根据上面结果改
param_test3 = {
        'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=4,
                                                  min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                        param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(train_x,train_y)
print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)

xgb2 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelFit(xgb2,train_x,train_y)

# tuning subsample and colsample_bytree
param_test4 = {
        'subsample':[i/10.0 for i in range(6,10)],
        'colsample_bytree':[i/10.0 for i in range(6,10)]
}
# n_estimators根据xgb2改
gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=177, max_depth=4,
                                                  min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                        param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(train_x,train_y)
print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)

param_test5 = {
        'subsample':[i/100.0 for i in range(75,90,5)],
        'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=177, max_depth=4,
                                                  min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
                        param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
gsearch5.fit(train_x,train_y)
print(gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_)

# tuning Regularization Parameters
param_test6 = {
        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=177, max_depth=4,
                                                  min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                        param_grid = param_test6, scoring='roc_auc',iid=False, cv=5)
gsearch6.fit(train_x,train_y)
print(gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_)

param_test7 = {
        'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}
gsearch7 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=177, max_depth=4,
                                                  min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
                        param_grid = param_test7, scoring='roc_auc',iid=False, cv=5)
gsearch7.fit(train_x,train_y)
print(gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_)

xgb3 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.005,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelFit(xgb3,train_x,train_y)

xgb4 = XGBClassifier(
        learning_rate =0.01,
        n_estimators=5000,
        max_depth=4,
        min_child_weight=6,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.005,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelFit(xgb4,train_x,train_y)


