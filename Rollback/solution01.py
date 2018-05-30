# -*- coding: utf-8 -*-
"""
Created on Sat May 26 09:45:16 2018

@author: wxk06
"""

import os
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt

input_dir = os.path.join(os.pardir, 'Kesci-data')
print('Input files:\n{}'.format(os.listdir(input_dir)))
print('Loading data sets...')

# =============================================================================
# # add columns name
# app_launch_df = pd.read_table(os.path.join(input_dir, 'app_launch_log.txt'),header=None)
# user_register_df = pd.read_table(os.path.join(input_dir, 'user_register_log.txt'),header=None)
# video_create_df = pd.read_table(os.path.join(input_dir, 'video_create_log.txt'),header=None)
# user_activity_df = pd.read_table(os.path.join(input_dir, 'user_activity_log.txt'),header=None)
#  
# app_launch_df.columns = ['user_id','day']
# user_register_df.columns = ['user_id','register_day','register_type','device_type']
# video_create_df.columns = ['user_id','day']
# user_activity_df.columns = ['user_id','day','page','video_id','author_id','action_type']
# 
# pickle.dump(app_launch_df,open(os.path.join(input_dir, 'app_launch.pkl'),"wb")) 
# pickle.dump(user_register_df,open(os.path.join(input_dir, 'user_register.pkl'),"wb")) 
# pickle.dump(video_create_df,open(os.path.join(input_dir, 'video_create.pkl'),"wb")) 
# pickle.dump(user_activity_df,open(os.path.join(input_dir, 'user_activity.pkl'),"wb")) 
# =============================================================================

launch_df = pickle.load(open(os.path.join(input_dir, 'app_launch.pkl'),"rb"))
register_df = pickle.load(open(os.path.join(input_dir, 'user_register.pkl'),"rb"))
video_df = pickle.load(open(os.path.join(input_dir, 'video_create.pkl'),"rb"))
activity_df = pickle.load(open(os.path.join(input_dir, 'user_activity.pkl'),"rb"))

print('preparing data sets...')
def select_df(start_day,end_day,original_df):
    df_selected = original_df.loc[(original_df['day'] >= start_day) & (original_df['day'] <= end_day)]
    return df_selected

def cal_freq(df,agg_funs,groupby_name,columns_names):
    freq = df.groupby(groupby_name).agg(agg_funs)
    freq.columns = columns_names
    return freq

def create_data(start_day,end_day,launch_df,video_df,activity_df,register_df):
    launch_df = select_df(start_day,end_day,launch_df)
    video_df = select_df(start_day,end_day,video_df)
    activity_df = select_df(start_day,end_day,activity_df)
    agg_funs = {'user_id': 'mean', 'day': 'count'}
    launch_freq = cal_freq(launch_df,agg_funs,groupby_name='user_id',columns_names=['user_id', 'launch_count'])
    video_freq = cal_freq(video_df,agg_funs,groupby_name='user_id',columns_names=['user_id', 'video_count'])
    activity_freq = cal_freq(activity_df,agg_funs,groupby_name='user_id',columns_names=['user_id', 'activity_count'])
    merged_df = register_df.merge(launch_freq, on='user_id', right_index=True, how='left')
    merged_df = merged_df.merge(video_freq, on='user_id', right_index=True, how='left')
    merged_df = merged_df.merge(activity_freq, on='user_id', right_index=True, how='left')
    return merged_df

# training data
x_train_start = 17; x_train_end = 23; y_train_start = 24; y_train_end = 30;
merged_df_x_train = create_data(x_train_start,x_train_end,launch_df,video_df,activity_df,register_df)
merged_df_y_train = create_data(y_train_start,y_train_end,launch_df,video_df,activity_df,register_df)
merged_df_y_train = merged_df_y_train.fillna(0)
merged_df_y_train['total_count'] = np.sum(merged_df_y_train[['video_count','activity_count']],axis = 1)
merged_df_y_train.loc[merged_df_y_train['total_count'] > 1, 'total_count'] = 1
drop_elements = ['launch_count','video_count','activity_count','register_day','register_type','device_type']
merged_df_y_train = merged_df_y_train.drop(drop_elements,axis=1)
merged_df_train = merged_df_x_train.merge(merged_df_y_train, on='user_id', left_index=True, how='left')
merged_df_train = merged_df_train.fillna(0)
x_train = merged_df_train.drop(['user_id','total_count'],axis=1).values
y_train = merged_df_train['total_count'].values

# test data
x_test_start = 24; x_test_end = 30; y_test_start = 24; y_test_end = 30;
merged_df_x_test = create_data(x_test_start,x_test_end,launch_df,video_df,activity_df,register_df)
merged_df_y_test = create_data(y_test_start,y_test_end,launch_df,video_df,activity_df,register_df)
merged_df_y_test = merged_df_y_test.fillna(0)
merged_df_y_test['total_count'] = np.sum(merged_df_y_test[['video_count','activity_count']],axis = 1)
merged_df_y_test.loc[merged_df_y_test['total_count'] > 1, 'total_count'] = 1
drop_elements = ['launch_count','video_count','activity_count','register_day','register_type','device_type']
merged_df_y_test = merged_df_y_test.drop(drop_elements,axis=1)
merged_df_test = merged_df_x_test.merge(merged_df_y_test, on='user_id', left_index=True, how='left')
merged_df_test = merged_df_test.fillna(0)
x_test = merged_df_test.drop(['user_id','total_count'],axis=1).values
y_test = merged_df_test['total_count'].values


print('training...')
gbm = xgb.XGBClassifier(
        #learning_rate = 0.02,
        n_estimators= 2000,
        max_depth= 4,
        min_child_weight= 2,
        #gamma=1,
        gamma=0.9,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread= -1,
        scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)
active_id = pd.DataFrame(data=predictions,columns=['isactive'])
result = merged_df_test.loc[active_id['isactive'] > 0]['user_id']
result.to_csv('result.txt', index=False)

accuracy = np.sum((y_test == predictions),axis = 0)/len(predictions)

# F1 score
M = np.sum(predictions)
N = np.sum(y_test)
MandN = np.sum((predictions + y_test)>1)
precision = MandN/M
recall = MandN/N
F1_score = 2*precision*recall/(precision + recall)

print(accuracy,F1_score)

# =============================================================================
# def get_xgb_feat_importances(clf):
# 
#     if isinstance(clf, xgb.XGBModel):
#         # clf has been created by calling
#         # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()
#         fscore = clf.booster().get_fscore()
#     else:
#         # clf has been created by calling xgb.train.
#         # Thus, clf is an instance of xgb.Booster.
#         fscore = clf.get_fscore()
# 
#     feat_importances = []
#     for ft, score in fscore.iteritems():
#         feat_importances.append({'Feature': ft, 'Importance': score})
#     feat_importances = pd.DataFrame(feat_importances)
#     feat_importances = feat_importances.sort_values(
#         by='Importance', ascending=False).reset_index(drop=True)
#     # Divide the importances by the sum of all importances
#     # to get relative importances. By using relative importances
#     # the sum of all importances will equal to 1, i.e.,
#     # np.sum(feat_importances['importance']) == 1
#     feat_importances['Importance'] /= feat_importances['Importance'].sum()
#     # Print the most important features and their importances
# #    print feat_importances.head()
#     return feat_importances
# 
# feat_importances = get_xgb_feat_importances(gbm)
# #fig, ax = plt.subplots(figsize=(12,18))
# #xgb.plot_importance(gbm, max_num_features=50, height=0.8,ax=ax)
# #plt.show()
# =============================================================================

effect_record = [x_train_start,x_train_end,y_train_start,y_train_end,x_test_start,x_test_end,y_test_start,y_test_end,accuracy,F1_score]
effect_record = pd.DataFrame(data=effect_record).T
effect_record.to_csv('effect_record.csv', index=False, header=False, mode='a')














