# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 20:00:17 2018

@author: ASUS
"""

import os
import pandas as pd
import numpy as np

dataset_1_feat_dir = os.path.join(os.pardir, 'Kesci-data-dealt/dataset_1_feat')
dataset_1_label_dir = os.path.join(os.pardir, 'Kesci-data-dealt/dataset_1_label')
dataset_2_feat_dir = os.path.join(os.pardir, 'Kesci-data-dealt/dataset_2_feat')
dataset_2_label_dir = os.path.join(os.pardir, 'Kesci-data-dealt/dataset_2_label')
dataset_3_feat_dir = os.path.join(os.pardir, 'Kesci-data-dealt/dataset_3_feat')

train_path = os.path.join(os.pardir, 'Kesci-data-dealt/train_and_test/train.csv')
test_path = os.path.join(os.pardir, 'Kesci-data-dealt/train_and_test/test.csv')

def get_train_label(feat_path,label_path):
    feat_register = pd.read_csv(feat_path + '/register.csv', usecols=['user_id'])
    feat_launch = pd.read_csv(feat_path + '/launch.csv', usecols=['user_id'])
    feat_video = pd.read_csv(feat_path + '/video.csv', usecols=['user_id'])
    feat_activity = pd.read_csv(feat_path + '/activity.csv', usecols=['user_id'])
    feat_data_id = np.unique(pd.concat([feat_register,feat_launch,feat_video,feat_activity]))
    
#    label_register = pd.read_csv(label_path + '/register.csv', usecols=['user_id'])
    label_launch = pd.read_csv(label_path + '/launch.csv', usecols=['user_id'])
    label_video = pd.read_csv(label_path + '/video.csv', usecols=['user_id'])
    label_activity = pd.read_csv(label_path + '/activity.csv', usecols=['user_id'])
    label_data_id = np.unique(pd.concat([label_launch,label_video,label_activity]))
    
    train_label = []
    for i in feat_data_id:
        if i in label_data_id:
            train_label.append(1)
        else:
            train_label.append(0)
    train_data = pd.DataFrame()
    train_data['user_id'] = feat_data_id
    train_data['label'] = train_label
    return train_data

def get_test_id(feat_path):
    feat_register = pd.read_csv(feat_path + '/register.csv', usecols=['user_id'])
    feat_launch = pd.read_csv(feat_path + '/launch.csv', usecols=['user_id'])
    feat_video = pd.read_csv(feat_path + '/video.csv', usecols=['user_id'])
    feat_activity = pd.read_csv(feat_path + '/activity.csv', usecols=['user_id'])
    feat_data_id = np.unique(pd.concat([feat_register,feat_launch,feat_video,feat_activity]))
    test_data = pd.DataFrame()
    test_data['user_id'] = feat_data_id
    return test_data

def get_register_feature(register):
    feature = pd.DataFrame()
    feature['user_id'] = register['user_id'].drop_duplicates()
    feature['register_type'] = register['register_type']
    feature['device_type'] = register['device_type']
    return feature

def get_launch_feature(launch):
    feature = pd.DataFrame()
    feature['user_id'] = launch['user_id'].drop_duplicates()
    end_day = np.max(launch['launch_day'])
    launch_count = launch[['user_id']].groupby(['user_id']).size().rename('launch_count').reset_index()
    launch_day_diff = pd.concat([launch['user_id'],launch.groupby(['user_id']).diff().rename({'launch_day':'launch_day_diff'},axis='columns')],axis=1)
    launch_day_diff_max = launch_day_diff.groupby(['user_id'])['launch_day_diff'].max().rename('launch_day_diff_max').reset_index()
    launch_day_diff_min = launch_day_diff.groupby(['user_id'])['launch_day_diff'].min().rename('launch_day_diff_min').reset_index()
    launch_day_diff_mean = launch_day_diff.groupby(['user_id'])['launch_day_diff'].mean().rename('launch_day_diff_mean').reset_index()
    launch_day_diff_std = launch_day_diff.groupby(['user_id'])['launch_day_diff'].std().rename('launch_day_diff_std').reset_index()
#    launch_day_diff_kurt = launch_day_diff.groupby(['user_id'])['launch_day_diff'].kurt().rename('launch_day_diff_kurt').reset_index()
    launch_day_diff_skew = launch_day_diff.groupby(['user_id'])['launch_day_diff'].skew().rename('launch_day_diff_skew').reset_index()
    launch_day_diff_last = launch_day_diff.groupby(['user_id'])['launch_day_diff'].last().rename('launch_day_diff_last').reset_index()
    launch_rest_day = (launch.groupby(['user_id'])['launch_day'].max()-end_day).rename('launch_rest_day').reset_index()
    feature = pd.merge(feature,launch_count,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_max,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_min,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_mean,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_std,how='left',on='user_id')
#    feature = pd.merge(feature,launch_day_diff_kurt,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_skew,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_last,how='left',on='user_id')
    feature = pd.merge(feature,launch_rest_day,how='left',on='user_id')
    return feature

def get_video_feature(video):
    feature = pd.DataFrame()
    feature['user_id'] = video['user_id'].drop_duplicates()
    end_day = np.max(video['video_day'])
    video_count = video[['user_id']].groupby(['user_id']).size().rename('video_count').reset_index()
    video_day_diff = pd.concat([video['user_id'],video.groupby(['user_id']).diff().rename({'video_day':'video_day_diff'},axis='columns')],axis=1)
    video_day_diff_max = video_day_diff.groupby(['user_id'])['video_day_diff'].max().rename('video_day_diff_max').reset_index()
    video_day_diff_min = video_day_diff.groupby(['user_id'])['video_day_diff'].min().rename('video_day_diff_min').reset_index()
    video_day_diff_mean = video_day_diff.groupby(['user_id'])['video_day_diff'].mean().rename('video_day_diff_mean').reset_index()
    video_day_diff_std = video_day_diff.groupby(['user_id'])['video_day_diff'].std().rename('video_day_diff_std').reset_index()
#    video_day_diff_kurt = video_day_diff.groupby(['user_id'])['video_day_diff'].kurt().rename('video_day_diff_kurt').reset_index()
    video_day_diff_skew = video_day_diff.groupby(['user_id'])['video_day_diff'].skew().rename('video_day_diff_skew').reset_index()
    video_day_diff_last = video_day_diff.groupby(['user_id'])['video_day_diff'].last().rename('video_day_diff_last').reset_index()
    video_rest_day = (video.groupby(['user_id'])['video_day'].max()-end_day).rename('video_rest_day').reset_index()
    feature = pd.merge(feature,video_count,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_max,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_min,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_mean,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_std,how='left',on='user_id')
#    feature = pd.merge(feature,video_day_diff_kurt,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_skew,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_last,how='left',on='user_id')
    feature = pd.merge(feature,video_rest_day,how='left',on='user_id')
    return feature

def get_activity_feature(activity):
    feature = pd.DataFrame()
    feature['user_id'] = activity['user_id'].drop_duplicates()
    end_day = np.max(activity['activity_day'])
    activity_count = activity[['user_id']].groupby(['user_id']).size().rename('activity_count').reset_index()
#    activity_day_diff = pd.concat([activity['user_id'],activity.groupby(['user_id']).diff().rename({'activity_day':'activity_day_diff'},axis='columns')],axis=1)
#    activity_day_diff_max = activity_day_diff.groupby(['user_id'])['activity_day_diff'].max().rename('activity_day_diff_max').reset_index()
#    activity_day_diff_min = activity_day_diff.groupby(['user_id'])['activity_day_diff'].min().rename('activity_day_diff_min').reset_index()
#    activity_day_diff_mean = activity_day_diff.groupby(['user_id'])['activity_day_diff'].mean().rename('activity_day_diff_mean').reset_index()
#    activity_day_diff_std = activity_day_diff.groupby(['user_id'])['activity_day_diff'].std().rename('activity_day_diff_std').reset_index()
#    activity_day_diff_kurt = activity_day_diff.groupby(['user_id'])['activity_day_diff'].kurt().rename('activity_day_diff_kurt').reset_index()
#    activity_day_diff_skew = activity_day_diff.groupby(['user_id'])['activity_day_diff'].skew().rename('activity_day_diff_skew').reset_index()
#    activity_day_diff_last = activity_day_diff.groupby(['user_id'])['activity_day_diff'].last().rename('activity_day_diff_last').reset_index()
#    activity_rest_day = (activity.groupby(['user_id'])['activity_day'].max()-end_day).rename('activity_rest_day').reset_index()
    feature = pd.merge(feature,activity_count,how='left',on='user_id')
#    feature = pd.merge(feature,activity_day_diff_max,how='left',on='user_id')
#    feature = pd.merge(feature,activity_day_diff_min,how='left',on='user_id')
#    feature = pd.merge(feature,activity_day_diff_mean,how='left',on='user_id')
#    feature = pd.merge(feature,activity_day_diff_std,how='left',on='user_id')
#    feature = pd.merge(feature,activity_day_diff_kurt,how='left',on='user_id')
#    feature = pd.merge(feature,activity_day_diff_skew,how='left',on='user_id')
#    feature = pd.merge(feature,activity_day_diff_last,how='left',on='user_id')
#    feature = pd.merge(feature,activity_rest_day,how='left',on='user_id')

    author_count = activity.groupby('author_id').agg({'author_id': 'mean', 'activity_day': 'count'})
    author_count.columns = ['user_id','author_count']
    feature = pd.merge(feature,author_count,how='left',on='user_id')
    for i in range(6):
        action_freq = activity[['user_id','action_type']].loc[activity['action_type']==i]
        action_freq = action_freq.groupby('user_id').agg({'user_id': 'mean', 'action_type': 'count'})
        action_freq.columns = ['user_id','action_type_'+str(i)+'_count']
        feature = pd.merge(feature,action_freq,how='left',on='user_id')
    return feature
    
def deal_feature(path, user_id):
    register = pd.read_csv(path + '/register.csv')
    launch = pd.read_csv(path + '/launch.csv')
    video = pd.read_csv(path + '/video.csv')
    activity = pd.read_csv(path + '/activity.csv')
    feature = pd.DataFrame()
    feature['user_id'] = user_id
    
    print('getting video feature...')
    video_feature = get_video_feature(video)
    feature = pd.merge(feature, video_feature, on='user_id', how='left')
    
    print('getting register feature...')
    register_feature = get_register_feature(register)
    feature = pd.merge(feature, register_feature, on='user_id', how='left')
    
    print('getting launch feature...')
    launch_feature = get_launch_feature(launch)
    feature = pd.merge(feature, launch_feature, on='user_id', how='left')
    
    
    
    print('getting activity feature...')
    activity_feature = get_activity_feature(activity)
    feature = pd.merge(feature,activity_feature, on='user_id', how='left')
    
    feature = feature.fillna(0)
    return feature

def get_data_feature():
    print('Getting train data 1 ...')
    train_label_1 = get_train_label(dataset_1_feat_dir,dataset_1_label_dir)
    data_1 = deal_feature(dataset_1_feat_dir,train_label_1['user_id'])
    data_1['label'] = train_label_1['label']
    
    print('Getting train data 2 ...')
    train_label_2 = get_train_label(dataset_2_feat_dir,dataset_2_label_dir)
    data_2 = deal_feature(dataset_2_feat_dir,train_label_2['user_id'])
    data_2['label'] = train_label_2['label']
    
    train_data = pd.concat([data_1,data_2])
    train_data.to_csv(train_path,index=False)
    
    print('Getting test data...')
    test_id = get_test_id(dataset_3_feat_dir)
    test_data = deal_feature(dataset_3_feat_dir,test_id['user_id'])
    test_data.to_csv(test_path,index=False)


get_data_feature()














