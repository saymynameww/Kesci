# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 20:00:17 2018

@author: ASUS
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

time_start = datetime.now()
print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))

dataset_1_feat_dir = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/dataset_1_feat')
dataset_1_label_dir = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/dataset_1_label')
dataset_2_feat_dir = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/dataset_2_feat')
dataset_2_label_dir = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/dataset_2_label')
dataset_3_feat_dir = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/dataset_3_feat')
dataset_3_label_dir = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/dataset_3_label')
dataset_4_feat_dir = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/dataset_4_feat')

train_path = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/train.csv')
test_path = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/test.csv')

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
    end_day = np.max(register['register_day'])
    feature['register_day'] = register['register_day']
    feature['register_type'] = register['register_type']
    feature['device_type'] = register['device_type']
    register_rest_day = (register.groupby(['user_id'])['register_day'].max()-end_day).rename('register_rest_day').reset_index()
    feature = pd.merge(feature,register_rest_day,how='left',on='user_id')
    return feature

def get_launch_feature(launch):
    feature = pd.DataFrame()
    feature['user_id'] = launch['user_id'].drop_duplicates()
    end_day = np.max(launch['launch_day'])
    launch_total_count = launch[['user_id']].groupby(['user_id']).size().rename('launch_total_count').reset_index()
    launch_day_diff = pd.concat([launch['user_id'],launch.groupby(['user_id']).diff().rename({'launch_day':'launch_day_diff'},axis=1)],axis=1)
    launch_day_diff_max = launch_day_diff.groupby(['user_id'])['launch_day_diff'].max().rename('launch_day_diff_max').reset_index()
    launch_day_diff_min = launch_day_diff.groupby(['user_id'])['launch_day_diff'].min().rename('launch_day_diff_min').reset_index()
    launch_day_diff_mean = launch_day_diff.groupby(['user_id'])['launch_day_diff'].mean().rename('launch_day_diff_mean').reset_index()
    launch_day_diff_std = launch_day_diff.groupby(['user_id'])['launch_day_diff'].std().rename('launch_day_diff_std').reset_index()
    launch_day_diff_kurt = launch_day_diff.groupby(['user_id'])['launch_day_diff'].agg(lambda x: pd.Series.kurt(x)).rename('launch_day_diff_kurt').reset_index()
    launch_day_diff_skew = launch_day_diff.groupby(['user_id'])['launch_day_diff'].skew().rename('launch_day_diff_skew').reset_index()
    launch_day_diff_last = launch_day_diff.groupby(['user_id'])['launch_day_diff'].last().rename('launch_day_diff_last').reset_index()
    launch_last_day = launch.groupby(['user_id'])['launch_day'].max().rename('launch_last_day').reset_index()
    launch_rest_day = (launch.groupby(['user_id'])['launch_day'].max()-end_day).rename('launch_rest_day').reset_index()
    feature = pd.merge(feature,launch_total_count,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_max,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_min,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_mean,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_std,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_kurt,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_skew,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_last,how='left',on='user_id')
    feature = pd.merge(feature,launch_last_day,how='left',on='user_id')
    feature = pd.merge(feature,launch_rest_day,how='left',on='user_id')
    return feature

def get_video_feature(video):
    feature = pd.DataFrame()
    feature['user_id'] = video['user_id'].drop_duplicates()
    end_day = np.max(video['video_day'])
    video_total_count = video[['user_id']].groupby(['user_id']).size().rename('video_total_count').reset_index()
    video_day_diff = pd.concat([video['user_id'],video.groupby(['user_id']).diff().rename({'video_day':'video_day_diff'},axis=1)],axis=1)
    video_day_diff_max = video_day_diff.groupby(['user_id'])['video_day_diff'].max().rename('video_day_diff_max').reset_index()
    video_day_diff_min = video_day_diff.groupby(['user_id'])['video_day_diff'].min().rename('video_day_diff_min').reset_index()
    video_day_diff_mean = video_day_diff.groupby(['user_id'])['video_day_diff'].mean().rename('video_day_diff_mean').reset_index()
    video_day_diff_std = video_day_diff.groupby(['user_id'])['video_day_diff'].std().rename('video_day_diff_std').reset_index()
    video_day_diff_kurt = video_day_diff.groupby(['user_id'])['video_day_diff'].agg(lambda x: pd.Series.kurt(x)).rename('video_day_diff_kurt').reset_index()
    video_day_diff_skew = video_day_diff.groupby(['user_id'])['video_day_diff'].skew().rename('video_day_diff_skew').reset_index()
    video_day_diff_last = video_day_diff.groupby(['user_id'])['video_day_diff'].last().rename('video_day_diff_last').reset_index()
    video_last_day = video.groupby(['user_id'])['video_day'].max().rename('video_last_day').reset_index()
    video_rest_day = (video.groupby(['user_id'])['video_day'].max()-end_day).rename('video_rest_day').reset_index()
    video_everyday_count = video.groupby(['user_id','video_day']).agg({'user_id':'mean','video_day':'count'})
    video_day_most = video_everyday_count.groupby(['user_id'])['video_day'].max().rename('video_day_most').reset_index()
    video_day_mode = video_everyday_count.groupby(['user_id'])['video_day'].agg(lambda x: np.mean(pd.Series.mode(x))).rename('video_day_mode').reset_index()
    feature = pd.merge(feature,video_total_count,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_max,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_min,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_mean,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_std,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_kurt,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_skew,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_last,how='left',on='user_id')
    feature = pd.merge(feature,video_last_day,how='left',on='user_id')
    feature = pd.merge(feature,video_rest_day,how='left',on='user_id')
    feature = pd.merge(feature,video_day_most,how='left',on='user_id')
    feature = pd.merge(feature,video_day_mode,how='left',on='user_id')
    return feature

def get_activity_feature(activity):
    activity = activity.rename({'action_type':'action'},axis=1)
    feature = pd.DataFrame()
    feature['user_id'] = activity['user_id'].drop_duplicates()
    end_day = np.max(activity['activity_day'])
    activity_total_count = activity[['user_id']].groupby(['user_id']).size().rename('activity_total_count').reset_index()
    activity_day_unique = activity.groupby(['user_id','activity_day']).agg({'user_id':'mean','activity_day':'mean'})
    activity_day_diff = activity_day_unique.groupby(['user_id']).diff().rename({'activity_day':'activity_day_diff'},axis=1).reset_index().drop('activity_day',axis=1)
    activity_day_diff_max = activity_day_diff.groupby(['user_id'])['activity_day_diff'].max().rename('activity_day_diff_max').reset_index()
    activity_day_diff_min = activity_day_diff.groupby(['user_id'])['activity_day_diff'].min().rename('activity_day_diff_min').reset_index()
    activity_day_diff_mean = activity_day_diff.groupby(['user_id'])['activity_day_diff'].mean().rename('activity_day_diff_mean').reset_index()
    activity_day_diff_std = activity_day_diff.groupby(['user_id'])['activity_day_diff'].std().rename('activity_day_diff_std').reset_index()
    activity_day_diff_kurt = activity_day_diff.groupby(['user_id'])['activity_day_diff'].agg(lambda x: pd.Series.kurt(x)).rename('activity_day_diff_kurt').reset_index()
    activity_day_diff_skew = activity_day_diff.groupby(['user_id'])['activity_day_diff'].skew().rename('activity_day_diff_skew').reset_index()
    activity_day_diff_last = activity_day_diff.groupby(['user_id'])['activity_day_diff'].last().rename('activity_day_diff_last').reset_index()
    activity_last_day = activity.groupby(['user_id'])['activity_day'].max().rename('activity_last_day').reset_index()
    activity_rest_day = (activity_day_unique.groupby(['user_id'])['activity_day'].max()-end_day).rename('activity_rest_day').reset_index()
    page_count = activity.groupby(['user_id','page']).agg({'page':'count'}).rename({'page':'page_count'},axis=1).reset_index()
    page0_count = page_count[page_count.page==0].drop('page',axis=1).rename({'page_count':'page0_count'},axis=1)
    page1_count = page_count[page_count.page==1].drop('page',axis=1).rename({'page_count':'page1_count'},axis=1)
    page2_count = page_count[page_count.page==2].drop('page',axis=1).rename({'page_count':'page2_count'},axis=1)
    page3_count = page_count[page_count.page==3].drop('page',axis=1).rename({'page_count':'page3_count'},axis=1)
    page4_count = page_count[page_count.page==4].drop('page',axis=1).rename({'page_count':'page4_count'},axis=1)
    page_percent = pd.merge(activity_total_count,page0_count,how='left',on='user_id')
    page_percent = pd.merge(page_percent,page1_count,how='left',on='user_id')
    page_percent = pd.merge(page_percent,page2_count,how='left',on='user_id')
    page_percent = pd.merge(page_percent,page3_count,how='left',on='user_id')
    page_percent = pd.merge(page_percent,page4_count,how='left',on='user_id')
    page_percent['page0_pct'] = page_percent['page0_count']/page_percent['activity_total_count']
    page_percent['page1_pct'] = page_percent['page1_count']/page_percent['activity_total_count']
    page_percent['page2_pct'] = page_percent['page2_count']/page_percent['activity_total_count']
    page_percent['page3_pct'] = page_percent['page3_count']/page_percent['activity_total_count']
    page_percent['page4_pct'] = page_percent['page4_count']/page_percent['activity_total_count']
    page_percent = page_percent.drop('activity_total_count',axis=1)
    action_count = activity.groupby(['user_id','action']).agg({'action':'count'}).rename({'action':'action_count'},axis=1).reset_index()
    action0_count = action_count[action_count.action==0].drop('action',axis=1).rename({'action_count':'action0_count'},axis=1)
    action1_count = action_count[action_count.action==1].drop('action',axis=1).rename({'action_count':'action1_count'},axis=1)
    action2_count = action_count[action_count.action==2].drop('action',axis=1).rename({'action_count':'action2_count'},axis=1)
    action3_count = action_count[action_count.action==3].drop('action',axis=1).rename({'action_count':'action3_count'},axis=1)
    action4_count = action_count[action_count.action==4].drop('action',axis=1).rename({'action_count':'action4_count'},axis=1)
    action5_count = action_count[action_count.action==5].drop('action',axis=1).rename({'action_count':'action5_count'},axis=1)
    action_percent = pd.merge(activity_total_count,action0_count,how='left',on='user_id')
    action_percent = pd.merge(action_percent,action1_count,how='left',on='user_id')
    action_percent = pd.merge(action_percent,action2_count,how='left',on='user_id')
    action_percent = pd.merge(action_percent,action3_count,how='left',on='user_id')
    action_percent = pd.merge(action_percent,action4_count,how='left',on='user_id')
    action_percent = pd.merge(action_percent,action5_count,how='left',on='user_id')
    action_percent['action0_pct'] = action_percent['action0_count']/action_percent['activity_total_count']
    action_percent['action1_pct'] = action_percent['action1_count']/action_percent['activity_total_count']
    action_percent['action2_pct'] = action_percent['action2_count']/action_percent['activity_total_count']
    action_percent['action3_pct'] = action_percent['action3_count']/action_percent['activity_total_count']
    action_percent['action4_pct'] = action_percent['action4_count']/action_percent['activity_total_count']
    action_percent['action5_pct'] = action_percent['action5_count']/action_percent['activity_total_count']
    action_percent = action_percent.drop('activity_total_count',axis=1)
    video_id_count = activity.groupby(['user_id','video_id']).agg({'user_id':'mean','video_id':'count'})
    video_id_most = video_id_count.groupby(['user_id'])['video_id'].max().rename('video_id_most').reset_index() 
    author_id_count = activity.groupby(['user_id','author_id']).agg({'user_id':'mean','author_id':'count'})
    author_id_most = author_id_count.groupby(['user_id'])['author_id'].max().rename('author_id_most').reset_index()
    activity_count = activity.groupby(['user_id','activity_day']).agg({'activity_day':'count'}).rename({'activity_day':'activity_count'},axis=1).reset_index()
    activity_count_max = activity_count.groupby(['user_id'])['activity_count'].max().rename('activity_count_max').reset_index()
    activity_count_min = activity_count.groupby(['user_id'])['activity_count'].min().rename('activity_count_min').reset_index()
    activity_count_mean = activity_count.groupby(['user_id'])['activity_count'].mean().rename('activity_count_mean').reset_index()
    activity_count_std = activity_count.groupby(['user_id'])['activity_count'].std().rename('activity_count_std').reset_index()
    activity_count_kurt = activity_count.groupby(['user_id'])['activity_count'].agg(lambda x: pd.Series.kurt(x)).rename('activity_count_kurt').reset_index()
    activity_count_skew = activity_count.groupby(['user_id'])['activity_count'].skew().rename('activity_count_skew').reset_index()
    activity_count_last = activity_count.groupby(['user_id'])['activity_count'].last().rename('activity_count_last').reset_index()
    activity_count_diff = pd.concat([activity_count['user_id'],activity_count.groupby(['user_id']).diff().rename({'activity_count':'activity_count_diff'},axis=1)],axis=1).drop('activity_day',axis=1)
    activity_count_diff_max = activity_count_diff.groupby(['user_id'])['activity_count_diff'].max().rename('activity_count_diff_max').reset_index()
    activity_count_diff_min = activity_count_diff.groupby(['user_id'])['activity_count_diff'].min().rename('activity_count_diff_min').reset_index()
    activity_count_diff_mean = activity_count_diff.groupby(['user_id'])['activity_count_diff'].mean().rename('activity_count_diff_mean').reset_index()
    activity_count_diff_std = activity_count_diff.groupby(['user_id'])['activity_count_diff'].std().rename('activity_count_diff_std').reset_index()
    activity_count_diff_kurt = activity_count_diff.groupby(['user_id'])['activity_count_diff'].agg(lambda x: pd.Series.kurt(x)).rename('activity_count_diff_kurt').reset_index()
    activity_count_diff_skew = activity_count_diff.groupby(['user_id'])['activity_count_diff'].skew().rename('activity_count_diff_skew').reset_index()
    activity_count_diff_last = activity_count_diff.groupby(['user_id'])['activity_count_diff'].last().rename('activity_count_diff_last').reset_index()
    page_everyday_count = activity.groupby(['user_id','activity_day','page']).agg({'page':'count'}).rename({'page':'page_everyday_count'},axis=1).reset_index()
    page0_max = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'])['page_everyday_count'].max().rename('page0_max').reset_index()
    page0_min = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'])['page_everyday_count'].min().rename('page0_min').reset_index()
    page0_mean = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'])['page_everyday_count'].mean().rename('page0_mean').reset_index()
    page0_std = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'])['page_everyday_count'].std().rename('page0_std').reset_index()
    page0_kurt = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'])['page_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('page0_kurt').reset_index()
    page0_skew = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'])['page_everyday_count'].skew().rename('page0_skew').reset_index()
    page0_last = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'])['page_everyday_count'].last().rename('page0_last').reset_index()
    page1_max = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'])['page_everyday_count'].max().rename('page1_max').reset_index()
    page1_min = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'])['page_everyday_count'].min().rename('page1_min').reset_index()
    page1_mean = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'])['page_everyday_count'].mean().rename('page1_mean').reset_index()
    page1_std = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'])['page_everyday_count'].std().rename('page1_std').reset_index()
    page1_kurt = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'])['page_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('page1_kurt').reset_index()
    page1_skew = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'])['page_everyday_count'].skew().rename('page1_skew').reset_index()
    page1_last = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'])['page_everyday_count'].last().rename('page1_last').reset_index()
    page2_max = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'])['page_everyday_count'].max().rename('page2_max').reset_index()
    page2_min = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'])['page_everyday_count'].min().rename('page2_min').reset_index()
    page2_mean = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'])['page_everyday_count'].mean().rename('page2_mean').reset_index()
    page2_std = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'])['page_everyday_count'].std().rename('page2_std').reset_index()
    page2_kurt = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'])['page_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('page2_kurt').reset_index()
    page2_skew = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'])['page_everyday_count'].skew().rename('page2_skew').reset_index()
    page2_last = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'])['page_everyday_count'].last().rename('page2_last').reset_index()
    page3_max = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'])['page_everyday_count'].max().rename('page3_max').reset_index()
    page3_min = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'])['page_everyday_count'].min().rename('page3_min').reset_index()
    page3_mean = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'])['page_everyday_count'].mean().rename('page3_mean').reset_index()
    page3_std = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'])['page_everyday_count'].std().rename('page3_std').reset_index()
    page3_kurt = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'])['page_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('page3_kurt').reset_index()
    page3_skew = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'])['page_everyday_count'].skew().rename('page3_skew').reset_index()
    page3_last = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'])['page_everyday_count'].last().rename('page3_last').reset_index()
    page4_max = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'])['page_everyday_count'].max().rename('page4_max').reset_index()
    page4_min = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'])['page_everyday_count'].min().rename('page4_min').reset_index()
    page4_mean = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'])['page_everyday_count'].mean().rename('page4_mean').reset_index()
    page4_std = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'])['page_everyday_count'].std().rename('page4_std').reset_index()
    page4_kurt = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'])['page_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('page4_kurt').reset_index()
    page4_skew = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'])['page_everyday_count'].skew().rename('page4_skew').reset_index()
    page4_last = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'])['page_everyday_count'].last().rename('page4_last').reset_index()
    action_everyday_count = activity.groupby(['user_id','activity_day','action']).agg({'action':'count'}).rename({'action':'action_everyday_count'},axis=1).reset_index()
    action0_max = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'])['action_everyday_count'].max().rename('action0_max').reset_index()
    action0_min = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'])['action_everyday_count'].min().rename('action0_min').reset_index()
    action0_mean = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'])['action_everyday_count'].mean().rename('action0_mean').reset_index()
    action0_std = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'])['action_everyday_count'].std().rename('action0_std').reset_index()
    action0_kurt = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'])['action_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('action0_kurt').reset_index()
    action0_skew = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'])['action_everyday_count'].skew().rename('action0_skew').reset_index()
    action0_last = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'])['action_everyday_count'].last().rename('action0_last').reset_index()
    action1_max = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'])['action_everyday_count'].max().rename('action1_max').reset_index()
    action1_min = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'])['action_everyday_count'].min().rename('action1_min').reset_index()
    action1_mean = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'])['action_everyday_count'].mean().rename('action1_mean').reset_index()
    action1_std = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'])['action_everyday_count'].std().rename('action1_std').reset_index()
    action1_kurt = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'])['action_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('action1_kurt').reset_index()
    action1_skew = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'])['action_everyday_count'].skew().rename('action1_skew').reset_index()
    action1_last = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'])['action_everyday_count'].last().rename('action1_last').reset_index()
    action2_max = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'])['action_everyday_count'].max().rename('action2_max').reset_index()
    action2_min = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'])['action_everyday_count'].min().rename('action2_min').reset_index()
    action2_mean = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'])['action_everyday_count'].mean().rename('action2_mean').reset_index()
    action2_std = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'])['action_everyday_count'].std().rename('action2_std').reset_index()
    action2_kurt = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'])['action_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('action2_kurt').reset_index()
    action2_skew = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'])['action_everyday_count'].skew().rename('action2_skew').reset_index()
    action2_last = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'])['action_everyday_count'].last().rename('action2_last').reset_index()
    action3_max = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'])['action_everyday_count'].max().rename('action3_max').reset_index()
    action3_min = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'])['action_everyday_count'].min().rename('action3_min').reset_index()
    action3_mean = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'])['action_everyday_count'].mean().rename('action3_mean').reset_index()
    action3_std = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'])['action_everyday_count'].std().rename('action3_std').reset_index()
    action3_kurt = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'])['action_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('action3_kurt').reset_index()
    action3_skew = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'])['action_everyday_count'].skew().rename('action3_skew').reset_index()
    action3_last = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'])['action_everyday_count'].last().rename('action3_last').reset_index()
    action4_max = action_everyday_count[action_everyday_count.action==4].groupby(['user_id'])['action_everyday_count'].max().rename('action4_max').reset_index()
    action4_min = action_everyday_count[action_everyday_count.action==4].groupby(['user_id'])['action_everyday_count'].min().rename('action4_min').reset_index()
    action4_mean = action_everyday_count[action_everyday_count.action==4].groupby(['user_id'])['action_everyday_count'].mean().rename('action4_mean').reset_index()
    action4_std = action_everyday_count[action_everyday_count.action==4].groupby(['user_id'])['action_everyday_count'].std().rename('action4_std').reset_index()
    action4_kurt = action_everyday_count[action_everyday_count.action==4].groupby(['user_id'])['action_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('action4_kurt').reset_index()
    action4_skew = action_everyday_count[action_everyday_count.action==4].groupby(['user_id'])['action_everyday_count'].skew().rename('action4_skew').reset_index()
    action4_last = action_everyday_count[action_everyday_count.action==4].groupby(['user_id'])['action_everyday_count'].last().rename('action4_last').reset_index()
    action5_max = action_everyday_count[action_everyday_count.action==5].groupby(['user_id'])['action_everyday_count'].max().rename('action5_max').reset_index()
    action5_min = action_everyday_count[action_everyday_count.action==5].groupby(['user_id'])['action_everyday_count'].min().rename('action5_min').reset_index()
    action5_mean = action_everyday_count[action_everyday_count.action==5].groupby(['user_id'])['action_everyday_count'].mean().rename('action5_mean').reset_index()
    action5_std = action_everyday_count[action_everyday_count.action==5].groupby(['user_id'])['action_everyday_count'].std().rename('action5_std').reset_index()
    action5_kurt = action_everyday_count[action_everyday_count.action==5].groupby(['user_id'])['action_everyday_count'].agg(lambda x: pd.Series.kurt(x)).rename('action5_kurt').reset_index()
    action5_skew = action_everyday_count[action_everyday_count.action==5].groupby(['user_id'])['action_everyday_count'].skew().rename('action5_skew').reset_index()
    action5_last = action_everyday_count[action_everyday_count.action==5].groupby(['user_id'])['action_everyday_count'].last().rename('action5_last').reset_index()
    most_activity_day = activity_count.groupby('user_id').apply(lambda x: x[x.activity_count==x.activity_count.max()]).rename({'activity_day':'most_activity_day'},axis=1).drop('activity_count',axis=1).groupby('user_id')['most_activity_day'].max().reset_index()
    author_count = activity.groupby('author_id').agg({'author_id': 'mean', 'activity_day': 'count'}).rename({'author_id':'user_id','activity_day': 'author_count'},axis=1)
    
    feature = pd.merge(feature,activity_total_count,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_diff_max,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_diff_min,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_diff_mean,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_diff_std,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_diff_kurt,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_diff_skew,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_diff_last,how='left',on='user_id')
    feature = pd.merge(feature,activity_last_day,how='left',on='user_id')
    feature = pd.merge(feature,activity_rest_day,how='left',on='user_id')
    feature = pd.merge(feature,page_percent,how='left',on='user_id')
    feature = pd.merge(feature,action_percent,how='left',on='user_id')
    feature = pd.merge(feature,video_id_most,how='left',on='user_id')
    feature = pd.merge(feature,author_id_most,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_max,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_min,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_mean,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_std,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_kurt,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_skew,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_last,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_diff_max,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_diff_min,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_diff_mean,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_diff_std,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_diff_kurt,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_diff_skew,how='left',on='user_id')
    feature = pd.merge(feature,activity_count_diff_last,how='left',on='user_id')
    feature = pd.merge(feature,page0_max,how='left',on='user_id')
    feature = pd.merge(feature,page0_min,how='left',on='user_id')
    feature = pd.merge(feature,page0_mean,how='left',on='user_id')
    feature = pd.merge(feature,page0_std,how='left',on='user_id')
    feature = pd.merge(feature,page0_kurt,how='left',on='user_id')
    feature = pd.merge(feature,page0_skew,how='left',on='user_id')
    feature = pd.merge(feature,page0_last,how='left',on='user_id')
    feature = pd.merge(feature,page1_max,how='left',on='user_id')
    feature = pd.merge(feature,page1_min,how='left',on='user_id')
    feature = pd.merge(feature,page1_mean,how='left',on='user_id')
    feature = pd.merge(feature,page1_std,how='left',on='user_id')
    feature = pd.merge(feature,page1_kurt,how='left',on='user_id')
    feature = pd.merge(feature,page1_skew,how='left',on='user_id')
    feature = pd.merge(feature,page1_last,how='left',on='user_id')
    feature = pd.merge(feature,page2_max,how='left',on='user_id')
    feature = pd.merge(feature,page2_min,how='left',on='user_id')
    feature = pd.merge(feature,page2_mean,how='left',on='user_id')
    feature = pd.merge(feature,page2_std,how='left',on='user_id')
    feature = pd.merge(feature,page2_kurt,how='left',on='user_id')
    feature = pd.merge(feature,page2_skew,how='left',on='user_id')
    feature = pd.merge(feature,page2_last,how='left',on='user_id')
    feature = pd.merge(feature,page3_max,how='left',on='user_id')
    feature = pd.merge(feature,page3_min,how='left',on='user_id')
    feature = pd.merge(feature,page3_mean,how='left',on='user_id')
    feature = pd.merge(feature,page3_std,how='left',on='user_id')
    feature = pd.merge(feature,page3_kurt,how='left',on='user_id')
    feature = pd.merge(feature,page3_skew,how='left',on='user_id')
    feature = pd.merge(feature,page3_last,how='left',on='user_id')
    feature = pd.merge(feature,page4_max,how='left',on='user_id')
    feature = pd.merge(feature,page4_min,how='left',on='user_id')
    feature = pd.merge(feature,page4_mean,how='left',on='user_id')
    feature = pd.merge(feature,page4_std,how='left',on='user_id')
    feature = pd.merge(feature,page4_kurt,how='left',on='user_id')
    feature = pd.merge(feature,page4_skew,how='left',on='user_id')
    feature = pd.merge(feature,page4_last,how='left',on='user_id')
    feature = pd.merge(feature,action0_max,how='left',on='user_id')
    feature = pd.merge(feature,action0_min,how='left',on='user_id')
    feature = pd.merge(feature,action0_mean,how='left',on='user_id')
    feature = pd.merge(feature,action0_std,how='left',on='user_id')
    feature = pd.merge(feature,action0_kurt,how='left',on='user_id')
    feature = pd.merge(feature,action0_skew,how='left',on='user_id')
    feature = pd.merge(feature,action0_last,how='left',on='user_id')
    feature = pd.merge(feature,action1_max,how='left',on='user_id')
    feature = pd.merge(feature,action1_min,how='left',on='user_id')
    feature = pd.merge(feature,action1_mean,how='left',on='user_id')
    feature = pd.merge(feature,action1_std,how='left',on='user_id')
    feature = pd.merge(feature,action1_kurt,how='left',on='user_id')
    feature = pd.merge(feature,action1_skew,how='left',on='user_id')
    feature = pd.merge(feature,action1_last,how='left',on='user_id')
    feature = pd.merge(feature,action2_max,how='left',on='user_id')
    feature = pd.merge(feature,action2_min,how='left',on='user_id')
    feature = pd.merge(feature,action2_mean,how='left',on='user_id')
    feature = pd.merge(feature,action2_std,how='left',on='user_id')
    feature = pd.merge(feature,action2_kurt,how='left',on='user_id')
    feature = pd.merge(feature,action2_skew,how='left',on='user_id')
    feature = pd.merge(feature,action2_last,how='left',on='user_id')
    feature = pd.merge(feature,action3_max,how='left',on='user_id')
    feature = pd.merge(feature,action3_min,how='left',on='user_id')
    feature = pd.merge(feature,action3_mean,how='left',on='user_id')
    feature = pd.merge(feature,action3_std,how='left',on='user_id')
    feature = pd.merge(feature,action3_kurt,how='left',on='user_id')
    feature = pd.merge(feature,action3_skew,how='left',on='user_id')
    feature = pd.merge(feature,action3_last,how='left',on='user_id')
    feature = pd.merge(feature,action4_max,how='left',on='user_id')
    feature = pd.merge(feature,action4_min,how='left',on='user_id')
    feature = pd.merge(feature,action4_mean,how='left',on='user_id')
    feature = pd.merge(feature,action4_std,how='left',on='user_id')
    feature = pd.merge(feature,action4_kurt,how='left',on='user_id')
    feature = pd.merge(feature,action4_skew,how='left',on='user_id')
    feature = pd.merge(feature,action4_last,how='left',on='user_id')
    feature = pd.merge(feature,action5_max,how='left',on='user_id')
    feature = pd.merge(feature,action5_min,how='left',on='user_id')
    feature = pd.merge(feature,action5_mean,how='left',on='user_id')
    feature = pd.merge(feature,action5_std,how='left',on='user_id')
    feature = pd.merge(feature,action5_kurt,how='left',on='user_id')
    feature = pd.merge(feature,action5_skew,how='left',on='user_id')
    feature = pd.merge(feature,action5_last,how='left',on='user_id')
    feature = pd.merge(feature,most_activity_day,how='left',on='user_id')
    feature = pd.merge(feature,author_count,how='left',on='user_id')
    return feature
    
def deal_feature(path, user_id):
    register = pd.read_csv(path + '/register.csv')
    launch = pd.read_csv(path + '/launch.csv')
    video = pd.read_csv(path + '/video.csv')
    activity = pd.read_csv(path + '/activity.csv')
    feature = pd.DataFrame()
    feature['user_id'] = user_id
    
    print('getting register feature...')
    register_feature = get_register_feature(register)
    feature = pd.merge(feature, register_feature, on='user_id', how='left')
    
    print('getting launch feature...')
    launch_feature = get_launch_feature(launch)
    feature = pd.merge(feature, launch_feature, on='user_id', how='left')
    
    print('getting video feature...')
    video_feature = get_video_feature(video)
    feature = pd.merge(feature, video_feature, on='user_id', how='left')
    
    print('getting activity feature...')
    activity_feature = get_activity_feature(activity)
    feature = pd.merge(feature,activity_feature, on='user_id', how='left')
    
    feature['last_launch_sub_register'] = feature['launch_last_day'] - feature['register_day']
    feature['last_video_sub_register'] = feature['video_last_day'] - feature['register_day']
    feature['last_activity_sub_register'] = feature['activity_last_day'] - feature['register_day']
    
    feature = feature.fillna(0)
    return feature

def get_data_feature():
    print('Feature engineering...')
    print('Getting train data 1 ...')
    train_label_1 = get_train_label(dataset_1_feat_dir,dataset_1_label_dir)
    data_1 = deal_feature(dataset_1_feat_dir,train_label_1['user_id'])
    data_1['label'] = train_label_1['label']
    
    print('Getting train data 2 ...')
    train_label_2 = get_train_label(dataset_2_feat_dir,dataset_2_label_dir)
    data_2 = deal_feature(dataset_2_feat_dir,train_label_2['user_id'])
    data_2['label'] = train_label_2['label']
    
    print('Getting train data 3 ...')
    train_label_3 = get_train_label(dataset_3_feat_dir,dataset_3_label_dir)
    data_3 = deal_feature(dataset_3_feat_dir,train_label_3['user_id'])
    data_3['label'] = train_label_3['label']
    
    train_data = pd.concat([data_1,data_2,data_3])
    train_data.to_csv(train_path,index=False)
    
    print('Getting test data...')
    test_id = get_test_id(dataset_4_feat_dir)
    test_data = deal_feature(dataset_4_feat_dir,test_id['user_id'])
    test_data.to_csv(test_path,index=False)


get_data_feature()


time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')











