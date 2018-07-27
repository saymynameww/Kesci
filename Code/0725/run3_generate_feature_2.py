# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:06:37 2018

@author: Administrator
"""

"""
get_action4/5 deleted
action4/5_everyday_count_stat deleted
action4/5_slide_count deleted
"""
# -*- coding: utf-8 -*-
## generate feature 
import os
import gc
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

time_start = datetime.now()
print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))

dataset_1_feat_dir = os.path.join(os.pardir,os.pardir,os.pardir, '1Kesci-data/dataset_1_feat')
dataset_1_label_dir = os.path.join(os.pardir,os.pardir,os.pardir, '1Kesci-data/dataset_1_label')
dataset_2_feat_dir = os.path.join(os.pardir,os.pardir,os.pardir, '1Kesci-data/dataset_2_feat')
dataset_2_label_dir = os.path.join(os.pardir,os.pardir,os.pardir, '1Kesci-data/dataset_2_label')
dataset_3_feat_dir = os.path.join(os.pardir,os.pardir,os.pardir, '1Kesci-data/dataset_3_feat')

train_path = os.path.join(os.pardir,os.pardir,os.pardir, '1Kesci-data/train_and_test/train.csv')
test_path = os.path.join(os.pardir,os.pardir,os.pardir, '1Kesci-data/train_and_test/test.csv')
# train_path = '/home/kesci/work/train.csv'
# test_path = '/home/kesci/work/test.csv'

def get_train_label(feat_path,label_path):
    feat_register_userid = pd.read_csv(feat_path + '/register.csv', usecols=['user_id'])
    feat_launch_userid = pd.read_csv(feat_path + '/launch.csv', usecols=['user_id'])
    feat_video_userid = pd.read_csv(feat_path + '/video.csv', usecols=['user_id'])
    feat_activity_userid = pd.read_csv(feat_path + '/activity.csv', usecols=['user_id'])
    feat_userid = pd.DataFrame()
    feat_userid['user_id'] = np.unique(pd.concat([feat_register_userid,feat_launch_userid,feat_video_userid,feat_activity_userid]))
    
    label_launch_userid = pd.read_csv(label_path + '/launch.csv', usecols=['user_id'])
    label_video_userid = pd.read_csv(label_path + '/video.csv', usecols=['user_id'])
    label_activity_userid = pd.read_csv(label_path + '/activity.csv', usecols=['user_id'])
    label_userid = pd.DataFrame()
    label_userid['user_id'] = np.unique(pd.concat([label_launch_userid,label_video_userid,label_activity_userid]))
    label_userid['label'] = 1
    
    train_label = pd.merge(feat_userid,label_userid,on=['user_id'],how='left',copy=False).fillna(0).astype(int)
    return train_label

def get_test_id(feat_path):
    register_userid = pd.read_csv(feat_path + '/register.csv', usecols=['user_id'])
    launch_userid = pd.read_csv(feat_path + '/launch.csv', usecols=['user_id'])
    video_userid = pd.read_csv(feat_path + '/video.csv', usecols=['user_id'])
    activity_userid = pd.read_csv(feat_path + '/activity.csv', usecols=['user_id'])
    feat_userid = np.unique(pd.concat([register_userid,launch_userid,video_userid,activity_userid]))
    test_userid = pd.DataFrame()
    test_userid['user_id'] = feat_userid
    return test_userid

def get_register_feature(register):
    feature = pd.DataFrame()
    feature['user_id'] = register['user_id'].drop_duplicates()
    begin_day = np.min(register['register_day'])
    end_day = np.max(register['register_day'])
    feature['register_day'] = register['register_day']
    feature['register_type'] = register['register_type']
    feature['device_type'] = register['device_type']
    feature['register_rest_day'] = register['register_day']-end_day
    feature['register_rest_day'] = -feature['register_rest_day']+1
    feature['divisor_day'] = feature['register_rest_day']
    feature.loc[feature['divisor_day']>16,'divisor_day']=16
    return feature

def launch_slide_window(launch,suffix):
    launch_slide_count = launch.groupby(['user_id'],as_index=False)['user_id'].agg({'launch_slide_count'+suffix:'count'})
    return launch_slide_count

def get_launch_feature(launch):
    feature = pd.DataFrame()
    feature['user_id'] = launch['user_id'].drop_duplicates()
    end_day = np.max(launch['launch_day'])
    #滑窗 窗口内launch次数
    launch_slide_count_all=launch_slide_window(launch,'_all')
    launch_slide_count_1=launch_slide_window(launch[(launch['launch_day']>=end_day) & (launch['launch_day']<=end_day)],'_1')
    launch_slide_count_2=launch_slide_window(launch[(launch['launch_day']>=end_day-1) & (launch['launch_day']<=end_day)],'_2')
    launch_slide_count_3=launch_slide_window(launch[(launch['launch_day']>=end_day-2) & (launch['launch_day']<=end_day)],'_3')
    launch_slide_count_5=launch_slide_window(launch[(launch['launch_day']>=end_day-4) & (launch['launch_day']<=end_day)],'_5')
    launch_slide_count_7=launch_slide_window(launch[(launch['launch_day']>=end_day-6) & (launch['launch_day']<=end_day)],'_7')
    launch_slide_count_10=launch_slide_window(launch[(launch['launch_day']>=end_day-9) & (launch['launch_day']<=end_day)],'_10')
    launch_slide_count = pd.merge(launch_slide_count_all,launch_slide_count_1,on=['user_id'],how='left',copy=False)
    launch_slide_count = pd.merge(launch_slide_count,launch_slide_count_2,on=['user_id'],how='left',copy=False)
    launch_slide_count = pd.merge(launch_slide_count,launch_slide_count_3,on=['user_id'],how='left',copy=False)
    launch_slide_count = pd.merge(launch_slide_count,launch_slide_count_5,on=['user_id'],how='left',copy=False)
    launch_slide_count = pd.merge(launch_slide_count,launch_slide_count_7,on=['user_id'],how='left',copy=False)
    launch_slide_count = pd.merge(launch_slide_count,launch_slide_count_10,on=['user_id'],how='left',copy=False)
    #launch日期特征
    launch_day_stat = launch.groupby(['user_id'],as_index=False)['launch_day'].agg({'launch_day_max':np.max,'launch_day_min':np.min,'launch_day_mean':np.mean,'launch_day_std':np.std,'launch_day_kurt':lambda x: pd.Series.kurt(x),'launch_day_skew':lambda x: pd.Series.skew(x)}) 
    launch_day_stat['launch_rest_day'] = -(launch_day_stat['launch_day_max']-end_day)
    #launch相隔天数特征
    launch_day_diff = pd.concat([launch['user_id'],launch.groupby(['user_id']).diff().rename({'launch_day':'launch_day_diff'},axis=1)],axis=1)
    launch_day_diff_stat = launch_day_diff.groupby(['user_id'],as_index=False)['launch_day_diff'].agg({'launch_day_diff_max':np.max,'launch_day_diff_min':np.min,'launch_day_diff_mean':np.mean,'launch_day_diff_std':np.std,'launch_day_diff_kurt':lambda x: pd.Series.kurt(x),'launch_day_diff_skew':lambda x: pd.Series.skew(x)}) 
    launch_day_diff_stat['launch_day_diff_last'] = launch_day_diff.groupby(['user_id'],as_index=False)['launch_day_diff'].last()['launch_day_diff']
    #MERGE
    feature = pd.merge(feature,launch_slide_count,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_stat,how='left',on='user_id')
    feature = pd.merge(feature,launch_day_diff_stat,how='left',on='user_id')
    return feature

def video_slide_window(video,suffix):
    video_slide_count = video.groupby(['user_id'],as_index=False)['user_id'].agg({'video_slide_count'+suffix:'count'})
    return video_slide_count

def get_video_feature(video):
    feature = pd.DataFrame()
    feature['user_id'] = video['user_id'].drop_duplicates()
    end_day = np.max(video['video_day'])
    #滑窗 窗口内video次数
    video_slide_count_all=video_slide_window(video,'_all')
    video_slide_count_1=video_slide_window(video[(video['video_day']>=end_day) & (video['video_day']<=end_day)],'_1')
    video_slide_count_2=video_slide_window(video[(video['video_day']>=end_day-1) & (video['video_day']<=end_day)],'_2')
    video_slide_count_3=video_slide_window(video[(video['video_day']>=end_day-2) & (video['video_day']<=end_day)],'_3')
    video_slide_count_5=video_slide_window(video[(video['video_day']>=end_day-4) & (video['video_day']<=end_day)],'_5')
    video_slide_count_7=video_slide_window(video[(video['video_day']>=end_day-6) & (video['video_day']<=end_day)],'_7')
    video_slide_count_10=video_slide_window(video[(video['video_day']>=end_day-9) & (video['video_day']<=end_day)],'_10')
    video_slide_count = pd.merge(video_slide_count_all,video_slide_count_1,on=['user_id'],how='left',copy=False)
    video_slide_count = pd.merge(video_slide_count,video_slide_count_2,on=['user_id'],how='left',copy=False)
    video_slide_count = pd.merge(video_slide_count,video_slide_count_3,on=['user_id'],how='left',copy=False)
    video_slide_count = pd.merge(video_slide_count,video_slide_count_5,on=['user_id'],how='left',copy=False)
    video_slide_count = pd.merge(video_slide_count,video_slide_count_7,on=['user_id'],how='left',copy=False)
    video_slide_count = pd.merge(video_slide_count,video_slide_count_10,on=['user_id'],how='left',copy=False)
    #video日期特征
    video_day_stat = video.groupby(['user_id'],as_index=False)['video_day'].agg({'video_day_max':np.max,'video_day_min':np.min,'video_day_mean':np.mean,'video_day_std':np.std,'video_day_kurt':lambda x: pd.Series.kurt(x),'video_day_skew':lambda x: pd.Series.skew(x)}) 
    video_day_stat['video_rest_day'] = -(video_day_stat['video_day_max']-end_day)
    #video相隔天数特征
    video_day_diff = pd.concat([video['user_id'],video.groupby(['user_id']).diff().rename({'video_day':'video_day_diff'},axis=1)],axis=1)
    video_day_diff_stat = video_day_diff.groupby(['user_id'],as_index=False)['video_day_diff'].agg({'video_day_diff_max':np.max,'video_day_diff_min':np.min,'video_day_diff_mean':np.mean,'video_day_diff_std':np.std,'video_day_diff_kurt':lambda x: pd.Series.kurt(x),'video_day_diff_skew':lambda x: pd.Series.skew(x)}) 
    video_day_diff_stat['video_day_diff_last'] = video_day_diff.groupby(['user_id'],as_index=False)['video_day_diff'].last()['video_day_diff']
    video_unique = video.groupby(['user_id','video_day']).agg({'user_id':'mean','video_day':'mean'}).rename({'video_day': 'video_day_unique'},axis=1)
    #video去重后日期特征 video_day_unique_max=video_day_max  video_day_unique_min=video_day_min
    video_day_unique_stat = video_unique.groupby(['user_id'],as_index=False)['video_day_unique'].agg({'video_day_unique_mean':np.mean,'video_day_unique_std':np.std,'video_day_unique_kurt':lambda x: pd.Series.kurt(x),'video_day_unique_skew':lambda x: pd.Series.skew(x)}) 
    #video去重后相隔天数特征
    video_day_unique_diff = pd.concat([video_unique['user_id'],video_unique.groupby(['user_id']).diff().rename({'video_day_unique':'video_day_unique_diff'},axis=1)],axis=1)
    video_day_unique_diff_stat = video_day_unique_diff.groupby(['user_id'],as_index=False)['video_day_unique_diff'].agg({'video_day_unique_diff_max':np.max,'video_day_unique_diff_min':np.min,'video_day_unique_diff_mean':np.mean,'video_day_unique_diff_std':np.std,'video_day_unique_diff_kurt':lambda x: pd.Series.kurt(x),'video_day_unique_diff_skew':lambda x: pd.Series.skew(x)}) 
    video_day_unique_diff_stat['video_day_unique_diff_last'] = video_day_unique_diff.groupby(['user_id'],as_index=False)['video_day_unique_diff'].last()['video_day_unique_diff']
    #按天统计特征
    video_everyday_count = video.groupby(['user_id','video_day'],as_index=False)['video_day'].agg({'video_everyday_count':'count'})
    video_everyday_count_stat = video_everyday_count.groupby(['user_id'],as_index=False)['video_everyday_count'].agg({'video_everyday_count_max':np.max,'video_everyday_count_min':np.min,'video_everyday_count_mean':np.mean,'video_everyday_count_std':np.std,'video_everyday_count_kurt':lambda x: pd.Series.kurt(x),'video_everyday_count_skew':lambda x: pd.Series.skew(x)})
    video_everyday_count_stat['video_everyday_count_last'] = video_everyday_count.groupby(['user_id'],as_index=False)['video_everyday_count'].last()['video_everyday_count']
    video_day_stat['video_day_mode'] = video_everyday_count.groupby(['user_id'],as_index=False)['video_everyday_count'].agg({'video_day_mode':lambda x: np.mean(pd.Series.mode(x))})['video_day_mode']
    #最多video的是哪一天
    video_day_stat['most_video_day'] = video_everyday_count.groupby(['user_id'],as_index=False).apply(lambda x: x[x.video_everyday_count==x.video_everyday_count.max()]).rename({'video_day':'most_video_day'},axis=1).drop('video_everyday_count',axis=1).groupby('user_id')['most_video_day'].max().reset_index()['most_video_day']
    video_day_stat['most_video_day_sub_end_day'] = -(video_day_stat['most_video_day']-end_day)
    #MERGE
    feature = pd.merge(feature,video_slide_count,how='left',on='user_id')
    feature = pd.merge(feature,video_day_stat,how='left',on='user_id')
    feature = pd.merge(feature,video_day_diff_stat,how='left',on='user_id')
    feature = pd.merge(feature,video_day_unique_stat,how='left',on='user_id')
    feature = pd.merge(feature,video_day_unique_diff_stat,how='left',on='user_id')
    feature = pd.merge(feature,video_everyday_count_stat,how='left',on='user_id')
    return feature

def activity_slide_window(activity,suffix):
    activity_slide_count = activity.groupby(['user_id'],as_index=False)['user_id'].agg({'activity_slide_count'+suffix:'count'})
    #每个page次数和占比
    page_slide_count_temp = activity.groupby(['user_id','page'],as_index=False)['page'].agg({'page_count':'count'})
    page0_slide_count = page_slide_count_temp[page_slide_count_temp.page==0].drop('page',axis=1).rename({'page_count':'page0_slide_count'+suffix},axis=1)
    page1_slide_count = page_slide_count_temp[page_slide_count_temp.page==1].drop('page',axis=1).rename({'page_count':'page1_slide_count'+suffix},axis=1)
    page2_slide_count = page_slide_count_temp[page_slide_count_temp.page==2].drop('page',axis=1).rename({'page_count':'page2_slide_count'+suffix},axis=1)
    page3_slide_count = page_slide_count_temp[page_slide_count_temp.page==3].drop('page',axis=1).rename({'page_count':'page3_slide_count'+suffix},axis=1)
    page4_slide_count = page_slide_count_temp[page_slide_count_temp.page==4].drop('page',axis=1).rename({'page_count':'page4_slide_count'+suffix},axis=1)
    page_slide_count = pd.merge(activity_slide_count,page0_slide_count,how='left',on='user_id')
    page_slide_count = pd.merge(page_slide_count,page1_slide_count,how='left',on='user_id')
    page_slide_count = pd.merge(page_slide_count,page2_slide_count,how='left',on='user_id')
    page_slide_count = pd.merge(page_slide_count,page3_slide_count,how='left',on='user_id')
    page_slide_count = pd.merge(page_slide_count,page4_slide_count,how='left',on='user_id')
    page_slide_count['page0_slide_pct'+suffix] = page_slide_count['page0_slide_count'+suffix]/page_slide_count['activity_slide_count'+suffix]
    page_slide_count['page1_slide_pct'+suffix] = page_slide_count['page1_slide_count'+suffix]/page_slide_count['activity_slide_count'+suffix]
    page_slide_count['page2_slide_pct'+suffix] = page_slide_count['page2_slide_count'+suffix]/page_slide_count['activity_slide_count'+suffix]
    page_slide_count['page3_slide_pct'+suffix] = page_slide_count['page3_slide_count'+suffix]/page_slide_count['activity_slide_count'+suffix]
    page_slide_count['page4_slide_pct'+suffix] = page_slide_count['page4_slide_count'+suffix]/page_slide_count['activity_slide_count'+suffix]
#    page_slide_count = page_slide_count.drop('activity_slide_count'+suffix,axis=1)
    #每个action次数和占比
    action_slide_count_temp = activity.groupby(['user_id','action'],as_index=False)['action'].agg({'action_count':'count'})
    action0_slide_count = action_slide_count_temp[action_slide_count_temp.action==0].drop('action',axis=1).rename({'action_count':'action0_slide_count'+suffix},axis=1)
    action1_slide_count = action_slide_count_temp[action_slide_count_temp.action==1].drop('action',axis=1).rename({'action_count':'action1_slide_count'+suffix},axis=1)
    action2_slide_count = action_slide_count_temp[action_slide_count_temp.action==2].drop('action',axis=1).rename({'action_count':'action2_slide_count'+suffix},axis=1)
    action3_slide_count = action_slide_count_temp[action_slide_count_temp.action==3].drop('action',axis=1).rename({'action_count':'action3_slide_count'+suffix},axis=1)
#    action4_slide_count = action_slide_count_temp[action_slide_count_temp.action==4].drop('action',axis=1).rename({'action_count':'action4_slide_count'+suffix},axis=1)
#    action5_slide_count = action_slide_count_temp[action_slide_count_temp.action==5].drop('action',axis=1).rename({'action_count':'action5_slide_count'+suffix},axis=1)
    action_slide_count = pd.merge(activity_slide_count,action0_slide_count,how='left',on='user_id')
    action_slide_count = pd.merge(action_slide_count,action1_slide_count,how='left',on='user_id')
    action_slide_count = pd.merge(action_slide_count,action2_slide_count,how='left',on='user_id')
    action_slide_count = pd.merge(action_slide_count,action3_slide_count,how='left',on='user_id')
#    action_slide_count = pd.merge(action_slide_count,action4_slide_count,how='left',on='user_id')
#    action_slide_count = pd.merge(action_slide_count,action5_slide_count,how='left',on='user_id')
    action_slide_count['action0_slide_pct'+suffix] = action_slide_count['action0_slide_count'+suffix]/action_slide_count['activity_slide_count'+suffix]
    action_slide_count['action1_slide_pct'+suffix] = action_slide_count['action1_slide_count'+suffix]/action_slide_count['activity_slide_count'+suffix]
    action_slide_count['action2_slide_pct'+suffix] = action_slide_count['action2_slide_count'+suffix]/action_slide_count['activity_slide_count'+suffix]
    action_slide_count['action3_slide_pct'+suffix] = action_slide_count['action3_slide_count'+suffix]/action_slide_count['activity_slide_count'+suffix]
#    action_slide_count['action4_slide_pct'+suffix] = action_slide_count['action4_slide_count'+suffix]/action_slide_count['activity_slide_count'+suffix]
#    action_slide_count['action5_slide_pct'+suffix] = action_slide_count['action5_slide_count'+suffix]/action_slide_count['activity_slide_count'+suffix]
    action_slide_count = action_slide_count.drop('activity_slide_count'+suffix,axis=1)
    
    get_activity_slide_count = activity.groupby(['author_id'],as_index=False)['author_id'].agg({'get_activity_slide_count'+suffix:'count'})
    #被每个page次数和占比
    get_page_slide_count_temp = activity.groupby(['author_id','page'],as_index=False)['page'].agg({'get_page_count':'count'})
    get_page0_slide_count = get_page_slide_count_temp[get_page_slide_count_temp.page==0].drop('page',axis=1).rename({'get_page_count':'get_page0_slide_count'+suffix},axis=1)
    get_page1_slide_count = get_page_slide_count_temp[get_page_slide_count_temp.page==1].drop('page',axis=1).rename({'get_page_count':'get_page1_slide_count'+suffix},axis=1)
    get_page2_slide_count = get_page_slide_count_temp[get_page_slide_count_temp.page==2].drop('page',axis=1).rename({'get_page_count':'get_page2_slide_count'+suffix},axis=1)
    get_page3_slide_count = get_page_slide_count_temp[get_page_slide_count_temp.page==3].drop('page',axis=1).rename({'get_page_count':'get_page3_slide_count'+suffix},axis=1)
    get_page4_slide_count = get_page_slide_count_temp[get_page_slide_count_temp.page==4].drop('page',axis=1).rename({'get_page_count':'get_page4_slide_count'+suffix},axis=1)
    get_page_slide_count = pd.merge(get_activity_slide_count,get_page0_slide_count,how='left',on='author_id')
    get_page_slide_count = pd.merge(get_page_slide_count,get_page1_slide_count,how='left',on='author_id')
    get_page_slide_count = pd.merge(get_page_slide_count,get_page2_slide_count,how='left',on='author_id')
    get_page_slide_count = pd.merge(get_page_slide_count,get_page3_slide_count,how='left',on='author_id')
    get_page_slide_count = pd.merge(get_page_slide_count,get_page4_slide_count,how='left',on='author_id')
    get_page_slide_count['get_page0_slide_pct'+suffix] = get_page_slide_count['get_page0_slide_count'+suffix]/get_page_slide_count['get_activity_slide_count'+suffix]
    get_page_slide_count['get_page1_slide_pct'+suffix] = get_page_slide_count['get_page1_slide_count'+suffix]/get_page_slide_count['get_activity_slide_count'+suffix]
    get_page_slide_count['get_page2_slide_pct'+suffix] = get_page_slide_count['get_page2_slide_count'+suffix]/get_page_slide_count['get_activity_slide_count'+suffix]
    get_page_slide_count['get_page3_slide_pct'+suffix] = get_page_slide_count['get_page3_slide_count'+suffix]/get_page_slide_count['get_activity_slide_count'+suffix]
    get_page_slide_count['get_page4_slide_pct'+suffix] = get_page_slide_count['get_page4_slide_count'+suffix]/get_page_slide_count['get_activity_slide_count'+suffix]
#    get_page_slide_count = get_page_slide_count.drop('get_activity_slide_count'+suffix,axis=1)
    #被每个action次数和占比
    get_action_slide_count_temp = activity.groupby(['author_id','action'],as_index=False)['action'].agg({'get_action_count':'count'})
    get_action0_slide_count = get_action_slide_count_temp[get_action_slide_count_temp.action==0].drop('action',axis=1).rename({'get_action_count':'get_action0_slide_count'+suffix},axis=1)
    get_action1_slide_count = get_action_slide_count_temp[get_action_slide_count_temp.action==1].drop('action',axis=1).rename({'get_action_count':'get_action1_slide_count'+suffix},axis=1)
    get_action2_slide_count = get_action_slide_count_temp[get_action_slide_count_temp.action==2].drop('action',axis=1).rename({'get_action_count':'get_action2_slide_count'+suffix},axis=1)
    get_action3_slide_count = get_action_slide_count_temp[get_action_slide_count_temp.action==3].drop('action',axis=1).rename({'get_action_count':'get_action3_slide_count'+suffix},axis=1)
#    get_action4_slide_count = get_action_slide_count_temp[get_action_slide_count_temp.action==4].drop('action',axis=1).rename({'get_action_count':'get_action4_slide_count'+suffix},axis=1)
#    get_action5_slide_count = get_action_slide_count_temp[get_action_slide_count_temp.action==5].drop('action',axis=1).rename({'get_action_count':'get_action5_slide_count'+suffix},axis=1)
    get_action_slide_count = pd.merge(get_activity_slide_count,get_action0_slide_count,how='left',on='author_id')
    get_action_slide_count = pd.merge(get_action_slide_count,get_action1_slide_count,how='left',on='author_id')
    get_action_slide_count = pd.merge(get_action_slide_count,get_action2_slide_count,how='left',on='author_id')
    get_action_slide_count = pd.merge(get_action_slide_count,get_action3_slide_count,how='left',on='author_id')
#    get_action_slide_count = pd.merge(get_action_slide_count,get_action4_slide_count,how='left',on='author_id')
#    get_action_slide_count = pd.merge(get_action_slide_count,get_action5_slide_count,how='left',on='author_id')
    get_action_slide_count['get_action0_slide_pct'+suffix] = get_action_slide_count['get_action0_slide_count'+suffix]/get_action_slide_count['get_activity_slide_count'+suffix]
    get_action_slide_count['get_action1_slide_pct'+suffix] = get_action_slide_count['get_action1_slide_count'+suffix]/get_action_slide_count['get_activity_slide_count'+suffix]
    get_action_slide_count['get_action2_slide_pct'+suffix] = get_action_slide_count['get_action2_slide_count'+suffix]/get_action_slide_count['get_activity_slide_count'+suffix]
    get_action_slide_count['get_action3_slide_pct'+suffix] = get_action_slide_count['get_action3_slide_count'+suffix]/get_action_slide_count['get_activity_slide_count'+suffix]
#    get_action_slide_count['get_action4_slide_pct'+suffix] = get_action_slide_count['get_action4_slide_count'+suffix]/get_action_slide_count['get_activity_slide_count'+suffix]
#    get_action_slide_count['get_action5_slide_pct'+suffix] = get_action_slide_count['get_action5_slide_count'+suffix]/get_action_slide_count['get_activity_slide_count'+suffix]
    get_action_slide_count = get_action_slide_count.drop('get_activity_slide_count'+suffix,axis=1)    
    
    #看video次数特征 
    repeat_video_count = activity.groupby(['user_id','video_id'],as_index=False)['video_id'].agg({'repeat_video_count':'count'})
    repeat_video_count_stat = repeat_video_count.groupby(['user_id'],as_index=False)['repeat_video_count'].agg({'repeat_video_count_max'+suffix:np.max,'repeat_video_count_min'+suffix:np.min,'repeat_video_count_mean'+suffix:np.mean,'repeat_video_count_std'+suffix:np.std,'repeat_video_count_kurt'+suffix:lambda x: pd.Series.kurt(x),'repeat_video_count_skew'+suffix:lambda x: pd.Series.skew(x)})
    repeat_video_count_stat['watch_video_count'+suffix] = repeat_video_count.groupby(['user_id'],as_index=False)['video_id'].agg({'watch_video_count'+suffix:'count'})['watch_video_count'+suffix]
    #看author次数特征
    repeat_author_count = activity.groupby(['user_id','author_id'],as_index=False)['author_id'].agg({'repeat_author_count':'count'})
    repeat_author_count_stat = repeat_author_count.groupby(['user_id'],as_index=False)['repeat_author_count'].agg({'repeat_author_count_max'+suffix:np.max,'repeat_author_count_min'+suffix:np.min,'repeat_author_count_mean'+suffix:np.mean,'repeat_author_count_std'+suffix:np.std,'repeat_author_count_kurt'+suffix:lambda x: pd.Series.kurt(x),'repeat_author_count_skew'+suffix:lambda x: pd.Series.skew(x)})
    repeat_author_count_stat['watch_author_count'+suffix] = repeat_author_count.groupby(['user_id'],as_index=False)['author_id'].agg({'watch_author_count'+suffix:'count'})['watch_author_count'+suffix]
    
    #观众数特征 
#    get_repeated_user_count = activity.groupby(['author_id','user_id'],as_index=False)['user_id'].agg({'get_repeated_user_count':'count'})
#    get_repeated_user_count_stat = get_repeated_user_count.groupby(['author_id'],as_index=False)['get_repeated_user_count'].agg({'get_repeated_user_count_max'+suffix:np.max,'get_repeated_user_count_min'+suffix:np.min,'get_repeated_user_count_mean'+suffix:np.mean,'get_repeated_user_count_std'+suffix:np.std,'get_repeated_user_count_kurt'+suffix:lambda x: pd.Series.kurt(x),'get_repeated_user_count_skew'+suffix:lambda x: pd.Series.skew(x)})
#    get_repeated_user_count_stat['fans_count'+suffix] = get_repeated_user_count.groupby(['author_id'],as_index=False)['user_id'].agg({'fans_count'+suffix:'count'})['fans_count'+suffix]
    get_repeated_user_count_stat = activity.groupby(['author_id']).agg({'author_id':'mean'})
    #被看video次数特征
#    get_repeated_video_count = activity.groupby(['author_id','video_id'],as_index=False)['video_id'].agg({'get_repeated_video_count':'count'})
#    get_repeated_video_count_stat = get_repeated_video_count.groupby(['author_id'],as_index=False)['get_repeated_video_count'].agg({'get_repeated_video_count_max'+suffix:np.max,'get_repeated_video_count_min'+suffix:np.min,'get_repeated_video_count_mean'+suffix:np.mean,'get_repeated_video_count_std'+suffix:np.std,'get_repeated_video_count_kurt'+suffix:lambda x: pd.Series.kurt(x),'get_repeated_video_count_skew'+suffix:lambda x: pd.Series.skew(x)})
#    get_repeated_video_count_stat['video_count'+suffix] = get_repeated_video_count.groupby(['author_id'],as_index=False)['video_id'].agg({'video_count'+suffix:'count'})['video_count'+suffix]
    get_repeated_video_count_stat = activity.groupby(['author_id']).agg({'author_id':'mean'})
    return page_slide_count, action_slide_count, get_page_slide_count, get_action_slide_count, repeat_video_count_stat, repeat_author_count_stat, get_repeated_user_count_stat, get_repeated_video_count_stat
    
def get_activity_feature(activity):
    activity = activity.rename({'action_type':'action'},axis=1)
    feature = pd.DataFrame()
    feature['user_id'] = activity['user_id'].drop_duplicates()
    end_day = np.max(activity['activity_day'])
    #滑窗
    page_slide_count_all,action_slide_count_all,get_page_slide_count_all,get_action_slide_count_all,repeat_video_count_stat_all,repeat_author_count_stat_all,get_repeated_user_count_stat_all,get_repeated_video_count_stat_all = activity_slide_window(activity,'_all')
    page_slide_count_1,action_slide_count_1,get_page_slide_count_1,get_action_slide_count_1,repeat_video_count_stat_1,repeat_author_count_stat_1,get_repeated_user_count_stat_1,get_repeated_video_count_stat_1 = activity_slide_window(activity[(activity['activity_day']>=end_day) & (activity['activity_day']<=end_day)],'_1')
    page_slide_count_2,action_slide_count_2,get_page_slide_count_2,get_action_slide_count_2,repeat_video_count_stat_2,repeat_author_count_stat_2,get_repeated_user_count_stat_2,get_repeated_video_count_stat_2 = activity_slide_window(activity[(activity['activity_day']>=end_day-1) & (activity['activity_day']<=end_day)],'_2')
    page_slide_count_3,action_slide_count_3,get_page_slide_count_3,get_action_slide_count_3,repeat_video_count_stat_3,repeat_author_count_stat_3,get_repeated_user_count_stat_3,get_repeated_video_count_stat_3 = activity_slide_window(activity[(activity['activity_day']>=end_day-2) & (activity['activity_day']<=end_day)],'_3')
    page_slide_count_5,action_slide_count_5,get_page_slide_count_5,get_action_slide_count_5,repeat_video_count_stat_5,repeat_author_count_stat_5,get_repeated_user_count_stat_5,get_repeated_video_count_stat_5 = activity_slide_window(activity[(activity['activity_day']>=end_day-4) & (activity['activity_day']<=end_day)],'_5')
    page_slide_count_7,action_slide_count_7,get_page_slide_count_7,get_action_slide_count_7,repeat_video_count_stat_7,repeat_author_count_stat_7,get_repeated_user_count_stat_7,get_repeated_video_count_stat_7 = activity_slide_window(activity[(activity['activity_day']>=end_day-6) & (activity['activity_day']<=end_day)],'_7')
    page_slide_count_10,action_slide_count_10,get_page_slide_count_10,get_action_slide_count_10,repeat_video_count_stat_10,repeat_author_count_stat_10,get_repeated_user_count_stat_10,get_repeated_video_count_stat_10 = activity_slide_window(activity[(activity['activity_day']>=end_day-9) & (activity['activity_day']<=end_day)],'_10')
    page_slide_count = pd.merge(page_slide_count_all,page_slide_count_1,on=['user_id'],how='left',copy=False)
    page_slide_count = pd.merge(page_slide_count,page_slide_count_2,on=['user_id'],how='left',copy=False)
    page_slide_count = pd.merge(page_slide_count,page_slide_count_3,on=['user_id'],how='left',copy=False)
    page_slide_count = pd.merge(page_slide_count,page_slide_count_5,on=['user_id'],how='left',copy=False)
    page_slide_count = pd.merge(page_slide_count,page_slide_count_7,on=['user_id'],how='left',copy=False)
    page_slide_count = pd.merge(page_slide_count,page_slide_count_10,on=['user_id'],how='left',copy=False)
    action_slide_count = pd.merge(action_slide_count_all,action_slide_count_1,on=['user_id'],how='left',copy=False)
    action_slide_count = pd.merge(action_slide_count,action_slide_count_2,on=['user_id'],how='left',copy=False)
    action_slide_count = pd.merge(action_slide_count,action_slide_count_3,on=['user_id'],how='left',copy=False)
    action_slide_count = pd.merge(action_slide_count,action_slide_count_5,on=['user_id'],how='left',copy=False)
    action_slide_count = pd.merge(action_slide_count,action_slide_count_7,on=['user_id'],how='left',copy=False)
    action_slide_count = pd.merge(action_slide_count,action_slide_count_10,on=['user_id'],how='left',copy=False)
    get_page_slide_count = pd.merge(get_page_slide_count_all,get_page_slide_count_1,on=['author_id'],how='left',copy=False)
    get_page_slide_count = pd.merge(get_page_slide_count,get_page_slide_count_2,on=['author_id'],how='left',copy=False)
    get_page_slide_count = pd.merge(get_page_slide_count,get_page_slide_count_3,on=['author_id'],how='left',copy=False)
    get_page_slide_count = pd.merge(get_page_slide_count,get_page_slide_count_5,on=['author_id'],how='left',copy=False)
    get_page_slide_count = pd.merge(get_page_slide_count,get_page_slide_count_7,on=['author_id'],how='left',copy=False)
    get_page_slide_count = pd.merge(get_page_slide_count,get_page_slide_count_10,on=['author_id'],how='left',copy=False)
    get_action_slide_count = pd.merge(get_action_slide_count_all,get_action_slide_count_1,on=['author_id'],how='left',copy=False)
    get_action_slide_count = pd.merge(get_action_slide_count,get_action_slide_count_2,on=['author_id'],how='left',copy=False)
    get_action_slide_count = pd.merge(get_action_slide_count,get_action_slide_count_3,on=['author_id'],how='left',copy=False)
    get_action_slide_count = pd.merge(get_action_slide_count,get_action_slide_count_5,on=['author_id'],how='left',copy=False)
    get_action_slide_count = pd.merge(get_action_slide_count,get_action_slide_count_7,on=['author_id'],how='left',copy=False)
    get_action_slide_count = pd.merge(get_action_slide_count,get_action_slide_count_10,on=['author_id'],how='left',copy=False)
    repeat_video_count_stat = pd.merge(repeat_video_count_stat_all,repeat_video_count_stat_1,on=['user_id'],how='left',copy=False)
    repeat_video_count_stat = pd.merge(repeat_video_count_stat,repeat_video_count_stat_2,on=['user_id'],how='left',copy=False)
    repeat_video_count_stat = pd.merge(repeat_video_count_stat,repeat_video_count_stat_3,on=['user_id'],how='left',copy=False)
    repeat_video_count_stat = pd.merge(repeat_video_count_stat,repeat_video_count_stat_5,on=['user_id'],how='left',copy=False)
    repeat_video_count_stat = pd.merge(repeat_video_count_stat,repeat_video_count_stat_7,on=['user_id'],how='left',copy=False)
    repeat_video_count_stat = pd.merge(repeat_video_count_stat,repeat_video_count_stat_10,on=['user_id'],how='left',copy=False)
    repeat_author_count_stat = pd.merge(repeat_author_count_stat_all,repeat_author_count_stat_1,on=['user_id'],how='left',copy=False)
    repeat_author_count_stat = pd.merge(repeat_author_count_stat,repeat_author_count_stat_2,on=['user_id'],how='left',copy=False)
    repeat_author_count_stat = pd.merge(repeat_author_count_stat,repeat_author_count_stat_3,on=['user_id'],how='left',copy=False)
    repeat_author_count_stat = pd.merge(repeat_author_count_stat,repeat_author_count_stat_5,on=['user_id'],how='left',copy=False)
    repeat_author_count_stat = pd.merge(repeat_author_count_stat,repeat_author_count_stat_7,on=['user_id'],how='left',copy=False)
    repeat_author_count_stat = pd.merge(repeat_author_count_stat,repeat_author_count_stat_10,on=['user_id'],how='left',copy=False)
    get_repeated_user_count_stat = pd.merge(get_repeated_user_count_stat_all,get_repeated_user_count_stat_1,on=['author_id'],how='left',copy=False)
    get_repeated_user_count_stat = pd.merge(get_repeated_user_count_stat,get_repeated_user_count_stat_2,on=['author_id'],how='left',copy=False)
    get_repeated_user_count_stat = pd.merge(get_repeated_user_count_stat,get_repeated_user_count_stat_3,on=['author_id'],how='left',copy=False)
    get_repeated_user_count_stat = pd.merge(get_repeated_user_count_stat,get_repeated_user_count_stat_5,on=['author_id'],how='left',copy=False)
    get_repeated_user_count_stat = pd.merge(get_repeated_user_count_stat,get_repeated_user_count_stat_7,on=['author_id'],how='left',copy=False)
    get_repeated_user_count_stat = pd.merge(get_repeated_user_count_stat,get_repeated_user_count_stat_10,on=['author_id'],how='left',copy=False)
    get_repeated_video_count_stat = pd.merge(get_repeated_video_count_stat_all,get_repeated_video_count_stat_1,on=['author_id'],how='left',copy=False)
    get_repeated_video_count_stat = pd.merge(get_repeated_video_count_stat,get_repeated_video_count_stat_2,on=['author_id'],how='left',copy=False)
    get_repeated_video_count_stat = pd.merge(get_repeated_video_count_stat,get_repeated_video_count_stat_3,on=['author_id'],how='left',copy=False)
    get_repeated_video_count_stat = pd.merge(get_repeated_video_count_stat,get_repeated_video_count_stat_5,on=['author_id'],how='left',copy=False)
    get_repeated_video_count_stat = pd.merge(get_repeated_video_count_stat,get_repeated_video_count_stat_7,on=['author_id'],how='left',copy=False)
    get_repeated_video_count_stat = pd.merge(get_repeated_video_count_stat,get_repeated_video_count_stat_10,on=['author_id'],how='left',copy=False)
    del page_slide_count_all,action_slide_count_all,get_page_slide_count_all,get_action_slide_count_all,repeat_video_count_stat_all,repeat_author_count_stat_all,get_repeated_user_count_stat_all,get_repeated_video_count_stat_all
    del page_slide_count_1,action_slide_count_1,get_page_slide_count_1,get_action_slide_count_1,repeat_video_count_stat_1,repeat_author_count_stat_1,get_repeated_user_count_stat_1,get_repeated_video_count_stat_1
    del page_slide_count_2,action_slide_count_2,get_page_slide_count_2,get_action_slide_count_2,repeat_video_count_stat_2,repeat_author_count_stat_2,get_repeated_user_count_stat_2,get_repeated_video_count_stat_2
    del page_slide_count_3,action_slide_count_3,get_page_slide_count_3,get_action_slide_count_3,repeat_video_count_stat_3,repeat_author_count_stat_3,get_repeated_user_count_stat_3,get_repeated_video_count_stat_3
    del page_slide_count_5,action_slide_count_5,get_page_slide_count_5,get_action_slide_count_5,repeat_video_count_stat_5,repeat_author_count_stat_5,get_repeated_user_count_stat_5,get_repeated_video_count_stat_5
    del page_slide_count_7,action_slide_count_7,get_page_slide_count_7,get_action_slide_count_7,repeat_video_count_stat_7,repeat_author_count_stat_7,get_repeated_user_count_stat_7,get_repeated_video_count_stat_7
    del page_slide_count_10,action_slide_count_10,get_page_slide_count_10,get_action_slide_count_10,repeat_video_count_stat_10,repeat_author_count_stat_10,get_repeated_user_count_stat_10,get_repeated_video_count_stat_10
    gc.collect()
    feature = pd.merge(feature,page_slide_count,how='left',on='user_id')
    feature = pd.merge(feature,action_slide_count,how='left',on='user_id')
    feature = pd.merge(feature,get_page_slide_count.rename({'author_id':'user_id'},axis=1),how='left',on='user_id')
    feature = pd.merge(feature,get_action_slide_count.rename({'author_id':'user_id'},axis=1),how='left',on='user_id')
    feature = pd.merge(feature,repeat_video_count_stat,how='left',on='user_id')
    feature = pd.merge(feature,repeat_author_count_stat,how='left',on='user_id')
    feature = pd.merge(feature,get_repeated_user_count_stat.rename({'author_id':'user_id'},axis=1),how='left',on='user_id')
    feature = pd.merge(feature,get_repeated_video_count_stat.rename({'author_id':'user_id'},axis=1),how='left',on='user_id')
    del page_slide_count,action_slide_count,get_page_slide_count,get_action_slide_count,repeat_video_count_stat,repeat_author_count_stat,get_repeated_user_count_stat,get_repeated_video_count_stat
    gc.collect()
    print('activity slide feature done.')
    
    #activity日期特征
    activity_day_stat = activity.groupby(['user_id'],as_index=False)['activity_day'].agg({'activity_day_max':np.max,'activity_day_min':np.min,'activity_day_mean':np.mean,'activity_day_std':np.std,'activity_day_kurt':lambda x: pd.Series.kurt(x),'activity_day_skew':lambda x: pd.Series.skew(x)}) 
    activity_day_stat['activity_rest_day'] = -(activity_day_stat['activity_day_max']-end_day)
    #activity相隔天数特征
    activity_day_diff = pd.concat([activity['user_id'],activity.groupby(['user_id']).diff().rename({'activity_day':'activity_day_diff'},axis=1)],axis=1)
    activity_day_diff_stat = activity_day_diff.groupby(['user_id'],as_index=False)['activity_day_diff'].agg({'activity_day_diff_max':np.max,'activity_day_diff_min':np.min,'activity_day_diff_mean':np.mean,'activity_day_diff_std':np.std,'activity_day_diff_kurt':lambda x: pd.Series.kurt(x),'activity_day_diff_skew':lambda x: pd.Series.skew(x)}) 
    activity_day_diff_stat['activity_day_diff_last'] = activity_day_diff.groupby(['user_id'],as_index=False)['activity_day_diff'].last()['activity_day_diff']
    activity_unique = activity.groupby(['user_id','activity_day']).agg({'user_id':'mean','activity_day':'mean'}).rename({'activity_day': 'activity_day_unique'},axis=1)
    #activity去重后日期特征 activity_day_unique_max=activity_day_max  activity_day_unique_min=activity_day_min
    activity_day_unique_stat = activity_unique.groupby(['user_id'],as_index=False)['activity_day_unique'].agg({'activity_day_unique_mean':np.mean,'activity_day_unique_std':np.std,'activity_day_unique_kurt':lambda x: pd.Series.kurt(x),'activity_day_unique_skew':lambda x: pd.Series.skew(x)}) 
    #activity去重后相隔天数特征
    activity_day_unique_diff = pd.concat([activity_unique['user_id'],activity_unique.groupby(['user_id']).diff().rename({'activity_day_unique':'activity_day_unique_diff'},axis=1)],axis=1)
    activity_day_unique_diff_stat = activity_day_unique_diff.groupby(['user_id'],as_index=False)['activity_day_unique_diff'].agg({'activity_day_unique_diff_max':np.max,'activity_day_unique_diff_min':np.min,'activity_day_unique_diff_mean':np.mean,'activity_day_unique_diff_std':np.std,'activity_day_unique_diff_kurt':lambda x: pd.Series.kurt(x),'activity_day_unique_diff_skew':lambda x: pd.Series.skew(x)}) 
    activity_day_unique_diff_stat['activity_day_unique_diff_last'] = activity_day_unique_diff.groupby(['user_id'],as_index=False)['activity_day_unique_diff'].last()['activity_day_unique_diff']
    #activity按天统计特征
    activity_everyday_count = activity.groupby(['user_id','activity_day'],as_index=False)['activity_day'].agg({'activity_everyday_count':'count'})
    activity_everyday_count_stat = activity_everyday_count.groupby(['user_id'],as_index=False)['activity_everyday_count'].agg({'activity_everyday_count_max':np.max,'activity_everyday_count_min':np.min,'activity_everyday_count_mean':np.mean,'activity_everyday_count_std':np.std,'activity_everyday_count_kurt':lambda x: pd.Series.kurt(x),'activity_everyday_count_skew':lambda x: pd.Series.skew(x)})
    activity_everyday_count_stat['activity_everyday_count_last'] = activity_everyday_count.groupby(['user_id'],as_index=False)['activity_everyday_count'].last()['activity_everyday_count']
    activity_day_stat['activity_day_mode'] = activity_everyday_count.groupby(['user_id'],as_index=False)['activity_everyday_count'].agg({'activity_day_mode':lambda x: np.mean(pd.Series.mode(x))})['activity_day_mode']
    #最多activity的是哪一天
    activity_day_stat['most_activity_day'] = activity_everyday_count.groupby(['user_id'],as_index=False).apply(lambda x: x[x.activity_everyday_count==x.activity_everyday_count.max()]).rename({'activity_day':'most_activity_day'},axis=1).drop('activity_everyday_count',axis=1).groupby('user_id')['most_activity_day'].max().reset_index()['most_activity_day']
    activity_day_stat['most_activity_day_sub_end_day'] = -(activity_day_stat['most_activity_day']-end_day)
    del activity_day_diff,activity_day_unique_diff,activity_everyday_count
    gc.collect()
    feature = pd.merge(feature,activity_day_stat,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_diff_stat,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_unique_stat,how='left',on='user_id')
    feature = pd.merge(feature,activity_day_unique_diff_stat,how='left',on='user_id')
    feature = pd.merge(feature,activity_everyday_count_stat,how='left',on='user_id')
    del activity_day_stat,activity_day_diff_stat,activity_day_unique_stat,activity_day_unique_diff_stat,activity_everyday_count_stat
    gc.collect()
    print('activity date feature done.')
    
#    #被activity日期特征
#    get_activity_day_stat = activity.groupby(['author_id'],as_index=False)['activity_day'].agg({'get_activity_day_max':np.max,'get_activity_day_min':np.min,'get_activity_day_mean':np.mean,'get_activity_day_std':np.std,'get_activity_day_kurt':lambda x: pd.Series.kurt(x),'get_activity_day_skew':lambda x: pd.Series.skew(x)}) 
#    get_activity_day_stat['get_activity_rest_day'] = -(get_activity_day_stat['get_activity_day_max']-end_day)
#    #被activity相隔天数特征
#    get_activity_day_diff = pd.concat([activity['author_id'],activity.groupby(['author_id']).diff().rename({'activity_day':'get_activity_day_diff'},axis=1)],axis=1)
#    get_activity_day_diff_stat = get_activity_day_diff.groupby(['author_id'],as_index=False)['get_activity_day_diff'].agg({'get_activity_day_diff_max':np.max,'get_activity_day_diff_min':np.min,'get_activity_day_diff_mean':np.mean,'get_activity_day_diff_std':np.std,'get_activity_day_diff_kurt':lambda x: pd.Series.kurt(x),'get_activity_day_diff_skew':lambda x: pd.Series.skew(x)}) 
#    get_activity_day_diff_stat['get_activity_day_diff_last'] = get_activity_day_diff.groupby(['author_id'],as_index=False)['get_activity_day_diff'].last()['get_activity_day_diff']
#    get_activity_unique = activity.groupby(['author_id','activity_day']).agg({'author_id':'mean','activity_day':'mean'}).rename({'activity_day': 'get_activity_day_unique'},axis=1)
#    #get_activity去重后日期特征 get_activity_day_unique_max=get_activity_day_max  get_activity_day_unique_min=get_activity_day_min
#    get_activity_day_unique_stat = get_activity_unique.groupby(['author_id'],as_index=False)['get_activity_day_unique'].agg({'get_activity_day_unique_mean':np.mean,'get_activity_day_unique_std':np.std,'get_activity_day_unique_kurt':lambda x: pd.Series.kurt(x),'get_activity_day_unique_skew':lambda x: pd.Series.skew(x)}) 
#    #被activity去重后相隔天数特征
#    get_activity_day_unique_diff = pd.concat([get_activity_unique['author_id'],get_activity_unique.groupby(['author_id']).diff().rename({'get_activity_day_unique':'get_activity_day_unique_diff'},axis=1)],axis=1)
#    get_activity_day_unique_diff_stat = get_activity_day_unique_diff.groupby(['author_id'],as_index=False)['get_activity_day_unique_diff'].agg({'get_activity_day_unique_diff_max':np.max,'get_activity_day_unique_diff_min':np.min,'get_activity_day_unique_diff_mean':np.mean,'get_activity_day_unique_diff_std':np.std,'get_activity_day_unique_diff_kurt':lambda x: pd.Series.kurt(x),'get_activity_day_unique_diff_skew':lambda x: pd.Series.skew(x)}) 
#    get_activity_day_unique_diff_stat['get_activity_day_unique_diff_last'] = get_activity_day_unique_diff.groupby(['author_id'],as_index=False)['get_activity_day_unique_diff'].last()['get_activity_day_unique_diff']
#    #被activity按天统计特征
#    get_activity_everyday_count = activity.groupby(['author_id','activity_day'],as_index=False)['activity_day'].agg({'get_activity_everyday_count':'count'})
#    get_activity_everyday_count_stat = get_activity_everyday_count.groupby(['author_id'],as_index=False)['get_activity_everyday_count'].agg({'get_activity_everyday_count_max':np.max,'get_activity_everyday_count_min':np.min,'get_activity_everyday_count_mean':np.mean,'get_activity_everyday_count_std':np.std,'get_activity_everyday_count_kurt':lambda x: pd.Series.kurt(x),'get_activity_everyday_count_skew':lambda x: pd.Series.skew(x)})
#    get_activity_everyday_count_stat['get_activity_everyday_count_last'] = get_activity_everyday_count.groupby(['author_id'],as_index=False)['get_activity_everyday_count'].last()['get_activity_everyday_count']
#    get_activity_day_stat['get_activity_day_mode'] = get_activity_everyday_count.groupby(['author_id'],as_index=False)['get_activity_everyday_count'].agg({'get_activity_day_mode':lambda x: np.mean(pd.Series.mode(x))})['get_activity_day_mode']
#    #最多被activity的是哪一天
#    get_activity_day_stat['most_get_activity_day'] = get_activity_everyday_count.groupby(['author_id'],as_index=False).apply(lambda x: x[x.get_activity_everyday_count==x.get_activity_everyday_count.max()]).rename({'activity_day':'most_get_activity_day'},axis=1).drop('get_activity_everyday_count',axis=1).groupby('author_id')['most_get_activity_day'].max().reset_index()['most_get_activity_day']
#    get_activity_day_stat['most_get_activity_day_sub_end_day'] = -(get_activity_day_stat['most_get_activity_day']-end_day)
    # feature = pd.merge(feature,get_activity_day_stat.rename({'author_id':'user_id'},axis=1),how='left',on='user_id')
    # feature = pd.merge(feature,get_activity_day_diff_stat.rename({'author_id':'user_id'},axis=1),how='left',on='user_id')
    # feature = pd.merge(feature,get_activity_day_unique_stat.rename({'author_id':'user_id'},axis=1),how='left',on='user_id')
    # feature = pd.merge(feature,get_activity_day_unique_diff_stat.rename({'author_id':'user_id'},axis=1),how='left',on='user_id')
    # feature = pd.merge(feature,get_activity_everyday_count_stat.rename({'author_id':'user_id'},axis=1),how='left',on='user_id')
    # del get_activity_day_diff,get_activity_day_unique_diff,get_activity_everyday_count
    # del get_activity_day_stat,get_activity_day_diff_stat,get_activity_day_unique_stat,get_activity_day_unique_diff_stat,get_activity_everyday_count_stat
    # gc.collect()
    
    #每个page按天统计特征
    page_everyday_count = activity.groupby(['user_id','activity_day','page'],as_index=False)['activity_day'].agg({'page_everyday_count':'count'})
    page0_everyday_count_stat = page_everyday_count[page_everyday_count.page==0].groupby(['user_id'],as_index=False)['page_everyday_count'].agg({'page0_everyday_count_max':np.max,'page0_everyday_count_min':np.min,'page0_everyday_count_mean':np.mean,'page0_everyday_count_std':np.std,'page0_everyday_count_kurt':lambda x: pd.Series.kurt(x),'page0_everyday_count_skew':lambda x: pd.Series.skew(x)})
    page0_everyday_count_stat['page0_everyday_count_last'] = page_everyday_count.groupby(['user_id'],as_index=False)['page_everyday_count'].last()['page_everyday_count']
    page1_everyday_count_stat = page_everyday_count[page_everyday_count.page==1].groupby(['user_id'],as_index=False)['page_everyday_count'].agg({'page1_everyday_count_max':np.max,'page1_everyday_count_min':np.min,'page1_everyday_count_mean':np.mean,'page1_everyday_count_std':np.std,'page1_everyday_count_kurt':lambda x: pd.Series.kurt(x),'page1_everyday_count_skew':lambda x: pd.Series.skew(x)})
    page1_everyday_count_stat['page1_everyday_count_last'] = page_everyday_count.groupby(['user_id'],as_index=False)['page_everyday_count'].last()['page_everyday_count']
    page2_everyday_count_stat = page_everyday_count[page_everyday_count.page==2].groupby(['user_id'],as_index=False)['page_everyday_count'].agg({'page2_everyday_count_max':np.max,'page2_everyday_count_min':np.min,'page2_everyday_count_mean':np.mean,'page2_everyday_count_std':np.std,'page2_everyday_count_kurt':lambda x: pd.Series.kurt(x),'page2_everyday_count_skew':lambda x: pd.Series.skew(x)})
    page2_everyday_count_stat['page2_everyday_count_last'] = page_everyday_count.groupby(['user_id'],as_index=False)['page_everyday_count'].last()['page_everyday_count']
    page3_everyday_count_stat = page_everyday_count[page_everyday_count.page==3].groupby(['user_id'],as_index=False)['page_everyday_count'].agg({'page3_everyday_count_max':np.max,'page3_everyday_count_min':np.min,'page3_everyday_count_mean':np.mean,'page3_everyday_count_std':np.std,'page3_everyday_count_kurt':lambda x: pd.Series.kurt(x),'page3_everyday_count_skew':lambda x: pd.Series.skew(x)})
    page3_everyday_count_stat['page3_everyday_count_last'] = page_everyday_count.groupby(['user_id'],as_index=False)['page_everyday_count'].last()['page_everyday_count']
    page4_everyday_count_stat = page_everyday_count[page_everyday_count.page==4].groupby(['user_id'],as_index=False)['page_everyday_count'].agg({'page4_everyday_count_max':np.max,'page4_everyday_count_min':np.min,'page4_everyday_count_mean':np.mean,'page4_everyday_count_std':np.std,'page4_everyday_count_kurt':lambda x: pd.Series.kurt(x),'page4_everyday_count_skew':lambda x: pd.Series.skew(x)})
    page4_everyday_count_stat['page4_everyday_count_last'] = page_everyday_count.groupby(['user_id'],as_index=False)['page_everyday_count'].last()['page_everyday_count']
    feature = pd.merge(feature,page0_everyday_count_stat,how='left',on='user_id')
    feature = pd.merge(feature,page1_everyday_count_stat,how='left',on='user_id')
    feature = pd.merge(feature,page2_everyday_count_stat,how='left',on='user_id')
    feature = pd.merge(feature,page3_everyday_count_stat,how='left',on='user_id')
    feature = pd.merge(feature,page4_everyday_count_stat,how='left',on='user_id')
    del page_everyday_count,page0_everyday_count_stat,page1_everyday_count_stat,page2_everyday_count_stat,page3_everyday_count_stat,page4_everyday_count_stat
    gc.collect()
    #每个action按天统计特征
    action_everyday_count = activity.groupby(['user_id','activity_day','action'],as_index=False)['activity_day'].agg({'action_everyday_count':'count'})
    action0_everyday_count_stat = action_everyday_count[action_everyday_count.action==0].groupby(['user_id'],as_index=False)['action_everyday_count'].agg({'action0_everyday_count_max':np.max,'action0_everyday_count_min':np.min,'action0_everyday_count_mean':np.mean,'action0_everyday_count_std':np.std,'action0_everyday_count_kurt':lambda x: pd.Series.kurt(x),'action0_everyday_count_skew':lambda x: pd.Series.skew(x)})
    action0_everyday_count_stat['action0_everyday_count_last'] = action_everyday_count.groupby(['user_id'],as_index=False)['action_everyday_count'].last()['action_everyday_count']
    action1_everyday_count_stat = action_everyday_count[action_everyday_count.action==1].groupby(['user_id'],as_index=False)['action_everyday_count'].agg({'action1_everyday_count_max':np.max,'action1_everyday_count_min':np.min,'action1_everyday_count_mean':np.mean,'action1_everyday_count_std':np.std,'action1_everyday_count_kurt':lambda x: pd.Series.kurt(x),'action1_everyday_count_skew':lambda x: pd.Series.skew(x)})
    action1_everyday_count_stat['action1_everyday_count_last'] = action_everyday_count.groupby(['user_id'],as_index=False)['action_everyday_count'].last()['action_everyday_count']
    action2_everyday_count_stat = action_everyday_count[action_everyday_count.action==2].groupby(['user_id'],as_index=False)['action_everyday_count'].agg({'action2_everyday_count_max':np.max,'action2_everyday_count_min':np.min,'action2_everyday_count_mean':np.mean,'action2_everyday_count_std':np.std,'action2_everyday_count_kurt':lambda x: pd.Series.kurt(x),'action2_everyday_count_skew':lambda x: pd.Series.skew(x)})
    action2_everyday_count_stat['action2_everyday_count_last'] = action_everyday_count.groupby(['user_id'],as_index=False)['action_everyday_count'].last()['action_everyday_count']
    action3_everyday_count_stat = action_everyday_count[action_everyday_count.action==3].groupby(['user_id'],as_index=False)['action_everyday_count'].agg({'action3_everyday_count_max':np.max,'action3_everyday_count_min':np.min,'action3_everyday_count_mean':np.mean,'action3_everyday_count_std':np.std,'action3_everyday_count_kurt':lambda x: pd.Series.kurt(x),'action3_everyday_count_skew':lambda x: pd.Series.skew(x)})
    action3_everyday_count_stat['action3_everyday_count_last'] = action_everyday_count.groupby(['user_id'],as_index=False)['action_everyday_count'].last()['action_everyday_count']
#    action4_everyday_count_stat = action_everyday_count[action_everyday_count.action==4].groupby(['user_id'],as_index=False)['action_everyday_count'].agg({'action4_everyday_count_max':np.max,'action4_everyday_count_min':np.min,'action4_everyday_count_mean':np.mean,'action4_everyday_count_std':np.std,'action4_everyday_count_kurt':lambda x: pd.Series.kurt(x),'action4_everyday_count_skew':lambda x: pd.Series.skew(x)})
#    action4_everyday_count_stat['action4_everyday_count_last'] = action_everyday_count.groupby(['user_id'],as_index=False)['action_everyday_count'].last()['action_everyday_count']
#    action5_everyday_count_stat = action_everyday_count[action_everyday_count.action==5].groupby(['user_id'],as_index=False)['action_everyday_count'].agg({'action5_everyday_count_max':np.max,'action5_everyday_count_min':np.min,'action5_everyday_count_mean':np.mean,'action5_everyday_count_std':np.std,'action5_everyday_count_kurt':lambda x: pd.Series.kurt(x),'action5_everyday_count_skew':lambda x: pd.Series.skew(x)})
#    action5_everyday_count_stat['action5_everyday_count_last'] = action_everyday_count.groupby(['user_id'],as_index=False)['action_everyday_count'].last()['action_everyday_count']
    feature = pd.merge(feature,action0_everyday_count_stat,how='left',on='user_id')
    feature = pd.merge(feature,action1_everyday_count_stat,how='left',on='user_id')
    feature = pd.merge(feature,action2_everyday_count_stat,how='left',on='user_id')
    feature = pd.merge(feature,action3_everyday_count_stat,how='left',on='user_id')
#    feature = pd.merge(feature,action4_everyday_count_stat,how='left',on='user_id')
#    feature = pd.merge(feature,action5_everyday_count_stat,how='left',on='user_id')
    del action_everyday_count,action0_everyday_count_stat,action1_everyday_count_stat,action2_everyday_count_stat,action3_everyday_count_stat,action4_everyday_count_stat,action5_everyday_count_stat
    gc.collect()
    print('activity type feature done.')

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
    
    del register_feature,launch_feature,video_feature
    gc.collect()
    
    print('getting activity feature...')
    activity_feature = get_activity_feature(activity)
    feature = pd.merge(feature,activity_feature, on='user_id', how='left')
    
    #最后一次与注册日之差
    feature['last_launch_sub_register'] = feature['launch_day_max'] - feature['register_day']
    feature['last_video_sub_register'] = feature['video_day_max'] - feature['register_day']
    feature['last_activity_sub_register'] = feature['activity_day_max'] - feature['register_day']
    #被看video数占拍视频数
    # feature['get_watched_video_pct'] = feature['video_count_all']/feature['video_slide_count_all']
    #平均 除以最大可能天数
    feature['mean_launch_slide_count_all'] = feature['launch_slide_count_all']/feature['divisor_day']
    feature['mean_video_slide_count_all'] = feature['video_slide_count_all']/feature['divisor_day']
    feature['mean_activity_slide_count_all'] = feature['activity_slide_count_all']/feature['divisor_day']
    feature['mean_page0_slide_count_all'] = feature['page0_slide_count_all']/feature['divisor_day']
    feature['mean_page1_slide_count_all'] = feature['page1_slide_count_all']/feature['divisor_day']
    feature['mean_page2_slide_count_all'] = feature['page2_slide_count_all']/feature['divisor_day']
    feature['mean_page3_slide_count_all'] = feature['page3_slide_count_all']/feature['divisor_day']
    feature['mean_page4_slide_count_all'] = feature['page4_slide_count_all']/feature['divisor_day']
    feature['mean_action0_slide_count_all'] = feature['action0_slide_count_all']/feature['divisor_day']
    feature['mean_action1_slide_count_all'] = feature['action1_slide_count_all']/feature['divisor_day']
    feature['mean_action2_slide_count_all'] = feature['action2_slide_count_all']/feature['divisor_day']
    feature['mean_action3_slide_count_all'] = feature['action3_slide_count_all']/feature['divisor_day']
    feature['mean_action4_slide_count_all'] = feature['action4_slide_count_all']/feature['divisor_day']
    feature['mean_action5_slide_count_all'] = feature['action5_slide_count_all']/feature['divisor_day']
    feature['mean_get_activity_slide_count_all'] = feature['get_activity_slide_count_all']/feature['divisor_day']
    feature['mean_get_page0_slide_count_all'] = feature['get_page0_slide_count_all']/feature['divisor_day']
    feature['mean_get_page1_slide_count_all'] = feature['get_page1_slide_count_all']/feature['divisor_day']
    feature['mean_get_page2_slide_count_all'] = feature['get_page2_slide_count_all']/feature['divisor_day']
    feature['mean_get_page3_slide_count_all'] = feature['get_page3_slide_count_all']/feature['divisor_day']
    feature['mean_get_page4_slide_count_all'] = feature['get_page4_slide_count_all']/feature['divisor_day']
    feature['mean_get_action0_slide_count_all'] = feature['get_action0_slide_count_all']/feature['divisor_day']
    feature['mean_get_action1_slide_count_all'] = feature['get_action1_slide_count_all']/feature['divisor_day']
    feature['mean_get_action2_slide_count_all'] = feature['get_action2_slide_count_all']/feature['divisor_day']
    feature['mean_get_action3_slide_count_all'] = feature['get_action3_slide_count_all']/feature['divisor_day']
#    feature['mean_get_action4_slide_count_all'] = feature['get_action4_slide_count_all']/feature['divisor_day']
#    feature['mean_get_action5_slide_count_all'] = feature['get_action5_slide_count_all']/feature['divisor_day']
    #规则
    feature.loc[(feature['launch_slide_count_all']==1) & (feature['register_rest_day']>7),'launch_only_once']=1
    
#    feature = feature.fillna(0)
    return feature

def get_data_feature():
    print('Feature engineering...')
    print('Getting train data 1 ...')
    train_label_1 = get_train_label(dataset_1_feat_dir,dataset_1_label_dir)
    data_1 = deal_feature(dataset_1_feat_dir,train_label_1['user_id'])
    data_1['label'] = train_label_1['label']
    print('train1 size:',data_1.shape)
    print('Getting train data 2 ...')
    train_label_2 = get_train_label(dataset_2_feat_dir,dataset_2_label_dir)
    data_2 = deal_feature(dataset_2_feat_dir,train_label_2['user_id'])
    data_2['label'] = train_label_2['label']
    print('train2 size:',data_2.shape)
    
    train_data = pd.concat([data_1,data_2])
    print('train size:',train_data.shape)
    train_data.to_csv(train_path,index=False)
    del data_1,data_2,train_data
    gc.collect()
    
    print('Getting test data...')
    test_id = get_test_id(dataset_3_feat_dir)
    test_data = deal_feature(dataset_3_feat_dir,test_id['user_id'])
    test_data.to_csv(test_path,index=False)


get_data_feature()


time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')

for key in list(globals().keys()): 
    if not key.startswith("__"): 
        globals().pop(key) 