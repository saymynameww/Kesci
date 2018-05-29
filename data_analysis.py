# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:40:11 2018

@author: wxk06
"""

import os
import pandas as pd
import numpy as np
import pickle

input_dir = os.path.join(os.pardir, 'Kesci-data')
print('Input files:\n{}'.format(os.listdir(input_dir)))
print('Loading data sets...')

register_df = pickle.load(open(os.path.join(input_dir, 'user_register.pkl'),"rb"))
launch_df = pickle.load(open(os.path.join(input_dir, 'app_launch.pkl'),"rb"))
video_df = pickle.load(open(os.path.join(input_dir, 'video_create.pkl'),"rb"))
activity_df = pickle.load(open(os.path.join(input_dir, 'user_activity.pkl'),"rb"))


def data_during(start_day,end_day):
    register_df_selected = register_df.loc[(register_df['register_day'] >= start_day) & (register_df['register_day'] <= end_day)]
    launch_df_selected = launch_df.loc[(launch_df['day'] >= start_day) & (launch_df['day'] <= end_day)]
    video_df_selected = video_df.loc[(video_df['day'] >= start_day) & (video_df['day'] <= end_day)]
    activity_df_selected = activity_df.loc[(activity_df['day'] >= start_day) & (activity_df['day'] <= end_day)]
    register_day = register_df_selected.groupby('user_id').agg({'user_id': 'mean', 'register_day': 'mean'})
    register_day.columns = ['user_id', 'register_day']
    launch_freq = launch_df_selected.groupby('user_id').agg({'user_id': 'mean', 'day': 'count'})
    launch_freq.columns = ['user_id', 'launch_count']
    video_freq = video_df_selected.groupby('user_id').agg({'user_id': 'mean', 'day': 'count'})
    video_freq.columns = ['user_id', 'video_count']
    activity_freq = activity_df_selected.groupby('user_id').agg({'user_id': 'mean', 'day': 'count'})
    activity_freq.columns = ['user_id', 'activity_count']
    return register_day,launch_freq,video_freq,activity_freq

for i in range(1,31):
    register_day,launch_freq,video_freq,activity_freq = data_during(i,i+1)


#    launch_freq = cal_freq(launch_df,agg_funs,groupby_name='user_id',columns_names=['user_id', 'launch_count'])
#    video_freq = cal_freq(video_df,agg_funs,groupby_name='user_id',columns_names=['user_id', 'video_count'])
#    activity_freq = cal_freq(activity_df,agg_funs,groupby_name='user_id',columns_names=['user_id', 'activity_count'])
#    merged_df = register_df.merge(launch_freq, on='user_id', right_index=True, how='left')
#    merged_df = merged_df.merge(video_freq, on='user_id', right_index=True, how='left')
#    merged_df = merged_df.merge(activity_freq, on='user_id', right_index=True, how='left')
#    return merged_df









