# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 18:47:58 2018

@author: Administrator
"""

import os
import pandas as pd
from datetime import datetime

time_start = datetime.now()
print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))

print('Generating datasets...')
input_dir = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/sorted_data')
print('Input files:\n{}'.format(os.listdir(input_dir)))
dataset_1_feat_dir = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/dataset_1_feat')
dataset_1_label_dir = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/dataset_1_label')
dataset_2_feat_dir = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/dataset_2_feat')
dataset_2_label_dir = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/dataset_2_label')
dataset_3_feat_dir = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/dataset_3_feat')

register = pd.read_csv(input_dir + '/user_register_log.csv')
launch = pd.read_csv(input_dir + '/app_launch_log.csv')
video = pd.read_csv(input_dir + '/video_create_log.csv')
activity = pd.read_csv(input_dir + '/user_activity_log.csv')

def cut_data_on_time(output_path,begin_day,end_day):
    temp_register = register[(register['register_day'] >= 1) & (register['register_day'] <= end_day)]
    temp_launch = launch[(launch['launch_day'] >= begin_day) & (launch['launch_day'] <= end_day)]
    temp_video = video[(video['video_day'] >= begin_day) & (video['video_day'] <= end_day)]
    temp_activity = activity[(activity['activity_day'] >= begin_day) & (activity['activity_day'] <= end_day)]
    
    temp_register.to_csv(output_path + '/register.csv',index=False)
    temp_launch.to_csv(output_path + '/launch.csv',index=False)
    temp_video.to_csv(output_path + '/video.csv',index=False)
    temp_activity.to_csv(output_path + '/activity.csv',index=False)

def generate_dataset():
    print('Cutting train data set 1 ...')
    begin_day = 1
    end_day = 16
    cut_data_on_time(dataset_1_feat_dir,begin_day,end_day)
    begin_day = 17
    end_day = 23
    cut_data_on_time(dataset_1_label_dir,begin_day,end_day)
    print('Cutting train data set 2 ...')
    begin_day = 8
    end_day = 23
    cut_data_on_time(dataset_2_feat_dir,begin_day,end_day)
    begin_day = 24
    end_day = 30
    cut_data_on_time(dataset_2_label_dir,begin_day,end_day)
    print('Cutting test data set...')
    begin_day = 15
    end_day = 30
    cut_data_on_time(dataset_3_feat_dir,begin_day,end_day)
    
generate_dataset()
print('Dataset generated.')

time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')
