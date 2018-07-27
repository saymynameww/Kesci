# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:35:30 2018

@author: Administrator
"""

##preprocess
import os
import pandas as pd
from datetime import datetime

time_start = datetime.now()
print('Start time:',time_start.strftime('%Y-%m-%d %H:%M:%S'))

input_dir = os.path.join(os.pardir,os.pardir,os.pardir, '1Kesci-data/original_data')
output_dir = os.path.join(os.pardir,os.pardir,os.pardir, '1Kesci-data/sorted_data')

app_launch_log = pd.read_csv(input_dir + '/app_launch_log.txt',sep='\t', header=None)
app_launch_log = app_launch_log.sort_values(by=[0,1])
app_launch_log = app_launch_log.rename({0:'user_id',1:'launch_day'},axis=1)
app_launch_log.to_csv(output_dir +'/app_launch_log.csv',index=False)

user_activity_log = pd.read_csv(input_dir + '/user_activity_log.txt',sep='\t', header=None)
user_activity_log = user_activity_log.sort_values(by=[0,1])
user_activity_log = user_activity_log.rename({0:'user_id',1:'activity_day',2:'page',3:'video_id',4:'author_id',5:'action_type'},axis=1)
user_activity_log.to_csv(output_dir +'/user_activity_log.csv',index=False)

user_register_log = pd.read_csv(input_dir + '/user_register_log.txt',sep='\t', header=None)
user_register_log = user_register_log.sort_values(by=[0,1])
user_register_log = user_register_log.rename({0:'user_id',1:'register_day',2:'register_type',3:'device_type'},axis=1)
user_register_log.to_csv(output_dir +'/user_register_log.csv',index=False)

video_create_log = pd.read_csv(input_dir + '/video_create_log.txt',sep='\t', header=None)
video_create_log = video_create_log.sort_values(by=[0,1])
video_create_log = video_create_log.rename({0:'user_id',1:'video_day'},axis=1)
video_create_log.to_csv(output_dir +'/video_create_log.csv',index=False)

time_end = datetime.now()
print('End time:',time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('Total time:',"%.2f" % ((time_end-time_start).seconds/60),'minutes')

for key in list(globals().keys()): 
    if not key.startswith("__"): 
        globals().pop(key) 