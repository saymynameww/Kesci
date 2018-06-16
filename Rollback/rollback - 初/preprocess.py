# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 18:16:49 2018

@author: NURBS
"""

import os
import pandas as pd

input_dir = os.path.join(os.pardir, 'Kesci-data')
output_dir = os.path.join(os.pardir, 'Kesci-data-dealt/sorted_data')

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