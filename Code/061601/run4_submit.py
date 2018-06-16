# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 20:19:13 2018

@author: Administrator
"""

import pandas as pd
from datetime import datetime
import os
import sys
from save_log import Logger

def submit(submit_mode,submit_threshold=0,submit_user_num=23727):
    if submit_mode==1:
        active_user_id = result[result['result'] >= submit_threshold]
        print('submit_threshold:',submit_threshold)
        print('predict_user_num:',len(active_user_id))
    elif submit_mode==2:
        active_user_id = result.sort_values(by='result', axis=0, ascending=False).iloc[0:submit_user_num,:]
        print('submit_threshold:',active_user_id.iloc[-1,1])
        print('predict_user_num:',len(active_user_id))
    else:
        print('submit mode error!')
    return active_user_id

result = pd.read_csv(os.path.join(os.pardir,os.pardir,'result/lgb_result.csv'))
stdout_backup = sys.stdout
sys.stdout = Logger("train_info.txt")
submit_txt_name = 'submit_result_'+str(datetime.now().date().month)+str(datetime.now().date().day)+str(datetime.now().time().hour)+str(datetime.now().time().minute)+'.txt'
print(submit_txt_name)
active_user_id = submit(submit_mode=2,submit_threshold=0.4,submit_user_num=23727)
sys.stdout = stdout_backup
del active_user_id['result']
active_user_id.to_csv(submit_txt_name, index=False, header=False)