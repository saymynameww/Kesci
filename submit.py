# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 20:19:13 2018

@author: Administrator
"""

import pandas as pd
from datetime import datetime

result = pd.read_csv('result/lgb_result.csv')
active_user_id = result[result['result'] >= 0.4]
#active_user_id = result.sort_values(by='result', axis=0, ascending=False).iloc[0:23760,:]
print('threshold:',active_user_id.iloc[-1,1])
print(len(active_user_id))

del active_user_id['result']
active_user_id.to_csv('result/submit_result_'+str(datetime.now().date().month)+str(datetime.now().date().day)+str(datetime.now().time().hour)+str(datetime.now().time().minute)+'.txt', index=False, header=False)