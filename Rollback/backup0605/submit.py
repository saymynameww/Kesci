# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 20:19:13 2018

@author: Administrator
"""

import pandas as pd

lgb = pd.read_csv('result/lgb_result.csv')
print(lgb)
result = lgb[lgb['result'] >= 0.4]
print(len(result))
del result['result']
result.to_csv('result/submit_result.txt', index=False, header=False)