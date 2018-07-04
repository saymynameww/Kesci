# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:16:06 2018

@author: Administrator
"""

import os

#cmd = "python ./preprocess.py"
#os.system(cmd)

#cmd = "python ./run1_generate_dataset.py"
#os.system(cmd)

cmd = "python ./run2_generate_feature.py"
os.system(cmd)

cmd = "python ./run3_train_lgb.py"
os.system(cmd)

cmd = "python ./run4_submit.py"
os.system(cmd)

