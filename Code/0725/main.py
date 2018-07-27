# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:16:06 2018

@author: Administrator
"""

import os

cmd = "python ./run1_preprocess.py"
os.system(cmd)

cmd = "python ./run2_generate_dataset_2plus1.py"
os.system(cmd)

cmd = "python ./run3_generate_feature_2.py"
os.system(cmd)

cmd = "python ./run4_train_lgb.py"
os.system(cmd)

#cmd = "python ./run4_submit.py"
#os.system(cmd)

