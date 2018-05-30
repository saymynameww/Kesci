# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:11:05 2018

@author: ASUS
"""

import pandas as pd
df1 = pd.DataFrame({'A':[1,3,5,7],'B':[2,4,6,8],'C':[5,6,7,8]})
df2 = pd.DataFrame({'A':[3,1,6,9,56],'B':[4,2,9,10,76],'D':[1,2,3,4,25]})
df = df1.merge(df2,how='outer',on=['A','B'],left_index=True)