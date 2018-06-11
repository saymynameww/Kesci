# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:11:05 2018

@author: ASUS
"""

import pandas as pd
df1 = pd.DataFrame({'A':[1,1,2,2],'B':[2,2,6,8],'C':[8,7,6,5]})
df2 = pd.DataFrame({'A':[3,1,6,9,56],'B':[4,2,9,10,76],'D':[1,2,3,4,25]})
#df = df1.merge(df2,how='outer',on=['A','B'],left_index=True)
#t = df1.groupby('A').agg({'C':'min'})

#t = df1.groupby(['A'])['C'].min().rename('T').reset_index()

t = df1.groupby(['A']).agg({'A':'mean','B':'mode'})