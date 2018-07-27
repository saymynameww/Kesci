# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 00:09:36 2018

@author: Administrator
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.noise import GaussianDropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from keras.callbacks import EarlyStopping

train_path = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/train.csv')
test_path = os.path.join(os.pardir,os.pardir,os.pardir, 'Kesci-data-dealt/train_and_test/test.csv')
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train_feature = train.drop(['user_id','label'],axis=1)
#train_feature = train[['register_day','register_type', 'device_type', 'register_rest_day']]
train_label = train['label']
test_feature = test.drop(['user_id'],axis=1)
#test_feature = test[['register_day','register_type', 'device_type', 'register_rest_day']]
train_feature, offline_test_feature, train_label, offline_test_label = train_test_split(train_feature, train_label, test_size=0.1,random_state=624)

len_x = train_feature.shape[1]

print("\nSetting up neural network model...")
nn = Sequential()
nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = len_x))
nn.add(PReLU())
nn.add(Dropout(.4))
nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units = 26, kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(1, kernel_initializer='normal'))
nn.compile(loss='mae', optimizer=Adam(lr=4e-2, decay=1e-3), metrics=['accuracy']) #lr=4e-3, decay=1e-4

print("\nFitting neural network model...")
#nn.fit(np.array(train_feature), np.array(train_label), batch_size = 32, epochs = 70, verbose=2)
nn.fit(np.array(train_feature), np.array(train_label), batch_size = 32, epochs = 70, verbose=1, validation_data=(offline_test_feature, offline_test_label), callbacks = [EarlyStopping(monitor='val_acc', patience=20)])

#评估模型
score = nn.evaluate(offline_test_feature, offline_test_label, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print("\nPredicting with neural network model...")
#print("x_test.shape:",x_test.shape)
y_pred_ann = nn.predict(offline_test_feature)

print( "\nPreparing results for write..." )
nn_pred = y_pred_ann.flatten()
print( "Type of nn_pred is ", type(nn_pred) )
print( "Shape of nn_pred is ", nn_pred.shape )
auc_score = roc_auc_score(offline_test_label,nn_pred)
print( "Auc_score is ", auc_score )


print( "\nNeural Network predictions:" )
print( pd.DataFrame(nn_pred).head() )