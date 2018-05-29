# -*- coding: utf-8 -*-
"""
Created on Sat May 26 22:39:36 2018

@author: wxk06
"""

## code backup

# =============================================================================
## no use function
# agg_funs = {'user_id': 'mean', 'day': 'count'}
# launch_freq = launch_df_train.groupby('user_id').agg(agg_funs)
# launch_freq.columns = ['user_id', 'launch_count']
# video_freq = video_df_train.groupby('user_id').agg(agg_funs)
# video_freq.columns = ['user_id', 'video_count']
# activity_freq = activity_df_train.groupby('user_id').agg(agg_funs)
# activity_freq.columns = ['user_id', 'activity_count']
# merged_df_train = register_df.merge(launch_freq, on='user_id', right_index=True, how='left')
# merged_df_train = merged_df_train.merge(video_freq, on='user_id', right_index=True, how='left')
# merged_df_train = merged_df_train.merge(activity_freq, on='user_id', right_index=True, how='left')
# =============================================================================



# =============================================================================
## prepare training data
# launch_df_train = select_df(1,train_day,launch_df)
# video_df_train = select_df(1,train_day,video_df)
# activity_df_train = select_df(1,train_day,activity_df)
# agg_funs = {'user_id': 'mean', 'day': 'count'}
# launch_freq_train = cal_freq(launch_df_train,agg_funs,groupby_name='user_id',columns_names=['user_id', 'launch_count'])
# video_freq_train = cal_freq(video_df_train,agg_funs,groupby_name='user_id',columns_names=['user_id', 'video_count'])
# activity_freq_train = cal_freq(activity_df_train,agg_funs,groupby_name='user_id',columns_names=['user_id', 'activity_count'])
# merged_df_train = register_df.merge(launch_freq_train, on='user_id', right_index=True, how='left')
# merged_df_train = merged_df_train.merge(video_freq_train, on='user_id', right_index=True, how='left')
# merged_df_train = merged_df_train.merge(activity_freq_train, on='user_id', right_index=True, how='left')
# 
# 
# launch_df_test = select_df(train_day+1,train_day+test_day,launch_df)
# video_df_test = select_df(train_day+1,train_day+test_day,video_df)
# activity_df_test = select_df(train_day+1,train_day+test_day,activity_df)
# launch_freq_test = cal_freq(launch_df_test,agg_funs,groupby_name='user_id',columns_names=['user_id', 'launch_count'])
# video_freq_test = cal_freq(video_df_test,agg_funs,groupby_name='user_id',columns_names=['user_id', 'video_count'])
# activity_freq_test = cal_freq(activity_df_test,agg_funs,groupby_name='user_id',columns_names=['user_id', 'activity_count'])
# merged_df_test = launch_freq_test.merge(video_freq_test, on='user_id', right_index=True, how='left')
# merged_df_test = merged_df_test.merge(activity_freq_test, on='user_id', right_index=True, how='left')
# 
# merged_df_test = merged_df_test.fillna(0)
# merged_df_test['total_count'] = np.sum(merged_df_test[['video_count','activity_count']],axis = 1)
# merged_df_test.loc[merged_df_test['total_count'] > 1, 'total_count'] = 1
# merged_df_test = merged_df_test.drop(['launch_count','video_count','activity_count'],axis=1)
# 
# merged_df_train = merged_df_train.merge(merged_df_test, on='user_id', left_index=True, how='left')
# merged_df_train = merged_df_train.fillna(0)
# x_train = merged_df_train.drop(['user_id','total_count'],axis =1).values
# y_train = merged_df_train['total_count']
# =============================================================================

# =============================================================================
# # how many users have launched, video, activity
# isdup = pd.DataFrame.duplicated(app_launch_df,subset='user_id',keep='first')
# launch_user_num = app_launch_df.shape[0] - np.sum(isdup)
# isdup = pd.DataFrame.duplicated(video_create_df,subset='user_id',keep='first')
# video_user_num = video_create_df.shape[0] - np.sum(isdup)
# isdup = pd.DataFrame.duplicated(user_activity_df,subset='user_id',keep='first')
# activity_user_num = user_activity_df.shape[0] - np.sum(isdup)
# =============================================================================

df.to_csv('A.txt', index=False)

t = activity_df[['action_type']].drop_duplicates()


l = launch_df.loc[launch_df['user_id'].drop_duplicates().index]
l2 = register_df.merge(l, on='user_id',left_index=True, how='left')

merged_df_y_train['total_count'] = np.sum(merged_df_y_train[['launch_count','video_count','activity_count']],axis = 1)


# =============================================================================
# # 7 days active
# userPre = activity_df[activity_df.day>=24]
# sub = userPre[['user_id']].drop_duplicates()
# sub.to_csv('sub.txt', index=False)
# result_list = list(result)
# t1 = sub.isin(result_list)
# print(np.sum(t1))
# sub['active_7_days'] = 1
# merged_sub = merged_df_test.merge(sub, on='user_id', left_index=True, how='left')
# merged_sub = merged_sub.fillna(0)
# m = merged_sub.loc[merged_sub['active_7_days'] != merged_sub['total_count']]
# 
# userPre_video = video_df[video_df.day>=24]
# sub_video = userPre_video[['user_id']].drop_duplicates()
# sub_video.to_csv('sub_video.txt', index=False)
# =============================================================================
