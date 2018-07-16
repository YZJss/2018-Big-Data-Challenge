import numpy as np
import pandas as pd

one_dataSet_train_path = '\\one_dataSet_train_'
one_dataSet_test_path = '\\one_dataSet_test_'
two_dataSet_train_path = '\\two_dataSet_train_'
two_dataSet_test_path = '\\two_dataSet_test_'
three_dataSet_train_path = '\\three_dataSet_train_'

def get_train_label(train_path,test_path):
    #data1 data2 打标 0 1
    train_reg = pd.read_csv(train_path + 'register.csv',usecols=['user_id'])
    train_app = pd.read_csv(train_path + 'app.csv',usecols=['user_id'])
    train_video = pd.read_csv(train_path + 'video.csv',usecols=['user_id'])
    train_act = pd.read_csv(train_path + 'activity.csv',usecols=['user_id'])
    train_data_id = np.unique(pd.concat([train_reg,train_app,train_video,train_act]))

    test_reg = pd.read_csv(test_path + 'register.csv', usecols=['user_id'])
    test_app = pd.read_csv(test_path + 'app.csv', usecols=['user_id'])
    test_video = pd.read_csv(test_path + 'video.csv', usecols=['user_id'])
    test_act = pd.read_csv(test_path + 'activity.csv', usecols=['user_id'])
    test_data_id = np.unique(pd.concat([test_reg, test_app, test_video, test_act]))

    train_label=[]
    for i in train_data_id:
        if i in test_data_id:
            train_label.append(1)
        else:
            train_label.append(0)
    train_data = pd.DataFrame()
    train_data['user_id'] = train_data_id
    train_data['label'] = train_label
    print("打标完成")
    return train_data
def get_test(test_path):
    #data3 user_id
    test_reg = pd.read_csv(test_path + 'register.csv', usecols=['user_id'])
    test_app = pd.read_csv(test_path + 'app.csv', usecols=['user_id'])
    test_video = pd.read_csv(test_path + 'video.csv', usecols=['user_id'])
    test_act = pd.read_csv(test_path + 'activity.csv', usecols=['user_id'])
    test_data_id = np.unique(pd.concat([test_reg, test_app, test_video, test_act]))
    test_data = pd.DataFrame()
    test_data['user_id'] = test_data_id
    return test_data
#feature
def get_app_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['app_min_day'] = row['app_day'].min()
    feature['app_max_day'] = row['app_day'].max()
    feature['app_count'] = row['app_day'].count()
    feature['app_mean_day'] = row['app_day'].mean()
    feature['app_var_day'] = row['app_day'].var()
    feature['app_std_day'] = row['app_day'].std()
    feature['app_ske_day'] = row['app_day'].skew()
    feature['app_kurt_day'] = row['app_day'].kurt()
    try:
        feature['app_last2_day'] = sorted(list(row['app_day']))[-2]
    except IndexError :
        feature['app_last2_day'] = sorted(list(row['app_day']))[-1]
    feature['app_last_day'] = sorted(list(row['app_day']))[-1]
    feature['app_diff_mean_day'] = row['app_day'].diff().mean()
    feature['app_diff_var_day'] = row['app_day'].diff().var()
    feature['app_diff_ske_day'] = row['app_day'].diff().skew()
    feature['app_diff_kurt_day'] = row['app_day'].diff().kurt()
    feature['app_diff_min_day'] = row['app_day'].diff().min()
    feature['app_diff_max_day'] = row['app_day'].diff().max()

    return feature
def get_video_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['video_min_day'] = row['video_day'].min()
    feature['video_max_day'] = row['video_day'].max()
    feature['video_count'] = row['video_day'].count()
    feature['video_mean_day'] = row['video_day'].mean()
    feature['video_var_day'] = row['video_day'].var()
    feature['video_std_day'] = row['video_day'].std()
    feature['video_ske_day'] = row['video_day'].skew()
    feature['video_kurt_day'] = row['video_day'].kurt()
    try:
        feature['video_last2_day'] = sorted(list(row['video_day']))[-2]
    except IndexError :
        feature['video_last2_day'] = sorted(list(row['video_day']))[-1]
    feature['video_last_day'] = sorted(list(row['video_day']))[-1]
    feature['video_diff_mean_day'] = row['video_day'].diff().mean()
    feature['video_diff_var_day'] = row['video_day'].diff().var()
    feature['video_diff_ske_day'] = row['video_day'].diff().skew()
    feature['video_diff_kurt_day'] = row['video_day'].diff().kurt()
    feature['video_diff_min_day'] = row['video_day'].diff().min()
    feature['video_diff_max_day'] = row['video_day'].diff().max()
    return feature
def get_activity_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['act_day_count'] = row['activity_day'].count()
    feature['act_min_day'] = row['activity_day'].min()
    feature['act_max_day'] = row['activity_day'].max()
    feature['act_mean_day'] = row['activity_day'].mean()
    feature['act_std_day'] = row['activity_day'].std()
    feature['act_var_day'] = row['activity_day'].var()
    feature['act_ske_day'] = row['activity_day'].skew()
    feature['act_kurt_day'] = row['activity_day'].kurt()
    try:
        feature['act_last2_day'] = sorted(list(row['activity_day']))[-2]
    except IndexError :
        feature['act_last2_day'] = sorted(list(row['activity_day']))[-1]
    feature['act_last_day'] = sorted(list(row['activity_day']))[-1]
    feature['act_vid_count'] = row['video_id'].count()
    feature['act_aid_count'] = row['author_id'].count()
    feature['act_page_count'] = row['page'].count()
    feature['act_type_count'] = row['action_type'].count()

    feature['act_diff_mean_day'] = row['activity_day'].diff().mean()
    feature['act_diff_var_day'] = row['activity_day'].diff().var()
    feature['act_diff_ske_day'] = row['activity_day'].diff().skew()
    feature['act_diff_kurt_day'] = row['activity_day'].diff().kurt()
    feature['act_diff_min_day'] = row['activity_day'].diff().min()
    feature['act_diff_max_day'] = row['activity_day'].diff().max()
    return feature

def page_0_count(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['page_0_count'] = row['page'].count()
    return feature
def page_1_count(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['page_1_count'] = row['page'].count()
    return feature
def page_2_count(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['page_2_count'] = row['page'].count()
    return feature
def page_3_count(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['page_3_count'] = row['page'].count()
    return feature
def page_4_count(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['page_4_count'] = row['page'].count()
    return feature

def page_0_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['page_0_max'] = row['page_0_count'].max()
    feature['page_0_min'] = row['page_0_count'].min()
    feature['page_0_mean'] = row['page_0_count'].mean()
    feature['page_0_kur'] = row['page_0_count'].kurt()
    feature['page_0_ske'] = row['page_0_count'].skew()
    feature['page_0_std'] = row['page_0_count'].std()
    feature['page_0_last'] = sorted(list(row['page_0_count']))[-1]
    return feature
def page_1_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['page_1_max'] = row['page_1_count'].max()
    feature['page_1_min'] = row['page_1_count'].min()
    feature['page_1_mean'] = row['page_1_count'].mean()
    feature['page_1_kur'] = row['page_1_count'].kurt()
    feature['page_1_ske'] = row['page_1_count'].skew()
    feature['page_1_std'] = row['page_1_count'].std()
    feature['page_1_last'] = sorted(list(row['page_1_count']))[-1]
    return feature
def page_2_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['page_2_max'] = row['page_2_count'].max()
    feature['page_2_min'] = row['page_2_count'].min()
    feature['page_2_mean'] = row['page_2_count'].mean()
    feature['page_2_kur'] = row['page_2_count'].kurt()
    feature['page_2_ske'] = row['page_2_count'].skew()
    feature['page_2_std'] = row['page_2_count'].std()
    feature['page_2_last'] = sorted(list(row['page_2_count']))[-1]
    return feature
def page_3_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['page_3_max'] = row['page_3_count'].max()
    feature['page_3_min'] = row['page_3_count'].min()
    feature['page_3_mean'] = row['page_3_count'].mean()
    feature['page_3_kur'] = row['page_3_count'].kurt()
    feature['page_3_ske'] = row['page_3_count'].skew()
    feature['page_3_std'] = row['page_3_count'].std()
    feature['page_3_last'] = sorted(list(row['page_3_count']))[-1]
    return feature
def page_4_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['page_4_max'] = row['page_4_count'].max()
    feature['page_4_min'] = row['page_4_count'].min()
    feature['page_4_mean'] = row['page_4_count'].mean()
    feature['page_4_kur'] = row['page_4_count'].kurt()
    feature['page_4_ske'] = row['page_4_count'].skew()
    feature['page_4_std'] = row['page_4_count'].std()
    feature['page_4_last'] = sorted(list(row['page_4_count']))[-1]
    return feature

def type_0_count(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['type_0_count'] = row['action_type'].count()
    return feature
def type_1_count(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['type_1_count'] = row['action_type'].count()
    return feature
def type_2_count(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['type_2_count'] = row['action_type'].count()
    return feature
def type_3_count(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['type_3_count'] = row['action_type'].count()
    return feature
def type_4_count(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['type_4_count'] = row['action_type'].count()
    return feature
def type_5_count(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['type_5_count'] = row['action_type'].count()
    return feature
def type_0_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['type_0_max'] = row['type_0_count'].max()
    feature['type_0_min'] = row['type_0_count'].min()
    feature['type_0_mean'] = row['type_0_count'].mean()
    feature['type_0_kur'] = row['type_0_count'].kurt()
    feature['type_0_ske'] = row['type_0_count'].skew()
    feature['type_0_std'] = row['type_0_count'].std()
    feature['type_0_last'] = sorted(list(row['type_0_count']))[-1]
    return feature
def type_1_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['type_1_max'] = row['type_1_count'].max()
    feature['type_1_min'] = row['type_1_count'].min()
    feature['type_1_mean'] = row['type_1_count'].mean()
    feature['type_1_kur'] = row['type_1_count'].kurt()
    feature['type_1_ske'] = row['type_1_count'].skew()
    feature['type_1_std'] = row['type_1_count'].std()
    feature['type_1_last'] = sorted(list(row['type_1_count']))[-1]
    return feature
def type_2_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['type_2_max'] = row['type_2_count'].max()
    feature['type_2_min'] = row['type_2_count'].min()
    feature['type_2_mean'] = row['type_2_count'].mean()
    feature['type_2_kur'] = row['type_2_count'].kurt()
    feature['type_2_ske'] = row['type_2_count'].skew()
    feature['type_2_std'] = row['type_2_count'].std()
    feature['type_2_last'] = sorted(list(row['type_2_count']))[-1]
    return feature
def type_3_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['type_3_max'] = row['type_3_count'].max()
    feature['type_3_min'] = row['type_3_count'].min()
    feature['type_3_mean'] = row['type_3_count'].mean()
    feature['type_3_kur'] = row['type_3_count'].kurt()
    feature['type_3_ske'] = row['type_3_count'].skew()
    feature['type_3_std'] = row['type_3_count'].std()
    feature['type_3_last'] = sorted(list(row['type_3_count']))[-1]
    return feature
def type_4_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['type_4_max'] = row['type_4_count'].max()
    feature['type_4_min'] = row['type_4_count'].min()
    feature['type_4_mean'] = row['type_4_count'].mean()
    feature['type_4_kur'] = row['type_4_count'].kurt()
    feature['type_4_ske'] = row['type_4_count'].skew()
    feature['type_4_std'] = row['type_4_count'].std()
    feature['type_4_last'] = sorted(list(row['type_4_count']))[-1]
    return feature
def type_5_feature(row):
    feature = pd.Series()
    feature['user_id'] = list(row['user_id'])[0]
    feature['type_5_max'] = row['type_5_count'].max()
    feature['type_5_min'] = row['type_5_count'].min()
    feature['type_5_mean'] = row['type_5_count'].mean()
    feature['type_5_kur'] = row['type_5_count'].kurt()
    feature['type_5_ske'] = row['type_5_count'].skew()
    feature['type_5_std'] = row['type_5_count'].std()
    feature['type_5_last'] = sorted(list(row['type_5_count']))[-1]
    return feature

def deal_feature(path,user_id):
    app = pd.read_csv(path + 'app.csv')
    video = pd.read_csv(path + 'video.csv')
    reg = pd.read_csv(path + 'register.csv')
    act = pd.read_csv(path + 'activity.csv')
    feature = pd.DataFrame()
    feature['user_id'] = user_id

    gp_app = app.groupby(['user_id'])['app_day'].unique()
    def count_continue2(t):
        s = gp_app[t]
        ans = 0
        for i in s:
            if i + 1 in s:
                ans = ans + 1
        return ans
    feature['app_2days'] = app['user_id'].apply(count_continue2)#统计连续2天登陆app

    app_feature = app.groupby('user_id', sort=True).apply(get_app_feature)
    app_feature['app_day_ax_in'] = app_feature['app_max_day'] - app_feature['app_min_day']  # max-min
    app_feature['app_last_max'] = app_feature['app_last_day'] - app_feature['app_max_day']  # last-max
    app_feature['app_last2_sub'] = app_feature['app_last_day'] - app_feature['app_last2_day'] #[-1]-[-2]
    app_feature['reg_max_day'] = np.max(reg['register_day'])
    app_feature['app_sub_reg_day'] = app_feature['app_last_day'] - app_feature['reg_max_day'] #app_max-reg
    app_last_count = app[app.app_day == app.app_day.max()][['user_id']].groupby(['user_id']).size().rename(
        'app_last_count').reset_index()
    feature = pd.merge(feature, pd.DataFrame(app_last_count), on='user_id', how='left')
    feature = pd.merge(feature, pd.DataFrame(app_feature), on='user_id', how='left')
    del feature['reg_max_day']
    print('app表特征提取完毕')

    feature = pd.merge(feature,pd.DataFrame(reg),on='user_id',how='left')
    print('register表特征提取完毕')

    video_feature = video.groupby('user_id',sort=True).apply(get_video_feature)
    video_feature['video_day_ax_in'] = video_feature['video_max_day'] - video_feature['video_min_day']  # max-min
    video_feature['video_last_max'] = video_feature['video_last_day'] - video_feature['video_max_day']  # last-max
    video_feature['video_last2_sub'] = video_feature['video_last_day'] - video_feature['video_last2_day']  # [-1]-[-2]
    video_feature['reg_max_day'] = np.max(reg['register_day'])
    video_feature['video_sub_reg_day'] = video_feature['video_last_day'] - video_feature['reg_max_day'] #video_max-reg
    video_last_count = video[video.video_day == video.video_day.max()][['user_id']].groupby(['user_id']).size().rename(
        'video_last_count').reset_index()
    feature = pd.merge(feature, pd.DataFrame(video_last_count), on='user_id', how='left')
    feature = pd.merge(feature,pd.DataFrame(video_feature),on='user_id',how='left')
    del feature['reg_max_day']
    print('video表特征提取完毕')

    act_feature = act.groupby('user_id',sort=True).apply(get_activity_feature)
    act_feature['act_day_ax_in'] = act_feature['act_max_day'] - act_feature['act_min_day']  # max-min
    act_feature['act_last_max'] = act_feature['act_last_day'] - act_feature['act_max_day']  # last-max
    act_feature['act_last2_sub'] = act_feature['act_last_day'] - act_feature['act_last2_day']  # [-1]-[-2]
    act_feature['reg_max_day'] = np.max(reg['register_day'])
    act_feature['act_sub_reg_day'] = act_feature['act_last_day'] - act_feature['reg_max_day'] #act_max-reg
    act_last_count = act[act.activity_day == act.activity_day.max()][['user_id']].groupby(['user_id']).size().rename(
        'act_last_count').reset_index()
    feature = pd.merge(feature, pd.DataFrame(act_last_count), on='user_id', how='left')
    feature = pd.merge(feature,pd.DataFrame(act_feature),on='user_id',how='left')
    del feature['reg_max_day']
    print('activity表特征提取完毕')

    act_page0 = act[(act.page == 0)].groupby(['user_id', 'activity_day']).apply(page_0_count)
    act_page_0f = act_page0.groupby(['user_id']).apply(page_0_feature)
    feature = pd.merge(feature, pd.DataFrame(act_page_0f), on='user_id', how='left')

    act_page1 = act[(act.page == 1)].groupby(['user_id', 'activity_day']).apply(page_1_count)
    act_page_1f = act_page1.groupby(['user_id']).apply(page_1_feature)
    feature = pd.merge(feature, pd.DataFrame(act_page_1f), on='user_id', how='left')

    act_page2 = act[(act.page == 2)].groupby(['user_id', 'activity_day']).apply(page_2_count)
    act_page_2f = act_page2.groupby(['user_id']).apply(page_2_feature)
    feature = pd.merge(feature, pd.DataFrame(act_page_2f), on='user_id', how='left')

    act_page3 = act[(act.page == 3)].groupby(['user_id', 'activity_day']).apply(page_3_count)
    act_page_3f = act_page3.groupby(['user_id']).apply(page_3_feature)
    feature = pd.merge(feature, pd.DataFrame(act_page_3f), on='user_id', how='left')

    act_page4 = act[(act.page == 4)].groupby(['user_id', 'activity_day']).apply(page_4_count)
    act_page_4f = act_page4.groupby(['user_id']).apply(page_4_feature)
    feature = pd.merge(feature, pd.DataFrame(act_page_4f), on='user_id', how='left')

    print('page表特征提取完毕')

    act_type0 = act[(act.action_type == 0)].groupby(['user_id', 'activity_day']).apply(type_0_count)
    act_type_0f = act_type0.groupby(['user_id']).apply(type_0_feature)
    feature = pd.merge(feature, pd.DataFrame(act_type_0f), on='user_id', how='left')

    act_type1 = act[(act.action_type == 1)].groupby(['user_id', 'activity_day']).apply(type_1_count)
    act_type_1f = act_type1.groupby(['user_id']).apply(type_1_feature)
    feature = pd.merge(feature, pd.DataFrame(act_type_1f), on='user_id', how='left')

    act_type2 = act[(act.action_type == 2)].groupby(['user_id', 'activity_day']).apply(type_2_count)
    act_type_2f = act_type2.groupby(['user_id']).apply(type_2_feature)
    feature = pd.merge(feature, pd.DataFrame(act_type_2f), on='user_id', how='left')

    act_type3 = act[(act.action_type == 3)].groupby(['user_id', 'activity_day']).apply(type_3_count)
    act_type_3f = act_type3.groupby(['user_id']).apply(type_3_feature)
    feature = pd.merge(feature, pd.DataFrame(act_type_3f), on='user_id', how='left')

    act_type4 = act[(act.action_type == 4)].groupby(['user_id', 'activity_day']).apply(type_4_count)
    act_type_4f = act_type4.groupby(['user_id']).apply(type_4_feature)
    feature = pd.merge(feature, pd.DataFrame(act_type_4f), on='user_id', how='left')

    act_type5 = act[(act.action_type == 5)].groupby(['user_id', 'activity_day']).apply(type_5_count)
    act_type_5f = act_type5.groupby(['user_id']).apply(type_5_feature)
    feature = pd.merge(feature, pd.DataFrame(act_type_5f), on='user_id', how='left')

    print('action_type表特征提取完毕')
    act_0page = act[act.page == 0][['user_id']].groupby(['user_id']).size().rename('act_0page_count').reset_index()
    feature = pd.merge(feature, pd.DataFrame(act_0page), on='user_id', how='left')
    act_1page = act[act.page == 1][['user_id']].groupby(['user_id']).size().rename('act_1page_count').reset_index()
    feature = pd.merge(feature, pd.DataFrame(act_1page), on='user_id', how='left')
    act_2page = act[act.page == 2][['user_id']].groupby(['user_id']).size().rename('act_2page_count').reset_index()
    feature = pd.merge(feature, pd.DataFrame(act_2page), on='user_id', how='left')
    act_3page = act[act.page == 3][['user_id']].groupby(['user_id']).size().rename('act_3page_count').reset_index()
    feature = pd.merge(feature, pd.DataFrame(act_3page), on='user_id', how='left')
    act_4page = act[act.page == 4][['user_id']].groupby(['user_id']).size().rename('act_4page_count').reset_index()
    feature = pd.merge(feature, pd.DataFrame(act_4page), on='user_id', how='left')

    act_0type = act[act.action_type == 0][['user_id']].groupby(['user_id']).size().rename(
        'act_type0_count').reset_index()
    feature = pd.merge(feature, pd.DataFrame(act_0type), on='user_id', how='left')
    act_1type = act[act.action_type == 1][['user_id']].groupby(['user_id']).size().rename(
        'act_type1_count').reset_index()
    feature = pd.merge(feature, pd.DataFrame(act_1type), on='user_id', how='left')
    act_2type = act[act.action_type == 2][['user_id']].groupby(['user_id']).size().rename(
        'act_type2_count').reset_index()
    feature = pd.merge(feature, pd.DataFrame(act_2type), on='user_id', how='left')
    act_3type = act[act.action_type == 3][['user_id']].groupby(['user_id']).size().rename(
        'act_type3_count').reset_index()
    feature = pd.merge(feature, pd.DataFrame(act_3type), on='user_id', how='left')
    act_4type = act[act.action_type == 4][['user_id']].groupby(['user_id']).size().rename(
        'act_type4_count').reset_index()
    feature = pd.merge(feature, pd.DataFrame(act_4type), on='user_id', how='left')
    act_5type = act[act.action_type == 5][['user_id']].groupby(['user_id']).size().rename(
        'act_type5_count').reset_index()
    feature = pd.merge(feature, pd.DataFrame(act_5type), on='user_id', how='left')

    feature['0page_div_sum'] = feature['act_0page_count'] / feature['act_page_count']
    feature['1page_div_sum'] = feature['act_1page_count'] / feature['act_page_count']
    feature['2page_div_sum'] = feature['act_2page_count'] / feature['act_page_count']
    feature['3page_div_sum'] = feature['act_3page_count'] / feature['act_page_count']
    feature['4page_div_sum'] = feature['act_4page_count'] / feature['act_page_count']

    feature['0type_div_sum'] = feature['act_type0_count'] / feature['act_type_count']
    feature['1type_div_sum'] = feature['act_type1_count'] / feature['act_type_count']
    feature['2type_div_sum'] = feature['act_type2_count'] / feature['act_type_count']
    feature['3type_div_sum'] = feature['act_type3_count'] / feature['act_type_count']
    feature['4type_div_sum'] = feature['act_type4_count'] / feature['act_type_count']
    feature['5type_div_sum'] = feature['act_type5_count'] / feature['act_type_count']
    print('page type div提取完毕')
    print(feature.shape)
    return feature
def get_data_feature():
    one_train_data = get_train_label(one_dataSet_train_path, one_dataSet_test_path)
    one_feature = deal_feature(one_dataSet_train_path, one_train_data['user_id'])
    one_feature['label'] = one_train_data['label']
    print('第一组特征提取完毕')

    two_train_data = get_train_label(two_dataSet_train_path, two_dataSet_test_path)
    two_feature = deal_feature(two_dataSet_train_path, two_train_data['user_id'])
    two_feature['label'] = two_train_data['label']
    print('第二组特征提取完毕')

    one_feature.to_csv('\\data1.csv', index=False)
    two_feature.to_csv('\\data2.csv', index=False)
    # train_feature = pd.concat([one_feature,two_feature])
    # train_feature.to_csv('\\test.csv',index=False)

    test_data=get_test(three_dataSet_train_path)
    test_feature = deal_feature(three_dataSet_train_path,test_data['user_id'])
    test_feature.to_csv('\\test.csv',index=False)

if __name__ == '__main__':
    get_data_feature()