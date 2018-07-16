import pandas as pd

one_dataSet_train_path = '\\one_dataSet_train_'
one_dataSet_test_path = '\\one_dataSet_test_'
two_dataSet_train_path = '\\two_dataSet_train_'
two_dataSet_test_path = '\\two_dataSet_test_'
three_dataSet_train_path = '\\three_dataSet_train_'

data_register=pd.read_csv("\\user_register_log.txt",sep="\t",names=["user_id","register_day","register_type","device_type"])
data_app=pd.read_csv("\\app_launch_log.txt",sep="\t",names=["user_id","app_day"])
data_video=pd.read_csv("\\video_create_log.txt",sep="\t",names=["user_id","video_day"])
data_activity=pd.read_csv("\\user_activity_log.txt",sep="\t",names=["user_id","activity_day","page","video_id","author_id","action_type"])
#分割数据集
def cut_data_as_time(new_dataSet_path,begin_day,end_day,reg_begin_day,reg_end_day):
    #分割数据集
    temp_register = data_register[(data_register['register_day']>=reg_begin_day) & (data_register['register_day']<=reg_end_day)]
    temp_video = data_video[(data_video['video_day']>=begin_day) & (data_video['video_day']<=end_day)]
    temp_app = data_app[(data_app['app_day']>=begin_day) & (data_app['app_day']<=end_day)]
    temp_activity = data_activity[(data_activity['activity_day']>=begin_day) & (data_activity['activity_day']<=end_day)]

    temp_register.to_csv(new_dataSet_path + 'register.csv',index=False)
    temp_video.to_csv(new_dataSet_path + 'video.csv', index=False)
    temp_app.to_csv(new_dataSet_path + 'app.csv', index=False)
    temp_activity.to_csv(new_dataSet_path + 'activity.csv', index=False)
def generate_dataSet():
    #1-16预测17-23用户是否活跃  data1
    begin_day = 1
    end_day = 16
    reg_begin_day = 1
    reg_end_day = 16
    cut_data_as_time(one_dataSet_train_path,begin_day,end_day,reg_begin_day,reg_end_day)
    begin_day = 17
    end_day = 23
    reg_begin_day = 17
    reg_end_day = 23
    cut_data_as_time(one_dataSet_test_path,begin_day, end_day,reg_begin_day,reg_end_day)
    #8-23预测24-30用户是否活跃  data2
    begin_day = 8
    end_day = 23
    reg_begin_day = 1
    reg_end_day = 23
    cut_data_as_time(two_dataSet_train_path,begin_day, end_day,reg_begin_day,reg_end_day)
    begin_day = 24
    end_day = 30
    reg_begin_day = 24
    reg_end_day = 30
    cut_data_as_time(two_dataSet_test_path,begin_day, end_day,reg_begin_day,reg_end_day)
    #15-30预测30-37用户是否活跃  data3
    begin_day = 15
    end_day = 30
    reg_begin_day = 1
    reg_end_day = 30
    cut_data_as_time(three_dataSet_train_path,begin_day, end_day,reg_begin_day,reg_end_day)
    print("成功划分三个数据集")

if __name__ == '__main__':
    generate_dataSet()
