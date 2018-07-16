import lightgbm as lgb
import pandas as pd
from sklearn import model_selection
import sklearn.metrics

test = pd.read_csv('\\test.csv',index_col= False)
train = pd.read_csv('\\train.csv',index_col=False)
label = train['label']
feature =[]#筛选特征
test_feature = test[feature]
train_feature = train[feature]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_feature,label,test_size=0.2,random_state=2018)
lgb_train = lgb.Dataset(train_feature,label)
lgb_eval = lgb.Dataset(X_test,Y_test,reference=lgb_train)
params = {
    'task':'train',
    'boosting_type':'gbdt',
    'objective':'binary',
    'metric': {'auc', 'binary_logloss'},
    'learning_rate':0.09,
    'num_leaves': 24,
    'max_depth': 5,
    'max_bin': 90,
    'colsample_bytree': 0.9
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=80,
                valid_sets=lgb_eval)
gbm.save_model('\\lgb_model.txt')
temp = gbm.predict(X_test)
temp[temp>0.4]=1
temp[temp<=0.4]=0
print('结果：'+str(sklearn.metrics.f1_score(Y_test,temp)))
print('特征重要性：'+ str(list(gbm.feature_importance())))

#预测test
pre = gbm.predict(test_feature)
df_result = pd.DataFrame()
df_result['user_id'] = test['user_id']
df_result['result'] = pre
res = df_result[df_result['result']>=0.4]
print(len(res))
df_result.to_csv('\\df_result.csv',index=False)
