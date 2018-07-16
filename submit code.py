import pandas as pd
lgb = pd.read_csv('\\df_result.csv')
print(lgb)
res = lgb[lgb['result']>=0.4]
print(len(res))
del res['result']
res.to_csv('\\result.txt',index=False,header=False)