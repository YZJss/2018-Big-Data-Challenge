#[2018中国高校计算机大赛——大数据挑战赛](https://www.kesci.com/apps/home/competition/5ab8c36a8643e33f5138cba4/content)
##赛事背景
 2018中国高校计算机大赛——大数据挑战赛（以下简称“大赛”）是由教育部高等学校计算机类专业教学指导委员会、教育部高等学校软件工程专业教学指导委员会、教育部高等学校大学计算机课程教学指导委会、全国高等学校计算机教育研究会主办，由清华大学和北京快手科技有限公司联合承办，以脱敏和采样后的数据信息为基础开展的高端算法竞赛。大赛面向全球高校在校生开放，旨在通过竞技的方式提升人们对数据分析与处理的算法研究与技术应用能力，探索大数据的核心科学与技术问题，尝试创新大数据技术，推动大数据的产学研用，本次大赛鼓励高校教师参与指导。
 
  本次大赛基于脱敏和采样后的数据信息，预测未来一段时间活跃的用户。参赛队伍需要设计相应的算法进行数据分析和处理，比赛结果按照指定的评价指标使用在线评测数据进行评测和排名，得分最优者获胜。
##赛题与评审介绍
###数据说明
大赛提供脱敏和采样后用户行为数据，日期信息进行统一编号，第一天编号为 01， 第二天为 02， 以此类推，所有文件中列使用 tab 分割。
####1.注册日志（user_register_log.txt）
|列名|类型|说明|示例|
|:---|:---|:---|:---|
|user_id|Int|用户唯一标识（脱敏后）|666|
|register_day|String|日期|01, 02 ..  30|
|register_type|Int|来源渠道（脱敏后）|0|
|device type|Int|设备类型（脱敏后）|0|
####2.APP 启动日志（app_launch_log.txt）
|列名|类型|说明|示例|
|:---|:---|:---|:---|
|user_id|Int|用户唯一标识（脱敏后）|666|
|day|String|日期|01, 02 ..  30|
####3.拍摄日志（video_create_log.txt）
|列名|类型|说明|示例|
|:---|:---|:---|:---|
|user_id|Int|用户唯一标识（脱敏后）|666|
|day|String|拍摄日期|01, 02 ..  30|
####4.行为日志（user_activity_log.txt）
|列名|类型|说明|示例|
|:---|:---|:---|:---|
|user_id|Int|用户唯一标识（脱敏后）|666|
|day|String|日期|01, 02 ..  30|
|page|Int|行为发生的页面。每个数字分别对应“关注页”、”个人主页“、”发现页“、”同城页“或”其他页“中的一个|1|
|video_id|Int|video id（脱敏后）|333|
|author_id|Int|作者 id（脱敏后）|999|
|action_type|Int|用户行为类型。每个数字分别对应“播放“、”关注“、”点赞“、”转发“、”举报“和”减少此类作品“中的一个|1|

###数据下载
[百度云链接](https://pan.baidu.com/s/14QAHFmxISgXPss1pjsOtDw)
####密码：v428
###评估标准
初赛

设参赛选手提交的用户集合为 M，实际上未来 7 天内使用过快手的用户集合为 N ，且集合 N 是提供给选手的注册用户的子集。选手提交结果的 F1 Score 定义为：
![图片加载失败]()

最终使用 F1 Score 作为参赛选手得分。F1 Score 越大，代表结果越优，排名越靠前。
#####ps：第一次参加比赛，感谢大佬们在科赛讨论区的分享，学到很多。
  

