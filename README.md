# Recommend-system
Douban Movies Recommendation based on FM \ NeuralFM \ LFM \ SVD++ and Tensorflow. This is a project in Social Network Analysis@FDU.   

利用豆瓣电影数据进行社交网络分析和推荐系统搭建的课程Project。数据可视化demo可以进入4_Visualization文件夹查看。

## 环境要求
Tensorflow > r1.0  
Python 3.6  
PyEcharts  
Gephi  

## 数据
MySQL数据表：  
Movie 表 - 997个电影  
         - 电影ID、豆瓣电影评分、导演、主演、电影类型、出版国家、电影简介  
User 表 - 1000个用户  
        - 用户ID、用户评分、关注用户、评论文本信息  
数据预处理：  
User Bias - 用户历史平均评分  
Movie Bias - 电影历史平均评分  
Doc2vec - 电影剧情简介表示成353维向量  
TF-IDF - 电影类型进行IF-IDF编码，电影导演和主要进行One-Hot编码  

数据集中 1000 个用户共评论过 997 个电影，共有 40381 条有效评分。  

## 爬虫  
利用豆瓣API爬取电影相关信息  
Movie 信息 - 电影ID、豆瓣电影评分、上映年份、导演、主演、电影类型、上映国家、电影标签、打标签人数  
爬虫后数据新增了上映年份和电影标签信息，可以考虑加入时间维度变量和进行标签化的推荐。  

本文对用户评分数目较多的电影进行爬取，最终爬取电影 3004 部，得到 1000 个用户在 3004 部电影上有效评分 333623 条。  
评分数据从原来的 40381 条有效评分扩展到 333623 条有效评分，评分矩阵充实度从 4% 变为 11%。  

## 社交网络关系
研究1000个用户关注信息构成的社交网络。  
用户度分布和电影流行性分布近似服从于幂律分布；  
平均聚类系数为0.079较小，远低于Facebook等社交网络；  
平均路径长度为3.944，与Facebook的平均路径长度近似；  
Pagerank只有一个峰值较高；  
中间中心性分布图：社交网络中信息传播的关键人物，仅占全部用户的一小部分。  

## 社交网络可视化
数据可视化demo可以进入4_Visualization文件夹查看。  
以力导向算法对点进行社交网络布局，Fast Unfolding算法进行社区划分，并选取两个主要社区进行分析。  

## 训练集和测试集
每个用户的评分中选取 10% 作为测试集，90% 作为训练集。  
评价指标为：FCP(Top-N)、RMSE(回归)、MAE(回归)  

## Surprise经典推荐算法
协同过滤(BaselineOnly)、基础版协同过滤(KNNBasic)、均值协同过滤(KNNWithMeans)、SVD、SVD++

## FM模型
利用 libfm 进行算法求解，





