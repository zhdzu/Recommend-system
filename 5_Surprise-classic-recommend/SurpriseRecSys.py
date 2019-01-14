# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:38:16 2018

@author: CynthiaWang
"""
from surprise import Dataset, print_perf, Reader
from surprise.model_selection import cross_validate
import os

# 指定文件所在路径
file_path = os.path.expanduser('inputSurprise.csv')
# 告诉文本阅读器，文本的格式是怎么样的
reader = Reader(line_format='user item rating', sep=',')
# 加载数据
data = Dataset.load_from_file(file_path, reader=reader)


#data = Dataset.load_builtin('ml-100k')
### 使用NormalPredictor
from surprise import NormalPredictor
algo = NormalPredictor()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE','FCP'], cv=3)
print_perf(perf)

### 使用BaselineOnly
from surprise import BaselineOnly
algo = BaselineOnly()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE','FCP'], cv=3)
print_perf(perf)

### 使用基础版协同过滤
from surprise import KNNBasic, evaluate
algo = KNNBasic()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE','FCP'], cv=3)
print_perf(perf)

### 使用均值协同过滤
from surprise import KNNWithMeans, evaluate
algo = KNNWithMeans()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE','FCP'], cv=3)
print_perf(perf)

### 使用协同过滤baseline
from surprise import KNNBaseline, evaluate
algo = KNNBaseline()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE','FCP'], cv=3)
print_perf(perf)

### 使用SVD
from surprise import SVD, evaluate
algo = SVD()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE','FCP'], cv=3)
print_perf(perf)

### 使用SVD++
from surprise import SVDpp, evaluate
algo = SVDpp()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE','FCP'], cv=3)
print_perf(perf)

### 使用NMF
from surprise import NMF
algo = NMF()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE','FCP'], cv=3)
print_perf(perf)