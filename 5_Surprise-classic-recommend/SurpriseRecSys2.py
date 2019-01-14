# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:46:36 2018

@author: Administrator
"""
'''
## movielens推荐实例
#-*- coding:utf-8 -*-
from __future__ import (absolute_import, division, print_function,unicode_literals)
import os
import io
from surprise import KNNBaseline
from surprise import Dataset

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a， %d %b %Y %H:%M:%S')


# 训练推荐模型 步骤:1
def getSimModle():
    # 默认载入movielens数据集
    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()
    #使用pearson_baseline方式计算相似度  False以item为基准计算相似度 本例为电影之间的相似度
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    ##使用KNNBaseline算法
    algo = KNNBaseline(sim_options=sim_options)
    #训练模型
    algo.fit(trainset)
    return algo


# 获取id到name的互相映射  步骤:2
def read_item_names():
    """
    获取电影名到电影id 和 电影id到电影名的映射
    """
    file_name = (os.path.expanduser('~') +
                 '/.surprise_data/ml-100k/ml-100k/u.item')
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]
    return rid_to_name, name_to_rid


# 基于之前训练的模型 进行相关电影的推荐  步骤：3
def showSimilarMovies(algo, rid_to_name, name_to_rid):
    # 获得电影Toy Story (1995)的raw_id
    toy_story_raw_id = name_to_rid['Toy Story (1995)']
    logging.debug('raw_id=' + toy_story_raw_id)
    #把电影的raw_id转换为模型的内部id
    toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)
    logging.debug('inner_id=' + str(toy_story_inner_id))
    #通过模型获取推荐电影 这里设置的是10部
    toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, 10)
    logging.debug('neighbors_ids=' + str(toy_story_neighbors))
    #模型内部id转换为实际电影id
    neighbors_raw_ids = [algo.trainset.to_raw_iid(inner_id) for inner_id in toy_story_neighbors]
    #通过电影id列表 或得电影推荐列表
    neighbors_movies = [rid_to_name[raw_id] for raw_id in neighbors_raw_ids]
    print('The 10 nearest neighbors of Toy Story are:')
    for movie in neighbors_movies:
        print(movie)


if __name__ == '__main__':
    # 获取id到name的互相映射
    rid_to_name, name_to_rid = read_item_names()

    # 训练推荐模型
    algo = getSimModle()

    ##显示相关电影
    showSimilarMovies(algo, rid_to_name, name_to_rid)
'''

from surprise import Dataset, print_perf, Reader
from surprise.model_selection import cross_validate
import os

# 指定文件所在路径
file_path = os.path.expanduser('inputSurprise.csv')
# 告诉文本阅读器，文本的格式是怎么样的
reader = Reader(line_format='user item rating', sep=',')
# 加载数据
data = Dataset.load_from_file(file_path, reader=reader)
### 使用SVD++
from surprise import SVDpp, evaluate
algo = SVDpp()
perf = cross_validate(algo, data, measures=['RMSE', 'MAE','FCP'], cv=3)
print_perf(perf)