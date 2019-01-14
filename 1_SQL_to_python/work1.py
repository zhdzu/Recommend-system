# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 22:59:19 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
import re
from pandas.core.frame import DataFrame
movie = pd.read_csv('movie.csv')
user = pd.read_csv('user.csv')

# print(user.dtypes) 查看数据框数据类型
# rates是str数据类型
rate = re.findall('\d*\d',user.rates[0])
rate = [int(x) for x in rate]

def getscore(rate):
    rates_movie = list()
    rates_score = list()
    user_id = [1 for x in range(int(len(rate)/2))]
    for i in range(len(rate)):
        if i%2==0:
            rates_movie.append(rate[i])
        elif i%2==1:
            rates_score.append(rate[i])
        else:
            print('error!')
    return rates_movie,rates_score,user_id


# np.sum(score>0)可以查看大于0的个数
def getuserscore(rate):
    score = np.zeros((1,1000))
    rates_movie,rates_score,user_id = getscore(rate)
    rates_movie_inset,rates_score_inset = zip(*((i,j) for i,j in zip(rates_movie,rates_score) 
                                                 if i in set(movie.id)))
    rates_movie_inset = list(rates_movie_inset)
    rates_score_inset = list(rates_score_inset)
    for i in range(len(rates_movie_inset)):
        for j in range(1000):
            if rates_movie_inset[i] == movie.id[j]:
                score[0,j] = rates_score_inset[i]
    return score


def getfullscore():
    score = np.zeros((1000,1000))
    for i in range(1000):
        rate = re.findall('\d*\d',user.rates[i])
        rate = [int(x) for x in rate]
        score_user = getuserscore(rate)
        score[i,:] = score_user
    return score

score = getfullscore()


def scoreshape(score):
    score_user = list()
    score_movie = list()
    for i in range(1000):
        score_user.append(np.sum(score[i,:]>0))
    for j in range(1000):
        score_movie.append(np.sum(score[:,j]>0))
    return score_user,score_movie

score_user,score_movie = scoreshape(score)

np.savetxt('score.csv', score, delimiter = ',')
np.savetxt('score_user.csv', score_user, delimiter = ',')
np.savetxt('score_movie.csv', score_movie, delimiter = ',')