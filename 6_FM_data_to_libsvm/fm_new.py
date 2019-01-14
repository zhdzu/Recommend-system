# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:44:34 2018

@author: Administrator
"""

import pandas as pd
import numpy as np

data = pd.read_csv('usermovierate.csv')
df = pd.read_csv('spider.csv')

del data['Unnamed: 0']
data = data.astype('int')

# 将电影 ID 转化为 1,2,...
l1 = sorted(list(set(data.movie_id)))
l2 = [i+1 for i in range(3004)]
dic = dict(zip(l1,l2))

data['movie_id_new'] = data['movie_id'].map(dic)



# 转化为libsvm格式
# 0-999列为用户ID，1000-4003列为电影ID
libsvm1 = []
def getlibsvm(d):
    for i in range(len(d)):
        libsvm1.append(d.rate[i])
        libsvm1.append(str(d.user_id[i]-1)+':'+str(1))
        libsvm1.append(str(d.movie_id_new[i]+999)+':'+str(1))
    return libsvm1
libsvm1 = getlibsvm(data)

libsvm = pd.DataFrame(np.array(libsvm1).reshape(len(data),3))
libsvm.columns = ['rate','user','movie']
libsvm.to_csv('fm1_libsvm.csv',header=None,index=None)


# 分成训练集和测试集
from sklearn.model_selection import train_test_split
train, test = train_test_split(libsvm, test_size=0.1, random_state=0)

# 保存成txt格式
fw = open("fm1_train.txt", 'w') 
for i in range(len(train)):
        fw.write(train.iloc[i,0]+' '+train.iloc[i,1]+' '+train.iloc[i,2])
        fw.write('\n')
fw.close()

fw = open("fm1_test.txt", 'w') 
for i in range(len(test)):
        fw.write(test.iloc[i,0]+' '+test.iloc[i,1]+' '+test.iloc[i,2])
        fw.write('\n')
fw.close()




# 构建特征二的 FM 模型
movie_bias = data[['movie_id_new','rate']].groupby(['movie_id_new']).agg('mean').reset_index()
data = pd.merge(data, movie_bias, how='left', on='movie_id_new')
data.columns = ['user_id', 'movie_id', 'rate', 'movie_id_new', 'movie_bias']

# 转化为libsvm格式
# 0-999列为用户ID，1000-4003列为电影ID，4004列为电影平均分
libsvm2 = []
def getlibsvm(d):
    for i in range(len(d)):
        libsvm2.append(d.rate[i])
        libsvm2.append(str(d.user_id[i]-1)+':'+str(1))
        libsvm2.append(str(d.movie_id_new[i]+999)+':'+str(1))
        libsvm2.append(str(4004)+':'+str(d.movie_bias[i]))
    return libsvm2
libsvm2 = getlibsvm(data)

libsvm = pd.DataFrame(np.array(libsvm2).reshape(len(data),4))
libsvm.columns = ['rate','user','movie','movie_bias']
libsvm.to_csv('fm2_libsvm.csv',header=None,index=None)


# 分成训练集和测试集
from sklearn.model_selection import train_test_split
train, test = train_test_split(libsvm, test_size=0.1, random_state=0)

# 保存成txt格式
fw = open("fm2_train.txt", 'w') 
for i in range(len(train)):
        fw.write(train.iloc[i,0]+' '+train.iloc[i,1]+' '+train.iloc[i,2]+' '+train.iloc[i,3])
        fw.write('\n')
fw.close()

fw = open("fm2_test.txt", 'w') 
for i in range(len(test)):
        fw.write(test.iloc[i,0]+' '+test.iloc[i,1]+' '+test.iloc[i,2]+' '+test.iloc[i,3])
        fw.write('\n')
fw.close()




# 构建特征三的 FM 模型
user_bias = data[['user_id','rate']].groupby(['user_id']).agg('mean').reset_index()
data = pd.merge(data, user_bias, how='left', on='user_id')
data.columns = ['user_id', 'movie_id', 'rate', 'movie_id_new', 'movie_bias', 'user_bias']

# 转化为libsvm格式
# 0-999列为用户ID，1000-4003列为电影ID，4004列为电影平均分，4005列为用户平均分
libsvm3 = []
def getlibsvm(d):
    for i in range(len(d)):
        libsvm3.append(d.rate[i])
        libsvm3.append(str(d.user_id[i]-1)+':'+str(1))
        libsvm3.append(str(d.movie_id_new[i]+999)+':'+str(1))
        libsvm3.append(str(4004)+':'+str(d.movie_bias[i]))
        libsvm3.append(str(4005)+':'+str(d.user_bias[i]))
    return libsvm3
libsvm3 = getlibsvm(data)

libsvm = pd.DataFrame(np.array(libsvm3).reshape(len(data),5))
libsvm.columns = ['rate','user','movie','movie_bias','user_bias']
libsvm.to_csv('fm3_libsvm.csv',header=None,index=None)


# 分成训练集和测试集
from sklearn.model_selection import train_test_split
train, test = train_test_split(libsvm, test_size=0.1, random_state=0)

# 保存成txt格式
fw = open("fm3_train.txt", 'w') 
for i in range(len(train)):
        fw.write(train.iloc[i,0]+' '+train.iloc[i,1]+' '+train.iloc[i,2]+
                 ' '+train.iloc[i,3]+' '+train.iloc[i,4])
        fw.write('\n')
fw.close()

fw = open("fm3_test.txt", 'w') 
for i in range(len(test)):
        fw.write(test.iloc[i,0]+' '+test.iloc[i,1]+' '+test.iloc[i,2]+
                 ' '+test.iloc[i,3]+' '+test.iloc[i,4])
        fw.write('\n')
fw.close()