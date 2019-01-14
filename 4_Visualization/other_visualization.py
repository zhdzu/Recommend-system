# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 19:20:49 2018

@author: 张媛媛 18210980079
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
df = pd.read_csv('D:\\social networks\\gephi_output3.csv')
user = pd.read_csv('D:\\social networks\\user.csv')
f = open("D:\\social networks\\movie.csv")
movie = pd.read_csv(f)  
# In[1]:
# 计算度分布的x,y
def degree(user_degree):
    x=[]
    y=[]
    for i in range(max(user_degree)+1):
        if sum(user_degree==i)==0:
            continue
        x.append(i)
        y.append(sum(user_degree==i))
    return x,y
# In[2]:
# 度分布图
user_degree = df["degree"]
x, y = degree(user_degree)
fig = plt.figure()
plt.scatter(x=x,y=y)
plt.title('degree distribution')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Degree')
plt.ylabel('number')
plt.show()
# In[3]:
# 计算电影流行度的x,y
def popular(user,movie):
    movie_id = []
    movie_id_ = []
    movie_id_inset = []
    for i in range(len(user)):
        str = user.rates[i]
        movie_id = re.findall(r"{u'(.+?)[': ]",str)
        movie_id_s = re.findall(r", u'(.+?)[': ]",str)
        movie_id.extend(movie_id_s)
        movie_id_.extend(movie_id)
    for j in range(len(movie_id_)):
        if int(movie_id_[j]) in set(movie.id):
            movie_id_inset.append(movie_id_[j])  
    x = []
    for m_id in set(movie_id_inset):
        x.append(movie_id_inset.count(m_id))
    x = sorted(x,reverse=True)
    y = []
    for count in set(x):
        y.append(x.count(count))
    x = sorted(list(set(x)))
    return x,y
# In[4]:
# 电影流行度
x,y = popular(user,movie)
fig = plt.figure()
plt.scatter(x=x,y=y)
plt.title('Movie Popularity ')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Popularity')
plt.ylabel('number')
plt.show()

# In[5]:
#pagerank distribution
y = df.pageranks
plt.hist(y)
plt.title('PageRank Distribution')
plt.yscale('log')
plt.xlabel('PageRank')
plt.ylabel('numbel')
plt.show()

# In[6]:
#Betweenness Centralit distribution
y = df.betweenesscentrality
plt.hist(y)
plt.title('Betweeness Centrality Distribution')
plt.yscale('log')
plt.xlabel('betweeness centrality')
plt.ylabel('numbel')
plt.show()










