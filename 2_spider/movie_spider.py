 # coding: utf-8

import json
# import uniout
import time
from datetime import datetime
import re
import requests

import pandas as pd
import numpy as np

movie = pd.read_csv('movie.csv')
user = pd.read_csv('user.csv')

user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:64.0) Gecko/20100101 Firefox/64.0'
user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:64.0) Gecko/20100101 Firefox/64.0'
user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0'
user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:60.0) Gecko/20100101 Firefox/60.0'
user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:63.0) Gecko/20100101 Firefox/63.0'

def movie_spider(url, number):
    hdr = {'User-Agent' : user_agent,
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Language':'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
           'Connection': 'keep-alive'} 
    req = requests.get(url, headers=hdr)
    res = req.text
    lists = json.loads(res)
    data = ()
    try:
        rate = lists[u'rating'][u'average']
        data += (rate,)
    except:
        data += ('None',)
    try:
        ratersnum = lists[u'rating'][u'numRaters']
        data += (ratersnum,)
    except:
        data += ('None',)
    try:
        name = lists[u'attrs'][u'title'][0]
        data += (name,)
    except:
        data += ('None',)
    try:
        year = lists[u'attrs'][u'year'][0]
        data += (year,)
    except:
        data += ('None',)
    try:
        director = lists[u'attrs'][u'director'][0]
        data += (director,)
    except:
        data += ('None',)
    try:
        actors = ''
        for i in range(4):
            a = lists[u'attrs'][u'cast'][i]
            actors = actors + ',' + a
        data += (actors,)
    except:
        data += ('None',)
    try:
        types = lists[u'attrs'][u'movie_type']
        t = ''
        for i in range(len(types)):
            a = lists[u'attrs'][u'movie_type'][i]
            t = t + ',' + a
        data += (t,)
    except:
        data += ('None',)
    try:
        countries = lists[u'attrs'][u'country']
        country = ''
        for i in range(len(countries)):
            a = lists[u'attrs'][u'country'][i]
            country = country + ',' + a
        data += (country,)
    except:
        data += ('None',)
    try:
        tag1 = str(lists[u'tags'][0]['count'])+","+lists[u'tags'][0]['name']
        data += (tag1,)
    except:
        data += ('None',)
    try:
        tag2 = str(lists[u'tags'][1]['count'])+','+lists[u'tags'][1]['name']
        data += (tag2,)
    except:
        data += ('None',)
    try:
        tag3 = str(lists[u'tags'][2]['count'])+','+lists[u'tags'][2]['name']
        data += (tag3,)
    except:
        data += ('None',)
    try:
        tag4 = str(lists[u'tags'][3]['count'])+','+lists[u'tags'][3]['name']
        data += (tag4,)
    except:
        data += ('None',)
    try:
        tag5 = str(lists[u'tags'][4]['count'])+','+lists[u'tags'][4]['name']
        data += (tag5,)
    except:
        data += ('None',)
    try:
        tag6 = str(lists[u'tags'][5]['count'])+','+lists[u'tags'][5]['name']
        data += (tag6,)
    except:
        data += ('None',)
    try:
        tag7 = str(lists[u'tags'][6]['count'])+','+lists[u'tags'][6]['name']
        data += (tag7,)
    except:
        data += ('None',)
    data += (number,)
#     movies = []
#     for l in lists:
#         title = l['title'].encode('utf-8')
#         movie = (l['rate'], title, l['url'], l['id'])
#         movies.append(movie)
#     print rate,'\n',year,'\n',ty,'\n',actors,'\n',countries,'\n',summary,'\n',directors,'\n'
#     print data,len(data)
    return data



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

rates_movie = []
rates_score = []
user_id = []
for i in range(1000):
    rate = re.findall('\d*\d',user.rates[i])
    rate = [int(x) for x in rate]
    rm,rs,ui = getscore(rate)
    rates_movie.extend(rm)
    rates_score.extend(rs)
    u = [(i+1)*item for item in ui]
    user_id.extend(u)


from pandas.core.frame import DataFrame
c={"user_id" : user_id,
   "movie_id" : rates_movie,
   "score" : rates_score}#将列表a，b转换成字典
all_rate=DataFrame(c)#将字典转换成为数据框
all_rate.to_csv('all_rate.csv')

movies = list(set(rates_movie))
from collections import Counter
d = Counter(rates_movie)
d = dict(d)
d = sorted(d.items(),key = lambda x:x[1], reverse=True)
movie_d = [d[i][0] for i in range(len(d))]

movie1 = movie_d[1700:1800]
all_movie1=[]
for n,m in enumerate(movie1):
    number = m
    id_ = str(m)
    url = 'https://api.douban.com/v2/movie/' +id_
    data = movie_spider(url, number)
    all_movie1.append(data)
    now = datetime.now()
    print ('Movie:%s is finished: %s out of %s. %s' % (id_,n+1,len(movies),now))
    if (n+1)%1 == 0:
        time.sleep(3)

df_movie = pd.DataFrame(np.zeros((len(all_movie1),16)),
                        columns=['score','num','name','year','directer','actors','type','country',
                                 'tag1','tag2','tag3','tag4','tag5','tag6','tag7','id'])

for i in range(len(all_movie1)):
    for j in range(16):
        df_movie.iloc[[i],[j]] = all_movie1[i][j]

df_movie['id'] = df_movie['id'].astype('int')
df_movie.to_csv('movie_d_1700_1800.csv')
    
    
    
 