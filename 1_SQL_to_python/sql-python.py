# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:57:07 2018

@author: Administrator
"""

import pandas as pd
from sqlalchemy import create_engine
import pymysql
engine = create_engine('mysql+pymysql://root:zhdzu2357@localhost:3306/douban',
                       convert_unicode=True, encoding='utf-8', connect_args={"charset":"utf8"})
df_movie = pd.read_sql('movie', engine)
df_user = pd.read_sql('user', engine)
df_movie.to_csv('movie.csv')
df_user.to_csv('user.csv')