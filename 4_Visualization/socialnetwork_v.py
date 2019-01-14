# -*- coding: utf-8 -*-
""" 这个文件用于生成可视化的网页，使用Gephi输出的csv文件"""
#pyecharts处理不了太复杂的关系图，可以借用： networkx 库
import pandas as pd
from pyecharts import Graph
import math
# In[1]:
df = pd.read_csv('D:\\social networks\\gephi_output3.csv')
nodes = []
links = []
cat = []
for indexs in df.index:
    cat.append(str(df.loc[indexs]['modularity_class']))
    nodes.append({"name": df.loc[indexs]['Id'],
                  "symbolSize": 5 + 10 * math.sqrt(df.loc[indexs]['degree']),
                  "category": str(df.loc[indexs]['modularity_class'])
                  })
cat = list(set(cat))
cat2 = []
for i in cat:
    cat2.append({"name": i})

# In[2]:
df2 = pd.read_csv('D:\\social networks\\gephi_input.csv')    
import pickle
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
#定义有向图
g = nx.DiGraph()
edges = []
for i in range(len(df2)):
    edges.append([
        df2.loc[i, "Source"],
        df2.loc[i, "Target"]
    ])
# In[3]:
#仅仅是社交网络关系    
for i in range(len(df)):
    g.add_node(df.Id[i])
# add edges
for i in range(len(df2)):
    g.add_edge(edges[i][0], edges[i][1])

g_data = json_graph.node_link_data(g)

graph = Graph("豆瓣关注关系图", width=1600, height=900)
graph.add("", nodes=g_data['nodes'], links=g_data['links'],categories=cat2,is_legend_show=False,line_curve=0.2,label_pos="right")
graph.render("relationship.html") 
  
# In[4]:
# 分割社区后
links = []
for i in range(len(df2)):
    links.append({"source": edges[i][0], "target": edges[i][1]})
graph = Graph("豆瓣关注关系图", width=1600, height=900)
graph.add("", nodes, links, cat2, label_pos="right", graph_layout='force',
          graph_repulsion=50000, is_legend_show=False,
          line_curve=0.2, label_text_color=None)
graph.render("relationship_diagram.html")    

# In[5]:
# 第一个主要社区
df = pd.read_csv('D:\\social networks\\gephi_output5.csv')
def fF(df):
    nodes = []
    links = []
    cat = []
    for indexs in df.index:
        cat.append(str(df.loc[indexs]['modularity_class']))
        nodes.append({"name": df.loc[indexs]['Id'],
                      "symbolSize": 5 + 10 * math.sqrt(df.loc[indexs]['degree']),
                      })
    cat = list(set(cat))
    cat2 = []
    for i in cat:
        cat2.append({"name": i})
    return nodes,cat2
nodes,cat2 = fF(df)
links = []
for i in range(len(edges)):
    if edges[i][0] in set(df.Id) and edges[i][1] in set(df.Id):
        links.append({"source": edges[i][0], "target": edges[i][1]})
graph = Graph("豆瓣关注关系--主要社区", width=1600, height=900)
graph.add("", nodes, links, cat2, label_pos="right", graph_layout='force',
          graph_repulsion=500, is_legend_show=False,
          line_curve=0.2, label_text_color=None)
graph.render("relationshipP1.html")    

# In[6]:
# 第二个主要社区
df = pd.read_csv('D:\\social networks\\gephi_output6.csv')
nodes,cat2 = fF(df)
links = []
for i in range(len(edges)):
    if edges[i][0] in set(df.Id) and edges[i][1] in set(df.Id):
        links.append({"source": edges[i][0], "target": edges[i][1]})
graph = Graph("豆瓣关注关系--主要社区", width=1600, height=900)
graph.add("", nodes, links, cat2, label_pos="right", graph_layout='force',
          graph_repulsion=500, is_legend_show=False,
          line_curve=0.2, label_text_color=None)
graph.render("relationshipP2.html")  