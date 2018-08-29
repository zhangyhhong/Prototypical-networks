#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 15:26:05 2018

@author: alibaba
"""

import pandas as pd
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
test = pd.read_csv('fewshot_test.csv')
test.head()

coun_label = Counter(test.label)
def get_plotxy(coun_label):
    label_num_count = Counter(coun_label.values())
    sort_train = sorted(label_num_count.items(),key = lambda items:items[0])
    keys,values = zip(*sort_train)
    return keys,values
test_k,test_v = get_plotxy(coun_label)

#柱状图2
plt.figure(figsize=(9,5))
plt.bar(test_k,test_v,color = 'lightgreen', label='test')
for a,b in zip(test_k,test_v):  
 plt.text(a+0.4, b+10, '%.0f' % b, ha='center', va= 'bottom',fontsize=7) 
plt.title('test label frequence')
plt.xlabel('sample size per label')
plt.ylabel('frequence')
plt.legend(loc = "best")
plt.show()

result_kmeans = pd.read_csv('result_kmeans.txt')
result_kmeans.tail()

result_kmeans_1 = result_kmeans[(result_kmeans.n_clusters==1)&(result_kmeans.choose_option==0)]
result_kmeans_1.sort_values('n_proto')

result_kmeans_2_min = result_kmeans[(result_kmeans.n_clusters==2) & (result_kmeans.choose_option==0)]
result_kmeans_2_min.sort_values('n_proto')

result_kmeans_2_cho = result_kmeans[(result_kmeans.n_clusters==2) & (result_kmeans.choose_option==1)]
result_kmeans_2_cho.sort_values('n_proto')

result_kmeans_2_rand = result_kmeans[(result_kmeans.n_clusters==2) & (result_kmeans.choose_option==2)]
result_kmeans_2_rand.sort_values('n_proto')

result_kmeans_3_min = result_kmeans[(result_kmeans.n_clusters==3) & (result_kmeans.choose_option==0)]
result_kmeans_3_min.sort_values('n_proto')

result_kmeans_3_cho = result_kmeans[(result_kmeans.n_clusters==3) & (result_kmeans.choose_option==1)]
result_kmeans_3_cho.sort_values('n_proto')

result_kmeans_3_rand = result_kmeans[(result_kmeans.n_clusters==3) & (result_kmeans.choose_option==2)]
result_kmeans_3_rand.sort_values('n_proto')

result_mean = pd.read_csv('result_mean.txt',names = ['method','AUC','n_proto','n_clusters','choose_option','nb_test','cost'])

result_min = pd.read_csv('result_min.txt',names = ['method','AUC','n_proto','n_clusters','choose_option','nb_test','cost'])


'''
dist方法为对每个embedding后的向量和current向量对比，然后求{mean，max，min}作为最终距离
k-means-N表示对embedding后的向量聚成N类，然后求和current向量距离的最小值
k-means-N-cho表示取聚类最多样本的类的中心，求和current向量距离作为最终距离
'''
#model_name,AUC,n_proto,n_test,cost_time
plt.figure(figsize=(12,6))
plt.plot(result_mean['n_proto'],result_mean['AUC'],'c-*',label='dist mean')
plt.plot(result_min['n_proto'],result_min['AUC'],'r-*',label='dist min')

plt.plot(result_kmeans_1['n_proto'],result_kmeans_1['AUC'],'b-*',label='kmeans1')

plt.plot(result_kmeans_2_min['n_proto'],result_kmeans_2_min['AUC'],'m-*',label='kmeans2 min')
plt.plot(result_kmeans_2_cho['n_proto'],result_kmeans_2_cho['AUC'],'m--*',label='kmeans2 cho')
plt.plot(result_kmeans_2_rand['n_proto'],result_kmeans_2_rand['AUC'],'m-.*',label='kmeans2 rand')

plt.plot(result_kmeans_3_min['n_proto'],result_kmeans_3_min['AUC'],'g-*',label='kmeans3 min')
plt.plot(result_kmeans_3_cho['n_proto'],result_kmeans_3_cho['AUC'],'g--*',label='kmeans3 cho')
plt.plot(result_kmeans_3_rand['n_proto'],result_kmeans_3_rand['AUC'],'g-.*',label='kmeans3 rand')

plt.title('AUC vs n_proto')
plt.xlabel('n_proto')
plt.ylabel('AUC')
plt.axis([0,20.5,0.75,1])
plt.legend(loc = "best",ncol=2)
plt.show()
print('kmeans2 min两类中心距离取最小')
print('kmeans2 cho两类中选择样本多的中心')
print('kmeans2 rand两类中心随机选择一个中心')


def get_max(result_mean):
    return {str(result_mean['method'][0]): max(result_mean['AUC'])}

def get_max2(result_kmeans):
    return {str(result_kmeans.method.iloc[0])+str(result_kmeans.n_clusters.iloc[0])+'_'+str(result_kmeans.choose_option.iloc[0]): max(result_kmeans['AUC'])}

results_max = []
results_max.append(get_max(result_mean))
results_max.append(get_max(result_min))

results_max.append(get_max2(result_kmeans_1))
results_max.append(get_max2(result_kmeans_2_min))
results_max.append(get_max2(result_kmeans_2_cho))
results_max.append(get_max2(result_kmeans_2_rand))

results_max.append(get_max2(result_kmeans_3_min))
results_max.append(get_max2(result_kmeans_3_cho))
results_max.append(get_max2(result_kmeans_3_rand))

method = list(map(lambda x:list(x.keys())[0],results_max))
max_auc = list(map(lambda x:list(x.values())[0],results_max))

max_AUC_summary = pd.DataFrame({'method':method,'max_AUC':max_auc})
print(max_AUC_summary)
print('kmeans2_0两类中心距离取最小')
print('kmeans2_1两类中选择样本多的中心')
print('kmeans2_2两类中心随机选择一个中心')