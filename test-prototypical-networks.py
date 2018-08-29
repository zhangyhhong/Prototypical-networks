#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:01:29 2018

@author: luoxi
用于测试不同方法下AUC随n_proto的变化
方法：dist min,dist mean,kmeans
"""
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import keras
from sklearn.metrics import roc_curve
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import roc_auc_score
import time

from keras.models import Sequential,Model
from keras.layers import Dropout, Activation,Flatten
from keras.layers import Dense,Lambda,Input,LSTM
from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras import backend as K
from keras.models import load_model

nb_test = 5000   #构造测试样本的个数
same_rate = 0.9  #样本中的相同配对所占的比例

nb_proto_lis = list(range(10,21))  #测试时用到的record数据条数
AUC_method = 'kmeans'  #k-means,max，min，mean

n_clusters_lis = [2,3]
choose_option_lis=[1,2]  #choose_option：0不选一个，1认真选择一个，2随机选择一个

orig_model_weights = 'base_model_80.hdf5'
#fs_model_weights = 'wsqet_60_5_3_8000_835.hdf5'

def get_label(yc):
     yc_list = list(set(yc))
     lables = dict(zip(yc_list,range(len(yc_list))))
     ys = [ lables[i] for i in  yc]
     return ys


def get_label2index(data):  
    '''获取label和index对应的字典'''    
    label_index_dict = {}
    all_labels = set(data.label)       
    for la in all_labels:
       label_index_dict[la] = list(data.index[data.label==la])
       
    nb_classes = len(all_labels)
    print('the number of labels:',nb_classes)
    return label_index_dict      

def get_model(emb_dim):
    '''原始模型base-80'''
    inp_l = Input(shape=(im_width, im_height),dtype='float32')
    inp_r = Input(shape=(im_width, im_height),dtype='float32')
    x_l = Bidirectional(LSTM(256,return_sequences=True))(inp_l)
    #x_l = Flatten()(x_l)
    x_r = Bidirectional(LSTM(256,return_sequences=True))(inp_r)
    avg_pool_l = GlobalAveragePooling1D()(x_l)
    max_pool_l = GlobalMaxPooling1D()(x_l)

    avg_pool_r = GlobalAveragePooling1D()(x_r)
    max_pool_r = GlobalMaxPooling1D()(x_r)
    #x_r = Flatten()(x_r)
    conc = concatenate([avg_pool_l, max_pool_l,avg_pool_r,max_pool_r])
    f = Dense(emb_dim, name='features')(conc)
    normalize_feature = Lambda(lambda  x: K.l2_normalize(x,axis=1))(f)

    model = Model(input=[inp_l, inp_r], output=normalize_feature)
    return model

def generate_test(data,label2index,sample_num=0,same_rate = 0.9,pro_num=3):
     '''data为最后一行是label的数据，返回配对的测试集数据index和是否不同的标签，pro_num表示原型样本个数'''
     labels = list(label2index.keys())
     nb_labels = len(labels)
     nb_same = int(same_rate*sample_num)
     same_per_label = nb_same//nb_labels  #整除
     nb_surplus_same_label = nb_same % nb_labels  #取余
     surplus_same_label = np.random.choice(labels,nb_surplus_same_label)
     ind_1 = [];ind_2 = []
     repeat_sample = []
     if same_per_label>0:
         for la in labels:  #对每个类
             for i in range(same_per_label):  #每个类要抽的次数
                 try:
                     ind = list(np.random.choice(list(label2index[la]),pro_num+1,replace=False))
                 except:
                     ind = list(np.random.choice(list(label2index[la]),1))
                     remain_lis = list(label2index[la])
                     remain_lis.remove(ind[0])
                     ind.extend(list(np.random.choice(remain_lis,pro_num,replace=True)))
                     repeat_sample.append(la)  #记录重复抽样类别
                     #print('类别%s重复抽样' % la)
                 ind_1.extend(ind[1:])
                 ind_2.append(ind[0])
     if nb_surplus_same_label>0:
         for la in surplus_same_label:
             try:
                 ind = list(np.random.choice(list(label2index[la]),pro_num+1,replace=False))
             except:
                 ind = list(np.random.choice(list(label2index[la]),1))
                 remain_lis = list(label2index[la])
                 remain_lis.remove(ind[0])
                 ind.extend(list(np.random.choice(remain_lis,pro_num,replace=True)))
                 repeat_sample.append(la)  #记录重复抽样类别
                     #print('类别%s重复抽样' % la)
             ind_1.extend(ind[1:])
             ind_2.append(ind[0])                 

     nb_diff = sample_num - nb_same  #不一样的类别
     proto_diff_label = np.random.choice(labels,nb_diff)
     for la in proto_diff_label:
         try:
             ind_diff_1 = list(np.random.choice(list(label2index[la]),pro_num,replace=False))
         except:
             ind_diff_1 = list(np.random.choice(list(label2index[la]),pro_num,replace=True))
             repeat_sample.append(la)
             #print('类别%s重复抽样' % la)
         ind_1.extend(ind_diff_1)
     print('每个原型样本数为%s' % pro_num)
     print('测试集共抽样%s对样本，其中重复抽样%s次'% (sample_num,len(repeat_sample)))
     ind_diff_2 = list(np.random.choice(data.index,nb_diff,replace=False)) #直接抽出需要的样本数
     ind_2.extend(ind_diff_2)
 
     label1 = list(data.iloc[ind_1[::pro_num],-1])  #每pro_num取一行
     label2 = list(data.iloc[ind_2,-1])
     diff_label = [0 if label1[i]==label2[i] else 1 for i in range(len(label1))] #不相同则为1
     
     return ind_1,ind_2,diff_label



def get_km_func(n_clusters,choose_option):
    '''choose_option = 0  #0不选一个，1认真选择一个，2随机选择一个'''
    def get_kmeans_center(X):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        if choose_option==1:
            labels = kmeans.labels_
            cho = Counter(labels).most_common(1)[0][0]  #选择聚类最多的中心
            return kmeans.cluster_centers_[cho]
        elif choose_option==2:
            cho = np.random.choice(range(n_clusters))  #随机选择一条
            return kmeans.cluster_centers_[cho]
        else:
            return kmeans.cluster_centers_
    return get_kmeans_center



def get_auc2(model,emb1,emb2,method='max',csv_name=False,n_clusters=2,choose_option=0):
    '''对结果取k-means,max，min，mean'''
    if method=='kmeans':
        nrow,ncol = emb1.shape
        emb11 = emb1.reshape(int(nrow/nb_proto),-1,ncol)
        get_kmeans_center1 = get_km_func(n_clusters,choose_option)
        emb11 = np.array(list(map(get_kmeans_center1,emb11))).reshape(-1,ncol)   #每条数据都是n_clusters个中心作为代表     
        if choose_option:  #为1或2
            euc_dist = np.sqrt(np.sum((emb11 - emb2)**2,axis=1)) #只有一个中心
        else:  #有多个中心
            emb2 = np.tile(emb2,n_clusters).reshape(emb2.shape[0]*n_clusters,emb2.shape[1])                  
            euc_dist = np.sqrt(np.sum((emb11 - emb2)**2,axis=1))
            n_row = euc_dist.shape[0]
            euc_dist = euc_dist.reshape(int(n_row/n_clusters),-1)            
            euc_dist = np.min(euc_dist,axis=1)                
    else:
        emb22 = np.tile(emb2,nb_proto).reshape(emb2.shape[0]*nb_proto,emb2.shape[1])    
        euc_dist = np.sqrt(np.sum((emb1 - emb22)**2,axis=1))
        n_row = euc_dist.shape[0]
        euc_dist = euc_dist.reshape(int(n_row/nb_proto),-1)
        if method=='max':
            euc_dist = np.max(euc_dist,axis=1)
        elif method=='min':
            euc_dist = np.min(euc_dist,axis=1)
        elif method=='mean':
            euc_dist = np.mean(euc_dist,axis=1)        
    score = roc_auc_score(diff_label,euc_dist)    
    final_test = pd.DataFrame(np.array([euc_dist,diff_label]).T,columns=['euc_dist_'+str(method),'diff_label'])  
    if csv_name:
        final_test.to_csv(csv_name)          
    return score 
    

if __name__ == '__main__':
    im_width, im_height = 200,4  #左/右输入的宽、高
    emb_dim = 128  
    
    #原始模型
    model_orig = get_model(emb_dim)
    model_orig.load_weights(orig_model_weights,by_name=True)  
    
#    #few shot model导入权重
#    left_input = Input(shape=(im_width, im_height), dtype='float32', name='left_input')
#    right_input = Input(shape=(im_width, im_height), dtype='float32', name='right_input')
#    submodel = get_model(emb_dim)
#    emb = submodel([left_input,right_input])
#    model_fs = Model([left_input, right_input], emb)
#    model_fs.load_weights(fs_model_weights)
    
    #生成数据对用于测试        
    test = pd.read_csv('fewshot_test.csv',index_col=0) 
    test.reset_index(inplace=True,drop=True)
    test['label'] = get_label(test.userid.tolist())

    test_x = test.drop(['index','userid','apdid','phonetype','media_id','rowkey','label'],axis=1)
    #print( test_x.head())     
    test_label2index = get_label2index(test)
    
    for nb_proto in nb_proto_lis:
        ind_1,ind_2,diff_label = generate_test(test,test_label2index,nb_test,same_rate,pro_num=nb_proto)            
                
        test1 = test_x.iloc[ind_1].values.reshape(len(ind_1),im_width, im_height*2).astype('float32')    
        test1_l = test1[:,:,:im_height];test1_r = test1[:,:,im_height:]
        
        test2 = test_x.iloc[ind_2].values.reshape(len(ind_2),im_width, im_height*2).astype('float32')    
        test2_l = test2[:,:,:im_height];test2_r = test2[:,:,im_height:]     
        
        #保存建模embedding之后的数据
        emb1 = model_orig.predict([test1_l,test1_r])
        emb2 = model_orig.predict([test2_l,test2_r])
        
        pick_name = 'data_test_'+str(nb_proto)+'.pickle'
        with open(pick_name,'wb') as f:
            pickle.dump(emb1,f)
            pickle.dump(emb2,f)
            pickle.dump(diff_label,f)
        
    #导入数据并求AUC
    for n_clusters in n_clusters_lis:
        for choose_option in choose_option_lis:
            for nb_proto in nb_proto_lis:         
                t1 = time.time()   
                
                pick_name = 'data_test_'+str(nb_proto)+'.pickle'              
                with open(pick_name,'rb') as f:
                    emb1 = pickle.load(f)
                    emb2 = pickle.load(f)
                    diff_label = pickle.load(f)
                
                csv_name = 'dist_orig_'+str(AUC_method)+str(n_clusters)+'_cho'+str(choose_option)+'_'+str(nb_proto)+'_new.csv'
                score_orig = get_auc2(model_orig,emb1,emb2,AUC_method,csv_name,n_clusters,choose_option)
                print('原始模型在测试集上的AUC：',score_orig)
    
    
                t2 = time.time();cost = int((t2-t1)/60)+1
                print('运行耗费时间:%s min'% cost)
                
                result_name = 'result_'+str(AUC_method)+'.txt'
                with open(result_name,'a') as f:
                    f.write(str(AUC_method)+','+str(score_orig)+','+str(nb_proto)+','
                            +str(n_clusters)+','+str(choose_option)+','+str(nb_test)+','+str(cost)+'\n')


    
        
        
        
 
