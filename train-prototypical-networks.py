#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:42:35 2018

@author: luoxi
训练原型网络
"""

# coding: utf-8
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential,Model
from keras.layers import Dropout, Activation,Flatten
from keras.layers import Dense,Lambda,Input,LSTM
from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras import backend as K
from keras.models import load_model
from collections import Counter
from sklearn.metrics import roc_auc_score
import os
import time

n_way = 20  #5,20,60
n_support,n_query = 5,3 
epoch = 600     #训练模型时抽样label的次数
epoch_start = 0
n_episodes = 100       #训练模型时每个epoch抽样次数
#fs_model_initi_weights = 'wsqet_60_5_3_8000_835.hdf5'  #模型接着训练
arr_name_load = 'train_wsq_20_5_3.npy'   #用于前500个epoch的训练数据
fs_model_initi_weights = 'base_model_80.hdf5'  #训练模型初始化

def get_label(data):
    '''根据userid得到label'''
    ids = list(set(data.userid))
    lable2id = dict(zip(ids,range(len(ids))))
    label = [lable2id[i] for i in data.userid]
    return label

def get_label2index(data):  
    '''获取label和index对应的字典'''    
    label_index_dict = {}
    all_labels = set(data.label)       
    for la in all_labels:
       label_index_dict[la] = list(data.index[data.label==la])
       
    nb_classes = len(all_labels)
    print('the number of labels:',nb_classes)
    return label_index_dict      


def get_train_batch(label2index,n_way,n_support,n_query): 
    '''返回一个batch训练集数据'''
    labels = list(label2index.keys())      
    epi_classes = np.random.choice(labels,n_way) #选取n_way个类别         
    repeat_sample = [];idx = []
    for cls in epi_classes:
        index_s = list(np.random.choice(label2index[cls],n_support,replace=False))
        try:
            index_q = list(np.random.choice(list(set(label2index[cls])-set(index_s)),n_query,replace=False))
        except:
            index_q = list(np.random.choice(list(set(label2index[cls])-set(index_s)),n_query,replace=True))
            #print('类别%s重复采样' % cls) 
            repeat_sample.append(cls)
        index_s.extend(index_q) 
        idx.extend(index_s)         
    if len(repeat_sample)>0: print('训练样本的query集重复抽样类别数：%s' % len(repeat_sample))
    return idx

def get_y():
    '''返回训练集的y'''
    y_s = [];y_q = []
    for i in range(n_way):
        y_s.extend([i]*n_support)
        y_q.extend([i]*n_query)
    y_s.extend(y_q)
    y_s = np.array(y_s)
    return y_s


def get_model(emb_dim):
    '''原始模型的嵌入层模型'''
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
    #normalize_feature = Lambda(lambda  x: K.l2_normalize(x))(f)

    model = Model(input=[inp_l, inp_r], output=normalize_feature)
    return model
    

def euc_distance(a,b):
    row_norms_A = tf.reduce_sum(tf.square(a), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.reduce_sum(tf.square(b), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.
    return row_norms_A - 2 * tf.matmul(a, tf.transpose(b)) + row_norms_B

def proto_loss(emb,y_input):
    '''求emb_s和emb_q的softmax(loss)'''
    emb_s = emb[0:n_way*n_support,:]
    emb_q = emb[n_way*n_support:,:]    
    y_s = K.cast(K.gather(y_input, [i for i in range(n_way*n_support)]), dtype='int32')
    y_q = K.cast(K.gather(y_input, [i for i in range(n_way*n_support, n_way*(n_support+n_query)) ]), dtype='int32')

    onehot_s = K.one_hot(y_s, n_way)
    centers =  K.squeeze(  K.dot(K.transpose(onehot_s), emb_s)/n_support, axis=1)

    dist_1 = euc_distance(emb_q,centers)

    onehot_q = K.cast(K.one_hot(y_q, n_way),'bool')
    onehot_q= K.squeeze(onehot_q, axis=1)
    loss_deno = K.logsumexp(-1 * dist_1,axis=1)
    loss_nume = tf.boolean_mask(dist_1, onehot_q)
    
    loss = K.mean(loss_nume + loss_deno)
    return loss   

if __name__ == '__main__':         
    t1 = time.time()
    im_width, im_height = 200,4  #左/右输入的宽、高
    emb_dim = 128      
     

    #few shot model
    left_input = Input(shape=(im_width, im_height), dtype='float32', name='left_input')
    right_input = Input(shape=(im_width, im_height), dtype='float32', name='right_input')
    y_input = Input(shape=(1, ),dtype='int32', name='y_input')

    submodel = get_model(emb_dim)
    emb = submodel([left_input,right_input])
    loss_layer = Lambda(lambda x : proto_loss(x[0],x[1]),name='loss_layer')
    loss= loss_layer([emb,y_input])

    model_train = Model(inputs=[left_input,right_input,y_input], outputs=loss) 
    model_train.compile(optimizer='Adadelta', loss=lambda y_true,y_pred: y_pred)  
    
    model_train.load_weights(fs_model_initi_weights,by_name=True)    
    
    #print(model_train.summary())
    
    #读取数据
    train =  pd.read_csv('fewshot_train.csv',index_col=0)
    train['label'] = get_label(train)
    train.reset_index(inplace=True,drop=True)
    train_x = train.drop(['userid','apdid','phonetype','media_id','rowkey','label'],axis=1)        
    train_label2index = get_label2index(train)
    
    #训练
    print('Training...')
    y_input = get_y()
    random_y = np.array(np.random.rand(n_way*(n_support+n_query)))
    
#   #老老实实训练边生成数据边训练    
#    all_loss = []
#    for ep in range(epoch_start+1,epoch+1):
#        res = []
#        for epi in range(n_episodes):
#            idx = get_train_batch(train_label2index,n_way,n_support,n_query)           
#            train_sq = train_x.iloc[idx]
#            train_sq = train_sq.values.reshape(len(idx),im_width, im_height*2).astype('float32')
#            
#            train_sq_l = train_sq[:,:,:im_height]
#            train_sq_r = train_sq[:,:,im_height:]
#            loss = model_train.train_on_batch([train_sq_l,train_sq_r,y_input],random_y)
#            res.append(loss)
#        m_loss = np.mean(res) 
#        all_loss.append(m_loss)
#        if ep%10==0:            
#            #all_loss.append(m_loss)
#            print('Epoch %s/%s: the mean loss is %s' % (ep,epoch,m_loss))
#        if ep%1000==0:            
#            t2 = time.time();cost = int((t2-t1)/60)+1
#            print('运行总耗费时间:%s min'% cost)
#            #few shot model的输出
#            model_fs = Model([left_input, right_input], emb)
#            model_name = 'wsqet_'+str(n_way)+'_'+str(n_support)+'_'+str(n_query)+'_'+str(ep)+'_'+str(cost)+'.hdf5'
#            model_fs.save_weights(model_name)
#            print('保存模型为：',model_name)
                   
  
        
    #生成训练样本
    idx_lis = []
    for i in range(10000):    
        idx = get_train_batch(train_label2index,n_way,n_support,n_query)
        idx_lis.append(idx)
    idx_arr = np.array(idx_lis)
    arr_name_save = 'train_wsq_'+str(n_way)+'_'+str(n_support)+'_'+str(n_query)+'.npy'
    np.save(arr_name_save,idx_arr)
    
    
    if epoch<=500:
        #使用训练样本进行训练
        #arr_name_load = 'train_wsq_20_5_3.npy'
        idx_arr = np.load(arr_name_load)    
        loss_lis = []
        for ep in [100,200,300,400,500]:
            for i in range(1,idx_arr.shape[0]+1):  #一万个epoch，算是100个epoch
                idx = idx_arr[i]
                train_sq = train_x.iloc[idx]
                train_sq = train_sq.values.reshape(len(idx),im_width, im_height*2).astype('float32')
                
                train_sq_l = train_sq[:,:,:im_height]
                train_sq_r = train_sq[:,:,im_height:]
                loss = model_train.train_on_batch([train_sq_l,train_sq_r,y_input],random_y)
            if i%100==0:
                loss_lis.append(loss)
                print('Epoch %s/%s: the loss is %s' % (int(i/100),epoch,loss)) 
            t2 = time.time();cost = int((t2-t1)/60)+1
            print('运行总耗费时间:%s min'% cost)
            #few shot model的输出
            model_fs = Model([left_input, right_input], emb)
            model_name = 'wsqet_'+str(n_way)+'_'+str(n_support)+'_'+str(n_query)+'_'+str(ep)+'_'+str(cost)+'.hdf5'
            model_fs.save_weights(model_name)
            print('保存模型为：',model_name) 
        epoch_start = 500
            #记录loss变化
        with open('loss.txt','a') as f:
            f.write(str(model_name)+',')
            for i in loss_lis:
                f.write(str(i)+',')            
            f.write('\n') 
    
    loss_lis2 = []
    for ep in range(epoch_start+1,epoch+1):
        for epi in range(100):
            idx = get_train_batch(train_label2index,n_way,n_support,n_query)           
            train_sq = train_x.iloc[idx]
            train_sq = train_sq.values.reshape(len(idx),im_width, im_height*2).astype('float32')
            
            train_sq_l = train_sq[:,:,:im_height]
            train_sq_r = train_sq[:,:,im_height:]
            loss = model_train.train_on_batch([train_sq_l,train_sq_r,y_input],random_y)
        loss_lis2.append(loss)
        print('Epoch %s/%s: the mean loss is %s' % (ep,epoch,loss))
        
        if ep%100==0:            
            t2 = time.time();cost = int((t2-t1)/60)+1
            print('运行总耗费时间:%s min'% cost)
            #few shot model的输出
            model_fs = Model([left_input, right_input], emb)
            model_name = 'wsqet_'+str(n_way)+'_'+str(n_support)+'_'+str(n_query)+'_'+str(ep)+'_'+str(cost)+'.hdf5'
            model_fs.save_weights(model_name)
            print('保存模型为：',model_name)        
    
    #记录loss变化
    with open('loss.txt','a') as f:
        f.write(str(model_name)+',')
        for i in loss_lis2:
            f.write(str(i)+',')            
        f.write('\n') 
 
        
    

           

