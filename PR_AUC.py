#!/usr/bin/env python
# -*- coding:utf8 -*-
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve,auc
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from scipy.stats import norm

class PRHelper():

    def Find_Optimal_Cutoff(self,target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------
        list type, with optimal cutoff value

        """
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
        roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

        return list(roc_t['threshold'])

    def show_thres(self,label,probality):
        best_threshold = self.Find_Optimal_Cutoff(label,probality)
        print(best_threshold)


    def __init__(self,fin):
        self._fin = fin
        self._fout = fin+".png"
        print(self._fin)

    def run(self):

        rs=pd.read_csv(self._fin,names=['probality','label'],header=0,index_col=0)
        #rs['label'] =rs['label'].apply(lambda x: 0  if x==1 else 1)
        #rs['probality'] =rs['probality'].apply(lambda x:1-x)
        print(rs.shape)

        same_df = rs[rs['label'] == 0]
        diff_df = rs[rs['label'] == 1]


        fpr, tpr, thresholds = roc_curve(rs['label'],rs['probality'])
        lauc = auc(fpr,tpr)

        self.show_thres(rs['label'],rs['probality'])

        precision, recall, thresholds = precision_recall_curve(rs['label'],rs['probality'])
        average_precision = average_precision_score(rs['label'],rs['probality'])
        print('AUC={0:0.2f}'.format(lauc))
        print('PRA={0:0.2f}'.format(average_precision))
        pr = pd.DataFrame()
        pr['precision'] = precision
        pr['recall'] = recall
        ths= [rs['probality'].min()]+thresholds.tolist()
        print(rs['probality'].min())
        print(len(thresholds))
        print(len(ths))
        print(pr.shape)
        pr['threshold'] = ths
        pr.to_csv(self._fin+'.pr',index=False)

        # Plot Precision-Recall curve

        fig = plt.figure(figsize=(28, 8), dpi=80)
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1])

        #plt.subplot(2, 2, 1)
        plt.subplot(gs[0])
        plt.plot(pr['recall'], pr['precision'], label='PR')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        ax = plt.gca()
        xmajorLocator   = MultipleLocator(0.1)
        ax.xaxis.set_major_locator(xmajorLocator)
        ymajorLocator   = MultipleLocator(0.1)
        ax.yaxis.set_major_locator(ymajorLocator)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('PR curve: avepr={0:0.2f}'.format(average_precision))
        plt.grid(True)
        plt.legend(loc=0,prop={'size':10})

        plt.subplot(gs[1])
        #plt.subplot(2, 2, 2)
        plt.plot(fpr, tpr, label='ROC',color='darkorange')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax = plt.gca()
        xmajorLocator   = MultipleLocator(0.1)
        ax.xaxis.set_major_locator(xmajorLocator)
        ymajorLocator   = MultipleLocator(0.1)
        ax.yaxis.set_major_locator(ymajorLocator)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('ROC curve: AUC={0:0.2f}'.format(lauc))
        plt.grid(True)
        plt.legend(loc=0,prop={'size':10})

        plt.subplot(gs[2])
        sns.distplot(same_df['probality'],norm_hist=False)
        sns.distplot(diff_df['probality'],norm_hist =False)
        plt.xlim([0.0, 2.0])
        plt.xlabel('score')
        plt.show()

        #fig.savefig(self._fout)

if __name__ == '__main__':
    fin  = sys.argv[1]
    helper = PRHelper(fin)
    helper.run()

#python PR_AUC.py dist.csv

