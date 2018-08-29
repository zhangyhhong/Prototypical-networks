# Prototypical networks

## 训练原型网络

train-prototypical-networks(train-prototypical-networks.py)用于训练。

## 测试网络

test-prototypical-networks(test-prototypical-networks.py)用于测试。

其中，测试数据是二分类问题，用于AUC来评估模型。一般的分类方法用AUC_method=‘min’计算，此处提供多个尝试的方法。

## 结果可视化

test.ipynb(test.ipynb)用于和源代码比较，看是否有效。

summary_result(summary_result.py)用于对多种方法生成的结果进行结果统计和可视化。

PR_AUC(PR_AUC.py)用于对结果进行PR和AUC的计算，以及可视化。

