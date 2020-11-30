# Unsupervised\_Text\_Classification
Implementation for paper "Text Classification by Bootstrapping with Keywords, EM and Shrinkage" http://www.cs.cmu.edu/~knigam/papers/keywordcat-aclws99.pdf
无监督文本分类，基础算法来自上述论文，在细节和效率上做出一定优化。

## 适用场景
对大批量文本进行快速的预标注, 以研究样本分布情况和分类标注规则, 提高后续的精细化标注和使用复杂模型进行迭代的效率

## 算法说明
关键词预分类 + 贝叶斯分类 + EM迭代 + shrinkage步骤

## Requirements
python >= 3.6
numpy
scikit-learn
jieba
tqdm
pickle

## 数据准备
### 待分类样本集
实验所用语料为网友收集的今日头条新闻语料，包含15个分类下的38万条样本。
https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset

### 初始关键词
1. 关键词格式说明
2. 构建方法

## 模块说明
1. 训练
2. 评估
3. 预测

## 实验记录 
1. 不同初始关键词
2. 不同向量化参数
3. shrinkage步骤的影响
4. 不同迭代轮数

## 总结 & TODO
1. 自定义预标注规则