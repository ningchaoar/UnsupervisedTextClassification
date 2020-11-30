# Unsupervised Text Classification
Implementation for paper ["Text Classification by Bootstrapping with Keywords, EM and Shrinkage"](http://www.cs.cmu.edu/~knigam/papers/keywordcat-aclws99.pdf)

无监督文本分类，基础算法来自上述论文，在细节和效率上做出一定优化。

## 适用场景
对大批量无标注文本进行快速的预分类, 以研究样本分布情况和分类标注规则, 提高后续的精细化标注和使用复杂模型进行迭代的效率。

## 算法说明
分类算法主要由以下四部分组成：  
**1. 关键词预分类**  
  使用初始关键词(seed keywords)对文本集进行预标注。  
**2. 多项式朴素贝叶斯**  
  在预标注的文本集上，对分类先验概率P(c)和单词的后验概率P(w|c)进行估计  
**3. hierachical shrinkage**  
  利用分类的层级关系，对在计算P(w|c)时将父分类样本也包含进去，缓解特征稀疏的问题  
**4. EM迭代**  
  步骤2和步骤3均利用EM迭代得到更新并最终收敛  

## Requirements
python >= 3.6  
numpy  
scikit-learn  
jieba  
tqdm  
pickle  

## 数据准备
### 待分类样本集
实验所用语料为网友收集的**今日头条新闻语料**，包含15个分类下的38万条样本。
语料介绍及下载详见[toutiao-text-classfication-dataset](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset)

### 初始关键词
**1. 关键词格式**  
  如示例关键词文件`resources/dict/words_toutiao_news.txt`所示:  
  每行对应一个分类，分类与关键词之间用"###"分隔，关键词与关键词之间用"|"分隔  
  分类层级的表示方法为：父分类与子分类之间用"/"分隔，父分类在前，子分类在后。  
    例：`新闻/经济/金融业###W1|W2|W3`  
    W1、W2、W3为分类**金融业**的关键词，其父分类为**经济**，而**经济**类再上一层是**新闻**类  
  注意：如果存在层级分类，那么需要准备层级上的每一个分类的关键词  

**2. 构建经验**  
  首先定义分类，然后可根据直觉、经验以及对样本的观察获取各分类的初始关键词。  
  按原论文所述，关键词的精确和召回之间，优先保证前者。即多一些特征显著的词，不要使用停用词。  
  原论文所做实验中，初始关键词覆盖率为41%，本项目实验中初始关键词覆盖率为39%。  
  
  如分类结果不佳，首先检查样本集中是否存在较多的未定义分类，或者存在容易混淆的分类，前者需要新增相应分类，后者可以考虑对易混分类进行统一或建立层级关系。最后考虑对关键词进行优化和补充。  

## 模块说明
准备完语料和关键词，按以下步骤进行训练
1. 读取&分词
2. 训练
3. 预测

## 实验记录 
1. 不同初始关键词
2. 不同向量化参数
3. shrinkage步骤的影响
4. 不同迭代轮数

## 总结 & TODO
1. 自定义预标注规则