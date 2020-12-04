# Unsupervised Text Classification
Implementation for paper ["Text Classification by Bootstrapping with Keywords, EM and Shrinkage"](http://www.cs.cmu.edu/~knigam/papers/keywordcat-aclws99.pdf)

无监督文本分类，基础算法来自上述论文，在细节和效率上做出一定优化。

## 适用场景
对大批量无标注文本进行快速的预分类，以研究样本分布情况和分类标注规则，提高后续的精细化标注和使用复杂模型进行迭代的效率。

## 算法说明
分类算法主要由以下四部分组成：  
**1. 关键词预标注**  
  使用初始关键词(seed keywords)对文本集进行预标注。  
**2. 多项式朴素贝叶斯**  
  在预标注的文本集上，对类别先验概率P(category)和单词的后验概率P(word|catrgory)进行估计，进而计算出P(category|document)  
**3. hierachical shrinkage**  
  利用分类的层级关系，在估计P(word|category)时将父分类样本包含进去，缓解特征稀疏的问题  
**4. EM迭代**  
  步骤2和步骤3涉及的参数均利用EM迭代进行更新并得到收敛  

## Requirements
python >= 3.6  
numpy >= 1.18.5  
scipy >= 1.4.1  
scikit-learn >= 0.21.0  
jieba  
tqdm  

## 如何开始训练
### 待分类样本集
实验所用样本集为网友收集的**今日头条新闻语料**，包含15个分类下的382688条样本。  
介绍及下载详见[toutiao-text-classfication-dataset](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset)  
下载并解压得到toutiao_cat_data.txt，其文件路径将在训练时作为参数传入main函数。

### 初始关键词
**1. 关键词格式**  
  如`resources/dict/words_toutiao_news.txt`所示:  
  每行对应一个分类，分类与关键词之间用"###"分隔，关键词与关键词之间用"|"分隔  
  分类层级的表示方法为：父分类与子分类之间用"/"分隔，父分类在前，子分类在后。  
    例：`新闻/国际/经济###W1|W2|W3`  
    W1、W2、W3是**经济**类别的关键词，其父分类为**国际**，而**国际**类再上一层是**新闻**类  
  注意：如果存在层级分类，那么需要准备层级上的每一个分类的关键词  

**2. 构建经验**  
  首先定义分类，然后可根据直觉、经验以及对样本的观察获取各分类的初始关键词。  
  按原论文所述，关键词的精确和召回之间，**优先保证精确**。即多一些特征显著的词，不要使用停用词。  
  原论文所做实验中，初始关键词覆盖率为41%，本项目实验中初始关键词覆盖率为39%。  
  
  如分类结果不佳，首先检查样本集中是否存在较多的未定义分类，或者存在容易混淆的分类，前者需要新增相应分类，后者可以考虑对易混分类进行统一或建立层级关系。最后考虑对关键词进行优化和补充。  

### 训练模型
`main.py`: **模型训练main函数**，给定关键词路径、样本路径、结果保存路径、模型保存路径和迭代轮数即可开始训练  
`utils.py`: 如使用其他格式的样本，需修改word_segment函数以适应样本格式  

### 其它模块说明
`report.py`: 打印评测报告  
`category.py`: 分类树类  
`predict.py`: 模型读取&预测类，Classifier.predict_text为单条预测接口，传入原始文本并返回概率最高的前N个结果  
`resources/dict/words_toutiao_news.txt`: 示例关键词文件  
`resources/cropus/toutiao_cat_data_example.txt`: 示例样本文件(头条新闻语料抽样)  

## 实验记录
目前在头条新闻语料上所取得的最好结果如下：
```
                    precision    recall  f1-score   support

        news_story       0.34      0.91      0.50      6273
      news_culture       0.86      0.71      0.78     28031
news_entertainment       0.93      0.75      0.83     39396
       news_sports       0.98      0.84      0.91     37568
      news_finance       0.54      0.39      0.45     27085
        news_house       0.79      0.90      0.84     17672
          news_car       0.94      0.90      0.92     35785
          news_edu       0.83      0.83      0.83     27058
         news_tech       0.81      0.72      0.76     41543
     news_military       0.81      0.66      0.72     24984
       news_travel       0.75      0.79      0.77     21422
        news_world       0.52      0.74      0.61     26909
             stock       0.02      0.71      0.05       340
  news_agriculture       0.83      0.85      0.84     19322
         news_game       0.90      0.91      0.90     29300

          accuracy                           0.77    382688
         macro avg       0.72      0.77      0.71    382688
      weighted avg       0.81      0.77      0.78    382688
```
使用关键词：
```
news_story###故事|事件|真实|民间|神话|传说|记录|儿媳|儿子|婆婆|结婚
news_culture###文化|历史|艺术|哲学|上联|下联|文艺|文学
news_entertainment###网红|热搜|综艺|明星|演员|奥斯卡|整容
news_sports###体育|运动|篮球|足球|乒乓球|排球|奥运会
news_finance###金融|理财|投资|银行|人民币|经济|GDP|资金|货币|融资|财富
news_house###房子|买房|购房|房产|房价
news_car###汽车|驾驶|买车|车辆|跑车
news_edu###教育|学校|大学|课程|教师|高中
news_tech###科技|科学|技术|电子|互联网|研究|手机|智能|5G|卫星
news_military###军事|战争|武器|装备|枪械|坦克|战斗机|导弹|潜艇|军舰
news_travel###旅行|旅游|驴友|自驾游|景点|景区|风景
news_world###世界|局势|政治|经济|美国|中国|联合国|国际
stock###股票|炒股|大盘|指数
news_agriculture###农业|三农|水稻|农村
news_game###游戏|手游|电脑|战队|电竞
```
使用参数：
```
CountVectorizer(analyzer="word", max_df=0.8, min_df=0.00001, token_pattern=r"(?u)\b\w+\b")
max_iters = 5
```
性能：  
内存占用峰值约1GB  
AMD 3700X每轮迭代耗时约40s，i5-7200U每轮迭代耗时约75s  

## 关键词&参数研究
**1. 初始关键词的影响**  
  大幅减少各分类关键词数:
  ```
  news_story###故事|事件|真实|民间
  news_culture###文化|历史|艺术|哲学|上联|下联
  news_entertainment###网红|热搜|综艺
  news_sports###体育|运动|篮球|足球
  news_finance###金融|理财|投资|银行
  news_house###房子|买房|购房|房产|房价
  news_car###汽车|驾驶|买车|车辆
  news_edu###教育|学校|大学
  news_tech###科技|科学|技术|电子|互联网
  news_military###军事|战争|武器|装备
  news_travel###旅行|旅游|驴友|自驾游|景点|景区
  news_world###世界|局势|政治|经济|美国|中国
  stock###股票|炒股
  news_agriculture###农业|三农|水稻|农村
  news_game###游戏|手游|电脑|战队
  ```
  ```
                      precision    recall  f1-score   support
  
          news_story       0.03      0.09      0.05      6273
        news_culture       0.81      0.72      0.77     28031
  news_entertainment       0.93      0.51      0.66     39396
         news_sports       0.98      0.78      0.87     37568
        news_finance       0.64      0.30      0.41     27085
          news_house       0.73      0.91      0.81     17672
            news_car       0.92      0.91      0.92     35785
            news_edu       0.84      0.87      0.86     27058
           news_tech       0.81      0.66      0.73     41543
       news_military       0.73      0.46      0.57     24984
         news_travel       0.73      0.80      0.76     21422
          news_world       0.43      0.75      0.55     26909
               stock       0.02      0.72      0.05       340
    news_agriculture       0.61      0.85      0.71     19322
           news_game       0.81      0.93      0.87     29300
  
            accuracy                           0.71    382688
           macro avg       0.67      0.69      0.64    382688
        weighted avg       0.77      0.71      0.72    382688
  ```
  初始关键词对分类效果有显著影响，当某分类召回率较低时，尝试增加关键词数量；优化一个分类的关键词，同时会小幅提升其他分类的效果。  
  如果存在易混淆分类，例如news_finance和stock，则调整关键词的作用不大，推荐处理办法是将混淆分类合并，或者增加层级关系。  

**2. 词典大小的影响**  
  改变CountVectorizer的max_df、min_df以及token_pattern参数，主要影响的是词典的大小。  
  测试结果如下：  
  | analyzer | 词典大小 | accuracy | macro f1 | weighted f1 |  
  | :---- | :---- | ----: | ----: | ----: |  
  | char | 5329 | 0.63 | 0.58 | 0.64 |  
  | word | 14800 | 0.75 | 0.70 | 0.77 |  
  | word | 27130 | 0.75 | 0.70 | 0.77 |  
  | word | 44654 | 0.76 | 0.71 | 0.78 |  
  | word(default) | 83278 | 0.77 | 0.72 | 0.78 |  
  | word | 232820 | 0.74 | 0.69 | 0.75 |  

  在合理的参数组合下(analyzer!='char')，最差和最好的结果差距在3%左右。  
  对于开放领域的中文语料，使用代码中的默认值即可。  
  备选组合：  
  `CountVectorizer(analyzer="word", max_df=0.8, min_df=0, token_pattern=r"(?u)\b\w\w+\b")`
  `CountVectorizer(analyzer="word", max_df=0.8, min_df=10, token_pattern=r"(?u)\b\w+\b")`

**3. shrinkage步骤的影响**  
  经测试，hierarchical shrinkage步骤在分类不具备层级关系时，效果不明显。  
  而在分类存在层级关系时，取消shrinkage步骤将使分类结果偏聚到子分类上。  
  使用shrinkage步骤将使训练速度大幅降低，但完全在可接受范围内，因此默认在任何情况下均使用shrinkage步骤。

**4. 不同迭代轮数的影响**  
  原论文以参数收敛作为EM迭代停止条件，但没有写明收敛的判定标准。  
  经测试，迭代1轮效果较差。迭代次数超过5次时，效果基本不再变化。  
  | 迭代轮数 | Accuracy | macro f1 | weighted f1 |  
  | :---- | ----: | ----: | ----: |  
  | 0(预标注) | 0.24 | 0.31 | 0.34 |  
  | 1 | 0.64 | 0.61 | 0.66 |  
  | 3 | 0.76 | 0.71 | 0.78 |  
  | 5(default) | 0.77 | 0.72 | 0.78 |  
  | 7 | 0.77 | 0.71 | 0.78 |  
  | 10 | 0.77 | 0.71 | 0.78 |  

  因此采用max_iters=5作为默认参数。
