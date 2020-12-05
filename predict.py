#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : ningchao
# @Time    : 20/11/23 15:08
import utils
import jieba
import numpy as np
from typing import *
from sklearn.feature_extraction.text import CountVectorizer


class Classifier:
    def __init__(self, model_dir: str):
        # 注意CountVectorizer的参数应与训练时一致
        vocab, self.p_c, self.p_w_c, self.labels = utils.load_model(model_dir)
        self.cv = CountVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", vocabulary=vocab)

    def predict_text(self, text: str, top_n: int=1) -> List[Tuple[str, float]]:
        """
        单条预测接口
        :param text: 待分类文本
        :param top_n: 返回概率最高的前N个预测结果, 默认为1
        :return: [(category_name, probability)]
        """
        # 分词模式与训练一致
        seg = " ".join(utils.jieba_segment(text, mode="search"))
        text_vec = self.cv.transform([seg])
        log_p_d_c = text_vec @ np.log(self.p_w_c)
        log_p_c_d = np.log(self.p_c).reshape(-1, 1) + log_p_d_c.T
        prob = utils.softmax(log_p_c_d)
        top_n_index = prob[:, 0].argsort()[::-1][:top_n]
        return [(self.labels[index], prob[:, 0][index]) for index in top_n_index]


if __name__ == "__main__":
    cls = Classifier("resources/model/toutiao_news_model")
    res = cls.predict_text("今年5G通信产业规模预计达5036亿元 同比增长128%", top_n=3)
    print(res)
