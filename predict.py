#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : ningchao
# @Time    : 20/11/23 15:08
import utils
import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class Classifier:
    def __init__(self, model_dir: str):
        vocab, self.p_c, self.p_w_c, self.labels = utils.load_model(model_dir)
        self.cv = CountVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", vocabulary=vocab)

    def predict_text(self, text: str):
        seg = " ".join(list(jieba.cut(text)))
        text_vec = self.cv.transform([seg])
        log_p_d_c = text_vec @ np.log(self.p_w_c)
        log_p_c_d = np.log(self.p_c).reshape(-1, 1) + log_p_d_c.T
        prob = utils.softmax(log_p_c_d)
        return self.labels[prob[:, 0].argmax()]


if __name__ == "__main__":
    cls = Classifier("resources/model/toutiao_news_model")
    res = cls.predict_text("世界互联网大会，互联网发展论坛圆满闭幕")
    print(res)
