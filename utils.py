#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : ningchao
# @Time    : 20/11/19 16:35
import os
import jieba
import pickle
import logging
import numpy as np
from typing import *
from tqdm import tqdm
from sklearn import metrics

from category import Category


def load_seed_keywords(keywords_file: str) -> Category:
    """
    获取初始关键词, 并构建分类树
    :param keywords_file: 关键词文件路径
    :return: 分类树root节点
    """
    category = Category("ROOT")
    category_list = []
    with open(keywords_file, "r", encoding="utf-8") as fr:
        for line in fr:
            categories, words = line.strip().split("\t")
            category_list.append(categories)
            categories = categories.split("/")
            words = set(words.split(" "))
            category.add_category(categories).set_keywords(words)
    category.set_category_list(category_list)
    logging.info("分类树: {}".format(category))
    return category


def load_data(file: str) -> List[str]:
    """
    读取待分类数据集
    :param file: 文件路径
    :return:
    """
    with open(file, "r", encoding="utf-8") as fr:
        datas = [line.strip() for line in fr]
    return datas


def word_segment(datas: List[str]) -> List[List[str]]:
    """
    对原始文档分词, 根据数据集格式调整这里的处理方式, 确保对文本进行分词
    :param datas: list of document
    :return: list of seged document
    """
    segs = []
    for data in tqdm(datas, desc="文档集预分词"):
        document = data.split("_!_", 3)[-1]
        document = document.replace("_!_", " ")
        seg = list(jieba.cut(document))
        segs.append(seg)
    return segs


def softmax(x):
    # 对log_p_c_d做归一化, 故取axis=0
    norm_x = x - x.max(axis=0)
    return np.exp(norm_x) / np.exp(norm_x).sum(axis=0)


def save_model(model_dir: str, vocabulary: Dict[str, int], p_c: np.ndarray, p_w_c: np.ndarray, labels: List[str]):
    """
    保存模型
    :param model_dir: 模型保存目录, 如目录非空则覆盖旧模型
    :param vocabulary: CountVectorizer词典
    :param p_c: P(C)
    :param p_w_c: P(W|C)
    :param labels: category list
    :return:
    """
    if not model_dir.endswith("/"):
        model_dir += "/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(model_dir + "vocab.pkl", "wb") as fw:
        pickle.dump(vocabulary, fw)
    np.save(model_dir + "category_prob.npy", p_c)
    np.save(model_dir + "word_prob.npy", p_w_c)
    with open(model_dir + "labels.txt", "w", encoding="utf-8") as fw:
        for l in labels:
            fw.write(l + "\n")
    logging.info("模型保存成功")


def load_model(model_dir: str):
    """
    读取模型
    :param model_dir: 模型保存目录
    :return: 词典, P(C), P(W|C), category list
    """
    if not model_dir.endswith("/"):
        model_dir += "/"
    with open(model_dir + "vocab.pkl", "rb") as fr:
        vocabulary = pickle.load(fr)
    p_c = np.load(model_dir + "category_prob.npy")
    p_w_c = np.load(model_dir + "word_prob.npy")
    with open(model_dir + "labels.txt", "r", encoding="utf-8") as fr:
        labels = [line.strip() for line in fr]

    return vocabulary, p_c, p_w_c, labels


def get_report(result_file: str, label_file: str):
    with open(label_file, "r", encoding="utf-8") as fr:
        labels = [line.strip() for line in fr]
    with open(result_file, "r", encoding="utf-8") as fr:
        fr = [line.strip().rsplit("\t", 1) for line in fr]
    y_predict = []
    y_label = []
    for line in tqdm(fr):
        raw_data, y = line
        label = raw_data.split("_!_")[2]
        y_predict.append(labels.index(y))
        y_label.append(labels.index(label))
    print(metrics.classification_report(y_label, y_predict, target_names=labels))


if __name__ == "__main__":
    get_report(result_file="resources/cropus/toutiao_cat_data_result.txt",
               label_file="resources/model/toutiao_news_model/labels.txt")
