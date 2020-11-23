#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : ningchao
# @Time    : 20/11/19 16:35
import jieba
import logging
import numpy as np
from typing import *
from tqdm import tqdm
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
    对原始文档分词, 根据数据集格式调整这里的处理, 确保对内容进行分词
    :param documents: list of document
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
