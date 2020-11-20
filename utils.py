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
    读取待分类样本集
    :param file: 样本文件路径
    :return: list of document
    """
    with open(file, "r", encoding="utf-8") as fr:
        documents = [line.strip() for line in fr]
    return documents


def word_segment(documents: List[str]) -> List[List[str]]:
    """
    TODO: 加入自定义分词器
    :param documents: list of document
    :return: list of seged document
    """
    segs = []
    for text in tqdm(documents, desc="样本集预分词"):
        text = text.strip().split("\t")[-1]
        seg = list(jieba.cut(text))
        segs.append(seg)
    return segs


def softmax(x):
    # 对log_p_c_d做归一化, 故取axis=0
    norm_x = x - x.max(axis=0)
    return np.exp(norm_x) / np.exp(norm_x).sum(axis=0)
