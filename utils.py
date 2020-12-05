#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : ningchao
# @Time    : 20/11/19 16:35
import os
import jieba
import pickle
import logging
import collections
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
            categories, words = line.strip().split("###")
            category_list.append(categories)
            categories = categories.split("/")
            words = set(words.split("|"))
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


def word_segment(datas: List[str], mode: str="search") -> List[List[str]]:
    """
    对原始文档分词, 根据数据集格式调整这里的处理方式, 确保对文本进行分词
    可以用自定义分词器替换jieba_segment
    :param datas: list of raw lines
    :param mode: jieba分词模式, "default"(粗粒度) or "search"(细粒度)
                 细粒度模式下, 一些长词会被再次切分(如 "计算机科学与技术" -> "计算机 科学 与 技术")
    :return: list of segmented documents
    """
    segs = []
    for data in tqdm(datas, desc="文档集预分词"):
        document = data.split("_!_", 3)[-1]
        document = document.replace("_!_", " ")
        seg = jieba_segment(document, mode=mode)
        segs.append(seg)
    logging.info("分词前: {}".format(datas[0]))
    logging.info("分词后: {}".format(segs[0]))
    logging.info("如所取字段与预期不一致, 需修改utils.word_segment, 确保其符合数据集格式")
    return segs


def jieba_segment(text: str, mode: str) -> List[str]:
    seg = list(jieba.tokenize(text, mode=mode))
    # build DAG
    graph = collections.defaultdict(list)
    for word in seg:
        graph[word[1]].append(word[2])

    def dfs(graph: Dict[int, List[int]], v: int, seen: List[int]=None, path: List[int]=None):
        if seen is None:
            seen = []
        if path is None:
            path = [v]
        seen.append(v)
        paths = []
        for t in graph[v]:
            if t not in seen:
                t_path = path + [t]
                paths.append(tuple(t_path))
                paths.extend(dfs(graph, t, seen, t_path))
        return paths

    longest_path = sorted(dfs(graph, 0), key=lambda x: len(x), reverse=True)[0]
    longest_seg = [text[longest_path[i]: longest_path[i + 1]] for i in range(len(longest_path) - 1)]
    return longest_seg


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
