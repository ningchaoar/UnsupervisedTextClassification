#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : ningchao
# @Time    : 19/7/20 14:29
# @Remark  : 测试关键步骤


def get_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as fr:
        cropus = [line.strip() for line in fr.readlines()]
    return cropus


def get_keywords(keywords_file):
    pass


def preliminary_labeling(cropus, category_to_keywords):
    """
    用关键词进行初步分类
    :param cropus: list of documents
    :param category_to_keywords: {C1:(w11, w12, ...), c2:(w21, 22, ...), ...}
    :return: category_to_documents
    """
    # 建立关键词到类别的字典
    keywords_to_category = {}
    for cate in category_to_keywords:
        for word in category_to_keywords[cate]:
            if word not in keywords_to_category:
                keywords_to_category[word] = cate

    category_to_documents = {}
    for cate in category_to_keywords:
        category_to_documents[cate] = []
    for i, line in enumerate(cropus):
        for word in keywords_to_category:
            if word in line:
                category = keywords_to_category[word]
                category_to_documents[category].append(i)

    return category_to_documents
