#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : ningchao
# @Time    : 20/11/7 15:58
import jieba
import random
import logging
import numpy as np
from typing import *
from tqdm import tqdm
from Category import Category
from sklearn.feature_extraction.text import CountVectorizer


def load_category_and_words(input_file: str) -> Category:
    root_category = Category("ROOT")
    category_list = []
    with open(input_file, "r", encoding="utf-8") as fr:
        for line in fr:
            categories, words = line.strip().split("\t")
            category_list.append(categories)
            categories = categories.split("/")
            words = set(words.split(" "))
            root_category.add_category(categories).set_keywords(words)
    root_category.set_category_list(category_list)
    return root_category


def load_unlabled_documents(file: str) -> List[str]:
    with open(file, "r", encoding="utf-8") as fr:
        documents = [line.strip() for line in fr]
    return documents


def seg_documents(documents: List[str]) -> List[List[str]]:
    segs = []
    for text in tqdm(documents, desc="文档预分词"):
        text = text.strip().split("\t")[-1]
        seg = list(jieba.cut(text))
        segs.append(seg)
    return segs


def preliminary_labeling(category_tree: Category, segs: List[List[str]]):
    """
    遍历文档，根据关键词做预标注, 同时得到计数词典
    :param category_tree: 分类树根节点
    :param segs: 文档分词结果
    :return: 返回词表({word: index})和文档计数稀疏矩阵(csr_matrix)
    """
    # 默认的token_pattern会过滤掉单字
    cv = CountVectorizer(analyzer="word", max_df=0.95, min_df=0.001, token_pattern=r"(?u)\b\w+\b")
    logging.info("初始化单词-文档计数矩阵")
    document_to_word_count = cv.fit_transform([" ".join(seg) for seg in segs])  # csr_matrix, shape=(N_documents, N_vocabulary)
    for i, seg in tqdm(enumerate(segs), desc="文档预标注"):
        category = None
        for word in seg:
            category = category_tree.find_category_by_word(word)
            if category is not None:
                break
        if category is not None:
            category.add_document(i)

    return cv.vocabulary_, document_to_word_count


def init_bayes_model(category_tree: Category, documents_size: int, vocab_size: int):
    # 初始化贝叶斯模型
    # 1. 对文档进行预分词(DONE)
    # 2. 遍历一遍文档, 完成预标注，并得到词表和N(w, d), 矩阵格式, 形状 = (词典大小, 文档数), 利用CountVectorizer得到稀疏矩阵(DONE)
    # 3. 遍历一遍分类树, 得到P(c)(公式-2)和P(c|d)(0 or 1), 后者是矩阵格式, 形状 = (文档数, 类别数)(DONE)
    # 4. 对于每一个词, 计算P(w|c)(公式-1)
    # 5. 开始迭代, 对每一个文档, 重新计算P(c|d)(公式-3)
    # 6. 更新分类树, 重新计算P(c)
    # 7. 重复步骤4-6, 直至收敛
    category_list = category_tree.get_category_list()
    category_size = len(category_list)
    category_prior_probability = np.zeros(category_size)  # 类别先验概率P(C)
    category_document_cond_probability = np.zeros(([documents_size, category_size]))  # 文档条件概率P(C|D)

    # 根据预标注结果生成先验概率和条件概率
    for i, category in tqdm(enumerate(category_list), desc="模型初始化"):
        category_path = category.split("/")
        category_documents = category_tree.find_category(category_path).get_documents()
        for document_index in category_documents:
            category_document_cond_probability[document_index][i] = 1.0
        category_prior_probability[i] = (1 + len(category_documents)) / (category_size + documents_size)  # Laplace smooth

    # 调整形状并转化为Matrix对象, 便于矩阵乘法
    category_prior_probability = np.matrix(category_prior_probability).T  # shape=(N_cate, 1)
    category_document_cond_probability = np.matrix(category_document_cond_probability).T  # shape=(N_cate, N_documents)
    word_category_cond_probability = np.matrix(np.zeros([vocab_size, len(category_list)]))  # shape=(N_vocab, N_cate)
    logging.info("总计{}/{}条样本得到预标注".format(category_document_cond_probability.sum(), documents_size))

    return category_prior_probability, category_document_cond_probability, word_category_cond_probability


def expectation_step(document_to_word_count, p_c, p_c_d, p_w_c):
    # 根据本轮P(C|D), 重新计算P(W|C)(公式1)和P(C)(公式2)
    category_to_word_count = p_c_d * document_to_word_count  # shape=(N_cate, N_vocabulary)
    category_size = p_c.shape[0]
    documents_size = document_to_word_count.shape[0]
    vocab_size = document_to_word_count.shape[1]
    for i in tqdm(range(vocab_size), desc="E-step"):
        for j in range(category_size):
            p_w_c[i, j] = (1 + category_to_word_count[j, i]) / (vocab_size + category_to_word_count[j].sum())
    for i in range(category_size):
        p_c[i, 0] = (1 + p_c_d[i].sum()) / (category_size + documents_size)


def maximization_step(document_to_word_count, p_c_d, p_c, p_w_c):
    # 根据本轮P(W|C)和P(C), 更新P(C|D)(公式3)
    # 公式3 中的乘法改为加法
    category_size = p_c.shape[0]
    documents_size = document_to_word_count.shape[0]
    for i in tqdm(range(category_size), desc="M-step"):
        for j in range(documents_size):
            val = p_c[i, 0]
            document_vec = document_to_word_count[j]
            doc_feature_row, doc_feature_col = document_vec.nonzero()
            for m, n in zip(doc_feature_row, doc_feature_col):
                val += p_w_c[n, i] * document_vec[m, n]
            p_c_d[i, j] = val
    return softmax(p_c_d)


def softmax(x):
    norm_x = np.exp(x - x.max())
    return norm_x / norm_x.sum()


def hierarchical_shrinkage_step():
    pass


if __name__ == "__main__":
    word_file = "resources/dict/words.txt"
    sms_file = "resources/cropus/sms.txt"
    logging.basicConfig(level=logging.INFO)
    category_tree = load_category_and_words(word_file)
    documents = load_unlabled_documents(sms_file)
    random.shuffle(documents)
    documents = documents[:20000]
    segs = seg_documents(documents)
    vocabulary, document_to_word_count = preliminary_labeling(category_tree, segs)
    p_c, p_c_d, p_w_c = init_bayes_model(category_tree, documents_size=len(documents), vocab_size=len(vocabulary))

    for i in tqdm(range(5), desc="EM迭代进度"):
        expectation_step(document_to_word_count, p_c, p_c_d, p_w_c)
        p_c_d = maximization_step(document_to_word_count, p_c_d, p_c, p_w_c)

    category_list = category_tree.get_category_list()
    fw = open("resources/cropus/sms_result_1111.txt", "w", encoding="utf-8")
    for i in range(len(documents)):
        prob = p_c_d[:, i]
        predict_category = category_list[prob.argmax()]
        fw.write(documents[i] + "\t" + predict_category + "\n")
    fw.close()


