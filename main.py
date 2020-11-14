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
from category import Category
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
    logging.info("分类树: {}".format(root_category))
    return root_category


def load_unlabled_documents(file: str) -> List[str]:
    with open(file, "r", encoding="utf-8") as fr:
        documents = [line.strip() for line in fr]
    return documents


def seg_documents(documents: List[str]) -> List[List[str]]:
    # TODO: 应支持自定义的分词器
    segs = []
    for text in tqdm(documents, desc="文档预分词"):
        text = text.strip().split("\t")[-1]
        seg = list(jieba.cut(text))
        segs.append(seg)
    return segs


def preliminary_labeling(category_tree: Category, segs: List[List[str]]):
    """
    遍历文档，根据关键词做预标注, 同时得到计数词典
    TODO: 应支持自定义的预标注规则, 例如命中两个词或匹配正则
    :param category_tree: 分类树根节点
    :param segs: 文档分词结果
    :return: 返回词表({word: index})和文档计数稀疏矩阵(csr_matrix)
    """
    # 默认的token_pattern会过滤掉单字
    cv = CountVectorizer(analyzer="word", max_df=0.8, min_df=0.001, token_pattern=r"(?u)\b\w\w+\b")
    logging.info("初始化单词-文档计数矩阵")
    # TODO: 转换成numpy array, 后面用到的矩阵也使用numpy array而不是matrix
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
            category_document_cond_probability[document_index, i] = 1.0
        category_prior_probability[i] = (1.0 + len(category_documents)) / (category_size + documents_size)  # Laplace smooth

    # 调整形状并转化为Matrix对象, 便于矩阵乘法
    category_prior_probability = np.matrix(category_prior_probability).T  # shape=(N_cate, 1)
    category_document_cond_probability = np.matrix(category_document_cond_probability).T  # shape=(N_cate, N_documents)
    word_category_cond_probability = np.matrix(np.zeros([vocab_size, len(category_list)]))  # shape=(N_vocab, N_cate)
    logging.info("总计{}/{}条样本得到预标注".format(int(category_document_cond_probability.sum()), documents_size))

    return category_prior_probability, category_document_cond_probability, word_category_cond_probability


def maximization_step(document_to_word_count, p_c, p_c_d, p_w_c):
    # 根据本轮P(C|D), 重新计算P(W|C)(公式1)和P(C)(公式2)
    category_to_word_count = p_c_d * document_to_word_count  # shape=(N_cate, N_vocabulary)
    category_size = p_c.shape[0]
    documents_size = document_to_word_count.shape[0]
    vocab_size = document_to_word_count.shape[1]
    for i in tqdm(range(category_size), desc="M-step(without shrinkage)"):
        for j in range(vocab_size):
            p_w_c[j, i] = (1 + category_to_word_count[i, j]) / (vocab_size + category_to_word_count[i].sum())
    for i in range(category_size):
        p_c[i, 0] = (1.0 + p_c_d[i].sum()) / (category_size + documents_size)


def maximization_step_with_shrinkage(category_tree: Category, document_to_word_count, p_c, p_c_d, p_w_c, lambda_matrix):
    # 每一个分类，存储对应的层级路径list，比如
    category_size = p_c.shape[0]
    documents_size = document_to_word_count.shape[0]
    vocab_size = document_to_word_count.shape[1]
    lambda_size = lambda_matrix.shape[1]  # max_depth + 2

    category_list = category_tree.get_category_list()

    category_to_word_count = p_c_d * document_to_word_count  # shape=(N_cate, N_vocabulary)
    p_w_c_k = np.zeros([vocab_size, category_size, lambda_size])

    for i in tqdm(range(category_size), desc="M-step(with shrinkage)"):
        category_path = category_list[i].split("/")
        dep_list = []
        category_depth = len(category_path)
        for k in range(category_depth):
            # 第一层只包含该分类自身, 然后沿着路径一直到ROOT节点(但不包含ROOT节点)
            dep_list.append(category_list.index("/".join(category_path)))
            category_to_word_count_hierarchy = category_to_word_count[dep_list].sum(axis=0)  # 将父分类的文本集也算入子分类中
            for j in range(vocab_size):
                p_w_c_k[j, i, k] = (1.0 + category_to_word_count_hierarchy[0, j]) / (vocab_size + category_to_word_count_hierarchy.sum())
            category_path = category_path[:-1]

    category_to_word_count_root = category_to_word_count.sum(axis=0)
    for i in range(vocab_size):
        p_w_c_k[i, :, -2] = (1.0 + category_to_word_count_root[0, i]) / (vocab_size + category_to_word_count_root.sum())
    p_w_c_k[:, :, -1] = 1.0 / vocab_size

    for i in range(vocab_size):
        p_w_c[i] = np.multiply(lambda_matrix, p_w_c_k[i]).sum(axis=1).T

    for i in range(category_size):
        p_c[i, 0] = (1 + p_c_d[i].sum()) / (category_size + documents_size)


def expectation_step(document_to_word_count, p_c, p_w_c):
    # 根据本轮P(W|C)和P(C), 更新P(C|D)(公式3)
    # 公式3 中的乘法改为加法
    logging.info("E-step")
    log_p_d_c = document_to_word_count * np.log(p_w_c)  # shape=(N_documents, N_cate)
    log_p_c_d = np.log(p_c) + log_p_d_c.T  # shape=(N_cate, N_documents)
    return softmax(log_p_c_d)


def expectation_step_with_shrinkage(document_to_word_count, p_c, p_w_c):
    logging.info("E-step(with shrinkage)")
    log_p_d_c = document_to_word_count * np.log(p_w_c)  # shape=(N_documents, N_cate)
    log_p_c_d = np.log(p_c) + log_p_d_c.T  # shape=(N_cate, N_documents)

    beta_matrix = np.matrix(np.zeros([]))

    return softmax(log_p_c_d)


def softmax(x):
    # 注意axis=0，对每一个文档的概率分布都减去对应的最大值
    norm_x = x - x.max(axis=0)
    return np.exp(norm_x) / np.exp(norm_x).sum(axis=0)


def hierarchical_shrinkage_init(category_tree: Category):
    # 构建一个lambda矩阵, 行为分类索引，列为分类树最大深度, 元素取值范围[0, 1], 每行求和等于1
    # 最后, 每个分类的p_w_c都通过矩阵乘法获得, 修正后的p_w_c = lambda * p_w_c_k (公式4)
    # p_w_c_k = path_matrix * p_w_c
    # 从叶节点到ROOT的最长路径, depth=0表示ROOT节点, depth=max_depth表示叶节点。
    # 在lambda_matrix中, 0列表示叶节点权重, max_depth列表示ROOT节点权重, max_depth+1列表示1/|V|的权重
    logging.info("初始化lambda矩阵")
    max_depth = Category.get_max_depth(category_tree)
    category_list = category_tree.get_category_list()
    category_size = len(category_list)

    lambda_matrix = np.matrix(np.zeros([category_size, max_depth + 2]))
    for i, path in enumerate(category_list):
        category_node = category_tree.find_category(path.split("/"))
        depth = category_node.get_depth()
        init_lambda_val = 1.0 / (depth + 2)
        for j in range(depth):
            lambda_matrix[i, j] = init_lambda_val
        lambda_matrix[i, max_depth] = init_lambda_val
        lambda_matrix[i, max_depth+1] = init_lambda_val

    return lambda_matrix


if __name__ == "__main__":
    word_file = "resources/dict/words_18w.txt"
    sms_file = "resources/cropus/18w_sms.txt"
    logging.basicConfig(level=logging.INFO)

    category_tree = load_category_and_words(word_file)
    documents = load_unlabled_documents(sms_file)
    # random.shuffle(documents)
    # documents = documents[:2000]
    segs = seg_documents(documents)
    vocabulary, document_to_word_count = preliminary_labeling(category_tree, segs)
    lambda_matrix = hierarchical_shrinkage_init(category_tree)
    p_c, p_c_d, p_w_c = init_bayes_model(category_tree, documents_size=len(documents), vocab_size=len(vocabulary))
    for _i in tqdm(range(5), desc="EM迭代进度"):
        # maximization_step(document_to_word_count, p_c, p_c_d, p_w_c)
        maximization_step_with_shrinkage(category_tree, document_to_word_count, p_c, p_c_d, p_w_c, lambda_matrix)
        p_c_d = expectation_step(document_to_word_count, p_c, p_w_c)

    category_list = category_tree.get_category_list()
    fw = open("resources/cropus/18w_sms_result.txt", "w", encoding="utf-8")
    for i in range(len(documents)):
        prob = p_c_d[:, i]
        predict_category = category_list[prob.argmax()]
        fw.write(documents[i] + "\t" + predict_category + "\n")
    fw.close()


