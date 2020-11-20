#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : ningchao
# @Time    : 20/11/7 15:58
# TODO: 优化命名, 优化项目结构, 完善注释, 优化内存和效率
# TODO: 增加模型保存和预测接口
import utils
import logging
import numpy as np
from typing import *
from tqdm import tqdm
from category import Category
from sklearn.feature_extraction.text import CountVectorizer


def preliminary_labeling(category_tree: Category, segs: List[List[str]]):
    """
    TODO: 应支持自定义的预标注逻辑, 或读取预标注结果
    遍历文档，根据初始关键词做预标注, 同时得到文档-词频表
    :param category_tree: 分类树根节点
    :param segs: 文档集分词结果
    :return: 返回单词索引表({word: index})和文档词频矩阵(documents_size, vocab_size)
    """
    # 默认的token_pattern会过滤掉单字
    cv = CountVectorizer(analyzer="word", max_df=0.8, min_df=0.0001, token_pattern=r"(?u)\b\w\w+\b")
    logging.info("初始化单词-文档计数矩阵")
    document_vectors = cv.fit_transform([" ".join(seg) for seg in segs])  # csr_matrix
    for i, seg in tqdm(enumerate(segs), desc="文档预标注"):
        category = None
        for word in seg:
            category = category_tree.find_category_by_word(word)
            if category is not None:
                break
        if category is not None:
            category.add_document(i)
    # TODO: document_vectors转化为ndarray对于开放领域的大语料(文档多，词多), 可能还是造成内存溢出
    return cv.vocabulary_, document_vectors.toarray()


def init_bayes_model(category_tree: Category, documents_size: int, vocab_size: int):
    """
    初始化模型所用参数
    :param category_tree: 分类树根节点
    :param documents_size: 文档数
    :param vocab_size: 单词数
    :return: P(C) -> (category_size, )
             P(C|D) -> (category_size, documents_size)
             P(W|C) -> (vocab_size, category_size)
    """
    category_list = category_tree.get_category_list()
    category_size = len(category_list)
    category_prior_probability = np.zeros(category_size)  # 类别先验概率P(C)
    category_document_cond_probability = np.zeros(([documents_size, category_size]))  # 文档条件概率P(C|D)

    # 根据预标注结果初始化P(C)和P(C|D)
    for c, category in tqdm(enumerate(category_list), desc="模型初始化"):
        category_path = category.split("/")
        category_documents = category_tree.find_category(category_path).get_documents()
        for document_index in category_documents:
            category_document_cond_probability[document_index, c] = 1.0
        category_prior_probability[c] = (1.0 + len(category_documents)) / (category_size + documents_size)  # using Laplace smooth

    category_document_cond_probability = category_document_cond_probability.T  # 转置便于矩阵乘法
    word_category_cond_probability = np.zeros([vocab_size, len(category_list)])
    logging.info("总计{}/{}条文档得到预标注".format(int(category_document_cond_probability.sum()), documents_size))

    return category_prior_probability, category_document_cond_probability, word_category_cond_probability


def maximization_step(document_vectors, p_c, p_c_d, p_w_c):
    # E-step更新P(C|D)后, 在M-step中更新P(W|C)(公式1)和P(C)(公式2)
    category_vectors = p_c_d @ document_vectors  # shape=(category_size, vocab_size)
    category_size = p_c.shape[0]
    documents_size = document_vectors.shape[0]  # 原论文documents_size = |D| + |H|
    vocab_size = document_vectors.shape[1]
    for c in tqdm(range(category_size), desc="Horizontal M-step"):
        for v in range(vocab_size):
            p_w_c[v, c] = (1 + category_vectors[c, v]) / (vocab_size + category_vectors[c].sum())
    for c in range(category_size):
        p_c[c] = (1.0 + p_c_d[c].sum()) / (category_size + documents_size)


def maximization_step_with_shrinkage(category_tree: Category, document_vectors, p_c, p_c_d, p_w_c, p_w_c_k, lambda_matrix, beta_matrix, iter: int):
    # E-step更新P(C|D)后, 在M-step中更新P(W|C)(公式1)和P(C)(公式2)
    documents_size, vocab_size = document_vectors.shape
    category_size, lambda_size = lambda_matrix.shape
    category_list = category_tree.get_category_list()
    # vertical M
    if iter > 0:
        shrinkage_maximization_step(lambda_matrix, beta_matrix, p_c_d)
    # horizontal M
    # update P^{α}(w|c)
    # TODO: 需要优化, 在下面的遍历中p_c_d只需要取出用得到的类别即可, ROOT节点可以单独算, 不要做full size的矩阵乘法
    for c in tqdm(range(category_size), desc="Horizontal M-step"):
        category_path = category_list[c].split("/")
        dep_list = []
        category_depth = len(category_path)
        for k in range(category_depth):
            # 第一层为该类自身, 然后沿着层级直到ROOT(不包含ROOT)
            dep_list.append(category_list.index("/".join(category_path)))
            category_vectors = p_c_d[dep_list] @ document_vectors  # 只需取出包含的类别
            if category_vectors.ndim == 1:
                category_vectors = category_vectors.reshape(1, -1)
            category_vector_hierarchy = category_vectors.sum(axis=0)  # 将父分类的文本集也算入子分类中
            category_vector_hierarchy_sum = category_vector_hierarchy.sum()
            for v in range(vocab_size):
                p_w_c_k[v, c, k] = (1.0 + category_vector_hierarchy[v]) / (vocab_size + category_vector_hierarchy_sum)
            category_path = category_path[:-1]
    category_vector_root = document_vectors.sum(axis=0)
    category_vector_root_sum = document_vectors.sum()
    for v in range(vocab_size):
        p_w_c_k[v, :, -2] = (1.0 + category_vector_root[v]) / (vocab_size + category_vector_root_sum)
    p_w_c_k[:, :, -1] = 1.0 / vocab_size
    # update p_w_c by function(4)
    for v in range(vocab_size):
        p_w_c[v] = (lambda_matrix * p_w_c_k[v]).sum(axis=1)
    # update p_c by function(2)
    for c in range(category_size):
        p_c[c] = (1 + p_c_d[c].sum()) / (category_size + documents_size)


def expectation_step_with_shrinkage(document_vectors, p_c, p_w_c, p_w_c_k, lambda_matrix, beta_matrix):
    # M-step更新P(W|C)和P(C)后, 在E-step中更新P(C|D)(公式3)
    logging.info("Horizontal E-step")
    # vertical E
    shrinkage_expectation_step(document_vectors, lambda_matrix, beta_matrix, p_w_c_k)
    # horizontal E
    # 求log将function(3)中累乘改为累加
    log_p_d_c = document_vectors @ np.log(p_w_c)  # shape=(documents_size, category_size)
    log_p_c_d = np.log(p_c).reshape(-1, 1) + log_p_d_c.T  # shape=(category_size, documents_size)
    return utils.softmax(log_p_c_d)


def hierarchical_shrinkage_init(category_tree: Category, document_vectors):
    # 构建一个lambda矩阵, 行为分类索引，列为分类树最大深度, 元素取值范围[0, 1], 每行求和等于1
    # 最后, 每个分类的p_w_c都通过矩阵乘法获得, 修正后的p_w_c = lambda @ p_w_c_k (公式4)
    # p_w_c_k = path_matrix * p_w_c
    # 从叶节点到ROOT的最长路径, depth=0表示ROOT节点, depth=max_depth表示叶节点。
    # 在lambda_matrix中, 0列表示叶节点权重, max_depth列表示ROOT节点权重, max_depth+1列表示1/|V|的权重
    # init λ
    logging.info("初始化shrinkage参数")
    max_depth = Category.get_max_depth(category_tree)
    category_list = category_tree.get_category_list()
    category_size = len(category_list)
    lambda_size = max_depth + 2
    lambda_matrix = np.zeros([category_size, lambda_size])
    for c, path in enumerate(category_list):
        category_node = category_tree.find_category(path.split("/"))
        depth = category_node.get_depth()
        init_lambda_val = 1.0 / (depth + 2)
        for k in range(depth):
            lambda_matrix[c, k] = init_lambda_val
        lambda_matrix[c, max_depth] = init_lambda_val
        lambda_matrix[c, max_depth+1] = init_lambda_val
    # init β
    documents_size, vocab_size = document_vectors.shape
    beta_matrix = np.zeros([documents_size, category_size, lambda_size])
    # init P^{α}(w|c)
    p_w_c_k = np.zeros([vocab_size, category_size, lambda_size])
    return lambda_matrix, beta_matrix, p_w_c_k


def shrinkage_maximization_step(lambda_matrix, beta_matrix, p_c_d):
    # 更新λ function(6)
    documents_size, category_size, lambda_size = beta_matrix.shape
    for c in tqdm(range(category_size), desc="Vertical M-step"):
        norm_val = p_c_d[c].sum()
        for k in range(lambda_size):
            lambda_matrix[c, k] = beta_matrix[:, c, k] @ p_c_d[c]
            lambda_matrix[c, k] /= norm_val


def shrinkage_expectation_step(document_vectors, lambda_matrix, beta_matrix, p_w_c_k):
    # 更新β function(5)
    documents_size, vocab_size = document_vectors.shape
    for d in tqdm(range(documents_size), desc="Vertical E-step"):
        document_vector_nonzero = document_vectors[d].nonzero()  # 只要求出该文档非零词的索引即可，词频不为零则词频会约掉，词频为零则无必要去求
        for v in document_vector_nonzero[0]:
            p_w_c_alpha = lambda_matrix * p_w_c_k[v]  # shape = (category_size, lambda_size)
            p_w_c_alpha = p_w_c_alpha / p_w_c_alpha.sum(axis=1).reshape(-1, 1)
            p_w_c_alpha /= document_vector_nonzero[0].shape[0]  # 公式6中分母上的K, 提前放在这里
            beta_matrix[d] += p_w_c_alpha


if __name__ == "__main__":
    word_file = "resources/dict/words_18w.txt"
    sms_file = "resources/cropus/18w_sms.txt"
    logging.basicConfig(level=logging.INFO)

    category_tree = utils.load_seed_keywords(word_file)
    documents = utils.load_data(sms_file)
    segs = utils.word_segment(documents)
    vocabulary, document_vectors = preliminary_labeling(category_tree, segs)
    p_c, p_c_d, p_w_c = init_bayes_model(category_tree, documents_size=len(documents), vocab_size=len(vocabulary))
    lambda_matrix, beta_matrix, p_w_c_k = hierarchical_shrinkage_init(category_tree, document_vectors)
    for _i in range(5):
        # TODO: 增加原文中根据收敛情况终止迭代的设计
        logging.info("EM迭代进度: {}/{}".format(_i, 5))
        maximization_step_with_shrinkage(category_tree, document_vectors, p_c, p_c_d, p_w_c, p_w_c_k, lambda_matrix, beta_matrix, _i)
        p_c_d = expectation_step_with_shrinkage(document_vectors, p_c, p_w_c, p_w_c_k, lambda_matrix, beta_matrix)

    category_list = category_tree.get_category_list()
    fw = open("resources/cropus/18w_sms_full_shrinkage_result_iter10.txt", "w", encoding="utf-8")
    for i in range(len(documents)):
        prob = p_c_d[:, i]
        predict_category = category_list[prob.argmax()]
        fw.write(documents[i] + "\t" + predict_category + "\n")
    fw.close()
