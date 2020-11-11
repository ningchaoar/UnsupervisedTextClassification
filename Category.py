#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : ningchao
# @Time    : 20/11/9 10:19
from typing import *


class Category:
    def __init__(self, name: str):
        self.name = name
        self.parent = None
        self.childs = {}  # dict in python>=3.7 is guaranteed to preserve order and is more performant than OrderedDict.
        self.keywords = set()
        self.documents = set()  # 存储文本索引
        self.category_list = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name
        else:
            return False

    def __repr__(self):
        if len(self.childs) == 0:
            return self.name + ": []"
        return self.name + ": " + str(list(self.childs.keys()))

    def get_hierarchical_path(self) -> List[str]:
        path = []
        parent = self.parent
        while parent is not None:
            path.insert(0, parent.name)
            parent = parent.parent
        path.append(self.name)
        return path

    def set_parent(self, parent: 'Category') -> None:
        self.parent = parent

    def add_child(self, child: 'Category', depth: int) -> None:
        child.set_parent(self)
        self.childs[child] = depth

    def set_keywords(self, keywords: Set[str]) -> None:
        self.keywords = keywords

    def get_parent(self) -> 'Category':
        return self.parent

    def get_childs(self) -> List['Category']:
        return list(self.childs.keys())

    def get_category_list(self):
        return self.category_list

    def set_category_list(self, category_list):
        self.category_list = category_list

    def add_document(self, document_index: int) -> None:
        self.documents.add(document_index)

    def get_documents(self) -> Set[int]:
        return self.documents

    def get_documents_size(self):
        return len(self.documents)

    def find_document(self, document_index: int) -> Optional['Category']:
        """
        传入文档索引查找包含该文档的类别
        :param document_index:
        :return:
        """
        if document_index in self.documents:
            return self
        else:
            for child in self.childs:
                category = child.find_document(document_index)
                if category is not None:
                    return category
            return None

    def find_category(self, categorie_path_list: List[str]) -> Optional['Category']:
        """
        在类别树中查找一个分类
        :param categorie_path_list: 分类层级list，父分类在前，子分类在后, 例: ["新闻", "经济", "金融业"]
        :return: 返回最后一级子分类 or None
        """
        if len(categorie_path_list) == 0:
            return None

        categories_copy = categorie_path_list.copy()
        if self.name == "ROOT":
            categories_copy.insert(0, "ROOT")

        if self.name == categories_copy[0]:
            if len(categories_copy) == 1:
                return self
            for child in self.childs:
                found = child.find_category(categories_copy[1:])
                if found is not None:
                    return found
        return None

    def add_category(self, categories: List[str]) -> 'Category':
        """
        向类别树中添加一个分类，若路径上的父级分类不存在，则一并添加
        :param categories: 分类层级list，父分类在前，子分类在后, 例: ["新闻", "经济", "金融业"]
        :return: 返回最后一级子分类
        """
        depth = len(categories)
        if depth == 0:
            return self

        leaf_category = self.find_category(categories)
        if leaf_category:
            return leaf_category

        category_path = categories[:-1]
        leaf_category = Category(categories[-1])
        parent = self.find_category(category_path)
        if parent is None:
            parent = self.add_category(category_path)
        parent.add_child(leaf_category, depth)

        return leaf_category

    def find_category_by_word(self, word: str) -> Optional['Category']:
        """
        传入单词搜索对应类别，同一路径上，子分类优先级更高。不同路径上，按照类别添加顺序选择。
        :param word:
        :return:
        """
        for child in self.childs:
            categoty = child.find_category_by_word(word)
            if categoty is not None:
                return categoty

        if word in self.keywords:
            return self

        return None


if __name__ == "__main__":
    root = Category("ROOT")
    root.add_category(["运营商", "流量提醒", "流量不足"])
    root.add_category(["运营商", "流量提醒", "流量优惠"])
    root.add_category(["运营商", "5G场景"])
    root.add_category(["银行"])
    root.add_category(["银行", "交易提醒"])
    root.add_category(["火车票", "购票提醒"])
    root.add_category(["火车票", "交易提醒"])
    print(root)
