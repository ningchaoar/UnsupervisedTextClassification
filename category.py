#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : ningchao
# @Time    : 20/11/9 10:19
from typing import *
from collections import OrderedDict


class Category:
    def __init__(self, name: str):
        self.name = name
        self.depth = 0
        self.parent = None
        self.childs = OrderedDict()  # 使用OrderedDict以兼容python<=3.5
        self.keywords = set()
        self.documents = set()  # 存储文本索引
        self.category_list = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other: 'Category'):
        if isinstance(other, self.__class__):
            return self.name == other.name
        else:
            return False

    def __repr__(self):
        if len(self.childs) == 0:
            return self.name + ": []"
        return self.name + ": " + str(list(self.childs.keys()))

    def get_hierarchical_path(self) -> List[str]:
        # path not including 'ROOT'
        path = []
        parent = self.parent
        while parent.name != "ROOT":
            path.insert(0, parent.name)
            parent = parent.parent
        path.append(self.name)
        return path

    def set_depth(self, depth: int) -> None:
        self.depth = depth

    def get_depth(self) -> int:
        return self.depth

    @staticmethod
    def get_max_depth(root_node: 'Category', cur_depth: int = 0) -> int:
        if len(root_node.get_childs()) == 0:
            return cur_depth
        cur_depth_copy = cur_depth
        for child in root_node.childs:
            child_depth = Category.get_max_depth(child, cur_depth_copy + 1)
            if child_depth > cur_depth:
                cur_depth = child_depth
        return cur_depth

    def set_parent(self, parent: 'Category') -> None:
        self.parent = parent

    def add_child(self, child: 'Category', depth: int) -> None:
        child.set_parent(self)
        child.set_depth(depth)
        self.childs[child] = depth

    def set_keywords(self, keywords: Set[str]) -> None:
        self.keywords = keywords

    def get_parent(self) -> 'Category':
        return self.parent

    def get_childs(self) -> List['Category']:
        return list(self.childs.keys())

    def get_category_list(self) -> List[str]:
        return self.category_list

    def set_category_list(self, category_list: List[str]):
        self.category_list = category_list

    def add_document(self, document_index: int) -> None:
        self.documents.add(document_index)

    def get_documents(self) -> Set[int]:
        return self.documents

    def get_documents_size(self) -> int:
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
