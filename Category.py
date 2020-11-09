#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : ningchao
# @Time    : 20/11/9 10:19
from typing import *


class Category:
    def __init__(self, name: str):
        self.name = name
        self.parent = None
        self.childs = set()
        self.keywords = set()

        # self.documents = []

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name
        else:
            return False

    def __repr__(self):
        if len(self.childs) == 0:
            return self.name + ": {}"
        return self.name + ": " + str(self.childs)

    def set_parent(self, parent: 'Category') -> None:
        self.parent = parent

    def add_child(self, child: 'Category') -> None:
        child.set_parent(self)
        self.childs.add(child)

    def set_keywords(self, keywords: Set[str]) -> None:
        self.keywords = keywords

    def get_parent(self) -> 'Category':
        return self.parent

    def get_childs(self) -> Set['Category']:
        return self.childs

    def find_category(self, categories: List[str]) -> Optional['Category']:
        """
        在类别树中查找一个分类
        :param categories: 分类层级list，父分类在前，子分类在后, 例: ["新闻", "经济", "金融业"]
        :return: 返回最后一级子分类 or None
        """
        if len(categories) == 0:
            return None

        categories_copy = categories.copy()
        if self.name == "ROOT":
            categories_copy.insert(0, "ROOT")

        if self.name == categories_copy[0]:
            if len(categories_copy) == 1:
                return self

            for child in self.get_childs():
                found = child.find_category(categories_copy[1:])
                if found is not None:
                    return found

        return None

    def add_category(self, categories: List[str]) -> 'Category':
        """
        向类别树中增加一个分类
        :param categories: 分类层级list，父分类在前，子分类在后, 例: ["新闻", "经济", "金融业"]
        :return: 返回最后一级子分类
        """
        if len(categories) == 0:
            return self

        leaf_category = self.find_category(categories)
        if leaf_category:
            return leaf_category

        category_path = categories[:-1]
        leaf_category = Category(categories[-1])
        parent = self.find_category(category_path)
        if parent is None:
            parent = self.add_category(category_path)
        parent.add_child(leaf_category)

        return leaf_category


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
