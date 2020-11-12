#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : ningchao
# @Time    : 20/11/9 18:16
import unittest
from category import Category


class TestMain(unittest.TestCase):

    def test_category_class(self):
        categort_tree = Category("ROOT")
        categort_tree.add_category(["运营商", "流量提醒", "流量不足"])
        categort_tree.add_category(["运营商", "流量提醒", "流量优惠"])
        categort_tree.add_category(["运营商", "5G场景"])
        categort_tree.add_category(["银行"])
        categort_tree.add_category(["银行", "交易提醒"])
        categort_tree.add_category(["火车票", "购票提醒"])
        categort_tree.add_category(["火车票", "交易提醒"])

        res = str(categort_tree)
        an = 'ROOT: [运营商: [流量提醒: [流量不足: [], 流量优惠: []], 5G场景: []], 银行: [交易提醒: []], 火车票: [购票提醒: [], 交易提醒: []]]'
        self.assertEqual(res, an)

    def test_init_category(self):

        def load_category_and_words(input_file: str) -> Category:
            root_category = Category("ROOT")
            with open(input_file, "r", encoding="utf-8") as fr:
                for line in fr:
                    categories, words = line.strip().split("\t")
                    categories = categories.split("/")
                    words = set(words.split(" "))
                    root_category.add_category(categories).set_keywords(words)
            return root_category

        file = "resources/dict/words_example.txt"
        category_tree = load_category_and_words(file)
        res = str(category_tree)
        an = 'ROOT: [运营商: [流量场景: [], 积分场景: [], 5G场景: []], 银行: [交易场景: [], 通知场景: []], 火车票: [购票场景: [], 抢票场景: []], 机票: [购票场景: [], 改签场景: []], 快递: [取件场景: [], 寄件场景: []]]'
        self.assertEqual(res, an)

        category = category_tree.find_category_by_word("流量")
        if category is not None:
            res = "<-".join(category.get_hierarchical_path())
        an = 'ROOT<-运营商<-流量场景'
        self.assertEqual(res, an)


if __name__ == '__main__':
    unittest.main()
