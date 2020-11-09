#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author  : ningchao
# @Time    : 20/11/7 15:58
from Category import Category


def load_category_and_words(input_file: str) -> Category:
    root_category = Category("ROOT")
    with open(input_file, "r", encoding="utf-8") as fr:
        for line in fr:
            categories, words = line.strip().split("\t")
            categories = categories.split("/")
            words = set(words.split(" "))
            root_category.add_category(categories).set_keywords(words)
    return root_category


if __name__ == "__main__":
    file = "resources/dict/words.txt"
    categort_tree = load_category_and_words(file)
    print(categort_tree)
