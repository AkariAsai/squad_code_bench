import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corenlp
import json
import pprint
import  ast

class DependencyNode:
    def __init__(self, word, postag="root"):
        self.word = word
        self.postag = postag
        self.children = None

class DependencyTree:
    def __init__(self, dep_lists):
        """
        Take a dependency tree represented returned from stanford core nlp and create tree.
        Input : [['root', 'ROOT', 'love'], ['nsubj', 'love', 'I'], ['dobj', 'love', 'pugs']]
        (The original sentence = "I love pugs")
        Output : Tree forms
            ROOT
            │
            love(root)
                ├──pugs(dobj)
                |
                └── I(nsubj)
        """
        self.org_list = dep_lists
        self.tree = self.create_tree(self.org_list)

    def create_tree(self, dep_lists):
        root = None
        root_word = dep_lists[0][2]
        root_node = DependencyNode(root_word)




    def insert(self, node, word, postag):
        if node is None:
            return self.createNode(word, postag)

        node.children = self.insert(node.children, word, postag)
        return node

    def create_dependency_tree(self, word_list):
        root = DependencyNode(word_list[0][2])
        for sublist in word_list:
            parent_word = sub[1]
            node = self.search_node(root,parent_word)
            self.insert(node, sub[2],sub[0])
        return root



    def search_node(self, node, parent_word):
        if not node:
            return None
        if node.word == parent_word:
            return node
        else:
            search_result = [self.search_node(child) for child in node.children]
            for result in search_result:
                if result:
                    return result



    def get_word_by_postag(self, node, targe_postag):
        if not node:
            return "Unfound"
        if node.postag == target_postag:
            return node.word

        else:
            search_results = [self.get_word_by_postag(child) for child in children]
            for result in search_result:
                if result!= "Unfound":
                    return result

    def get_postag_by_word(self, node, target_word):
        if not node:
            return "Unfound"
        if node.word == target_word:
            return node.postag

        else:
            search_results = [self.get_word_by_postag(child) for child in children]
            for result in search_results:
                if result!= "Unfound":
                    return result
