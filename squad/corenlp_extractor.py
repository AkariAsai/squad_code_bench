import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corenlp
import json
import pprint


class CoreNLPExtractor:
    def __init__(self):
        self.parser = self.connect_parser()

    def connect_parser(self):
        corenlp_dir = "stanford-corenlp-full-2016-10-31/"
        parser = corenlp.StanfordCoreNLP(corenlp_path=corenlp_dir)
        return parser

    def get_coreNLP_json(self, sentence):
        return json.loads(self.parser.parse(sentence))

    def convert_damn_corenlp_output(self, parse_str, json):
        parse = parse_str.split("[")[1:]
        word_list = []

        for word in parse:
            entity_dic = {}
            entity_list = word.split(" ")[:-1]
            for entity in entity_list:
                if len(entity.split("=")) == 2:
                    entity_dic[entity.split("=")[0]] = entity.split("=")[
                        1].replace("]", "")
            word_list.append(entity_dic)

        inital_word_dic = json["sentences"][0]['words'][0][1]
        word_list.append(inital_word_dic)

        return word_list

    def collect_constituency_dict(self, const_str):
        const = const_str.split(" ")
        const_dictionary = {}

        for i in range(len(const)):
            if const[i][0] != "(":
                const_tag = []
                for j in range(1, i):
                    if const[i - j][0] != "(" or i - j == 0:
                        break
                    else:
                        const_tag.append(const[i - j][1:])
                const_dictionary[const[i].replace(")", "")] = const_tag[::-1]
        return const_dictionary

    def collect_dependency_dict(self, json):
        deps = parse_tree_str = json["sentences"][0]['dependencies']
        dep_dictionary = {}
        for sublist in deps:
            if sublist[0] != "punct":
                dep_dictionary[sublist[2]] = sublist[0]
        return dep_dictionary

    def get_pos_entity(self, word_dic):
        pos, entity = {}, {}

        for item in word_dic:
            if item["NamedEntityTag"] != 'O':
                entity[item['Lemma']] = item["NamedEntityTag"]
            if 'Lenma' in item.keys() and 'PartOfSpeech' in item.keys():
                pos[item['Lemma']] = item['PartOfSpeech']

        return pos, entity

    def collect_word_info(self, sentence):
        json = self.get_coreNLP_json(sentence)

        parse_tree_str = json["sentences"][0]['parsetree']
        index = parse_tree_str.index("(")
        const_str = parse_tree_str[index:]
        parse_tree = parse_tree_str[:index]

        word_dic = self.convert_damn_corenlp_output(parse_tree, json)
        pos, entity = self.get_pos_entity(word_dic)
        const_dic = self.collect_constituency_dict(const_str)
        dep_dic = self.collect_dependency_dict(json)

        return pos, entity, const_dic, dep_dic


def main():
    cnlp_extractor = CoreNLPExtractor()
    print(cnlp_extractor.collect_word_info("when did Ed married with Nancy?"))


if __name__ == "__main__":
    main()
