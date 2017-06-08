import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import corenlp
import seaborn as sn
from sklearn.metrics import confusion_matrix
import re
import json
import pprint
from nltk import ngrams
from nltk.corpus import stopwords
from ast import literal_eval
import string
from text_similarity_calculator import TextSimilarityCalculator
from nltk.tokenize import word_tokenize
from corenlp_extractor import CoreNLPExtractor


class FeatureCreatorWord():
    def __init__(self, id, extractor, text_similarity_calculator):
        self.extractor = extractor
        self.id = id
        # get basic infomation of question
        self.question_df = self.load_target_csv(id)
        self.question = self.question_df["question"]
        self.answers = self.collect_answers()
        self.answer_type = self.question_df["AnswerType"]

        self.sentences = self.split_paragraph_into_sentences(
            self.question_df["context"])
        self.stop = set(stopwords.words('english'))
        self.text_similarity_calculator = text_similarity_calculator
        # get corenlp's constituency, dependency, postag and namedentity
        # extraction result.
        self.corenlp_info = self.get_coreNLP_information(self.question)
        self.question_pos = self.corenlp_info["pos"]
        self.question_entity = self.corenlp_info["entity"]
        self.question_const_dic = self.corenlp_info["const_dic"]
        self.question_dep_dic = self.corenlp_info["dep_dic"]

    def get_coreNLP_information(self, sentence):
        corenlp_info_dic = {}
        pos, entity, const_dic, dep_dic = self.extractor.collect_word_info(
            sentence)
        corenlp_info_dic["pos"] = pos
        corenlp_info_dic["entity"] = entity
        corenlp_info_dic["const_dic"] = const_dic
        corenlp_info_dic["dep_dic"] = dep_dic

        return corenlp_info_dic

    def load_target_csv(self, id):
        df_columns = ['id', 'question', 'answer_0_text', 'answer_0_start', 'answer_1_text', 'answer_1_start',
                      'answer_2_text', 'answer_2_start', 'title', 'context', 'paragraph_idx',
                      'AnswerType', 'entities', 'NNPs', 'NNs', 'VBs', 'WP']
        df = pd.read_csv("df_dev_updated.csv")
        df = df[df_columns]
        questions = df.loc[df['id'] == id].iloc[0]
        return questions

    def collect_answers(self):
        answer0 = self.question_df["answer_0_text"]
        answer1 = self.question_df["answer_1_text"]
        answer2 = self.question_df["answer_2_text"]

        return [answer0, answer1, answer2]

    def split_paragraph_into_sentences(self, paragraph):
        sentenceEnders = re.compile('[.!?]')
        sentenceList = sentenceEnders.split(paragraph)
        return sentenceList

    def entity_match(self, answer_type, candidate_type):
        if answer_type == "HUMAN":
            if "HUMAN" == candidate_type:
                return True
        elif answer_type == "LOCATION":
            if "LOCATION" == candidate_type:
                return True
        elif answer_type == "ENTITY":
            if "ENTITY" == candidate_type or "MISC" == candidate_type or 'ORGANIZATION' == candidate_type:
                return True
        elif answer_type == "NUMERIC":
            if "NUMBER" == candidate_type or "DURATION" == candidate_type or "DATE" == candidate_type:
                return True
        else:
            return False

    def isverbmatched(self, verbs, s_words):
        for verb in verbs:
            for token in s_words:
                if token.lower() == verb:
                    return True
        return False

    def find_word_index_in_sentence(self, word, sentence):
        token_sentence = sentence.split(' ')
        index = 0
        for token in token_sentence:
            if word.lower() == token.lower():
                return index
            index += 1
        return 100

    def dis_from_keyword(self, token, keywords, sentence):
        if keywords:
            word_index = self.find_word_index_in_sentence(token, sentence)
            keyword_indices = [self.find_word_index_in_sentence(
                keyword, sentence) for keyword in keywords]
            if min(keyword_indices) != 100:
                return min([abs(word_index - keyword_index) for keyword_index in keyword_indices])
            else:
                return 30
        else:
            return 30

    def with_punctuation(self, token):
        characters = token
        for c in characters:
            if c in string.punctuation:
                return True
        return False

    def get_words_by_postag(self, postag_dic, postag):
        return [key for key, val in postag_dic.items() if val == postag]

    def compare_const_similarity_q_sentence(self, sentence_const):
        match_count = 0
        for word, c_tag in sentence_const.items():
            if word in self.question_const_dic.keys() and c_tag == self.question_const_dic[word]:
                match_count += 1
        return match_count / len(sentence_const)

    def dep_matched(self, sentence_dep, dep_tag):
        if dep_tag == "root":
            print(sentence_dep)
            answer_root = [
                key for key, val in self.question_dep_dic.items() if val == 'root']
            sentence_root = [key for key,
                             val in sentence_dep.items() if val == 'root']
            if len(answer_root) >= 1 and len(sentence_root) >= 1:
                return answer_root[0] == sentence_root[0]
            else:
                return False

        elif dep_tag == "nmod":
            answer_deps = [
                key for key, val in self.question_dep_dic.items() if 'nmod' in val]
            sentence_deps = [key for key,
                             val in sentence_dep.items() if 'nmod' in val]
            return len([target_word for target_word in sentence_deps if target_word in answer_deps]) > 0
        else:
            answer_deps = [
                key for key, val in self.question_dep_dic.items() if val == dep_tag]
            sentence_deps = [key for key,
                             val in sentence_dep.items() if val == dep_tag]
            return len([target_word for target_word in sentence_deps if target_word in answer_deps]) > 0

    def create_feature_vectors(self):
        df_features = pd.DataFrame(columns=[
            "word", "question_id", "question", "answers",
            "shared_word_num", "num_of_keywords", "num_of_eneities",  "num_of_nnps", "num_of_nns",
            "answer_entity_type", "token_entity_type", "entity_matched",
            "min_distance_key", "verb_matched", "constituency_tag",
            "pos", "capital", "punctuation", "text_similarity",
            "subj_match", "root_match", "obj_match", "nmod_match",
            "dep_tag", "const_tag", "const_match_count"])

        q_info = self.question_df
        entities = self.question_entity.keys()
        nnps = self.get_words_by_postag(self.question_pos, "NNP")
        nns = self.get_words_by_postag(self.question_pos, "NN")
        vb = self.get_words_by_postag(self.question_pos, "VB")
        q_words = word_tokenize(self.question)

        for sentence in self.sentences:
            s_words = word_tokenize(sentence)
            # ignore senteces which doesn't contain any single words with
            # the questions.
            if len(sentence) > 0 and len([word for word in s_words if word in q_words]) > 1:
                sentence_corenlp_info = self.get_coreNLP_information(sentence)
                pos, entity = sentence_corenlp_info["pos"], sentence_corenlp_info["entity"]
                const, dep = sentence_corenlp_info["const_dic"], sentence_corenlp_info["dep_dic"]
                # The features of the selected sentence.
                shared_word_num = len(
                    [word for word in s_words if word in q_words])
                num_of_eneities = len(
                    [word for word in s_words if word in entities])
                num_of_nnps = len(
                    [word for word in s_words if word in nnps])
                num_of_nns = len(
                    [word for word in s_words if word in nns])
                verb_matched = self.isverbmatched(
                    vb, s_words)
                text_similarity = self.text_similarity_calculator.calculate_cos_similarity(
                    self.question, sentence)

                # syntactic structure
                const_match_count = self.compare_const_similarity_q_sentence(
                    const)
                root_matched = self.dep_matched(dep, "root")
                subj_match = self.dep_matched(dep, "nsubj")
                obj_match = self.dep_matched(dep, "dobj")
                nmod_match = self.dep_matched(dep, "nmod")

                for token in s_words:
                    if token not in self.stop and token not in string.punctuation and token != "``":
                        min_distance_key = min(self.dis_from_keyword(token, nnps, sentence), self.dis_from_keyword(
                            token, entities, sentence))
                        entity_type = "None"

                        if token in entity.keys():
                            entity_type = entity[token]
                        entity_matched = True if self.entity_match(
                            self.answer_type, entity_type) else False
                        pos_tag = pos[token.lower()] \
                            if token.lower() in pos.keys() else "SS"
                        dep_tag = dep[token] if token in dep.keys() else "NONE"
                        const_tag = const[token][0] if token in const.keys(
                        ) else "NONE"

                        df_features = df_features.append({"word": token, "question_id": self.id, "question": self.question, "answers": self.answers, "shared_word_num": shared_word_num,
                                                          "num_of_keywords": num_of_eneities + num_of_nnps, "num_of_eneities": num_of_eneities, "num_of_nnps": num_of_nnps, "num_of_nns": num_of_nns,
                                                          "answer_entity_type": q_info["AnswerType"], "token_entity_type": entity_type, "entity_matched": entity_matched,
                                                          "min_distance_key": min_distance_key, "verb_matched": verb_matched,
                                                          "pos": pos_tag, "capital": token.isupper(), "punctuation": self.with_punctuation(token),
                                                          "text_similarity": text_similarity, "subj_match": subj_match, "root_match": root_matched, "obj_match": obj_match, "nmod_match": nmod_match,
                                                          "dep_tag": dep_tag, "const_tag": const_tag, "const_match_count": const_match_count}, ignore_index=True)
        return df_features


def main():
    extractor = CoreNLPExtractor()
    df = pd.read_csv("df_dev_updated.csv", index_col=0)
    df = df.sample(frac=0.1, replace=True)
    text_similarity_calculator = TextSimilarityCalculator(
        df["context"].values + df["question"].values)
    creator = FeatureCreatorWord(
        "56e0812c231d4119001ac215", extractor, text_similarity_calculator)
    df = creator.create_feature_vectors()
    df.to_csv("df_feature_experiment.csv", index=False)


if __name__ == "__main__":
    main()
