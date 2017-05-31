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


class FeatureCreatorWord():
    def __init__(self, id, parser, text_similarity_calculator):
        self.parser = parser
        self.id = id
        self.question_df = self.load_target_csv(id)
        self.question = self.question_df["question"]
        self.answers = self. collect_answers()
        self.sentences = self.split_paragraph_into_sentences(
            self.question_df["context"])
        self.stop = set(stopwords.words('english'))
        self.text_similarity_calculator = text_similarity_calculator

    def load_target_csv(self, id):
        df_columns = ['id', 'question', 'answer_0_text', 'answer_0_start', 'answer_1_text', 'answer_1_start',
                      'answer_2_text', 'answer_2_start', 'title', 'context', 'paragraph_idx',
                      'AnswerType', 'entities', 'NNPs', 'NNs', 'VBs', 'WP']
        df = pd.read_csv("df_dev_updated.csv")
        df = df[df_columns]
        questions = df.loc[df['id'] == id].iloc[0]
        return questions

    def parse_by_corenlp(self, sentence):
        result_json = json.loads(self.parser.parse(sentence))
        return result_json

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

    def find_pos_entity(self, sentence):
        result_json = self.parse_by_corenlp(sentence)
        if len(result_json["sentences"]):
            result_sentence = [text[1:] for text in
                               result_json["sentences"][0]["parsetree"].split("] ")[:-1]]

            pos = {}
            entity = {}

            for j in range(len(result_sentence)):
                result_array = result_sentence[j].split(' ')

                for i in range(len(result_array)):
                    text = result_array[0].split('=')[1]
                    # if text not in self.stop:
                    tag = result_array[i].split('=')[0]
                    if tag == 'PartOfSpeech' and result_array[i].split('=')[1] != '.':
                        pos[text] = result_array[i].split('=')[1]

                    elif tag == 'NamedEntityTag' and result_array[i].split('=')[1] != 'O':
                        entity[text] = result_array[i].split('=')[1]
            return pos, entity
        return {}, []

    def with_punctuation(self, token):
        characters = token
        for c in characters:
            if c in string.punctuation:
                return True
        return False

    def create_feature_vectors(self):
        df_features = pd.DataFrame(columns=[
            "token", "question_id", "question", "answers",
            "shared_word_num", "num_of_keywords", "num_of_eneities",  "num_of_nnps", "num_of_nns",
            "answer_entity_type", "token_entity_type", "entity_matched",
            "min_distance_key", "verb_matched", "constituency_tag",
            "pos", "capital", "punctuation", "text_similarity"])

        q_info = self.question_df
        q_words = word_tokenize(self.question)
        entities = literal_eval(q_info["entities"]).keys()
        nnps = literal_eval(q_info["NNPs"])
        for sentence in self.sentences:
            s_words = sentence.split(" ")
            # ignore senteces which doesn't contain any single words with
            # the questions.
            if len(sentence) > 0 and len([word for word in s_words if word in q_words]) > 1:
                pos, entity = self.find_pos_entity(sentence)
                tokens = word_tokenize(sentence)
                # print(type(pos), entity)

                # The features of the selected sentence.
                shared_word_num = len(
                    [word for word in s_words if word in q_words])
                num_of_eneities = len(
                    [word for word in s_words if word in entities])
                num_of_nnps = len(
                    [word for word in s_words if word in literal_eval(q_info["NNPs"])])
                num_of_nns = len(
                    [word for word in s_words if word in literal_eval(q_info["NNs"])])
                verb_matched = self.isverbmatched(
                    literal_eval(q_info["VBs"]), s_words)
                text_similarity = self.text_similarity_calculator.calculate_cos_similarity(
                    self.question, sentence)

                for token in tokens:
                    # print(token)
                    # collect distance from
                    if token not in self.stop:
                        min_distance_key = min(self.dis_from_keyword(token, nnps, sentence), self.dis_from_keyword(
                            token, entities, sentence))
                        entity_type = "None"

                        if token in entity.keys():
                            entity_type = entity[token]
                        entity_matched = True if self.entity_match(
                            q_info["AnswerType"], entity_type) else False

                        pos_tag = pos[token.lower()] \
                            if token.lower() in pos.keys() else "SS"

                        df_features = df_features.append({"word": token, "question_id": self.id, "question": self.question, "answers": self.answers, "shared_word_num": shared_word_num,
                                                          "num_of_keywords": num_of_eneities + num_of_nnps, "num_of_eneities": num_of_eneities, "num_of_nnps": num_of_nnps, "num_of_nns": num_of_nns,
                                                          "answer_entity_type": q_info["AnswerType"], "token_entity_type": entity_type, "entity_matched": entity_matched,
                                                          "min_distance_key": min_distance_key, "verb_matched": verb_matched,
                                                          "pos": pos_tag, "capital": token.isupper(), "punctuation": self.with_punctuation(token),
                                                          "text_similarity": text_similarity}, ignore_index=True)
        return df_features


def main():
    # num of shared words shoukd be added. lematizer would be work>
    # rhw words postag sould be addede.
    # distance from the keywords should be dfixed.

    corenlp_dir = "stanford-corenlp-full-2016-10-31/"
    parser = corenlp.StanfordCoreNLP(corenlp_path=corenlp_dir)
    creator = FeatureCreatorWord("572fe41e04bcaa1900d76e4c", parser)
    df = creator.create_feature_vectors()
    df.to_csv("df_feature_experiment.csv", index=False)


if __name__ == "__main__":
    main()
