import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import corenlp
import json
import pprint
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import svm
from answertype_predictor import AnswertypePredictor


def read_csv(title, df):
    questions = df.loc[df['title'] == title]
    return questions


def parse_by_corenlp(parser, sentence):
    result_json = json.loads(parser.parse(sentence))
    return result_json


def find_pos_entity(parser, sentence, stop):
    result_json = parse_by_corenlp(parser, sentence)
    result_sentence = [text[1:] for text in
                       result_json["sentences"][0]["parsetree"].split("] ")[:-1]]
    pos = {}
    entity = {}
    for j in range(len(result_sentence)):
        result_array = result_sentence[j].split(' ')

        for i in range(len(result_array)):
            text = result_array[0].split('=')[1]
            if text not in stop:
                tag = result_array[i].split('=')[0]
                if tag == 'PartOfSpeech' and result_array[i].split('=')[1] != '.':
                    pos[text] = result_array[i].split('=')[1]

                elif tag == 'NamedEntityTag' and result_array[i].split('=')[1] != 'O':
                    entity[text] = result_array[i].split('=')[1]
    return pos, entity

def extract_postaged_words(parser, sentence):
    from nltk.corpus import stopwords
    stop = set(stopwords.words('english'))
    pos, entity = find_pos_entity(parser, sentence, stop)
    nnps = [k for (k, v) in pos.items() if v == 'NNP']
    nns = [k for (k, v) in pos.items() if v == 'NN']
    vbs = [k for (k, v) in pos.items() if 'VB' in v]
    wp = [k for (k, v) in pos.items() if v in ["WDT", "WP", "WP$", "WRB"]]

    return entity, nnps, nns, vbs, wp


def collect_keywords(parser, questions):
    df_keywords = pd.DataFrame(
        columns=['entities', 'NNPs', 'Nouns', 'Verbs', 'WP'])

    for question in questions:
        entity, nnps, nns, vbs, wp = extract_postaged_words(parser, question)
        df_keywords = df_keywords.append({'entities': entity, 'NNPs': nnps, 'NNs': nns,
                                          'VBs': vbs, 'WP': wp}, ignore_index=True)

    return df_keywords


def main():
    corenlp_dir = "stanford-corenlp-full-2016-10-31/"
    parser = corenlp.StanfordCoreNLP(corenlp_path=corenlp_dir)
    df_dev = pd.read_csv("dev_v1.csv")

    predictor = AnswertypePredictor()

    df_dev["AnswerType"] = [predictor.predict_answer_type(
        question, predictor.vectorizer, predictor.svm) for question in df_dev["question"]]

    df_keywords = collect_keywords(parser, df_dev["question"])
    df_keywords.to_csv("keywords.csv")
    df_keywords = pd.read_csv("keywords.csv")

    df_dev["entities"] = df_keywords["entities"]
    df_dev["NNPs"] = df_keywords["NNPs"]
    df_dev["NNs"] = df_keywords["NNs"]
    df_dev["VBs"] = df_keywords["VBs"]
    df_dev["WP"] = df_keywords["WP"]

    df_dev.to_csv("df_dev_updated.csv")


if __name__ == '__main__':
    main()
