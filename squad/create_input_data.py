import pandas as pd
import numpy as np
import corenlp
from nltk import ngrams
from text_similarity_calculator import TextSimilarityCalculator
from feature_creator import FeatureCreator
from create_answer_label import *
from feature_creator_word import FeatureCreatorWord
from corenlp_extractor import CoreNLPExtractor
import math


def creat_new_column_for_entity(df, is_token=True):
    if is_token:
        entity_column_name = "token_entity_type"
    else:
        entity_column_name = "answer_entity_type"

    for word in set(df[entity_column_name].values):
        new_column_name = entity_column_name + '_' + word
        df.loc[df[entity_column_name] == word, new_column_name] = 1
        df.loc[df[entity_column_name] != word, new_column_name] = 0


def create_new_column_for_dep_const(df, is_const=True):
    if is_const:
        entity_column_name = "const_tag"
    else:
        entity_column_name = "dep_tag"

    for word in set(df[entity_column_name].values):
        new_column_name = entity_column_name + '_' + word
        df.loc[df[entity_column_name] == word, new_column_name] = 1
        df.loc[df[entity_column_name] != word, new_column_name] = 0


def swap_bool_value_to_int(column_name, df):
    new_column_name = column_name + '_bool'
    df.loc[df[column_name] == True, new_column_name] = 1
    df.loc[df[column_name] == False, new_column_name] = 0


def create_new_columns_boolean_columns(df):
    for column in ["entity_matched", "verb_matched", "capital", "punctuation", "subj_match", "root_match", "obj_match", "nmod_match"]:
        swap_bool_value_to_int(column, df)
    creat_new_column_for_postage(df)


def creat_new_column_for_postage(df):
    pos_column_name = "pos"

    for word in set(df[pos_column_name].values):
        new_column_name = pos_column_name + '_' + word
        df.loc[df[pos_column_name] == word, new_column_name] = 1
        df.loc[df[pos_column_name] != word, new_column_name] = 0


def add_dummy_columns_qualitive_features(df):
    create_new_column_for_dep_const(df, True)
    create_new_column_for_dep_const(df, False)
    creat_new_column_for_entity(df, True)
    creat_new_column_for_entity(df, False)
    creat_new_column_for_postage(df)
    create_new_columns_boolean_columns(df)

    drop_columns = ["capital", "token_entity_type",
                    "answer_entity_type", "verb_matched", "entity_matched", "pos", "constituency_tag",
                    "subj_match", "root_match", "obj_match", "nmod_match", "const_tag", "dep_tag", "punctuation"]
    df.drop(drop_columns, inplace=True, axis=1)


def load_target_csv(id, df):
    questions = df.loc[df['id'] == id].iloc[0]
    return questions


def collect_answers(question_df):
    answer0 = question_df["answer_0_text"]
    answer1 = question_df["answer_1_text"]
    answer2 = question_df["answer_2_text"]

    return [answer for answer in [answer0, answer1, answer2] if answer is not np.nan]


def create_answer_dic(df, id_list):
    count = 0
    answer_dic = {}
    for id in id_list:
        count += 1
        question_df = load_target_csv(id, df)
        answer_dic[id] = collect_answers(question_df)
    print(answer_dic)
    return answer_dic


def add_answer_label_input_data(original_df, answers_dic):
    for question_id in set(original_df["question_id"]):
        temp_df = original_df.loc[original_df["question_id"] == question_id]
        create_class_label_fixed(temp_df, original_df,
                                 answers_dic[question_id])


def main():
    df = pd.read_csv("df_dev_updated.csv", index_col=0)
    df = df.sample(frac=0.1, replace=True)
    extractor = CoreNLPExtractor()
    # Create features for each word.
    text_similarity_calculator = TextSimilarityCalculator(
        df["context"].values + df["question"].values)

    answers_dic = {}

    df_feature = pd.DataFrame()
    index = 0
    for id in df.id.values:
        print("{0} : {1}".format(index, id))
        creator = FeatureCreatorWord(
            id,  extractor, text_similarity_calculator)
        answers_dic[id] = creator.answers
        df_id = creator.create_feature_vectors()
        df_feature = pd.concat([df_feature, df_id])
        index += 1
    df_feature.to_csv("01sampling_Word_0607.csv", index=False)

    # # Add Answer label for input data
    df_feature = pd.read_csv("01sampling_Word_0607.csv")
    df_columns = ['id', 'question', 'answer_0_text', 'answer_0_start', 'answer_1_text', 'answer_1_start',
                  'answer_2_text', 'answer_2_start', 'title', 'context', 'paragraph_idx',
                  'AnswerType', 'entities', 'NNPs', 'NNs', 'VBs', 'WP']
    df_answers = pd.read_csv("df_dev_updated.csv")
    df_answers = df_answers[df_columns]
    id_list = df_feature["question_id"]
    print(id_list)

    answers_dic = create_answer_dic(df_answers, list(set(id_list)))
    add_answer_label_input_data(df_feature, answers_dic)

    # Add dummy vals for categorial features.
    add_dummy_columns_qualitive_features(df_feature)

    df_feature.to_csv("01sampling_Word_0607.csv", index=False)

    print(df_feature.dtypes)


if __name__ == "__main__":
    main()
