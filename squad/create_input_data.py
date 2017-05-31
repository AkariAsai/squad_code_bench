import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import corenlp
from nltk import ngrams
from text_similarity_calculator import TextSimilarityCalculator
from feature_creator import FeatureCreator
from feature_creator_word import FeatureCreatorWord
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def creat_new_column_for_entity(df, is_token=True):
    if is_token:
        entity_column_name = "token_entity_type"
    else:
        entity_column_name = "answer_entity_type"

    for word in set(df[entity_column_name].values):
        new_column_name = entity_column_name + '_' + word
        df.loc[df[entity_column_name] == word, new_column_name] = 1
        df.loc[df[entity_column_name] != word, new_column_name] = 0


def swap_bool_value_to_int(column_name, df):
    new_column_name = column_name + '_bool'
    df.loc[df[column_name] == True, new_column_name] = 1
    df.loc[df[column_name] == False, new_column_name] = 0


def creat_new_column_for_postage(df):
    pos_column_name = "pos"

    for word in set(df[pos_column_name].values):
        new_column_name = pos_column_name + '_' + word
        df.loc[df[pos_column_name] == word, new_column_name] = 1
        df.loc[df[pos_column_name] != word, new_column_name] = 0


def create_new_columns_qualitive_date(df):
    for column in ["entity_matched", "verb_matched", "capital", "punctuation"]:
        swap_bool_value_to_int(column, df)
    creat_new_column_for_postage(df)


def flatten(nested_list):
    """2重のリストをフラットにする関数"""
    return [e for inner_list in nested_list for e in inner_list]


def create_class_label(df, original_df):
    answered = 0
    answers = df["answers"].iloc[0]

    if ", nan" in answers:
        answers = answers[:-6] + ']'
    if ", nan" in answers:
        answers = answers[:-6] + ']'
    answers = ast.literal_eval(answers)

    answer_word_list = flatten(
        [[word for word in answer.split(' ')] for answer in answers])
    for i in df.index:
        if df.loc[i, "word"] in [answer.split(' ')[0] for answer in answers]:
            original_df.loc[i, "category"] = "B"
            answered = 1
        elif df.loc[i, "word"] in [answer.split(' ')[-1] for answer in answers] and answered:
            original_df.loc[i, "category"] = "E"
        elif df.loc[i, "word"] in answer_word_list and answered:
            original_df.loc[i, "category"] = "M"
        else:
            original_df.loc[i, "category"] = "O"


def main():
    df = pd.read_csv("df_dev_updated.csv", index_col=0)
    df = df.sample(frac=0.1, replace=True)
    corenlp_dir = "stanford-corenlp-full-2016-10-31/"
    parser = corenlp.StanfordCoreNLP(corenlp_path=corenlp_dir)
    text_similarity_calculator = TextSimilarityCalculator(
        df["context"].values + df["question"].values)

    df_feature = pd.DataFrame()
    index = 0
    print(len(df))
    for id in df.id.values:
        print("{0} : {1}".format(index, id))
        creator = FeatureCreatorWord(id,  parser, text_similarity_calculator)
        df_id = creator.create_feature_vectors()
        df_feature = pd.concat([df_feature, df_id])
        index += 1

    df_feature.to_csv("01sampling_Word_0530.csv", index=False)


if __name__ == "__main__":
    main()
