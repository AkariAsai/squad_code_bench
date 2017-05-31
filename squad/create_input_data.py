import pandas as pd
import numpy as np
import corenlp
from nltk import ngrams
from text_similarity_calculator import TextSimilarityCalculator
from feature_creator import FeatureCreator
from create_answer_label import *
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


def create_new_columns_boolean_columns(df):
    for column in ["entity_matched", "verb_matched", "capital", "punctuation"]:
        swap_bool_value_to_int(column, df)
    creat_new_column_for_postage(df)


def creat_new_column_for_postage(df):
    pos_column_name = "pos"

    for word in set(df[pos_column_name].values):
        new_column_name = pos_column_name + '_' + word
        df.loc[df[pos_column_name] == word, new_column_name] = 1
        df.loc[df[pos_column_name] != word, new_column_name] = 0


def add_dummy_columns_qualitive_features(df):
    creat_new_column_for_entity(df, True)
    creat_new_column_for_entity(df, False)
    creat_new_column_for_postage(df)
    create_new_columns_boolean_columns(df)

    drop_columns = ["capital", "token_entity_type",
                    "answer_entity_type", "verb_matched", "entity_matched", "pos", "constituency_tag"]
    df.drop(drop_columns, inplace=True, axis=1)


def add_answer_label_input_data(original_df):
    for question_id in set(original_df["question_id"]):
        temp_df = original_df.loc[original_df["question_id"] == question_id]
        create_class_label_fixed(temp_df, original_df)


def main():
    df = pd.read_csv("df_dev_updated.csv", index_col=0)
    df = df.sample(frac=0.1, replace=True)
    corenlp_dir = "stanford-corenlp-full-2016-10-31/"
    parser = corenlp.StanfordCoreNLP(corenlp_path=corenlp_dir)

    # Create features for each word.
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

    # Add Answer label for input data
    add_answer_label_input_data(df_feature)

    df_feature = pd.read_csv("01sampling_Word_0530.csv")

    # Add dummy vals for categorial features.
    add_dummy_columns_qualitive_features(df_feature)
    print(df_feature.dtypes)

    df_feature.to_csv("01sampling_Word_0530.csv", index=False)


if __name__ == "__main__":
    main()
