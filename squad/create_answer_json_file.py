import pandas as pd
import numpy as np
import json
from nltk.corpus import stopwords
import string
from nltk import word_tokenize
import heapq


def create_answer_dic(y_pred, table_test):
    id_list = table_test.question_id.unique()

    answers = {id: [] for id in id_list}
    begun, ended = False, False

    answer, alt_answer = "", ""
    prev_id = 0
    table_test.loc[:, "category_B"] = [pred[0] for pred in y_pred]
    table_test.loc[:, "category_E"] = [pred[1] for pred in y_pred]
    table_test.loc[:, "category_O"] = [pred[2] for pred in y_pred]

    for question_id in id_list:
        temp_df = table_test.loc[table_test.question_id == question_id]
        threshold = 1 / 3
        for index, row in temp_df.iterrows():
            if max(temp_df.category_B) < threshold:
                threshold = min(heapq.nlargest(3, temp_df.category_B))
            if row.category_B >= threshold:
                if len(answer) > 0:
                    answer += " "
                answer += row.word
                begun, ended = True, False

            elif (begun and row.category_O >= 0.5 and not ended) or row.category_E > threshold:
                if row.word != ",":
                    if len(answer) > 0:
                        answer += " "
                    answer += row.word
                answers[question_id].append(clean_answer(answer))
                answer = ""
                begun, ended = False, True

            elif index == temp_df.index[-1]:
                answers[question_id].append(clean_answer(answer))
                answer = ""
                begun, ended = False, True

    return answers


def clean_answer(ans_str):
    if len(ans_str) > 0:
        if ans_str[-1] == ",":
            ans_str = ans_str.replace(" ,", "")

        capitalized_stop = [word.title()
                            for word in set(stopwords.words('english'))]
        ans_str = " ".join([word for word in word_tokenize(
            ans_str) if word not in capitalized_stop])

    return ans_str


def create_question_answer_table_from_dictionary(answer_dic, table):
    df_answer = pd.DataFrame()

    for question_id in answer_dic.keys():
        temp_df = table.loc[table["question_id"] == question_id].iloc[0]

        correct_answers = temp_df.answers
        question = temp_df.question
        predicted_answers = answer_dic[question_id]

        df1 = pd.DataFrame([[question_id, question, correct_answers, predicted_answers]], columns=[
                           "id", "question", "answers", "predicted_answers"])
        df_answer = df_answer.append(df1, ignore_index=True)

    df_answer.to_csv("answer_0601.csv")


def create_json_file_from_dictionary(answer_dic):
    with open("answer_0601.json", "w+") as outfile:
        json.dump(answer_dic, outfile, indent=4)
