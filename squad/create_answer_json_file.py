import pandas as pd
import numpy as np
import json


def create_answer_dic(y_pred, table_test):
    id_list = table_test.question_id.unique()
    answers = {id: [] for id in id_list}
    alt_answers = {id: [] for id in id_list}
    print(len(table_test))

    for pred, word, question_id in zip(y_pred, table_test.word, table_test.question_id):
        if pred.argmax(axis=0) == 0:
            answers[question_id].append(word)

        # set threshold lower for alternative answer in case there are no
        # "Begin the answer" label.
        elif pred.argmax(axis=0) != 0 and pred[0] > 0.30:
            alt_answers[question_id].append(word)

    return answers, alt_answers


def create_question_answer_table_from_dictionary(answer_dic, alt_answer_dic, table):
    df_answer = pd.DataFrame()

    for question_id in answer_dic.keys():
        temp_df = table.loc[table["question_id"] == question_id].iloc[0]

        correct_answers = temp_df.answers
        question = temp_df.question
        predicted_answers = answer_dic[question_id]
        alt_answers = alt_answer_dic[question_id]

        df1 = pd.DataFrame([[question_id, question, correct_answers, predicted_answers, alt_answers]], columns=[
                           "id", "question", "answers", "predicted_answers", "alternative_answers"])
        df_answer = df_answer.append(df1, ignore_index=True)

    df_answer.to_csv("answer_0601.csv")


def create_json_file_from_dictionary(answer_dic):
    with open("answer.json", "w+") as outfile:
        json.dump(answer_dic, outfile, indent=4)
