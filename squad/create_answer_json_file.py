import pandas as pd
import numpy as np
import json
from nltk.corpus import stopwords
import string
from nltk import word_tokenize


def create_answer_dic(y_pred, table_test):
    id_list = table_test.question_id.unique()

    answers = {id: [] for id in id_list}
    alt_answers = {id: [] for id in id_list}
    begun, ended = False, False
    alt_begun, alt_begun = False, False

    answer, alt_answer = "", ""
    prev_id = 0

    for pred, word, question_id in zip(y_pred, table_test.word, table_test.question_id):
        if pred.argmax(axis=0) == 0:
            if len(answer) > 0:
                answer += " "
            answer += word
            begun, ended = True, False

        elif (begun and pred[2] >= 0.5 and not ended) or pred.argmax(axis=0) == 1:
            if len(answer) > 0:
                answer += " "
            answer += word
            answers[question_id].append(clean_answer(answer))
            answer = ""
            begun, ended = False, True

        # set threshold lower for alternative answer in case there are no
        # "Begin the answer" label.
        elif pred.argmax(axis=0) != 0 and pred[0] > 0.30:
            if len(alt_answer) > 0:
                alt_answer += " "
            alt_answer += word
            alt_begun, alt_ended = True, False

        elif (alt_begun and pred[2] >= 0.5 and not alt_ended) or pred.argmax(axis=0) == 1:
            if len(alt_answer) > 0:
                alt_answer += " "
            alt_answer += word
            alt_answers[question_id].append(clean_answer(alt_answer))
            alt_answer = ""
            alt_begun, alt_ended = False, True

    return answers, alt_answers


def clean_answer(ans_str):
    if ans_str[-1] == ",":
        ans_str = ans_str.replace(" ,", "")

    capitalized_stop = [word.title()
                        for word in set(stopwords.words('english'))]
    ans_str = " ".join([word for word in word_tokenize(
        ans_str) if word not in capitalized_stop])

    return ans_str


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
    with open("answer_0601.json", "w+") as outfile:
        json.dump(answer_dic, outfile, indent=4)
