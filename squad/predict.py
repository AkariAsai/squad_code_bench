from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import pandas as pd
from create_answer_json_file import *


def create_X_y_data(df):
    columns = list(set(df) - set(['answers', 'question', 'question_id',
                                  'token', 'word', 'category', 'Unnamed: 0', 'Unnamed: 0.1']))
    index_columns = ['answers', 'question', 'question_id', 'word']

    X = df[columns]
    table = df[index_columns]
    y = df["category"]

    X = X.replace(np.nan, '0')

    return X, y, table


def return_class_label(y_pred, lr):
    y_pred_label = []
    classes = lr.classes_
    for i in range(len(y_pred)):
        y_pred_label.append(classes[np.argmax(y_pred[i])])
    return y_pred_label


def main():
    df = pd.read_csv("01sampling_Word_0530.csv")
    print(len(df.index))
    X, y, table = create_X_y_data(df)

    X = StandardScaler().fit_transform(X)

    # Split data into train/test. TODO: fix train.test split part.
    X_train, X_test = X[:50000], X[50000:]
    y_train, y_test = y[:50000], y[50000:]
    table_train, table_test = table[:40000], table[50000:]

    lr = LogisticRegression(class_weight="balanced", n_jobs=-1, C=1)
    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_test)

    y_pred_label = np.asarray(return_class_label(y_pred, lr))
    print(precision_score(y_test, y_pred_label, average='macro'))
    print(recall_score(y_test, y_pred_label, average='macro'))

    df_result = pd.DataFrame()
    df_train = pd.DataFrame()

    # save category prediction result to csv file.
    for word0, proba0, category0, answer0 in zip(table_test.word, y_pred, y_pred_label, table_test.answers):
        df2 = pd.DataFrame([[word0, proba0, answer0, category0]], columns=[
                           "word", "proba", "answer", "category"])
        df_result = df_result.append(df2, ignore_index=True)

    df_result.to_csv("result_0601.csv")

    # save final result to json/csv files.
    answers, alt_answers = create_answer_dic(y_pred, table_test)
    create_question_answer_table_from_dictionary(
        answers, alt_answers, table_test)
    create_json_file_from_dictionary(answers)


if __name__ == "__main__":
    main()
