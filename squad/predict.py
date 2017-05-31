from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import pandas as pd


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

    # Split data into train/test

    X_train, X_test = X[:50000], X[50000:]
    y_train, y_test = y[:50000], y[50000:]

    lr = LogisticRegression(class_weight="balanced", n_jobs=-1, C=0.8)
    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_test)

    y_pred_label = np.asarray(return_class_label(y_pred, lr))
    print(precision_score(y_test, y_pred_label, average='macro'))
    print(recall_score(y_test, y_pred_label, average='macro'))


if __name__ == "__main__":
    main()
