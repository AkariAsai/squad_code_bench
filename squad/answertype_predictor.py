# This is module to predict answer tyoe for questions.
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


class AnswertypePredictor():
    def __init__(self):
        self.annotate_df = self.load_annotate_df()
        self.X = self.annotate_df["question"].values
        self.y = self.annotate_df["TYPE"].values
        self.X_train = self.X[:180]
        self.X_test = self.X[180:]
        self.y_train = self.y[:180]
        self.y_test = self.y[180:]

        self.vectorizer, self.svm = self.create_SVM_models(self.X, self.y)

    def load_annotate_df(self):
        df_qd = pd.read_csv("qd_ml_experiment.csv")
        df_qd = df_qd.reindex(np.random.permutation(df_qd.index))
        return df_qd

    def create_RFC_models(self, X_train, y_train):
        vectorizer = TfidfVectorizer(stop_words='english', norm="l2")
        X_train_tfidf = vectorizer.fit_transform(X_train)
        terms = vectorizer.get_feature_names()

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=0, class_weight='balanced')
        clf.fit(X_train_tfidf, y_train)

        return vectorizer, clf

    def create_SVM_models(self, X_train, y_train):
        vectorizer = TfidfVectorizer(stop_words='english', norm="l2")
        X_train_tfidf = vectorizer.fit_transform(X_train)
        terms = vectorizer.get_feature_names()

        clf = svm.SVC(C=1.5, kernel='linear')
        clf.fit(X_train_tfidf, y_train)

        return vectorizer, clf

    def rule_base_predict(self, sentence):
        tokens = nltk.word_tokenize(sentence)

        if tokens[0] == 'Who':
            return 'HUMAN'
        elif tokens[0] == 'When':
            return 'NUMERIC'
        elif tokens[0] == 'Where':
            return 'LOCATION'

        first_two_tokens = ' '.join(tokens[:2])
        if first_two_tokens == 'How many' or first_two_tokens == "How much" or first_two_tokens == "What day" or first_two_tokens == "What year":
            return 'NUMERIC'

        return 'NONE'

    def predict_answer_type(self, question, vectorizer, clf):
        question_tfidf = vectorizer.transform([question])
        clf_predict = clf.predict(question_tfidf)[0]
        if self.rule_base_predict(question) != "NONE":
            clf_predict = self.rule_base_predict(question)

        return clf_predict


def main():
    predictor = AnswertypePredictor()
    print(predictor.predict_answer_type(
        "Who founded AFL?", predictor.vectorizer, predictor.svm))
    print(predictor.predict_answer_type(
        "How did Luther's tutors advise him to test what he learned?", predictor.vectorizer, predictor.svm))


if __name__ == "__main__":
    main()
