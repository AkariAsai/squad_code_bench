import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


class TextSimilarityCalculator():
    def __init__(self, X):
        self.X = X
        self.vectorizer, self.terms = self.create_tfidf_vectorizer()

    def create_tfidf_vectorizer(self):
        vectorizer = TfidfVectorizer(stop_words='english', norm="l2")
        vectorizer.fit_transform(self.X)
        terms = vectorizer.get_feature_names()
        return vectorizer, terms

    def vectorize_sentence(self, sentence):
        tfidf_sentence = self.vectorizer.transform([sentence])
        return tfidf_sentence

    def calculate_cos_similarity(self, sentence_a, sentence_b):
        tfidf_sentence_a = np.array(
            self.vectorize_sentence(sentence_a).toarray())
        tfidf_sentence_b = np.array(
            self.vectorize_sentence(sentence_b).toarray())

        dot = np.dot(tfidf_sentence_a[0], tfidf_sentence_b[0])
        a_norm, b_norm = np.linalg.norm(
            tfidf_sentence_a), np.linalg.norm(tfidf_sentence_b)

        return dot / (a_norm * b_norm)


def main():
    a = "I played football with Tom"
    b = "I played the piano with Tom"
    c = "He is a pretty good piano player"

    calculator = TextSimilarityCalculator([a, b, c])

    print(calculator.calculate_cos_similarity(a, b))
    print(calculator.calculate_cos_similarity(c, b))
    print(calculator.calculate_cos_similarity(a, c))


if __name__ == "__main__":
    main()
