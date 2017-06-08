import pandas as pd
import numpy as np
import nltk
from nltk import ngrams
from nltk import word_tokenize
import operator
import ast
from nltk.corpus import stopwords


def calculate_words_cooccurence(answer_tokenized, gram):
    co_occur = 0
    total_length = len(gram)
    for token in answer_tokenized:
        if token in gram:
            co_occur += 1

    return co_occur / total_length


def add_answer_label(answer, words):
    stop = set(stopwords.words('english'))
    answer_removed_stop = [
        token for token in word_tokenize(answer) if token not in stop]

    len_answer = len(answer_removed_stop)
    grams = ngrams(words, len_answer)
    score_dic = {}
    index = 0

    for gram in grams:
        score_dic[index] = calculate_words_cooccurence(
            answer_removed_stop, gram)
        index += 1

    # in case the len of words is smaller than the len of answer
    if len(words) < len_answer:
        labels = ['O' for word in words]
        labels[0] = "B"
        labels[-1] = "E"
        return labels

    answer_gram_index = max(score_dic, key=score_dic.get)

    labels = ['O' for word in words]
    begun, ended = False, False

    for i in range(len(words)):
        if i == answer_gram_index:
            labels[i] = 'B'
            begun = True
        elif len_answer > 1 and i == answer_gram_index + len(word_tokenize(answer)) - 1 and begun:
            labels[i] = 'E'
            ended = True
        # elif begun and not ended:
        #     labels[i] = 'M'

    return labels


def create_class_label_fixed(df, original_df, answers):
    answer_labels = add_answer_label(answers[0], list(df.word))

    # Also considering the 2nd and 3rd answer.
    for i in range(len(answers) - 1):
        print(df.word)
        print(answers)
        print(len(answers))
        alt_answer_labels = add_answer_label(answers[i + 1], list(df.word))

        for i in range(len(alt_answer_labels)):
            if answer_labels[i] == "O" and alt_answer_labels[i] != "O":
                answer_labels[i] = alt_answer_labels[i]

    index = 0
    for i in df.index:
        original_df.loc[i, "category"] = answer_labels[index]
        index += 1


def main():
    sentence = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the golden anniversary with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as Super Bowl L, so that the logo could prominently feature the Arabic numerals 50."
    words = word_tokenize(sentence)
    answer = "Denver Broncos"

    for word, label in zip(words, add_answer_label(answer, words)):
        if label is not "O":
            print(label, word)


if __name__ == "__main__":
    main()
