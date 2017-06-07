import ast
import pandas as pd
from nltk import word_tokenize


def calculate_words_cooccurence(answer, predicted):
    if len(predicted) == 0:
        return 0
    answer_tokenized = word_tokenize(answer)
    predicted_tokenize = word_tokenize(predicted)

    co_occur = 0
    for token in predicted_tokenize:
        if token in answer_tokenized:
            co_occur += 1

    return co_occur / len(answer_tokenized)


def eval_answer(answer_df):
    num_correct_answer = 0
    for index, row in answer_df.iterrows():
        answers = row.answers

        if ", nan" in answers:
            answers = answers[:-6] + ']'
        if ", nan" in answers:
            answers = answers[:-6] + ']'

        answers = ast.literal_eval(answers)
        predicted_answers = ast.literal_eval(row.predicted_answers)

        correct = False
        for answer in answers:
            co_occurence = [calculate_words_cooccurence(
                answer, predicted_answer) for predicted_answer in predicted_answers]
            if len(co_occurence) > 0 and max(co_occurence) > 0.8:
                correct = True
                break

        if correct == True:
            num_correct_answer += 1
            print(answers, predicted_answers, num_correct_answer)
    return num_correct_answer / len(answer_df.index)


def main():
    df = pd.read_csv("answer_0601.csv")
    print(eval_answer(df))


if __name__ == "__main__":
    main()
