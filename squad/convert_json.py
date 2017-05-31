import csv
import json

f = open("train-v1.1.json")
x = json.load(f)
f.close()

with open('train_v1.csv', 'w') as csvfile:
  fieldnames = ["id", "question", "answer_0_text", "answer_0_start",
                "answer_1_text", "answer_1_start",
                "answer_2_text", "answer_2_start",
                "title", "context", "paragraph_idx"]
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
  writer.writeheader()

  for wiki in x["data"]:
    paragraph_index = 0
    for paragraph in wiki["paragraphs"]:
      for qa in paragraph["qas"]:
        answer_num = len(qa["answers"])

        qa_answer_0_text = qa["answers"][0]["text"] if answer_num > 0 else None
        qa_answer_0_start = qa["answers"][0]["answer_start"] if answer_num > 0 else None
        qa_answer_1_text = qa["answers"][1]["text"] if answer_num > 1 else None
        qa_answer_1_start = qa["answers"][1]["answer_start"] if answer_num > 1 else None
        qa_answer_2_text = qa["answers"][2]["text"] if answer_num > 2 else None
        qa_answer_2_start = qa["answers"][2]["answer_start"] if answer_num > 2 else None

        writer.writerow({"id": qa["id"], "question": qa["question"],
                         "answer_0_text": qa_answer_0_text, "answer_0_start": qa_answer_0_start,
                         "answer_1_text": qa_answer_1_text, "answer_1_start": qa_answer_1_start,
                         "answer_2_text": qa_answer_2_text, "answer_2_start": qa_answer_2_start,
                         "title": wiki["title"], "context": paragraph["context"], "paragraph_idx": paragraph_index})
      paragraph_index += 1
