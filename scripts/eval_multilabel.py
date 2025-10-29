from transformers import pipeline
import torch
import pandas as pd
import sys

model_name = sys.argv[1]    # individual checkpoint
test_file = sys.argv[2]     # file with multilabel annotation

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-classification", model=model_name, device=device)
# print("Pipe loaded on device", device)

test_data = pd.read_csv(test_file, sep="\t", header=0, quoting=3)
test_data["text"] = test_data["text"].str.strip().astype(str)
sentences = list(test_data["text"])
gold_labels = [set([int(y) for y in x.split(",")]) for x in test_data["labels"]]
# print(sentences[:5])
# print(gold_labels[:5])

tok_args = {'padding': True, 'truncation': True, 'max_length': pipe.tokenizer.model_max_length}
predictions = pipe(sentences, **tok_args)
pred_labels_scores = [(int(x["label"].replace("LABEL_", "")), x["score"]) for x in predictions]

total, correct = 0, 0
for pred, gold in zip(pred_labels_scores, gold_labels):
    # if total < 10:
    #     print(pred[0], gold)
    total += 1
    correct += (pred[0] in gold)
acc = correct / total * 100

print(f"Model:         {model_name}")
print(f"Total samples: {total}")
print(f"Accuracy:      {acc:.3f}%")
print()
