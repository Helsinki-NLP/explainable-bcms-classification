from transformers import pipeline
from datasets import Dataset
import torch

import pandas as pd
import sys, json

model_name = sys.argv[1]
test_file = sys.argv[2]
out_file = sys.argv[3]
multilabel = len(sys.argv) > 4 and sys.argv[4] == "-ml"

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-classification", model=model_name, device=device)
print("Pipe loaded on device", device)
print("Multilabel mode:", multilabel)

test_data = pd.read_csv(test_file, sep="\t", header=0, quoting=3)
test_data["text"] = test_data["text"].str.strip().astype(str)
sentences = list(test_data["text"])
if multilabel:
    gold_labels = [set([int(y) for y in x.split(",")]) for x in test_data["labels"].astype(str)]
    num_labels = len(set([x for gold_label_set in gold_labels for x in gold_label_set]))
else:
    gold_labels = list(test_data["labels"])
    num_labels = len(set(gold_labels))

tok_args = {'padding': True, 'truncation': True, 'max_length': pipe.tokenizer.model_max_length}

def data_iterator():
    for i, row in test_data.iterrows():
        yield row["text"]

print("Predict full sentences")
predictions = pipe(data_iterator(), **tok_args)
pred_labels_scores = [(int(x["label"].replace("LABEL_", "")), x["score"]) for x in predictions]
pred_labels = [x[0] for x in pred_labels_scores]
pred_scores = [x[1] for x in pred_labels_scores]

# no confusion matrix in multilabel settings
if not multilabel:
    conf_matrix = pd.crosstab(gold_labels, pred_labels, rownames=['Gold'], colnames=['Predicted'], margins=True)
    print()
    print(conf_matrix)
    print()

print("Predict leave-one-out for correct predictions")
diffs = {}
i = 0
for sentence, goldlabel, full_predlabel, full_predscore in zip(sentences, gold_labels, pred_labels, pred_scores):
    # multilabel:  goldlabel is a set => test for inclusion
    # singlelabel: goldlabel is an int => test for equality
    if (multilabel and full_predlabel in goldlabel) or (not multilabel and full_predlabel == goldlabel):
        prob_diffs = {}

        # difference to initial implementation: if a word occurs several times, remove all its occurrences at once
        # might need to do this on tokenized lowercased text in the future
        words = [x for x in sentence.split(" ") if x != ""]
        unique_words = set(words)

        # if there is only one word and we remove it, there is nothing left :D
        if len(unique_words) > 1:
            inputs = []
            leftout = []
            for w in unique_words:
                rest = " ".join([x for x in words if x != w])
                inputs.append(rest)
                leftout.append(w)
            outputs = pipe(inputs, **tok_args)
            
            for loo_word, loo_prediction in zip(leftout, outputs):
                loo_predlabel = int(loo_prediction["label"].replace("LABEL_", ""))
                loo_predscore = loo_prediction["score"]
                score_diff = full_predscore - loo_predscore
                # negative difference means that the left-out word is not at all prototypical for the label, so we just skip this
                if score_diff > 0:
                    prob_diffs[loo_word] = score_diff

            # Sort words by probability difference and select top five
            top_words = sorted(prob_diffs.items(), key=lambda x: x[1], reverse=True)[:5]
            for w, sc in top_words:
                if w not in diffs:
                    diffs[w] = [[] for x in range(num_labels)]
                diffs[w][full_predlabel].append(sc)

outfile = open(out_file, 'w')
for word in sorted(diffs):
    obj = {"word": word}
    for i in range(num_labels):
        obj[f"cl{i}"] = diffs[word][i]
    outfile.write(json.dumps(obj) + "\n")
outfile.close()
