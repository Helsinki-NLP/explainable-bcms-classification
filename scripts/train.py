import sys, json, os, pathlib, shutil
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="HuggingFace model ID")
parser.add_argument("-train", help="path to csv training file")
parser.add_argument("-valid", help="path to csv validation file")
parser.add_argument("-outdir", help="path to directory for saved model files")
parser.add_argument("-instance_col", help="name of column containing instances")
parser.add_argument("-label_col", help="name of column containing labels")
parser.add_argument("-shuffle", action="store_true")
parser.add_argument("-epochs", type=int, default=10)
args = parser.parse_args()

train_df = pd.read_csv(args.train, sep="\t", header=0, quoting=3)
train_df["text"] = train_df[args.instance_col].str.strip()
labels = sorted(train_df[args.label_col].unique())
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
print(id2label)
print(label2id)
train_df["label"] = train_df[args.label_col].map(label2id)
train_df = train_df[['label', 'text']]
train_df = train_df.dropna()
train_ds = Dataset.from_pandas(train_df, split="train")
if args.shuffle:
	train_ds = train_ds.shuffle(seed=123)
print("train", len(train_ds))

valid_df = pd.read_csv(args.valid, sep="\t", header=0, quoting=3)
valid_df["text"] = valid_df[args.instance_col].str.strip()
valid_df["label"] = valid_df[args.label_col].map(label2id)
valid_df = valid_df[['label', 'text']]
valid_df = valid_df.dropna()
valid_ds = Dataset.from_pandas(valid_df, split="test")
if args.shuffle:
	valid_ds = valid_ds.shuffle(seed=123)
print("valid", len(valid_ds))

dataset = DatasetDict()
dataset['train'] = train_ds
dataset['test'] = valid_ds

model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(labels))
tokenizer = AutoTokenizer.from_pretrained(args.model)
# make sure model_max_length is set to a reasonable value
if tokenizer.model_max_length > model.config.max_position_embeddings:
	tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=model.config.max_position_embeddings)
# try to avoid saving errors with fine-tuned Bertic
if "bertic" in args.model:
	for param in model.parameters():
		param.data = param.data.contiguous()

def tokenize_function(instances):
	return tokenizer(instances["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
	output_dir=args.outdir,
	learning_rate=2e-5,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	num_train_epochs=args.epochs,
	weight_decay=0.01,
	eval_strategy="epoch",
	save_strategy="epoch",
	report_to="none",
	load_best_model_at_end=True,
	metric_for_best_model="accuracy"
)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
	logits, labels = eval_pred
	predictions = np.argmax(logits, axis=-1)
	print("gold", labels[:20])
	print("pred", predictions[:20])
	return accuracy_metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

#with open(f"{args.outdir}/labels.json", "w") as labelfile:
#	json.dump(label2id, labelfile)

print("Select best checkpoint")
checkpoints = os.listdir(args.outdir)
checkpoints = [x for x in checkpoints if x.startswith("checkpoint-")]
best_model_checkpoint = ""
best_metric = 0
for checkpoint in checkpoints:
	if "trainer_state.json" in os.listdir(args.outdir + "/" + checkpoint):
		state = json.load(open(args.outdir + "/" + checkpoint + "/trainer_state.json", "r"))
		if state["best_metric"] > best_metric:
			best_metric = state["best_metric"]
			best_model_checkpoint = state["best_model_checkpoint"].split("/")[-1]
print("Best checkpoint:", best_model_checkpoint, best_metric)
fd = os.open(args.outdir, os.O_RDONLY)
os.symlink(best_model_checkpoint, "best", dir_fd=fd)

for checkpoint in checkpoints:
	if checkpoint != best_model_checkpoint:
		path = pathlib.Path(args.outdir + "/" + checkpoint)
		shutil.rmtree(path)