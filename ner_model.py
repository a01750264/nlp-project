import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from accelerate import notebook_launcher

data = load_dataset('yelp_review_full')

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def tokenize(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_data = data.map(tokenize, batched=True)

small_train_dataset = tokenized_data['train'].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_data['test'].shuffle(seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=5)

training_args = TrainingArguments(output_dir='trainer_checkpoints', evaluation_strategy='epoch')
metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics
)

notebook_launcher(trainer.train())
