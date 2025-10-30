from models import init_model
from en_ko_dataset import get_dataset
from transformers import (
    DataCollatorForSeq2Seq,   
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch


model, tokenizer = init_model()
train_dataset = get_dataset()


def train_model(model, tokenizer, tokenized_datasets, data_collator):
    training_args = Seq2SeqTrainingArguments(
        output_dir = "results",
        eval_strategy = "epoch",  # evaluation_strategy -> eval_strategy (transformers 최신 버전)
        learning_rate = 2e-5,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        num_train_epochs = 1,
        weight_decay = 0.01,
        save_total_limit = 2,
        predict_with_generate = True
    )

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_datasets["train"],
        eval_dataset = tokenized_datasets["test"],
        data_collator = data_collator,
    )

    trainer.train()


def preprocessed_dataset(tokenized_datasets):
    tokenized_datasets = train_dataset.map(preprocess_function, batched = True, num_proc = 4, batch_size = 16)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)

    return tokenized_datasets, data_collator


def preprocess_function(examples):
    features = tokenizer(
        examples["english"],
        max_length = 128,
        truncation = True,
        padding = "max_length",
    )
    labels = tokenizer(
        examples["korean"],
        max_length = 128,
        truncation = True,
        padding = "max_length",
    )   

    features["label"] = labels["input_ids"]
    return features


if __name__ == "__main__":
    model, tokenizer = init_model()
    train_dataset = get_dataset()
    tokenized_datasets, data_collator = preprocessed_dataset(train_dataset)

    train_dataset = tokenized_datasets.train_test_split(test_size = 0.2)
    
    print(train_dataset)
    train_model(model, tokenizer, train_dataset, data_collator)


# 주말에 ㄱ
## OOM