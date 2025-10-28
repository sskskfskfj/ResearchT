from model.models import init_model
from data.en_ko_dataset import get_dataset
from transformers import (
    DataCollatorForSeq2Seq,   
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch


tokenizer, model = init_model()
train_dataset = get_dataset()


def preprocess_function(examples):
    features = tokenizer(
        examples["english"],
        max_length = 512,
        truncation = True,
        padding = "max_length",
    )
    labels = tokenizer(
        examples["korean"],
        max_length = 512,
        truncation = True,
        padding = "max_length",
    )   

    features["label"] = labels["input_ids"]
    return features


def train_model(model, tokenizer, tokenized_datasets, data_collator):
    training_args = Seq2SeqTrainingArguments(
        output_dir = "results",
        eval_strategy = "epoch",  # evaluation_strategy -> eval_strategy (transformers 최신 버전)
        learning_rate = 2e-5,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        num_train_epochs = 3,
        weight_decay = 0.01,
        save_total_limit = 2,
        predict_with_generate = True,
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
    tokenized_datasets = train_dataset.map(preprocess_function, batched = True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)

    return tokenized_datasets, data_collator


if __name__ == "__main__":
    tokenizer, model = init_model()
    train_dataset = get_dataset()
    tokenized_datasets, data_collator = preprocessed_dataset(train_dataset)
    train_model(model, tokenizer, tokenized_datasets, data_collator)



## OOV