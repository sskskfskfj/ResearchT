from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from config.secret import secret


HUGGINGFACE_TOKEN = secret.huggingface_token


def init_model(model_name : str = "facebook/nllb-200-distilled-1.3B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = HUGGINGFACE_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token = HUGGINGFACE_TOKEN)
    
    return tokenizer, model


def train_model(model, tokenizer, tokenized_datasets, data_collator):
    training_args = Seq2SeqTrainingArguments(
        output_dir = "results",
        evaluation_strategy = "epoch",
        learning_rate = 2e-5,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
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

if __name__ == "__main__":
    tokenizer, model = init_model()
    train_dataset = get_dataset()
    tokenized_datasets, data_collator = preprocessed_dataset(train_dataset)
    
    train_model(model, tokenizer, tokenized_datasets, data_collator)