from transformers import (
    AutoTokenizer,  
    AutoModelForSeq2SeqLM,
    pipeline,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

import dotenv
import os
import json


dotenv.load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


def init_model(model_name : str = "dhtocks/nllb-200-distilled-350M_en-ko"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token = HUGGINGFACE_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = HUGGINGFACE_TOKEN)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = init_model()
    input_text = """The dominant sequence transduction models are based on complex recurrent or
                    convolutional neural networks that include an encoder and a decoder."""
    model_input = tokenizer(input_text, return_tensors = "pt")

    translated = model.generate(**model_input)
    print(tokenizer.decode(translated[0], skip_special_tokens = True))