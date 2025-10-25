import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model_name = "Helsinki-NLP/opus-mt-small-en-ko"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cpu")
model.to(device)


def translate(text : str, max_len : int = 128) -> str:
    inputs = tokenizer(
        text, 
        return_tensors = "pt",
        padding = True,
        truncation = True,
        max_length = max_len
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length = max_len
        )
    print(outputs)

    return tokenizer.decode(outputs[0], skip_special_tokens = True)


if __name__ == "__main__":
    print(translate("Hello, world!"))