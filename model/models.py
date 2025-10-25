from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config.secret import secret


HUGGINGFACE_TOKEN = secret.huggingface_token


def init_teacher_model(model_name : str = "facebook/nllb-200-distilled-1.3B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = HUGGINGFACE_TOKEN, src_lang = "eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token = HUGGINGFACE_TOKEN)
    model.eval()
    
    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = init_teacher_model()

    encoded = tokenizer("I love machine learning", return_tensors = "pt", padding = True, truncation = True)
    generated = model.generate(**encoded, forced_bos_token_id = tokenizer.convert_tokens_to_ids("kor_Hang"))

    translated = tokenizer.decode(generated[0], skip_special_tokens = True)
    print(translated)