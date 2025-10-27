from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from config.secret import secret


HUGGINGFACE_TOKEN = secret.huggingface_token


def init_teacher_model(model_name : str = "facebook/nllb-200-distilled-1.3B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = HUGGINGFACE_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token = HUGGINGFACE_TOKEN)
    model.eval()
    
    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = init_teacher_model()

    text = """The dominant sequence transduction models are based on complex recurrent or
    convolutional neural networks that include an encoder and a decoder."""
    encoded = tokenizer(text, return_tensors = "pt", padding = True, truncation = True)
    generated = model.generate(**encoded, forced_bos_token_id = tokenizer.convert_tokens_to_ids("kor_Hang"))

    translated = tokenizer.decode(generated[0], skip_special_tokens = True)
    print(translated)


    translator = pipeline(
        "translation", 
        model="facebook/nllb-200-distilled-1.3B", 
        src_lang="eng_Latn", 
        tgt_lang="kor_Hang"
    )

    # Translate your text
    text_to_translate = "Hello, how are you?"
    result = translator(text_to_translate)
    print(result)