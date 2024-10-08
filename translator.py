from transformers import MarianMTModel, MarianTokenizer


def load_model_and_tokenizer(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


def translate(text, model, tokenizer):
    # Tokenize the text
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the translation
    translated = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)

    # Decode the output
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


if __name__ == "__main__":
    # Load model and tokenizer for English to French translation
    model_name = 'Helsinki-NLP/opus-mt-en-fr'
    tokenizer, model = load_model_and_tokenizer(model_name)

    # Get user input for translation
    english_text = input("Enter text to translate: ")
    french_translation = translate(english_text, model, tokenizer)
    print("Translation:", french_translation)
