import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Title of the app
st.title("Multilingual Translator")

# Supported languages with MarianMT Models (add/remove based on support)
language_pairs = {
    "English to Hindi": ("Helsinki-NLP/opus-mt-en-hi", "English", "Hindi"),
    "Hindi to English": ("Helsinki-NLP/opus-mt-hi-en", "Hindi", "English"),
    "English to French": ("Helsinki-NLP/opus-mt-en-fr", "English", "French"),
    "French to English": ("Helsinki-NLP/opus-mt-fr-en", "French", "English"),
    "English to German": ("Helsinki-NLP/opus-mt-en-de", "English", "German"),
    "German to English": ("Helsinki-NLP/opus-mt-de-en", "German", "English"),
    "English to Spanish": ("Helsinki-NLP/opus-mt-en-es", "English", "Spanish"),
    "Spanish to English": ("Helsinki-NLP/opus-mt-es-en", "Spanish", "English"),
    "English to Russian": ("Helsinki-NLP/opus-mt-en-ru", "English", "Russian"),
    "Russian to English": ("Helsinki-NLP/opus-mt-ru-en", "Russian", "English"),
    "English to Chinese": ("Helsinki-NLP/opus-mt-en-zh", "English", "Chinese"),
    "Chinese to English": ("Helsinki-NLP/opus-mt-zh-en", "Chinese", "English"),
    "English to Arabic": ("Helsinki-NLP/opus-mt-en-ar", "English", "Arabic"),
    "Arabic to English": ("Helsinki-NLP/opus-mt-ar-en", "Arabic", "English"),
}

# Sidebar for language selection
st.sidebar.header("Select Language Pair")
language_choice = st.sidebar.selectbox(
    "Choose translation direction", list(language_pairs.keys())
)

# Retrieve the model and tokenizer for the selected language pair
model_name, src_lang, tgt_lang = language_pairs[language_choice]
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Input box for the user to type in text
text_to_translate = st.text_area(f"Enter text in {src_lang}:")

# When the button is clicked
if st.button("Translate"):
    if text_to_translate:
        # Tokenize the input text
        translated = model.generate(**tokenizer(text_to_translate, return_tensors="pt", padding=True))
        translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        # Display the translated text
        st.text_area(f"Translated text in {tgt_lang}:", translated_text[0])
    else:
        st.warning("Please enter some text to translate!")

# Footer information
st.sidebar.write("Supported language pairs are provided by MarianMT models.")
st.sidebar.write("Powered by HuggingFace Transformers.")
