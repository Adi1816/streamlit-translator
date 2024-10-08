import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Define the models for each language
models = {
    "fr": {
        "model": "Helsinki-NLP/opus-mt-en-fr",
        "tokenizer": "Helsinki-NLP/opus-mt-en-fr"
    },
    "es": {
        "model": "Helsinki-NLP/opus-mt-en-es",
        "tokenizer": "Helsinki-NLP/opus-mt-en-es"
    },
    "de": {
        "model": "Helsinki-NLP/opus-mt-en-de",
        "tokenizer": "Helsinki-NLP/opus-mt-en-de"
    },
    # Add more languages here if needed
}

# Lazy load the model only when needed
@st.cache_resource
def load_model(language_code):
    model_name = models[language_code]["model"]
    tokenizer_name = models[language_code]["tokenizer"]
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer

# Streamlit app layout
st.title("Multi-language Translator")

# Create a dropdown for language selection
language_code = st.selectbox(
    "Select the target language:",
    options=list(models.keys()),
    format_func=lambda x: {"fr": "French", "es": "Spanish", "de": "German"}[x]  # Adjust to match your language codes
)

# Text input for translation
text_to_translate = st.text_area("Enter text to translate")

# Button to trigger translation
if st.button("Translate"):
    if text_to_translate:
        # Load the selected model and tokenizer
        model, tokenizer = load_model(language_code)

        # Encode the input text and perform the translation
        inputs = tokenizer(text_to_translate, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**inputs)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        # Display the translated text
        st.write("**Translated Text:**")
        st.success(translated_text)
    else:
        st.error("Please enter some text to translate.")
