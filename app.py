import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load the models and tokenizers for selected languages
models = {
    "English to French": "Helsinki-NLP/opus-mt-en-fr",
    "English to Spanish": "Helsinki-NLP/opus-mt-en-es",
    "English to German": "Helsinki-NLP/opus-mt-en-de",
    "English to Italian": "Helsinki-NLP/opus-mt-en-it",
    # "English to Portuguese": "Helsinki-NLP/opus-mt-en-pt",
    "English to Dutch": "Helsinki-NLP/opus-mt-en-nl",
    "English to Russian": "Helsinki-NLP/opus-mt-en-ru",
    "English to Chinese (Simplified)": "Helsinki-NLP/opus-mt-en-zh",
    # "English to Japanese": "Helsinki-NLP/opus-mt-en-ja",
    "English to Arabic": "Helsinki-NLP/opus-mt-en-ar",
}

def load_model_and_tokenizer(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate(text, model, tokenizer):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    translated = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Streamlit app
st.title('🌐 Language Translator')
st.markdown("""
    This app allows you to translate text from English to multiple languages.
    Select the target language from the dropdown menu and enter the text you want to translate.
""")

# Sidebar for additional options
st.sidebar.header("Options")
st.sidebar.write("Select the target language and enter the text for translation.")

# Dropdown for language selection
selected_language = st.selectbox('Choose the target language:', list(models.keys()))

# Load the selected model and tokenizer
model_name = models[selected_language]
tokenizer, model = load_model_and_tokenizer(model_name)

text = st.text_area('Enter text to translate:')
if st.button('Translate'):
    if text:
        translation = translate(text, model, tokenizer)
        st.success('Translation:')
        st.markdown(f"<h2 style='color:red;'>{translation}</h2>", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to translate.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by [Aditya Srivastava](https://adi1816.github.io/AdiInYourHeart/)")
