import streamlit as st
import torch
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, M2M100ForConditionalGeneration

if torch.cuda.is_available():
  generator = torch.Generator('cuda').manual_seed(0)
else:
  generator = torch.Generator().manual_seed(0)
  
# Mistral configuration
mistral_llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=20)

# M2m model configuration
m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
m2m_tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")

def translate_text(input_text, source_lang_code, target_lang_code):
    # Tokenize the input text
    model_inputs = m2m_tokenizer(input_text, return_tensors="pt", source_lang=source_lang_code, target_lang=target_lang_code)

    # Translate to the target language
    gen_tokens = m2m_model.generate(
        **model_inputs,
        forced_bos_token_id=m2m_tokenizer.get_lang_id(target_lang_code)
    )

    # Decode and return the translated text
    return m2m_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

def get_key_from_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    # If the target value is not found, return None or raise an exception
    return None  # You can modify this behavior as needed

# Page layout and styling
st.set_page_config(
    page_title="KeenSight - Language Translation",
)

# Streamlit app
st.image("logo.png", width=200)
st.title("Language Translation App")

# Sidebar for language selection
st.sidebar.header("Select Languages")

# Convert language codes to full names for display
top_languages = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "ro": "Romanian"
}

full_names = {code: name for code, name in top_languages.items()}
source_lang = st.sidebar.selectbox("Select Source Language", list(full_names.values()))
target_lang = st.sidebar.selectbox("Select Target Language", list(full_names.values()))

# Check if the selected source and target languages are one of "Hindi", "Chinese", or "Japanese"
use_other_model_source = source_lang in ["Hindi", "Chinese", "Japanese"]
use_other_model_target = target_lang in ["Hindi", "Chinese", "Japanese"]

# Text input for translation
input_text = st.text_area("Enter Text to Translate")

# Translate button
if st.button("Translate"):
    if input_text:
        if use_other_model_source or use_other_model_target:
            print("M2M")
            print(source_lang)
            print(target_lang)
            # Translate using the M2m model
            source_lang_code = get_key_from_value(top_languages,source_lang)
            target_lang_code = get_key_from_value(top_languages,target_lang)
            
            translated_text = translate_text(input_text, source_lang_code, target_lang_code)
        else:
            # Translate using Mistral
            print("General")
            map_template = f"<s>[INST] Translate the following text from {source_lang} to {target_lang}: {input_text} [/INST] </s>"
            map_prompt = PromptTemplate.from_template(map_template)
            translated_text = mistral_llm(map_template)
        
        st.success(f"Translated Text: {translated_text}")
    else:
        st.warning("Please enter text to translate.")