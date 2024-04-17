### KeenSight - Language Translation

This Streamlit app enables language translation using advanced language models. It leverages Mistral for general translation tasks and M2M100 for specific languages like Hindi, Chinese, and Japanese.

#### Installation

To run this app locally, follow these steps:

1. Install Streamlit:
   ```bash
   pip install streamlit
   ```

2. Install required dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Download the necessary models:
   - Mistral: `TheBloke/Mistral-7B-Instruct-v0.1-GGUF`
   - M2M100: `facebook/m2m100_418M`

#### Usage

1. Clone the repository or save the code to a local file.
2. Run the app using Streamlit:
   ```bash
   streamlit run app.py
   ```

#### Features

- **Language Translation**: Translate text between various languages including English, French, Spanish, German, Chinese, Japanese, Russian, Arabic, Hindi, and Romanian.
- **Advanced Models**: Utilizes Mistral for general translation and M2M100 for specific language pairs.
- **Interactive Interface**: Select source and target languages, input text, and click "Translate" to get the translated text.
- **Efficient Handling**: Handles different language models based on the selected languages for optimal translation results.

#### Requirements

- Python 3.6+
- Streamlit
- Torch
- Langchain
- CTransformers
- Transformers
- sentencepiece

#### Acknowledgements

- Mistral: [TheBloke/Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
- M2M100: [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M)

Feel free to modify and enhance this app according to your needs and preferences. Happy translating!