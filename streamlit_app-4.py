
import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
import nltk
from transformers import pipeline
import os
import re
import emoji
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Download NLTK 'punkt' data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Load the trained LSTM model
@st.cache_resource
def load_lstm_model():
    model = tf.keras.models.load_model('lstm_sentiment_topic_model.h5')
    return model

# Load the tokenizer object
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

# Load the sentiment_mapping dictionary
@st.cache_resource
def load_sentiment_mapping():
    with open('sentiment_mapping.pickle', 'rb') as handle:
        sentiment_mapping = pickle.load(handle)
    return sentiment_mapping

# Load the topic_mapping dictionary
@st.cache_resource
def load_topic_mapping():
    with open('topic_mapping.pickle', 'rb') as handle:
        topic_mapping = pickle.load(handle)
    return topic_mapping

# Initialize the IndoBERT sentiment pipeline
@st.cache_resource
def load_indobert_pipeline():
    model_name = 'mdhugol/indonesia-bert-sentiment-classification'
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
    return sentiment_pipeline

model = load_lstm_model()
tokenizer = load_tokenizer()
sentiment_mapping = load_sentiment_mapping()
topic_mapping = load_topic_mapping()
sentiment_pipeline = load_indobert_pipeline()

# Set maxlen from the training phase.
maxlen = 100

# Create reverse mappings for sentiment and topic labels
sentiment_mapping_reverse = {v: k for k, v in sentiment_mapping.items()}
topic_mapping_reverse = {v: k for k, v in topic_mapping.items()}

# --- Text Preprocessing Functions (Replicated from training phase) ---
# Ensure normalization_dict is explicitly defined for the standalone Streamlit app
normalization_dict = {
    'ae': 'saja','aja': 'saja','ajah': 'saja','aj': 'saja','jha': 'saja','sj': 'saja',
    'g': 'tidak','ga': 'tidak','gak': 'tidak','gk': 'tidak','kaga': 'tidak','kagak': 'tidak',
    'kg': 'tidak','ngga': 'tidak','Nggak': 'tidak','tdk': 'tidak','tak': 'tidak',
    'lgi': 'lagi','lg': 'lagi','donlod': 'download','pdhl': 'padahal','pdhal': 'padahal',
    'Coba2': 'coba-coba','tpi': 'tapi','tp': 'tapi','betmanfaat': 'bermanfaat',
    'gliran': 'giliran','kl': 'kalau','klo': 'kalau','gatau': 'tidak tau','bgt': 'banget',
    'hrs': 'harus','dll': 'dan lain-lain','dsb': 'dan sebagainya','trs': 'terus','trus': 'terus',
    'sangan': 'sangat','bs': 'bisa','bsa': 'bisa','gabisa': 'tidak bisa','gbsa': 'tidak bisa',
    'gada': 'tidak ada','gaada': 'tidak ada','gausah': 'tidak usah','bkn': 'bukan',
    'udh': 'sudah','udah': 'sudah','sdh': 'sudah','pertngahn': 'pertengahan',
    'ribet': 'ruwet','ribed': 'ruwet','sdangkan': 'sedangkan','lemot': 'lambat',
    'lag': 'lambat','ngelag': 'gangguan','yg': 'yang','dipakek': 'di pakai','pake': 'pakai',
    'kya': 'seperti','kyk': 'seperti','ngurus': 'mengurus','jls': 'jelas',
    'burik': 'buruk','payah':'buruk','krna': 'karena','dr': 'dari','smpe': 'sampai',
    'slalu': 'selalu','mulu': 'melulu','d': 'di','konek': 'terhubung','suruh': 'disuruh',
    'apk': 'aplikasi','app': 'aplikasi','apps': 'aplikasi','apl': 'aplikasi',
    'bapuk': 'jelek','bukak': 'buka','nyolong': 'mencuri','pas': 'ketika',
    'uodate': 'update','ato': 'atau','onlen': 'online','cmn': 'cuman','jele': 'jelek',
    'angel': 'susah','jg': 'juga','knp': 'kenapa','hbis': 'setelah','tololl': 'tolol','ny': 'nya',
    'skck':'skck','stnk':'stnk','sim':'sim','sp2hp':'sp2hp','propam':'propam','dumas':'dumas',
    'tilang':'tilang','e-tilang':'tilang','etilang':'tilang','surat kehilangan':'kehilangan'
}

def normalize_repeated_characters(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1", text)

def preprocess_text(text: str) -> str:
    text = str(text)
    text = normalize_repeated_characters(text)
    text = emoji.demojize(text)
    text = re.sub(r":[a-z_]+:", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\@\w+|#", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text)
    text = text.lower()
    for slang, standard in normalization_dict.items():
        text = re.sub(rf"\b{re.escape(slang.lower())}\b", standard.lower(), text)
    text = re.sub(r"\s+", " ").strip()
    return text

def predict_sentiment_and_topic(raw_text: str):
    cleaned_text = preprocess_text(raw_text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    lstm_sentiment_probs, lstm_topic_probs = model.predict(padded_sequence)
    lstm_sentiment_id = np.argmax(lstm_sentiment_probs, axis=1)[0]
    lstm_sentiment_label = sentiment_mapping_reverse.get(lstm_sentiment_id, "Unknown Sentiment")
    lstm_sentiment_score = float(lstm_sentiment_probs[0][lstm_sentiment_id])
    lstm_topic_id = np.argmax(lstm_topic_probs, axis=1)[0]
    lstm_topic_label = topic_mapping_reverse.get(lstm_topic_id, "Unknown Topic")
    lstm_topic_score = float(lstm_topic_probs[0][lstm_topic_id])

    indobert_result = sentiment_pipeline(cleaned_text)[0]
    indobert_sentiment_label = indobert_result['label']
    indobert_sentiment_score = float(indobert_result['score'])

    if indobert_sentiment_label == 'LABEL_0':
        indobert_sentiment_category = 'positive'
    elif indobert_sentiment_label == 'LABEL_1':
        indobert_sentiment_category = 'neutral'
    else:
        indobert_sentiment_category = 'negative'

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "lstm_sentiment": {
            "label": lstm_sentiment_label,
            "score": lstm_sentiment_score
        },
        "lstm_topic": {
            "label": lstm_topic_label,
            "score": lstm_topic_score
        },
        "indobert_sentiment": {
            "label": indobert_sentiment_category,
            "score": indobert_sentiment_score
        }
    }

# --- Streamlit App UI --- #
st.set_page_config(
    page_title="Analisis Sentimen & Topik Aplikasi",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Analisis Sentimen dan Topik Ulasan Aplikasi")
st.write("Aplikasi ini menganalisis sentimen dan topik dari ulasan pengguna menggunakan model LSTM multi-output dan pipeline IndoBERT.")

# Text input from user
user_input = st.text_area("Masukkan ulasan Anda di sini:", "Aplikasi ini sangat membantu dan cepat.", height=150)

if st.button("Analisis Ulasan"):
    if user_input:
        with st.spinner('Menganalisis ulasan...'):
            results = predict_sentiment_and_topic(user_input)

        st.subheader("Hasil Analisis")

        st.write(f"**Teks Asli:** {results['raw_text']}")
        st.write(f"**Teks Bersih (Preprocessed):** {results['cleaned_text']}")

        st.markdown("### Prediksi Model LSTM")
        st.info(f"**Sentimen (LSTM):** {results['lstm_sentiment']['label'].capitalize()} (Score: {results['lstm_sentiment']['score']:.4f})")
        st.info(f"**Topik (LSTM):** {results['lstm_topic']['label']} (Score: {results['lstm_topic']['score']:.4f})")

        st.markdown("### Prediksi IndoBERT Sentiment Pipeline")
        st.success(f"**Sentimen (IndoBERT):** {results['indobert_sentiment']['label'].capitalize()} (Score: {results['indobert_sentiment']['score']:.4f})")

    else:
        st.warning("Mohon masukkan teks ulasan untuk dianalisis.")

st.markdown("--- Jardana - Project Akhir MBKM --- ")
