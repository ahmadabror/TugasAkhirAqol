import os
import re
import json
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import emoji
import nltk
import gensim
from gensim import corpora

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# =========================================================
# ENV safe for deploy
# =========================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_HOME"] = "/tmp/hf"
os.makedirs("/tmp/hf", exist_ok=True)

try:
    from transformers import pipeline
    TRANSFORMERS_IMPORT_ERROR = ""
except Exception as e:
    pipeline = None
    TRANSFORMERS_IMPORT_ERROR = str(e)

HF_MODEL_ID = "mdhugol/indonesia-bert-sentiment-classification"

NLTK_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DIR)


# =========================================================
# Constants (samakan dengan training)
# =========================================================
MAX_SEQUENCE_LENGTH = 100  # sesuai kode awalmu
STOPWORD_PATH = "stopwordbahasa.txt"

# File names sesuai repo kamu
TOKENIZER_JSON = "tokenizer.json"
LSTM_SENTIMENT_MODEL = "lstm_sentiment_model.h5"

LDA_DICTIONARY = "lda_dictionary.gensim"

# ini yang IDEAL (harap ada)
LDA_MODEL = "lda_model_deploy.gensim"
# repo kamu terlihat punya ini juga
LDA_MODEL_STATE = "lda_model_deploy.gensim.state"

topic_name_map_lda = {
    0: "Kemudahan Pengurusan SKCK & Manfaat Aplikasi Polri",
    1: "Efisiensi Pendaftaran Online & Bantuan",
    2: "Isu Teknis, Error & Kendala Penggunaan Aplikasi",
    3: "Kepuasan Layanan Aplikasi & Akses Cepat",
}

sentiment_labels_map = {"LABEL_0": "positive", "LABEL_1": "neutral", "LABEL_2": "negative"}


# =========================================================
# NLTK ensure (punkt)
# =========================================================
def ensure_nltk_punkt() -> Tuple[bool, str]:
    try:
        nltk.data.find("tokenizers/punkt")
        return True, "OK"
    except LookupError:
        nltk.download("punkt", download_dir=NLTK_DIR, quiet=True, raise_on_error=False)
        try:
            nltk.data.find("tokenizers/punkt")
            return True, "OK"
        except LookupError as e:
            return False, str(e)


# =========================================================
# Stopwords, stemmer, normalization
# =========================================================
stop_factory = StopWordRemoverFactory()
stemmer = StemmerFactory().create_stemmer()

normalization_dict = {
    "ae": "saja", "aja": "saja", "ajah": "saja", "aj": "saja", "jha": "saja", "sj": "saja",
    "g": "tidak", "ga": "tidak", "gak": "tidak", "gk": "tidak", "kaga": "tidak", "kagak": "tidak",
    "kg": "tidak", "ngga": "tidak", "nggak": "tidak", "tdk": "tidak", "tak": "tidak",
    "lgi": "lagi", "lg": "lagi", "donlod": "download", "pdhl": "padahal", "pdhal": "padahal",
    "tpi": "tapi", "tp": "tapi",
    "gliran": "giliran", "kl": "kalau", "klo": "kalau", "gatau": "tidak tau", "bgt": "banget",
    "hrs": "harus", "dll": "dan lain-lain", "dsb": "dan sebagainya", "trs": "terus", "trus": "terus",
    "sangan": "sangat", "bs": "bisa", "bsa": "bisa", "gabisa": "tidak bisa", "gbsa": "tidak bisa",
    "gada": "tidak ada", "gaada": "tidak ada", "gausah": "tidak usah", "bkn": "bukan",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah",
    "ribet": "ruwet", "ribed": "ruwet", "sdangkan": "sedangkan", "lemot": "lambat",
    "ngelag": "gangguan", "yg": "yang", "dipakek": "di pakai", "pake": "pakai",
    "kya": "seperti", "kyk": "seperti", "ngurus": "mengurus", "jls": "jelas",
    "burik": "buruk", "payah": "buruk", "krna": "karena", "dr": "dari", "smpe": "sampai",
    "slalu": "selalu", "mulu": "melulu", "d": "di", "konek": "terhubung", "suruh": "disuruh",
    "apk": "aplikasi", "app": "aplikasi", "apps": "aplikasi", "apl": "aplikasi",
    "bapuk": "jelek", "bukak": "buka",
    "uodate": "update", "ato": "atau", "onlen": "online", "cmn": "cuman", "jele": "jelek",
    "angel": "susah", "jg": "juga", "knp": "kenapa", "hbis": "setelah", "ny": "nya",
    "skck": "skck", "stnk": "stnk", "sim": "sim", "sp2hp": "sp2hp", "propam": "propam", "dumas": "dumas",
    "tilang": "tilang", "e-tilang": "tilang", "etilang": "tilang", "surat kehilangan": "kehilangan",
}

more_stopword = ["dengan", "ia", "bahwa", "oleh", "nya", "dana"]


def build_stopwords() -> set:
    additional = []
    if os.path.exists(STOPWORD_PATH):
        with open(STOPWORD_PATH, "r", encoding="utf-8") as f:
            additional = [line.strip() for line in f if line.strip()]
    sw = set(stop_factory.get_stop_words())
    sw.update(more_stopword)
    sw.update(additional)
    return sw


# =========================================================
# Preprocess
# =========================================================
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

    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text_lda(cleaned_text: str, stop_words: set) -> str:
    t = stemmer.stem(cleaned_text)
    tokens = nltk.tokenize.word_tokenize(t)
    tokens = [x for x in tokens if x not in stop_words and len(x) > 2]
    return " ".join(tokens)


# =========================================================
# Load resources (cached)
# =========================================================
@st.cache_resource
def get_resources(use_indobert: bool):
    ok, msg = ensure_nltk_punkt()
    if not ok:
        raise RuntimeError(f"NLTK punkt belum siap: {msg}")

    stop_words = build_stopwords()

    # --- Tokenizer (JSON) ---
    if not os.path.exists(TOKENIZER_JSON):
        raise FileNotFoundError(f"File tidak ditemukan: {TOKENIZER_JSON}")
    with open(TOKENIZER_JSON, "r", encoding="utf-8") as f:
        tok = tokenizer_from_json(f.read())

    # --- LSTM Sentiment model ---
    if not os.path.exists(LSTM_SENTIMENT_MODEL):
        raise FileNotFoundError(f"File tidak ditemukan: {LSTM_SENTIMENT_MODEL}")
    lstm_model = load_model(LSTM_SENTIMENT_MODEL)

    # --- LDA assets ---
    lda_model = None
    lda_dict = None

    if os.path.exists(LDA_DICTIONARY):
        lda_dict = corpora.Dictionary.load(LDA_DICTIONARY)

    # Penting: harus ada file utama LDA_MODEL, bukan hanya .state
    if os.path.exists(LDA_MODEL):
        lda_model = gensim.models.LdaMulticore.load(LDA_MODEL)
    else:
        # kasih info kalau cuma .state yang ada
        if os.path.exists(LDA_MODEL_STATE):
            # tetap None, tapi jelaskan di UI nanti
            lda_model = None

    # --- IndoBERT optional ---
    indobert_pipe = None
    indobert_error = ""
    if use_indobert:
        if pipeline is None:
            indobert_error = f"Transformers pipeline gagal import: {TRANSFORMERS_IMPORT_ERROR}"
        else:
            try:
                indobert_pipe = pipeline(
                    "sentiment-analysis",
                    model=HF_MODEL_ID,
                    tokenizer=HF_MODEL_ID,
                    framework="pt",
                    device=-1,
                )
            except Exception as e:
                indobert_error = str(e)
                indobert_pipe = None

    return {
        "stop_words": stop_words,
        "tokenizer": tok,
        "lstm_model": lstm_model,
        "lda_model": lda_model,
        "lda_dictionary": lda_dict,
        "indobert_pipe": indobert_pipe,
        "indobert_error": indobert_error,
    }


# =========================================================
# Predict functions
# =========================================================
def predict_sentiment_lstm(cleaned_text: str, tok, lstm_model) -> str:
    seq = tok.texts_to_sequences([cleaned_text])
    pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    probs = lstm_model.predict(pad, verbose=0)

    # asumsi output softmax 3 kelas: [neg, neu, pos] (sesuaikan kalau urutanmu beda)
    idx = int(np.argmax(probs, axis=1)[0])
    class_names = ["negative", "neutral", "positive"]
    return class_names[idx] if idx < len(class_names) else "unknown"

def predict_topic_lda(cleaned_text: str, stop_words: set, lda_model, lda_dictionary) -> str:
    if lda_model is None or lda_dictionary is None:
        return "LDA tidak aktif (model/dictionary tidak lengkap)"

    text_lda = preprocess_text_lda(cleaned_text, stop_words)
    if not text_lda.strip():
        return "Tidak ada kata relevan setelah preprocessing"

    bow = lda_dictionary.doc2bow(text_lda.split())
    if not bow:
        return "Tidak ada kata relevan untuk LDA (BOW kosong)"

    dist = lda_model.get_document_topics(bow)
    if not dist:
        return "Topik tidak ditemukan"

    tid = max(dist, key=lambda x: x[1])[0]
    return topic_name_map_lda.get(tid, f"Unknown Topic {tid}")

def predict_sentiment_indobert(cleaned_text: str, indobert_pipe) -> Optional[str]:
    if indobert_pipe is None:
        return None
    if not cleaned_text.strip():
        return "neutral"
    out = indobert_pipe(cleaned_text)[0]
    label = out.get("label", "")
    return sentiment_labels_map.get(label, "neutral")


def nrs_percent(pos: int, neg: int, total: int) -> float:
    return ((pos - neg) / total * 100.0) if total else 0.0

def build_nrs_table(df: pd.DataFrame, topic_col="Topik LDA", sent_col="Sentimen") -> pd.DataFrame:
    tmp = df.copy()
    tmp[sent_col] = tmp[sent_col].fillna("neutral").astype(str)

    def bucket(s: str) -> str:
        s = s.lower()
        if "pos" in s: return "positive"
        if "neg" in s: return "negative"
        return "neutral"

    tmp["_bucket"] = tmp[sent_col].apply(bucket)
    grp = tmp.groupby(topic_col)["_bucket"].value_counts().unstack(fill_value=0)

    for c in ["positive", "neutral", "negative"]:
        if c not in grp.columns:
            grp[c] = 0

    grp["total"] = grp["positive"] + grp["neutral"] + grp["negative"]
    grp["NRS_%"] = grp.apply(lambda r: round(nrs_percent(int(r["positive"]), int(r["negative"]), int(r["total"])), 2), axis=1)
    grp = grp.reset_index().sort_values("NRS_%", ascending=False)
    return grp[[topic_col, "positive", "neutral", "negative", "total", "NRS_%"]]


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Analisis Sentimen & Topik (Polri Presisi)", layout="wide")

st.title("Dashboard Analisis Sentimen & Topik Ulasan Aplikasi Polri Presisi")
st.caption("Menggunakan LSTM (sentimen), LDA (topik), dan IndoBERT opsional (sentimen).")

tab_single, tab_upload, tab_diag = st.tabs(["📝 Analisis 1 Ulasan", "📄 Upload CSV/Excel", "🛠 Diagnostik"])


with tab_diag:
    st.subheader("Diagnostik File Repo")
    required = [TOKENIZER_JSON, LSTM_SENTIMENT_MODEL, LDA_DICTIONARY, STOPWORD_PATH]
    lda_required = [LDA_MODEL]  # ideal

    col1, col2 = st.columns(2)
    with col1:
        st.caption("File Penting")
        for f in required:
            st.write(("✅" if os.path.exists(f) else "❌"), f)

    with col2:
        st.caption("File LDA Model")
        st.write(("✅" if os.path.exists(LDA_MODEL) else "❌"), LDA_MODEL)
        st.write(("✅" if os.path.exists(LDA_MODEL_STATE) else "⚠️"), LDA_MODEL_STATE)
        if (not os.path.exists(LDA_MODEL)) and os.path.exists(LDA_MODEL_STATE):
            st.warning("Ada file .state tapi file utama LDA (.gensim) tidak ada → LDA tidak bisa diload.")

    st.divider()
    st.caption("Transformers (IndoBERT)")
    if pipeline is None:
        st.warning("Transformers pipeline gagal import → IndoBERT akan nonaktif.")
        st.code(TRANSFORMERS_IMPORT_ERROR)
    else:
        st.success("Transformers pipeline import OK.")

    ok, msg = ensure_nltk_punkt()
    st.caption("NLTK punkt")
    if ok:
        st.success("NLTK punkt OK.")
    else:
        st.error("NLTK punkt belum siap.")
        st.code(msg)


with tab_single:
    st.subheader("Analisis 1 Ulasan")
    use_indobert = st.checkbox("Gunakan IndoBERT (opsional)", value=True)

    user_text = st.text_area("Masukkan ulasan:", "Aplikasi ini membantu, tetapi kadang error.", height=140)

    if st.button("Analisis", type="primary"):
        res = get_resources(use_indobert=use_indobert)

        cleaned = preprocess_text(user_text)
        sent_lstm = predict_sentiment_lstm(cleaned, res["tokenizer"], res["lstm_model"])
        topic_lda = predict_topic_lda(cleaned, res["stop_words"], res["lda_model"], res["lda_dictionary"])

        sent_ib = predict_sentiment_indobert(cleaned, res["indobert_pipe"]) if use_indobert else None
        if sent_ib is None:
            sent_ib = "neutral"

        st.markdown("### Hasil")
        st.write(f"**Teks bersih:** {cleaned}")
        st.write(f"**Sentimen (LSTM):** {sent_lstm}")
        st.write(f"**Topik (LDA):** {topic_lda}")
        st.write(f"**Sentimen (IndoBERT):** {sent_ib}")

        if use_indobert and res["indobert_pipe"] is None and res["indobert_error"]:
            st.warning("IndoBERT gagal diaktifkan.")
            st.code(res["indobert_error"])


with tab_upload:
    st.subheader("Upload CSV/Excel → Output + NRS per Topik")
    use_indobert2 = st.checkbox("Gunakan IndoBERT untuk sentimen (disarankan)", value=True, key="use_indobert2")
    max_rows = st.number_input("Maks baris (hindari timeout)", 1, 200000, 1000)

    uploaded = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])
    if uploaded:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            xls = pd.ExcelFile(uploaded)
            sheet = st.selectbox("Pilih sheet:", xls.sheet_names)
            df = pd.read_excel(uploaded, sheet_name=sheet)

        st.caption("Preview")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        text_col = st.selectbox("Pilih kolom teks ulasan:", df.columns)

        if st.button("Generate Output", type="primary", key="gen_output"):
            res = get_resources(use_indobert=use_indobert2)

            work = df.copy().head(int(max_rows))
            work[text_col] = work[text_col].fillna("").astype(str)

            out_rows = []
            with st.spinner("Memproses..."):
                for review in work[text_col].tolist():
                    cleaned = preprocess_text(review)

                    # topik
                    topic_lda = predict_topic_lda(cleaned, res["stop_words"], res["lda_model"], res["lda_dictionary"])

                    # sentimen (prioritas IndoBERT jika aktif; kalau tidak, pakai LSTM)
                    if use_indobert2:
                        sent = predict_sentiment_indobert(cleaned, res["indobert_pipe"])
                        if sent is None:
                            sent = predict_sentiment_lstm(cleaned, res["tokenizer"], res["lstm_model"])
                    else:
                        sent = predict_sentiment_lstm(cleaned, res["tokenizer"], res["lstm_model"])

                    out_rows.append({
                        "Ulasan": review,
                        "Cleaned": cleaned,
                        "Topik LDA": topic_lda,
                        "Sentimen": sent,
                    })

            out_df = pd.DataFrame(out_rows)
            out_df["Score"] = out_df["Sentimen"].map(lambda s: 1 if s == "positive" else (-1 if s == "negative" else 0))

            st.divider()
            st.subheader("Output (Preview)")
            st.dataframe(out_df.head(50), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("NRS per Topik")
            nrs_df = build_nrs_table(out_df, topic_col="Topik LDA", sent_col="Sentimen")
            st.dataframe(nrs_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download Output (CSV)",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="output_ulasan.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download NRS per Topik (CSV)",
                data=nrs_df.to_csv(index=False).encode("utf-8"),
                file_name="nrs_per_topik.csv",
                mime="text/csv",
            )
