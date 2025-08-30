# --- Text Processing and Tokenization Imports ---
import streamlit as st
import nltk
import re
import numpy as np
import joblib
from fast_langdetect import detect
import dill
from nltk.corpus import stopwords
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Konlpy can be tricky on some systems, so we handle its import gracefully
try:
    from konlpy.tag import Okt, Mecab
except ImportError:
    print("Warning: konlpy not found. Korean tokenization will not be available.")
    Okt, Mecab = None, None

# --- ML/DL Model Imports ---
import tensorflow as tf
from tensorflow import keras
from sklearn.exceptions import NotFittedError


# ==============================================================================
# SETUP: ONE-TIME DOWNLOADS AND INITIALIZATIONS
# ==============================================================================
st.set_page_config(page_title="Multilingual Text Classifier", layout="wide", initial_sidebar_state="expanded")
# Use Streamlit's caching to avoid re-downloading and re-initializing on every run.
@st.cache_resource
@st.cache_resource
def initialize_resources():
    """Downloads NLTK data and sets up tokenizers."""
    print("--- Initializing Tokenization Resources ---")
    try:
        # 📦 Download required NLTK resources
        print("Downloading NLTK 'punkt' model...")
        nltk.download('wordnet')
        nltk.download('punkt-tab')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('punkt')
        print("Downloading NLTK 'stopwords' model...")
        nltk.download('stopwords')
        print("NLTK downloads complete.")

        # Get English stopwords
        eng_stop_words = set(stopwords.words('english'))
        # Load Bengali stopwords from NLTK
        bengali_stopwords = set(stopwords.words('bengali'))

        # 🛑 Define Hindi stopwords manually
        hindi_stopwords = set([
            "और", "के", "है", "यह", "था", "जो", "पर", "को",
            "में", "से", "भी", "थे", "तक", "लेकिन"
        ])

        # 🔤 Set up Hindi text normalizer
        factory = IndicNormalizerFactory()
        normalizer = factory.get_normalizer("hi")

        mecab_tokenizer = None
        if Mecab: # Only try if konlpy was imported
            try:
                # Try to initialize Mecab or Okt
                mecab_tokenizer = Mecab()
                print("Successfully initialized Mecab for Korean tokenization.")
            except Exception as konlpy_error:
                # If it fails, print a warning and continue without it.
                print(f"Warning: konlpy initialization failed: {konlpy_error}")
                print("Korean tokenization will use a basic fallback.")
                mecab_tokenizer = None # Ensure it is None

        print("--- Initialization Complete ---\n")
        return eng_stop_words, bengali_stopwords, hindi_stopwords, normalizer, mecab_tokenizer

    except Exception as e:
        # This outer block now catches other, more critical errors.
        st.error(f"A critical error occurred during resource initialization: {e}")
        return None, None, None, None, None

eng_stop_words, bengali_stopwords, hindi_stopwords, normalizer, mecab = initialize_resources()


# ==============================================================================
# ALL TOKENIZATION FUNCTIONS
# ==============================================================================

def tokenize_eng(text):
    tokens = nltk.word_tokenize(text)
    print(f"tokens: {tokens}")
    print (f"stop: {eng_stop_words}")
    return [word for word in tokens if word not in eng_stop_words]


def tokenize_hin(text):
    text = normalizer.normalize(text)
    tokens = list(indic_tokenize.trivial_tokenize(text, lang='hi'))
    return [t for t in tokens if t not in hindi_stopwords]


def tokenize_ben(text):
    tokens = list(indic_tokenize.trivial_tokenize(text, lang='bn'))
    return [t for t in tokens if t not in bengali_stopwords]


def tokenize_ara(text):
    return text.split()


def tokenize_ger(text):
    return nltk.word_tokenize(text, language='german')


def tokenize_rus(text):
    return nltk.word_tokenize(text, language='russian')


def tokenize_fre(text):
    return nltk.word_tokenize(text, language='french')


def tokenize_ita(text):
    return nltk.word_tokenize(text, language='italian')


def tokenize_spa(text):
    return nltk.word_tokenize(text, language='spanish')


def tokenize_kor(text):
    if mecab:
        return mecab.morphs(text)
    else:
        # Fallback if konlpy is not installed
        return text.split()


def tokenize_tur(text):
    return nltk.word_tokenize(text, language='turkish')


def tokenize_chi(text):
    text = re.sub(r" ", "", text).strip()
    return list(jieba.cut(text))


def tokenize_por(text):
    return nltk.word_tokenize(text, language='portuguese')


def tokenize_ind(text):
    return nltk.word_tokenize(text)


# ===== Mapping Dictionary =====
label_descriptions = {
    'ara_bully': '🇸🇦 Arabic 💢 Bully Speech',
    'ara_nonbully': '🇸🇦 Arabic 🤝 Non-Bully Speech',
    'ben_neutral': '🇧🇩 Bengali 😐 Neutral Speech',
    'ben_political': '🇧🇩 Bengali 🗳️ Political Speech',
    'ben_sexual': '🇧🇩 Bengali 🔞 Sexual Speech',
    'ben_threat': '🇧🇩 Bengali ⚠️ Threat Speech',
    'ben_troll': '🇧🇩 Bengali 🧌 Troll Speech',
    'chi_bully': '🇨🇳 Chinese 💢 Bully Speech',
    'chi_nonbully': '🇨🇳 Chinese 🤝 Non-Bully Speech',
    'eng_neutral': '🇬🇧 English 😐 Neutral Speech',
    'eng_race': '🇬🇧 English 🧬 Race-related Speech',
    'eng_religion': '🇬🇧 English 🛐 Religion-related Speech',
    'eng_sex': '🇬🇧 English 🔞 Sex-related Speech',
    'fre_bully': '🇫🇷 French 💢 Bully Speech',
    'fre_nonbully': '🇫🇷 French 🤝 Non-Bully Speech',
    'ger_bully': '🇩🇪 German 💢 Bully Speech',
    'ger_nonbully': '🇩🇪 German 🤝 Non-Bully Speech',
    'hin_defame': '🇮🇳 Hindi 🗣️ Defamation',
    'hin_fake': '🇮🇳 Hindi 📰 Fake News',
    'hin_hate': '🇮🇳 Hindi 💢 Hate Speech',
    'hin_neutral': '🇮🇳 Hindi 😐 Neutral Speech',
    'hin_offense': '🇮🇳 Hindi 🚫 Offensive Speech',
    'ind_bully': '🇮🇩 Indonesian 💢 Bully Speech',
    'ind_nonbully': '🇮🇩 Indonesian 🤝 Non-Bully Speech',
    'ita_bully': '🇮🇹 Italian 💢 Bully Speech',
    'ita_nonbully': '🇮🇹 Italian 🤝 Non-Bully Speech',
    'kor_bully': '🇰🇷 Korean 💢 Bully Speech',
    'kor_nonbully': '🇰🇷 Korean 🤝 Non-Bully Speech',
    'por_bully': '🇵🇹 Portuguese 💢 Bully Speech',
    'por_nonbully': '🇵🇹 Portuguese 🤝 Non-Bully Speech',
    'rus_bully': '🇷🇺 Russian 💢 Bully Speech',
    'rus_nonbully': '🇷🇺 Russian 🤝 Non-Bully Speech',
    'spa_bully': '🇪🇸 Spanish 💢 Bully Speech',
    'spa_nonbully': '🇪🇸 Spanish 🤝 Non-Bully Speech',
    'tur_bully': '🇹🇷 Turkish 💢 Bully Speech',
    'tur_nonbully': '🇹🇷 Turkish 🤝 Non-Bully Speech'
}

lang_descriptions = {
    'ara': '🇸🇦 Arabic',
    'ben': '🇧🇩 Bengali',
    'chi': '🇨🇳 Chinese',
    'eng': '🇬🇧 English',
    'fre': '🇫🇷 French',
    'ger': '🇩🇪 German',
    'hin': '🇮🇳 Hindi',
    'ind': '🇮🇩 Indonesian',
    'ita': '🇮🇹 Italian',
    'kor': '🇰🇷 Korean',
    'por': '🇵🇹 Portuguese',
    'rus': '🇷🇺 Russian',
    'spa': '🇪🇸 Spanish',
    'tur': '🇹🇷 Turkish'
}

tokenizers = {
    'eng': tokenize_eng, 'hin': tokenize_hin, 'ben': tokenize_ben,
    'ara': tokenize_ara, 'ger': tokenize_ger, 'rus': tokenize_rus,
    'fre': tokenize_fre, 'ita': tokenize_ita, 'spa': tokenize_spa,
    'kor': tokenize_kor, 'tur': tokenize_tur, 'chi': tokenize_chi,
    'por': tokenize_por, 'ind': tokenize_ind
}

langdetect_to_tokenizer = {
    'en': 'eng', 'fr': 'fre', 'hi': 'hin', 'tr': 'tur', 'bn': 'ben',
    'ko': 'kor', 'zh': 'chi', 'zh-cn': 'chi', 'it': 'ita', 'es': 'spa',
    'id': 'ind', 'de': 'ger', 'pt': 'por', 'ru': 'rus', 'ar': 'ara'
}

# Define supported tokenizer codes
SUPPORTED_LANGUAGES = set(langdetect_to_tokenizer.values())


# ==============================================================================
# PREPROCESSING AND PREDICTION
# ==============================================================================
def preprocess_text(text):
    """Basic cleaning before language-specific tokenization."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.replace("“", "").replace("”", "")
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Load Models and Preprocessing Objects ---
@st.cache_resource
def load_models_and_vectorizers():
    """Loads all ML/DL models and vectorizers from disk."""
    try:
        print("--- Loading Models and Vectorizers ---")
        models = {}
        models['svm'] = joblib.load('svm_finetuned_model.joblib')
        models['nb'] = joblib.load('nb_finetuned_model.joblib')
        models['cnn'] = keras.models.load_model('cnn_text_classifier.h5')

        with open("cnn_tokenizer.json", "r", encoding="utf-8") as f:
            cnn_tokenizer = tokenizer_from_json(f.read())
            MAXLEN = 100  # TRAINED USING THIS VALUE

        with open('tfidf.dill', 'rb') as f:
            tfidf_vectorizer = dill.load(f)

        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = joblib.load("label_encoder.pkl")

        print("--- Loading Complete ---\n")
        return models, tfidf_vectorizer, label_encoder, cnn_tokenizer, MAXLEN

    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Make sure all model and tokenizer files are in the correct directory.")
        return None, None, None, None, None
    except NotFittedError:
        st.error("\n[Error] The loaded TF-IDF vectorizer has not been fitted.")
        st.error("Please run the script to fit and save your vectorizer on your training data first.\n")
        return None, None, None, None, None


models, tfidf_vectorizer, label_encoder, cnn_tokenizer, MAXLEN = load_models_and_vectorizers()


# --- Prediction Functions ---

def predict_svm_nb_cnn(model, text_input, vectorizer, encoder, cnn_tokenizer=None, MAXLEN=None, is_cnn=False):
    """
    Makes a prediction for SVM, NB, or CNN models.
    Takes a pre-tokenized, space-joined string.
    """
    if is_cnn:
        seq = cnn_tokenizer.texts_to_sequences([text_input])
        pad = pad_sequences(seq, maxlen=MAXLEN, padding='post')
        probabilities = model.predict(pad, verbose=0)[0]
        confidence = np.max(probabilities)
        predicted_class_idx = np.argmax(probabilities)
        label = encoder.classes_[predicted_class_idx]
        return label, confidence
    else:
        # For SVM and Naive Bayes
        text_vectorized = vectorizer.transform([text_input])
        try:
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = np.max(probabilities)
            predicted_class_idx = np.argmax(probabilities)
            label = encoder.classes_[predicted_class_idx]
            return label, confidence
        except AttributeError:
            st.warning("Warning: .predict_proba() not available for the SVM model. Confidence score will be N/A.")
            st.warning("To fix this, retrain your SVM with the parameter `probability=True`.")
            label_idx = model.predict(text_vectorized)[0]
            label = encoder.classes_[label_idx]
            return label, None


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def format_prediction(label, confidence):
    """Returns formatted string with description and confidence."""
    description = label_descriptions.get(label, "Unknown Label")
    conf_str = f"({confidence * 100:.2f}%)" if confidence is not None else "(Confidence N/A)"
    return f"{description} {conf_str}"


def warn_if_mismatch(model_name, predicted_lang, detected_lang_code):
    if predicted_lang != detected_lang_code:
        detected_lang_name = lang_descriptions.get(f"{detected_lang_code}", detected_lang_code).strip()
        predicted_lang_name = lang_descriptions.get(f"{predicted_lang}", predicted_lang).strip()
        st.warning(
            f"⚠️ **Language Mismatch ({model_name}):** Detected language is **{detected_lang_name}**, but the model predicted a **{predicted_lang_name}** label.")


def extract_language_code(label: str) -> str:
    """Extracts the first three-letter language code from a label string."""
    return label.split('_')[0] if label else "Unknown"


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.sidebar.title("🤔 Multi-Lingual Cyberbully Detection")
st.sidebar.markdown("""
Reclaim your online peace. This app is your personal shield against digital toxicity, 
intelligently detecting and flagging harmful comments before they disrupt your space. 
Supporting multiple languages, it helps you ensure your online interactions are positive and respectful.
""")

# --- Model Selection ---
AVAILABLE_MODELS = {
    "svm": "Support Vector Machine",
    "nb": "Naive Bayes",
    "cnn": "CNN (Convolutional Neural Network)"
}

st.sidebar.subheader("Model Selection")
selected_models_display = st.sidebar.multiselect(
    "Choose which models to run:",
    options=list(AVAILABLE_MODELS.values()),
    default=list(AVAILABLE_MODELS.values())
)

# --- START OF FOOTER CODE ---
footer_html = """
   <style>
   .footer {
       position: fixed;
       left: 0;
       bottom: 0;
       width: 100%;
       background-color: rgb(38, 39, 48);
       color: white ;
       text-align: center;
       padding: 10px;
       border-left: 1px solid black;
   }
   </style>
   <div class="footer">
       <p>Copyright © 2025 Skye, Jin and Virgin</p>
   </div>
   """
st.markdown(footer_html, unsafe_allow_html=True)

with st.sidebar.expander("View supported languages"):
    # Get the list of languages from your dictionary
    languages = list(lang_descriptions.values())

    # Create 2 columns for a nice layout
    col1, col2 = st.columns(2)

    with col1:
        for lang in languages[0:7]:
            st.write("🌍" + lang[2:])
    with col2:
        for lang in languages[7:]:
            st.write("🌍" + lang[2:])

    st.write("Note: Unsupported languages may be processed using defaults")
# --------------------
# Map display names back to keys for logic
selected_model_keys = [key for key, value in AVAILABLE_MODELS.items() if value in selected_models_display]

st.image("dataset-cover.jpg", width=500)
st.title("🤖UnHateAI - Hate Text Detection and Classification")

user_sentence = st.text_area("Enter a sentence for classification:", "This is a test sentence.", height=150)

if st.button("🚀 Classify Sentence"):
    # --- Input Validations ---
    if not user_sentence or user_sentence.strip() == "":
        st.warning("Please enter a sentence to classify.")
    elif not selected_model_keys:
        st.warning("Please select at least one model from the sidebar.")
    elif models is None:  # Check if models loaded correctly
        st.error("Models are not loaded. Cannot perform classification. Check file paths and logs.")
    else:
        with st.spinner('Analyzing text...'):
            # --- Preprocessing Steps ---
            detected_lang_code = 'ara'  # Default
            with st.expander("📝 View Preprocessing Steps", expanded=False):
                cleaned_sentence = preprocess_text(user_sentence)
                st.info(f"**1. Cleaned Text:** `{cleaned_sentence}`")

                try:
                    detected_lang_info = detect(cleaned_sentence)
                    detected_lang = detected_lang_info['lang']
                    detected_lang_code = langdetect_to_tokenizer.get(detected_lang, 'ara')
                    if detected_lang_code not in SUPPORTED_LANGUAGES:
                        st.warning(
                            f"Detected language '{detected_lang}' is unsupported. Defaulting to basic tokenizer.")
                        detected_lang_code = 'ara'
                    st.info(f"**2. Language Detected:** '{detected_lang}' (using '{detected_lang_code}' tokenizer)")
                except Exception as e:
                    detected_lang_code = 'ara'
                    st.warning(f"**2. Language Detection Failed:** {e}. Defaulting to basic tokenizer.")

                tokenizer_func = tokenizers.get(detected_lang_code, tokenizers['ara'])
                tokens = tokenizer_func(cleaned_sentence)
                tokens = [tok for tok in tokens if tok and tok.strip()]
                st.info(f"**3. Tokenization:** `{tokens}`")

            # ensuring essential variables are available for prediction
            cleaned_sentence = preprocess_text(user_sentence)
            tokenizer_func = tokenizers.get(detected_lang_code, tokenizers['ara'])
            tokens = tokenizer_func(cleaned_sentence)
            tokens = [tok for tok in tokens if tok and tok.strip()]
            tokenized_for_models = tokens

            # --- Predictions and Display ---
            st.subheader("📊 Model Predictions")

            predictions = {}

            # --- Run Predictions for Selected Models ---
            if "svm" in selected_model_keys:
                pred, conf = predict_svm_nb_cnn(models['svm'], tokenized_for_models, tfidf_vectorizer, label_encoder)
                predictions['svm'] = (pred, conf)

            if "nb" in selected_model_keys:
                pred, conf = predict_svm_nb_cnn(models['nb'], tokenized_for_models, tfidf_vectorizer, label_encoder)
                predictions['nb'] = (pred, conf)

            if "cnn" in selected_model_keys:
                pred, conf = predict_svm_nb_cnn(models['cnn'], tokenized_for_models, tfidf_vectorizer, label_encoder,
                                                cnn_tokenizer, MAXLEN, is_cnn=True)
                predictions['cnn'] = (pred, conf)

            # --- Display Results ---
            if predictions:
                # Inject custom CSS
                st.markdown("""
                            <style>
                            div[data-testid="stMetricLabel"] > div { font-size: 1.1rem; }
                            div[data-testid="stMetricValue"] > div { font-size: 1.3rem; white-space: normal; overflow-wrap: break-word; }
                            </style>
                            """, unsafe_allow_html=True)

                cols = st.columns(len(predictions))
                for i, model_key in enumerate(sorted(predictions.keys())):
                    with cols[i]:
                        model_name = AVAILABLE_MODELS[model_key]
                        label, confidence = predictions[model_key]
                        st.metric(model_name, format_prediction(label, confidence))

                # --- Display Language Consistency Check ---
                st.subheader("Language Consistency Check🤔")
                consistency_warnings_found = False
                for model_key, (label, _) in predictions.items():
                    model_name = AVAILABLE_MODELS[model_key]
                    predicted_lang = extract_language_code(label)
                    if predicted_lang != detected_lang_code:
                        warn_if_mismatch(model_name, predicted_lang, detected_lang_code)
                        consistency_warnings_found = True

                if not consistency_warnings_found:
                    st.success("✅ All model predictions are consistent with the detected language.")