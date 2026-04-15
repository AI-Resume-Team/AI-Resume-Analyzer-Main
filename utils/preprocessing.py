import re
import string

# ---------------------------------------------------------------------------
# Lazy-load NLTK resources so the app still works if NLTK data isn't present
# ---------------------------------------------------------------------------

def _get_stopwords():
    try:
        from nltk.corpus import stopwords
        import nltk
        try:
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
    except ImportError:
        # Minimal fallback stopword list
        return {
            "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
            "they", "them", "this", "that", "the", "a", "an", "and", "or",
            "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "is", "are", "was", "were", "be", "been", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may",
        }


def _get_lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer
        import nltk
        try:
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("test")  # trigger resource load
            return lemmatizer
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            return WordNetLemmatizer()
    except ImportError:
        return None


def _tokenize(text):
    """Simple whitespace + punctuation tokenizer (NLTK-free fallback)."""
    try:
        import nltk
        try:
            return nltk.word_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            return nltk.word_tokenize(text)
    except ImportError:
        # Regex-based fallback
        return re.findall(r"\b\w+\b", text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
    1. Lowercase
    2. Remove special characters / punctuation (keep alphanumeric + spaces)
    3. Tokenize
    4. Remove stop words
    5. Lemmatize
    Returns a single cleaned string.
    """
    if not text:
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove special characters (keep letters, digits, spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Tokenize
    tokens = _tokenize(text)

    # 4. Remove stop words + single-char tokens
    stop_words = _get_stopwords()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # 5. Lemmatize
    lemmatizer = _get_lemmatizer()
    if lemmatizer:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def extract_keywords(text: str, top_n: int = 30) -> list:
    """
    Return the top_n most frequent meaningful words from the text
    after preprocessing.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    freq: dict = {}
    for word in cleaned.split():
        freq[word] = freq.get(word, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]

