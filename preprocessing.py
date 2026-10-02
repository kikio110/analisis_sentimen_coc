"""Preprocessing teks - sama dengan notebook (tokenisasi memakai split() agar tanpa dependensi NLTK)."""
import json, re, string
from pathlib import Path

RES = Path(__file__).parent / "resources"
SLANG = json.load(open(RES / "slangwords.json", encoding="utf-8"))
STOPWORDS = set(json.load(open(RES / "stopwords.json", encoding="utf-8")))
LEX_POS = json.load(open(RES / "lexicon_positive.json", encoding="utf-8"))
LEX_NEG = json.load(open(RES / "lexicon_negative.json", encoding="utf-8"))


def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"#[A-Za-z0-9]+", "", text)
    text = re.sub(r"RT[\s]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[0-9]+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip(" ")


def fix_slang(text):
    return " ".join(SLANG.get(w.lower(), w) for w in text.split())


def preprocess(text):
    """Return (list token tanpa stopword, kalimat akhir)."""
    text = fix_slang(clean_text(str(text)).lower())
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return tokens, " ".join(tokens)


def lexicon_sentiment(tokens):
    score = sum(LEX_POS.get(t, 0) for t in tokens) + sum(LEX_NEG.get(t, 0) for t in tokens)
    return score, ("positive" if score >= 0 else "negative")
