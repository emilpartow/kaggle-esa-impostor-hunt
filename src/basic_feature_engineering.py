"""
Feature engineering for paired texts (text_A, text_B).

This module computes a set of basic lexical, readability, and similarity features
for each pair of texts in a DataFrame. It is designed for scenarios where each
row contains two texts (A and B) and you want both per-text features and
pairwise comparisons (differences and similarities).

"""

import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Package for Named Entity Recognition (NER)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None  # If spaCy is not installed or model not available


def count_words(text):
    """Count word-like tokens using \w+ regex."""
    return len(re.findall(r"\w+", text))


def count_sentences(text):
    """Count sentence enders (., !, ?)."""
    return len(re.findall(r"[.!?]", text))


def count_citations(text):
    """
    Count bracketed numeric citations like [12] and DOI-looking strings.
    Returns the sum of both counts.
    """
    citations = re.findall(r"\[\d+\]", text)
    dois = re.findall(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
    return len(citations) + len(dois)


def avg_word_length(text):
    """Average character length of word-like tokens."""
    words = re.findall(r"\w+", text)
    if not words:
        return 0
    return np.mean([len(w) for w in words])


def count_syllables(word):
    """
    Very rough syllable count: count contiguous vowel clusters (aeiouy).
    This mirrors the original heuristic.
    """
    word = word.lower()
    return len(re.findall(r"[aeiouy]+", word))


def flesch_reading_ease(text):
    """
    Compute Flesch Reading Ease using the same simple syllable heuristic.
    Falls back to 0 if there are no words; ensures at least one sentence.
    """
    words = re.findall(r"\w+", text)
    sentences = max(1, len(re.findall(r"[.!?]", text)))
    syllables = sum([count_syllables(w) for w in words])
    if not words:
        return 0
    words_per_sentence = len(words) / sentences
    syllables_per_word = syllables / len(words)
    return 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word


def ends_with_punctuation(text):
    """Return 1 if text ends with ., !, or ?, else 0."""
    return int(text.strip()[-1:] in ".!?")


def count_entities(text):
    """
    Count named entities using spaCy if available.
    If spaCy isn't loaded, this function is still called but the caller
    chooses whether to apply it or set NaN (keeps original behavior).
    """
    if nlp is None:
        return np.nan
    doc = nlp(text)
    return len([ent for ent in doc.ents])


def count_numbers(text):
    """Count sequences of digits."""
    return len(re.findall(r"\d+", text))


def jaccard_sim(a, b):
    """
    Jaccard similarity between sets of lowercase word tokens in a and b.
    """
    set_a = set(re.findall(r"\w+", a.lower()))
    set_b = set(re.findall(r"\w+", b.lower()))
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def longest_sentence_length(text):
    """
    Length (in words) of the longest sentence, splitting on . ! ?
    """
    sentences = re.split(r"[.!?]", text)
    return max((len(s.split()) for s in sentences if s.strip()), default=0)


def max_repeat_word(text):
    """
    Maximum frequency of any single word (case-insensitive) in the text.
    """
    words = re.findall(r"\w+", text.lower())
    return max((words.count(w) for w in set(words)), default=0)


def basic_features(df):
    """
    Compute basic per-text features for text_A and text_B, then add
    pairwise differences and similarities.

    Per-text features (for each of text_A and text_B)
    -----------------------------------------------
    - length_chars
    - length_words
    - num_sentences
    - num_citations
    - avg_word_length
    - flesch_reading
    - ends_with_punct (0/1)
    - num_entities (requires spaCy; otherwise column filled with NaN scalar)
    - num_numbers
    - longest_sentence
    - max_repeat_word

    Pairwise features
    -----------------
    - *_diff: A minus B for most numeric features
    - cosine_sim_A_B: cosine similarity of TF-IDF vectors (try/except guarded)
    - jaccard_sim_A_B: Jaccard similarity of token sets

    Notes
    -----
    - If TF-IDF similarity computation fails, cosine_sim_A_B is set to 0.0.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns "text_A" and "text_B".

    Returns
    -------
    pandas.DataFrame
        A copy of df with all features added.
    """
    df = df.copy()

    # Per-text features for A and B
    for col in ["text_A", "text_B"]:
        df[col] = df[col].fillna("").astype(str)
        df[f"{col}_length_chars"] = df[col].apply(len)
        df[f"{col}_length_words"] = df[col].apply(count_words)
        df[f"{col}_num_sentences"] = df[col].apply(count_sentences)
        df[f"{col}_num_citations"] = df[col].apply(count_citations)
        df[f"{col}_avg_word_length"] = df[col].apply(avg_word_length)
        df[f"{col}_flesch_reading"] = df[col].apply(flesch_reading_ease)
        df[f"{col}_ends_with_punct"] = df[col].apply(ends_with_punctuation)
        # Keep original behavior: if spaCy is missing, assign scalar NaN for the entire column
        df[f"{col}_num_entities"] = df[col].apply(count_entities) if nlp is not None else np.nan
        df[f"{col}_num_numbers"] = df[col].apply(count_numbers)
        df[f"{col}_longest_sentence"] = df[col].apply(longest_sentence_length)
        df[f"{col}_max_repeat_word"] = df[col].apply(max_repeat_word)

    # A-B differences (preserve original set)
    df["length_chars_diff"] = df["text_A_length_chars"] - df["text_B_length_chars"]
    df["length_words_diff"] = df["text_A_length_words"] - df["text_B_length_words"]
    df["num_sentences_diff"] = df["text_A_num_sentences"] - df["text_B_num_sentences"]
    df["num_citations_diff"] = df["text_A_num_citations"] - df["text_B_num_citations"]
    df["avg_word_length_diff"] = df["text_A_avg_word_length"] - df["text_B_avg_word_length"]
    df["flesch_reading_diff"] = df["text_A_flesch_reading"] - df["text_B_flesch_reading"]
    df["ends_with_punct_diff"] = df["text_A_ends_with_punct"] - df["text_B_ends_with_punct"]
    if nlp is not None:
        df["num_entities_diff"] = df["text_A_num_entities"] - df["text_B_num_entities"]
    df["num_numbers_diff"] = df["text_A_num_numbers"] - df["text_B_num_numbers"]
    df["longest_sentence_diff"] = df["text_A_longest_sentence"] - df["text_B_longest_sentence"]
    df["max_repeat_word_diff"] = df["text_A_max_repeat_word"] - df["text_B_max_repeat_word"]

    # Cosine similarity of TF-IDF vectors (guarded)
    try:
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(df["text_A"].tolist() + df["text_B"].tolist())
        n = len(df)
        A_vecs = tfidf_matrix[:n]
        B_vecs = tfidf_matrix[n:]
        similarities = cosine_similarity(A_vecs, B_vecs).diagonal()
        df["cosine_sim_A_B"] = similarities
    except Exception as e:
        print("Could not compute cosine similarity:", e)
        df["cosine_sim_A_B"] = 0.0  # keep original fallback

    # Jaccard similarity of token sets
    df["jaccard_sim_A_B"] = [
        jaccard_sim(a, b) for a, b in zip(df["text_A"], df["text_B"])
    ]

    return df


if __name__ == "__main__":
    # Determine project root
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        root_dir = os.path.abspath("..")

    data_dir = os.path.join(root_dir, "data")

    for split in ["train", "test"]:
        path = os.path.join(data_dir, f"{split}.csv")
        df = pd.read_csv(path)
        df_feat = basic_features(df)
        df_feat.to_csv(path, index=False)
        print(f"File updated with all basic features: {path}")