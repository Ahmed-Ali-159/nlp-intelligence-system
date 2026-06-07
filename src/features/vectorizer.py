# PURPOSE: Convert cleaned text into numerical representations.
#
# The ML pipeline flow is:
#   raw text → text_cleaner.py → vectorizer.py → model
#
# Why is this a separate file from text_cleaner.py?
# Single Responsibility Principle:
#   - text_cleaner.py  = responsible for cleaning text
#   - vectorizer.py    = responsible for converting text to numbers
# This way you can change your cleaning strategy without touching
# your vectorizer, and vice versa.

import numpy as np
import joblib
# joblib: efficient serialization for large numpy arrays and sklearn objects.
# Better than pickle for ML artifacts because it handles sparse matrices well.

from pathlib import Path
# Path is cleaner than os.path for file operations.
# Path("data/models").mkdir(parents=True, exist_ok=True) is cleaner
# than os.makedirs("data/models", exist_ok=True)

from dataclasses import dataclass, field
from enum import Enum

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# CountVectorizer: implements Bag of Words
#   input:  ["great coffee", "bad coffee"]
#   output: sparse matrix where each column = one word in the vocabulary
#           each row = one document
#           each cell = how many times that word appears in that document
#
# TfidfVectorizer: implements TF-IDF
#   same interface as CountVectorizer but cells contain TF-IDF scores
#   instead of raw counts
#   TF  = term frequency    = how often word appears in THIS document
#   IDF = inverse doc freq  = log(total docs / docs containing this word)
#   TF-IDF = TF × IDF
#   Result: common words get low scores, rare-but-relevant words get high scores

from rank_bm25 import BM25Okapi
# BM25Okapi: the "Okapi BM25" ranking function — industry standard for search.
# Unlike BoW/TF-IDF which produce matrices, BM25 scores documents against queries.
# "Okapi" refers to the Okapi IR system at City University London where it was developed.


# ── ENUMS ─────────────────────────────────────────────────────────────────────

class VectorizerType(Enum):
    BOW = "bow"      # Bag of Words — for baseline classifier
    TFIDF = "tfidf"  # TF-IDF — for main sentiment classifier
    BM25 = "bm25"    # BM25 — for search engine


# ── CONFIG ────────────────────────────────────────────────────────────────────
# Why a config dataclass instead of just passing arguments directly?
# 1. MLflow can log the entire config as one object — full reproducibility
# 2. Easy to create multiple configs and compare them in experiments
# 3. Default values mean you only override what you need to change

@dataclass
class VectorizerConfig:
    vectorizer_type: VectorizerType = VectorizerType.TFIDF

    max_features: int = 50_000
    # Keep only the 50,000 most frequent words in the vocabulary.
    # Why cap it?
    #   - Amazon reviews have ~200k unique words (including typos, rare words)
    #   - Twitter has even more due to slang and misspellings
    #   - More features = larger matrix = slower training = more memory
    #   - Most words beyond top 50k add noise, not signal
    # This is a hyperparameter you'll tune in MLflow experiments later.

    ngram_range: tuple = field(default_factory=lambda: (1, 2))
    # (1, 1) = unigrams only:  ["great", "coffee", "bad"]
    # (1, 2) = unigrams + bigrams: ["great", "coffee", "great coffee", "bad"]
    # Why bigrams matter:
    #   "not good" as two unigrams = ["not", "good"] — the negation is lost
    #   "not good" as a bigram = ["not good"] — the negation is captured
    # Tradeoff: bigrams increase vocabulary size significantly

    min_df: int = 2
    # Ignore words that appear in fewer than 2 documents.
    # Why? A word appearing only once is likely a typo or extremely rare term.
    # It adds a column to the matrix but provides no generalizable signal.

    max_df: float = 0.95
    # Ignore words appearing in more than 95% of documents.
    # Why? A word in almost every document carries no discriminative power.
    # Example: if "product" appears in 98% of Amazon reviews, it doesn't
    # help distinguish positive from negative reviews.
    # This acts as an automatic dataset-specific stopword filter.


# ── MAIN CLASS ────────────────────────────────────────────────────────────────

class FeatureExtractor:
    def __init__(self, config: VectorizerConfig):
        self.config = config
        self.vectorizer = None  # not initialized until fit() is called
        self.is_fitted = False  # guard flag to prevent transform before fit

        # Build the sklearn vectorizer based on config type.
        # BM25 is NOT built here — it needs the actual corpus at fit() time.
        # sklearn vectorizers just need the config at init time.
        if config.vectorizer_type == VectorizerType.BOW:
            self.vectorizer = CountVectorizer(
                max_features=config.max_features,
                ngram_range=config.ngram_range,
                min_df=config.min_df,
                max_df=config.max_df,
            )
        elif config.vectorizer_type == VectorizerType.TFIDF:
            self.vectorizer = TfidfVectorizer(
                max_features=config.max_features,
                ngram_range=config.ngram_range,
                min_df=config.min_df,
                max_df=config.max_df,
                sublinear_tf=True,
                # sublinear_tf=True applies log(1 + tf) instead of raw tf.
                # Why? A word appearing 100 times is not 100x more important
                # than a word appearing once. Log scale brings extreme counts
                # closer together, giving a more balanced representation.
            )
        # BM25 case: vectorizer stays None until fit() is called


    # ── FIT ───────────────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> "FeatureExtractor":
        """
        Learn from the training corpus.

        For BoW/TF-IDF:
            Builds the vocabulary (what words exist) and computes
            IDF weights (how rare is each word across all documents).
            These statistics are learned ONCE from training data.

        For BM25:
            Builds the inverted index — a data structure mapping
            each word to the list of documents containing it,
            along with frequency statistics.

        CRITICAL: fit() must only be called on TRAINING data.
        Never fit on test data — that would leak information about
        the test set into your model (data leakage).

        Returns self to allow method chaining:
            matrix = extractor.fit(train_texts).transform(train_texts)
        """
        if self.config.vectorizer_type == VectorizerType.BM25:
            # BM25Okapi expects list of lists of tokens, not list of strings.
            # "great coffee taste" → ["great", "coffee", "taste"]
            # This is because BM25 needs access to individual tokens
            # for its internal frequency calculations.
            tokenized = [text.split() for text in texts]
            self.vectorizer = BM25Okapi(tokenized)
        else:
            # sklearn fit: learns vocabulary and IDF weights from corpus
            self.vectorizer.fit(texts)

        self.is_fitted = True
        return self  # enables method chaining


    # ── TRANSFORM ─────────────────────────────────────────────────────────────

    def transform(self, texts: list[str]):
        """
        Convert texts to numbers using the already-fitted vocabulary.

        For BoW/TF-IDF:
            Returns a sparse matrix of shape (n_documents, n_features).
            Sparse means most values are 0 — only words in the vocabulary
            that actually appear in a document get non-zero values.
            Sparse matrices are memory-efficient — storing 50k zeros per
            row would be wasteful; sparse format stores only non-zero values.

        For BM25:
            Returns the BM25 object itself. BM25 doesn't produce a matrix —
            it scores documents against a specific query at search time.
            See search() method below.

        Why separate fit() and transform()?
            fit()       on training data → learns statistics
            transform() on training data → converts train set
            transform() on test data    → converts test set using SAME vocabulary
            This ensures train and test use identical feature spaces.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform()")

        if self.config.vectorizer_type == VectorizerType.BM25:
            # BM25 doesn't transform to a matrix
            # The fitted BM25 object IS the search engine
            return self.vectorizer

        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]):
        """
        Fit and transform in a single call.
        Convenience method for training data only.

        Equivalent to: extractor.fit(texts).transform(texts)
        But slightly more efficient for sklearn vectorizers
        because they can combine the two passes internally.

        NEVER use on test data — always fit on train, transform on test.
        """
        return self.fit(texts).transform(texts)


    # ── SEARCH (BM25 specific) ─────────────────────────────────────────────────

    def search(self, query: str, texts: list[str], top_k: int = 5) -> list[dict]:
        """
        Given a search query, return the top_k most relevant documents.

        How BM25 scoring works:
            For each document, BM25 computes a relevance score based on:
            1. Term Frequency (TF): how often query words appear in the document
            2. IDF: how rare those words are across all documents
               (rare words that match are more significant than common ones)
            3. Document length normalization: a long document matching once
               is less relevant than a short document matching once

        Example:
            query = "great coffee"
            doc1  = "great coffee wonderful taste"   → high score (both words match)
            doc2  = "great service amazing"           → medium score (one word matches)
            doc3  = "terrible product waste money"    → score near 0 (no words match)

        Args:
            query:  the search string (will be split into tokens)
            texts:  the original documents to search over
                    (must be the same corpus used in fit())
            top_k:  number of results to return

        Returns:
            list of dicts sorted by relevance score descending:
            [{"rank": 1, "score": 3.14, "text": "..."}, ...]
        """
        if not self.is_fitted or self.config.vectorizer_type != VectorizerType.BM25:
            raise RuntimeError("Call fit() with BM25 config before searching")

        # Split query into tokens — same way we tokenized documents
        query_tokens = query.split()

        # Score every document in the corpus against the query
        # Returns a numpy array of shape (n_documents,)
        scores = self.vectorizer.get_scores(query_tokens)

        # argsort() returns indices that would sort the array ascending
        # [::-1] reverses to descending (highest score first)
        # [:top_k] takes only the top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "rank": rank + 1,          # 1-indexed for human readability
                "score": round(float(scores[idx]), 4),
                "text": texts[idx],
            }
            for rank, idx in enumerate(top_indices)
        ]


    # ── VOCABULARY INSPECTION ─────────────────────────────────────────────────

    def vocab_size(self) -> int:
        """
        Returns the number of unique features after fitting.
        Useful for logging to MLflow and understanding the feature space size.

        For Amazon reviews: expect ~30k-50k features (rich vocabulary)
        For Twitter: expect similar count but with more slang/noise tokens
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")
        if self.config.vectorizer_type == VectorizerType.BM25:
            # BM25 stores IDF values per unique term
            return len(self.vectorizer.idf)
        # sklearn stores vocabulary as a dict: word → column index
        return len(self.vectorizer.vocabulary_)


    # ── PERSISTENCE ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialize the fitted extractor to disk.

        Why save it?
        Training fits the vectorizer on hundreds of thousands of documents.
        You don't want to redo that every time the API starts up.
        Save once after training, load instantly at API startup.

        The saved file includes everything: the config, the vocabulary,
        the IDF weights, and the is_fitted flag.
        """
        if not self.is_fitted:
            raise RuntimeError("Nothing to save — call fit() first")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)  # serializes the entire FeatureExtractor object

    @classmethod
    def load(cls, path: str) -> "FeatureExtractor":
        """
        Load a previously saved extractor from disk.

        @classmethod means you call it on the class, not an instance:
            extractor = FeatureExtractor.load("models/tfidf.joblib")

        Used in FastAPI at startup:
            app.state.extractor = FeatureExtractor.load("models/tfidf.joblib")
        """
        return joblib.load(path)