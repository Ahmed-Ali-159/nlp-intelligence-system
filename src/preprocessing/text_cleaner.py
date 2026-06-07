# PURPOSE: Clean raw text from two very different datasets
# (Amazon reviews and Tweets) before feeding into ML models.
# The same tokens that confuse a model (HTML tags, URLs, @mentions)
# need to be removed — but each dataset has its own specific noise.

import re                          # regular expressions — for pattern-based cleaning
import string                      # gives us string.punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
import nltk                        # Natural Language Toolkit — tokenization, stemming, lemmatization
from nltk.tokenize import word_tokenize   # splits "hello world" → ["hello", "world"]
from nltk.corpus import stopwords         # common words to remove: "the", "is", "at", "a" ...
from nltk.stem import PorterStemmer, WordNetLemmatizer
# PorterStemmer:  "running" → "run", "studies" → "studi" (fast but rough)
# WordNetLemmatizer: "running" → "run", "studies" → "study" (slower but linguistically correct)
from enum import Enum              # lets us define fixed named options (like constants)
from dataclasses import dataclass  # clean way to define config objects without boilerplate

# ── NLTK data downloads ────────────────────────────────────────────────────────
# NLTK needs external data files to work. quiet=True suppresses the download logs.
# "punkt"     → rules for splitting text into sentences and words
# "punkt_tab" → updated tokenizer data (required in newer NLTK versions)
# "stopwords" → the list of common words to filter out
# "wordnet"   → dictionary the lemmatizer uses to find base word forms
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


# ── ENUMS: fixed named options ─────────────────────────────────────────────────
# Why enums? If you use plain strings like "amazon" or "twitter", a typo
# ("amazn") silently does the wrong thing. Enums make typos into errors.

class Dataset(Enum):
    AMAZON = "amazon"    # long structured reviews, HTML noise, star ratings
    TWITTER = "twitter"  # short noisy tweets, @mentions, #hashtags, URLs


class NormalizationMode(Enum):
    STEM = "stem"           # faster, more aggressive: "studies" → "studi"
    LEMMATIZE = "lemmatize" # slower, linguistically correct: "studies" → "study"
    NONE = "none"           # keep tokens as-is (useful for embeddings later)


# ── CONFIG: what settings to use for a given cleaning run ─────────────────────
# @dataclass automatically generates __init__, __repr__ etc. from the fields.
# This config object gets logged to MLflow so every experiment is reproducible.
@dataclass
class CleanerConfig:
    dataset: Dataset                                          # REQUIRED: which dataset
    normalization: NormalizationMode = NormalizationMode.LEMMATIZE  # default: lemmatize
    remove_stopwords: bool = True                             # default: remove stopwords
    min_token_length: int = 2                                 # drop single characters like "a", "i"


# ── MAIN CLASS ────────────────────────────────────────────────────────────────
class TextCleaner:
    def __init__(self, config: CleanerConfig):
        self.config = config

        # Load stopwords once at init — not inside clean() to avoid
        # reloading on every single document (would be very slow)
        self.stop_words = set(stopwords.words("english"))
        # set() makes lookup O(1) instead of O(n) — matters at 1.6M tweets

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    # ── LOW-LEVEL CLEANERS ────────────────────────────────────────────────────
    # Each method does exactly ONE thing.
    # Small focused methods = easy to test individually + easy to reorder.

    def _remove_html(self, text: str) -> str:
        """
        Amazon reviews often contain HTML artifacts from copy-paste or scraping.
        Examples: "great &amp; tasty", "good product<br/>", "&lt;3 stars&gt;"
        We remove both HTML tags (<br/>, <b>) and HTML entities (&amp;, &lt;)
        """
        text = re.sub(r"<[^>]+>", " ", text)   # remove tags:     <br/> → " "
        text = re.sub(r"&\w+;", " ", text)      # remove entities: &amp; → " "
        return text

    def _remove_urls(self, text: str) -> str:
        """
        Both datasets contain URLs but tweets far more so.
        "check http://example.com" → "check "
        \S+ means "one or more non-whitespace characters"
        """
        return re.sub(r"http\S+|www\.\S+", "", text)

    def _remove_mentions_hashtags(self, text: str) -> str:
        """
        Twitter-specific noise.
        @mentions: remove entirely — they identify users, not sentiment
        #hashtags: remove the # but KEEP the word — "great #food" → "great food"
                   because the word itself carries meaning
        """
        text = re.sub(r"@\w+", "", text)         # @john → ""
        text = re.sub(r"#(\w+)", r"\1", text)    # #awesome → "awesome"
        return text

    def _remove_punctuation(self, text: str) -> str:
        """
        Removes !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        str.maketrans("", "", string.punctuation) builds a translation table
        that maps every punctuation character to None (deletion).
        """
        return text.translate(str.maketrans("", "", string.punctuation))

    def _normalize_whitespace(self, text: str) -> str:
        """
        After all the removals above, we're left with irregular spacing.
        "hello   world  " → "hello world"
        \s+ matches one or more whitespace characters (spaces, tabs, newlines)
        """
        return re.sub(r"\s+", " ", text).strip()

    # ── NORMALIZATION ─────────────────────────────────────────────────────────

    def _normalize(self, token: str) -> str:
        """
        Reduce a word to its base form so that "running", "runs", "ran"
        all map to the same feature in our BoW/TF-IDF matrix.
        Which mode to use depends on the config passed at init.
        """
        if self.config.normalization == NormalizationMode.STEM:
            return self.stemmer.stem(token)       # fast, aggressive
        elif self.config.normalization == NormalizationMode.LEMMATIZE:
            return self.lemmatizer.lemmatize(token)  # slower, correct
        return token                              # NormalizationMode.NONE

    # ── MAIN PIPELINE ─────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """
        Orchestrates all cleaners in the correct order for the given dataset.
        ORDER MATTERS:
          - lowercase before punctuation removal (so URL detection works)
          - URL removal before punctuation (URLs contain dots and slashes)
          - punctuation removal before tokenization (cleaner splits)
          - stopword removal after tokenization (need individual tokens)
          - normalization last (operate on clean final tokens)
        """
        # Guard clause: handle None, empty strings, whitespace-only
        if not isinstance(text, str) or not text.strip():
            return ""

        # Step 1: lowercase everything so "Great" and "great" are the same token
        text = text.lower()

        # Step 2: remove URLs (common to both datasets)
        text = self._remove_urls(text)

        # Step 3: dataset-specific cleaning
        if self.config.dataset == Dataset.AMAZON:
            text = self._remove_html(text)              # Amazon has HTML artifacts
        elif self.config.dataset == Dataset.TWITTER:
            text = self._remove_mentions_hashtags(text) # Twitter has @mentions, #tags

        # Step 4: remove punctuation
        text = self._remove_punctuation(text)

        # Step 5: collapse multiple spaces into one
        text = self._normalize_whitespace(text)

        # Step 6: tokenize — split the string into a list of individual words
        # "hello world" → ["hello", "world"]
        tokens = word_tokenize(text)

        # Step 7: drop tokens that are too short (single chars add noise)
        tokens = [t for t in tokens if len(t) >= self.config.min_token_length]

        # Step 8: remove stopwords if configured
        # stopwords = ["the", "is", "at", "a", "an", ...] — carry no sentiment signal
        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        # Step 9: stem or lemmatize each token
        tokens = [self._normalize(t) for t in tokens]

        # Rejoin into a single string — this is what gets fed to TF-IDF/BoW
        return " ".join(tokens)

    def clean_batch(self, texts: list[str]) -> list[str]:
        """
        Convenience method to clean an entire list of texts at once.
        Used when processing a full DataFrame column:
            df["clean_text"] = cleaner.clean_batch(df["text"].tolist())
        """
        return [self.clean(t) for t in texts]