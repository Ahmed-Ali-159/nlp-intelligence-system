from src.features.vectorizer import FeatureExtractor, VectorizerConfig, VectorizerType


# ── sample corpus ─────────────────────────────────────────────────────────────
# Small but realistic — two distinct topics so BM25 ranking is meaningful
CORPUS = [
    "great coffee wonderful taste recommend",
    "terrible product waste money awful",
    "amazing food delicious fresh quality",
    "bad service slow cold disappointing",
    "love product best purchase ever made",
]

QUERY = "great coffee taste"


def test_bow_shape():
    """BoW matrix should have one row per document"""
    config = VectorizerConfig(
        vectorizer_type=VectorizerType.BOW,
        ngram_range=(1, 1),
        min_df=1,
    )
    extractor = FeatureExtractor(config)
    matrix = extractor.fit_transform(CORPUS)
    assert matrix.shape[0] == len(CORPUS)


def test_tfidf_shape():
    """TF-IDF matrix should have one row per document"""
    config = VectorizerConfig(
        vectorizer_type=VectorizerType.TFIDF,
        ngram_range=(1, 1),
        min_df=1,
    )
    extractor = FeatureExtractor(config)
    matrix = extractor.fit_transform(CORPUS)
    assert matrix.shape[0] == len(CORPUS)


def test_tfidf_values_between_0_and_1():
    """TF-IDF scores should be normalized between 0 and 1"""
    config = VectorizerConfig(
        vectorizer_type=VectorizerType.TFIDF,
        ngram_range=(1, 1),
        min_df=1,
    )
    extractor = FeatureExtractor(config)
    matrix = extractor.fit_transform(CORPUS)
    assert matrix.max() <= 1.0
    assert matrix.min() >= 0.0


def test_bm25_returns_ranked_results():
    """BM25 search should return top_k results in ranked order"""
    config = VectorizerConfig(vectorizer_type=VectorizerType.BM25)
    extractor = FeatureExtractor(config)
    extractor.fit(CORPUS)
    results = extractor.search(QUERY, CORPUS, top_k=3)

    assert len(results) == 3
    # scores should be in descending order
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_top_result_relevant():
    """The top BM25 result for 'great coffee' should be the coffee document"""
    config = VectorizerConfig(vectorizer_type=VectorizerType.BM25)
    extractor = FeatureExtractor(config)
    extractor.fit(CORPUS)
    results = extractor.search(QUERY, CORPUS, top_k=1)

    assert "coffee" in results[0]["text"]


def test_transform_before_fit_raises():
    """Calling transform() before fit() should raise a RuntimeError"""
    config = VectorizerConfig(vectorizer_type=VectorizerType.TFIDF)
    extractor = FeatureExtractor(config)
    try:
        extractor.transform(CORPUS)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


def test_vocab_size_positive():
    """Vocabulary size should be greater than 0 after fitting"""
    config = VectorizerConfig(
        vectorizer_type=VectorizerType.TFIDF,
        ngram_range=(1, 1),
        min_df=1,
    )
    extractor = FeatureExtractor(config)
    extractor.fit(CORPUS)
    assert extractor.vocab_size() > 0