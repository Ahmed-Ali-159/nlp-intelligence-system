from src.preprocessing.text_cleaner import TextCleaner, CleanerConfig, Dataset, NormalizationMode

def test_amazon_removes_html():
    cleaner = TextCleaner(CleanerConfig(dataset=Dataset.AMAZON))
    result = cleaner.clean("This is <br/> great &amp; tasty!")
    assert "<" not in result
    assert "&" not in result

def test_twitter_removes_mentions_and_urls():
    cleaner = TextCleaner(CleanerConfig(dataset=Dataset.TWITTER))
    result = cleaner.clean("@john check this http://example.com #awesome")
    assert "@" not in result
    assert "http" not in result
    assert "awesome" in result  # hashtag word kept

def test_empty_input():
    cleaner = TextCleaner(CleanerConfig(dataset=Dataset.AMAZON))
    assert cleaner.clean("") == ""
    assert cleaner.clean("   ") == ""

def test_normalization_lemmatize():
    cleaner = TextCleaner(CleanerConfig(
        dataset=Dataset.AMAZON,
        normalization=NormalizationMode.LEMMATIZE
    ))
    result = cleaner.clean("the dogs are running quickly")
    assert "dog" in result
    assert "run" in result