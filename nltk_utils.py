# Import Libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = SnowballStemmer(language='english')
english_stopwords = stopwords.words('english')

# Tokenize text and stem tokens
def tokenize(text):
    return [stemmer.stem(token) for token in word_tokenize(text)]

# Create a vectorizer to convert text to numerical features
def vectorizer():
    return TfidfVectorizer(
        tokenizer=tokenize,
        stop_words=english_stopwords
    )
