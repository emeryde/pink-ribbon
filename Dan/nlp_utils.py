"""
__file__
    nlp_utils.py
__description__
    This file provides functions to perform NLP task, e.g., TF-IDF and POS tagging.
__author__
    Dan Emery < emeryde@appstate.edu >
"""

import re
import nltk
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob
import numpy as np
import pandas as pd

from config_param import config

################
## Stop Words ##
################
stopwords = nltk.corpus.stopwords.words("english")
stopwords = set(stopwords)

##############
## Stemming ##
##############
if config.stemmer_type == "porter":
    english_stemmer = nltk.stem.PorterStemmer()
elif config.stemmer_type == "snowball":
    english_stemmer = nltk.stem.SnowballStemmer('english')


def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed


#############
## POS Tag ##
#############
token_pattern = r"(?u)\b\w\w+\b"


# token_pattern = r'\w{1,}'
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
def pos_tag_text(line,
                 token_pattern=token_pattern,
                 exclude_stopword=config.cooccurrence_word_exclude_stopword,
                 encode_digit=False):
    token_pattern = re.compile(token_pattern, flags=re.UNICODE | re.LOCALE)
    for name in ["query", "product_title", "product_description"]:
        l = line[name]
        ## tokenize
        tokens = [x.lower() for x in token_pattern.findall(l)]
        ## stem
        tokens = stem_tokens(tokens, english_stemmer)
        if exclude_stopword:
            tokens = [x for x in tokens if x not in stopwords]
        tags = pos_tag(tokens)
        tags_list = [t for w, t in tags]
        tags_str = " ".join(tags_list)
        # print tags_str
        line[name] = tags_str
    return line


############
## TF-IDF ##
############
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r'\w{1,}'
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
tfidf__norm = "l2"
tfidf__max_df = 0.75
tfidf__min_df = 3


def getTFV(token_pattern=token_pattern,
           norm=tfidf__norm,
           max_df=tfidf__max_df,
           min_df=tfidf__min_df,
           ngram_range=(1, 1),
           vocabulary=None,
           stop_words='english'):
    tfv = StemmedTfidfVectorizer(min_df=min_df, max_df=max_df, max_features=None,
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                                 stop_words=stop_words, norm=norm, vocabulary=vocabulary)
    return tfv


#########
## BOW ##
#########
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r'\w{1,}'
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
bow__max_df = 0.75
bow__min_df = 3


def getBOW(token_pattern=token_pattern,
           max_df=bow__max_df,
           min_df=bow__min_df,
           ngram_range=(1, 1),
           vocabulary=None,
           stop_words='english'):
    bow = StemmedCountVectorizer(min_df=min_df, max_df=max_df, max_features=None,
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range,
                                 stop_words=stop_words, vocabulary=vocabulary)
    return bow

def get_text_polarity(post):
    text = TextBlob(post)
    polarity = np.array([])
    for sentence in text.sentences:
        polarity = np.append(polarity, sentence.sentiment.polarity)
    return polarity.sum()

def get_text_subjectivity(post):
    text = TextBlob(post)
    subjectivity = np.array([])
    for sentence in text.sentences:
        subjectivity = np.append(subjectivity, sentence.sentiment.subjectivity)
    return subjectivity.sum()

def get_text_intensity(post):
    text = TextBlob(post)
    intensity = np.array([])
    for sentence in text.sentences:
        intensity = np.append(intensity, sentence.sentiment.polarity)
    return intensity.sum()

def text_length_words(text):
    try:
        if text == "missing":
            return 0
        else:
            words = [w for w in text.split(" ")]
            return len(words)
    except:
        return 0

def text_length_chars(text):
    return len(text)

def create_sentences_dataframe(df):
    text = TextBlob(df['Post'])
    res = pd.DataFrame()
    for sentence in text.sentences:
        tmp = pd.DataFrame({'ID': df['ID'], 'Date':df['Date'], 'Sentence':sentence})
        res = pd.concat(res, tmp)
    return res
