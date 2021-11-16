import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import re

# -- Tweet Preprocessing
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import emot
from spellchecker import SpellChecker
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from emot.emo_unicode import UNICODE_EMOJI

# Feature Engineering

# -- remove URL
def remove_url(tweet):
  url_pattern = re.compile(r'https?://\S+|www\.\S+')
  return url_pattern.sub(r'', tweet)

# -- remove HTML
def remove_html(tweet):
  return BeautifulSoup(tweet, 'lxml').text

# -- lowercase
def lower_case(tweet):
  return tweet.str.lower()

# -- covert emojis and emoticons to words
def convert_emoji(tweet):
  for emot in UNICODE_EMOJI:
    tweet = tweet.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
  return tweet

# -- remove special characters and non-ASCII characters
def remove_special_char(tweet):
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub(r"\x89Û_", "", tweet)
    tweet = re.sub(r"\x89ÛÒ", "", tweet)
    tweet = re.sub(r"\x89ÛÓ", "", tweet)
    tweet = re.sub(r"\x89ÛÏWhen", "When", tweet)
    tweet = re.sub(r"\x89ÛÏ", "", tweet)
    tweet = re.sub(r"\x89Û÷", "", tweet)
    tweet = re.sub(r"\x89Ûª", "", tweet)
    tweet = re.sub(r"\x89Û\x9d", "", tweet)
    tweet = re.sub(r"å_", "", tweet)
    tweet = re.sub(r"\x89Û¢", "", tweet)
    tweet = re.sub(r"\x89Û¢åÊ", "", tweet)
    tweet = re.sub(r"åÊ", "", tweet)
    tweet = re.sub(r"åÈ", "", tweet)
    tweet = re.sub(r"â²", "", tweet)
    tweet = re.sub(r"Ì©", "e", tweet)
    tweet = re.sub(r"å¨", "", tweet)
    tweet = re.sub(r"â¹", "", tweet)
    tweet = re.sub(r"â½", "", tweet)
    tweet = re.sub(r"â¾", "", tweet)
    tweet = re.sub(r"ã¼berweist", "", tweet)
    tweet = re.sub(r"ã¼cretsiz", "", tweet)
    tweet = re.sub(r"zã¼rich", "", tweet)
    tweet = re.sub(r"ã¼retime", "", tweet)
    tweet = re.sub(r"åÇ", "", tweet)
    tweet = re.sub(r"åÀ", "", tweet)
    tweet = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'mentioned', tweet)
    tweet = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'referance', tweet)
    tweet = re.sub(r'£|\$', 'money', tweet)
    tweet = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', ' ', tweet)
    tweet = re.sub(r'\d+(\.\d+)?', ' ', tweet) 
    tweet = re.sub(r'[^\w\d\s]', ' ', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = re.sub(r'^\s+|\s+?$', '', tweet.lower())
    tweet = re.sub(r"&gt;", ">", tweet)
    tweet = re.sub(r"&lt;", "<", tweet)
    tweet = re.sub(r"&amp;", "&", tweet) 
    tweet = re.sub(r"_", "  ", tweet)
    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        tweet = tweet.replace(p, f' {p} ')
    return str(tweet)

# -- spellcheck
def spellcheck(tweet):
  spell = SpellChecker(distance = 1, language='en')
  words = set(nltk.corpus.words.words())
  corrected_tweet= []
  misspelled_words = spell.unknown(tweet.split())
  for word in tweet.split():
    if word in misspelled_words:
      corrected_tweet.append(spell.correction(word))
    else:
      corrected_tweet.append(word)
  return " ".join(corrected_tweet)

# -- ensure English
def ensure_english(tweet):
  words = set(nltk.corpus.words.words())
  return " ".join(w for w in nltk.wordpunct_tokenize(tweet)\
                  if w.lower() in words or not w.isalpha())

# -- remove punctuation
def remove_punctuation(tweet):
  regular_punct = list(string.punctuation)
  for punctuation in regular_punct:
    if punctuation in tweet:
      tweet = tweet.replace(punctuation, ' ')
  return tweet.strip()
  
# -- remove stopwords
def remove_stopwords(tweet):
  en_stop =set(stopwords.words('english'))
  tweet = tweet.split()
  tweet = " ".join([word for word in tweet if not word in en_stop])
  return tweet

# -- tokenize
def tokenize(tweet):
  return word_tokenize(tweet)

# -- lematize
def lematize(tweet):
  lem = WordNetLemmatizer()
  return [lem.lemmatize(w) for w in tweet]

# -- finally, combine words
def combine_words(tweet):
  return ' '.join(tweet)


def clean_tweet(tweet):

  tweet = remove_url(tweet)
  tweet = remove_html(tweet)
  tweet = convert_emoji(tweet)
  tweet = remove_special_char(tweet)
  tweet = spellcheck(tweet)
  #tweet = ensure_english(tweet)
  tweet = remove_punctuation(tweet)
  tweet = remove_stopwords(tweet)
  tweet = tokenize(tweet)
  tweet = lematize(tweet)
  tweet = combine_words(tweet)

  return tweet