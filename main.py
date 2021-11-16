import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from utils import *
from baselines import *
# -- Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Set Data Paths
general = "data/general"
antiAsian = "data/antiAsian"
givenDataset = "data/given_dataset"

# Read in General Tweetes
generalTweets = pd.read_csv(os.path.join(general,'train_E6oV3lV.csv'))
generalTweets = generalTweets[['tweet', 'label']]

# Read in Prof's Tweets
givenTweetsB = pd.read_csv(os.path.join(givenDataset,'B_volunteer_labelled_data_20210913.csv'))
givenTweetsA = pd.read_csv(os.path.join(givenDataset,'A_volunteer_labelled_data_20210913.csv'))


X_train, X_test, y_train, y_test = train_test_split(generalTweets['tweet'], generalTweets['label'], test_size=0.20, random_state=42)

X_train = X_train.apply(clean_tweet)
X_test = X_test.apply(clean_tweet)

# Vectorize

# -- Bag of Words (unigrams)
cv_unigrams = CountVectorizer(ngram_range = (1,1))
X_train_bow = cv_unigrams.fit_transform(X_train)


# -- Bag of Words (bigrams)
cv_bigrams = CountVectorizer(ngram_range = (2,2))
X_train_bbow = cv_bigrams.fit_transform(X_train)


# -- TF-IDF
vec_tfidf = TfidfVectorizer(min_df = 2, max_df = 0.8, use_idf = True, ngram_range=(1, 1))
vec_tfidf.fit(X_train)
X_train_tfidf = vec_tfidf.fit_transform(X_train)


# -- NB with BOW unigram
mnb_bow = naive_bayes_model(X_train_bow, y_train)
# -- NB with BOW bigram
mnb_bbow = naive_bayes_model(X_train_bbow, y_train)
# -- NB with TF-IDF
mnb_tfidf = naive_bayes_model(X_train_tfidf, y_train)

# -- DT with BOW unigram
dt_bow = dt_model(X_train_bow, y_train)

# -- DT with BOW bigram
dt_bbow = dt_model(X_train_bbow, y_train)

# -- DT with TF-IDF
dt_tfidf = dt_model(X_train_tfidf, y_train)

# -- LR with BOW unigram
lr_bow = lr_model(X_train_bow, y_train)

# -- LR with BOW bigram
lr_bbow = lr_model(X_train_bbow, y_train)

# -- LR with TF-IDF
lr_tfidf = lr_model(X_train_tfidf, y_train)

# -- SVM with BOW unigram
svm_bow = svm_model(X_train_bow, y_train)
# -- SVM with BOW bigram
svm_bbow = svm_model(X_train_bbow, y_train)
# -- SVM with TF-IDF
svm_tfidf = svm_model(X_train_tfidf, y_train)
