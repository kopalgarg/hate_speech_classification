import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
import re
import torch
from utils import *
from baselines import *
from bert import *

# -- Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# -- Explainability
import shap
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer

# Set Data Paths
general = "data/general"
antiAsian = "data/antiAsian"
givenDataset = "data/given_dataset"

dataset = 'general'

if dataset=='general':
    # Read in General Tweetes
    generalTweets = pd.read_csv(os.path.join(general,'train_E6oV3lV.csv'))
    df = generalTweets[['tweet', 'label']]
else:
    # Read in Provided Tweets
    givenTweetsB = pd.read_csv(os.path.join(givenDataset,'B_volunteer_labelled_data_20210913.csv'))
    givenTweetsA = pd.read_csv(os.path.join(givenDataset,'A_volunteer_labelled_data_20210913.csv'))
    givenTweets = pd.concat([givenTweetsB, givenTweetsA])
    givenTweets = shuffle(givenTweets)
    # -- label encoding 
    encoding = {'not stigmatizing': '0',
            'stigmatizing - high': '1',
            'stigmatizing - low': '1',
            'stigmatizing - medium': '1',
            'unknown/irrelevant': 'NA'
           }
    labels = ['0', '1', 'NA']
    givenTweets['Does this tweet contain covid-related stigmatizing language against the people of Asian-descent?'].replace(encoding, inplace=True)
    df = givenTweets[givenTweets['Does this tweet contain covid-related stigmatizing language against the people of Asian-descent?'] != 'NA']

    df = df[['clean text of tweet','Does this tweet contain covid-related stigmatizing language against the people of Asian-descent?']]
    df.columns = ['tweet', 'label']

#df = df[1:100]
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['label'], test_size=0.20, random_state=42)

X_train = X_train.apply(clean_tweet)
X_test =  X_test.apply(clean_tweet)

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

print("NB")

# -- NB with BOW unigram
mnb_bow = naive_bayes_model(X_train_bow, y_train)
# -- NB with BOW bigram
mnb_bbow = naive_bayes_model(X_train_bbow, y_train)
# -- NB with TF-IDF
mnb_tfidf = naive_bayes_model(X_train_tfidf, y_train)

print("DT")
# -- DT with BOW unigram
dt_bow = dt_model(X_train_bow, y_train)

# -- DT with BOW bigram
dt_bbow = dt_model(X_train_bbow, y_train)

# -- DT with TF-IDF
dt_tfidf = dt_model(X_train_tfidf, y_train)

print("RF")

# -- RF with BOW unigram
rf_bow = dt_model(X_train_bow, y_train)

# -- RF with BOW bigram
rf_bbow = dt_model(X_train_bbow, y_train)

# -- RF with TF-IDF
rf_tfidf = dt_model(X_train_tfidf, y_train)

print("LR")

# -- LR with BOW unigram
lr_bow = lr_model(X_train_bow, y_train)

# -- LR with BOW bigram
lr_bbow = lr_model(X_train_bbow, y_train)

# -- LR with TF-IDF
lr_tfidf = lr_model(X_train_tfidf, y_train)

print("SVM")

# -- SVM with BOW unigram
svm_bow = svm_model(X_train_bow, y_train)
# -- SVM with BOW bigram
svm_bbow = svm_model(X_train_bbow, y_train)
# -- SVM with TF-IDF
svm_tfidf = svm_model(X_train_tfidf, y_train)

print("XGBoost")

# -- XGBoost with BOW unigram
xgb_bow = xgboost_model(X_train_bow, y_train)
# -- XGBoost with BOW bigram
xgb_bbow = xgboost_model(X_train_bbow, y_train)
# -- XGBoost with TF-IDF
xgb_tfidf = xgboost_model(X_train_tfidf, y_train)

# RNN

# BERT

# Explainability

# -- LIME
class_names=list(generalTweets.label.unique())
explainer = LimeTextExplainer(class_names=class_names)

c = make_pipeline(vec_tfidf, xgb_tfidf)
ls_X_train= list(X_train)

idx = 6

c.fit(X_train, y_train)

LIME_exp = explainer.explain_instance(ls_X_train[idx], c.predict_proba)

print('Document id: %d' % idx)
print('Tweet: ', ls_X_train[idx])
print('Probability =', c.predict_proba([ls_X_train[idx]]).round(3)[0,1])

LIME_exp.show_in_notebook(text=True)

# -- SHAP
SHAP_explainer = shap.KernelExplainer(lr_bow.predict, X_train_bow[1:100])
shap_vals = SHAP_explainer.shap_values(X_train_bow[1:5])
colour_test = pd.DataFrame(X_train_bow[1:5].todense())

shap.summary_plot(shap_vals, colour_test, feature_names=cv_unigrams.get_feature_names())