import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.test.gpu_device_name()
import re
import torch
# -- Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# -- Performance Metrics
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_score, recall_score,  accuracy_score, precision_recall_curve


# Train Baseline Models
# NB
def naive_bayes_model(feature_vector_x, feature_vector_y):
  alpha = [1e-10, 1e-5, 0.1, 1.0, 2.0, 5.0, 10.0]
  best_alpha = -1
  max_score = 0
  for a in alpha:
    mnb = MultinomialNB(alpha = a)
    scores = sklearn.model_selection.cross_val_score(mnb, feature_vector_x, feature_vector_y, cv = 5)
    if np.mean(scores)> max_score:
      best_alpha = a
      max_score = np.mean(scores)
    
    print('alpha =', a)
    print(np.mean(scores))
    print('\n')
  
  print('best alpha:', best_alpha)
  mnb = MultinomialNB(alpha = best_alpha)
  mnb.fit(feature_vector_x, feature_vector_y)
  print('train score:', mnb.score(feature_vector_x, feature_vector_y))
  return mnb

# DT
def dt_model(feature_vector_x, feature_vector_y):
  dtclassifier = DecisionTreeClassifier(criterion='entropy', max_depth=None)
  scores = cross_val_score(dtclassifier, feature_vector_x, feature_vector_y, cv = 10)
  dtclassifier.fit(feature_vector_x, feature_vector_y)
  print('train score:', accuracy_score(dtclassifier.predict(feature_vector_x), feature_vector_y))
  return dtclassifier

# LR
def lr_model(feature_vector_x, feature_vector_y):
  C_values = [0.001,0.01, 0.1,1,10,100]
  best_c = -1
  max_score = 0
  for c in C_values:
    lr = LogisticRegression(C = c, random_state=0, solver = 'lbfgs', multi_class='multinomial')
    lr.fit(feature_vector_x, feature_vector_y)
    scores = sklearn.model_selection.cross_val_score(lr, feature_vector_x, feature_vector_y, cv = 5)
    if np.mean(scores)> max_score:
      best_c = c
      max_score = np.mean(scores)
    
    print('c =', c)
    print(np.mean(scores))
    print('\n')

  lr = LogisticRegression(solver = 'lbfgs', multi_class='multinomial', C=c)
  lr.fit(feature_vector_x, feature_vector_y)
  print('train score:', accuracy_score(lr.predict(feature_vector_x), feature_vector_y))
  return lr

# SVM
def svm_model(feature_vector_x, feature_vector_y):
  params = {'C':[0.01, 0.1, 1, 10, 100],
       'kernel':['rbf', 'poly', 'linear', 'sigmoid']}
  classifier_linear = GridSearchCV(svm.SVC(), params, cv=10)
  classifier_linear.fit(feature_vector_x, feature_vector_y)
  print('train score:', accuracy_score(classifier_linear.predict(feature_vector_x), feature_vector_y))