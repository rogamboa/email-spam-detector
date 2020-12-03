#!/usr/bin/env python

import glob
import pandas as pd
from bs4 import BeautifulSoup
import unicodedata
import nltk
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import NMF

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()

def email_num_key_compare(email_path):
    match_pattern = r'[0-9]*$'
    email_num = re.search(match_pattern, email_path)
    return int(email_num.group(0))

def show_topics(nmf_model, feature_names, num_topics=10, num_top_words=10):
    word_dct = {}
    
    for i in range(num_topics):
        words_idx = nmf_model.components_[i].argsort()[:(-num_top_words)-1:-1]
        words = [feature_names[idx] for idx in words_idx]
        word_dct[f'Topic #{i+1}'] = words
        
    word_df = pd.DataFrame(word_dct)
        
    return word_df

def nmf_error_reconstruction(text_matrix, n_components):
    nmf = NMF(n_components=n_components)
    nmf.fit(text_matrix)
    return nmf.reconstruction_err_

emails_filepaths = sorted(glob.glob("data/emails/inmail.*"), key=email_num_key_compare)

num_emails = 100

col_names =  ['email', 'label'] 
emails_df = pd.DataFrame(columns = col_names) 

with open('data/labels') as f:
    labels_lst = [line.rstrip().split(' ')[0] for line in f]

for path, label in zip(emails_filepaths[:num_emails], labels_lst[:num_emails]):
    with open(path, 'rb') as file:
        html = BeautifulSoup(file.read(),"html.parser")
        email_text = html.get_text()
        
        new_row = {'filename': path[12:], 'email': email_text, 'label': label}
        emails_df = emails_df.append(new_row, ignore_index=True)

X = emails_df['email']
y = emails_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

nb_cv_pipeline = make_pipeline(
    CountVectorizer(binary=True),
    MultinomialNB()
)

nb_cv_pipeline.fit(X_train, y_train)
y_pred = nb_cv_pipeline.predict(X_test)

y_pred = nb_cv_pipeline.predict(X_train)
y_pred2 = nb_cv_pipeline.predict(X_test)

print(classification_report(y_train, y_pred))
print(classification_report(y_test, y_pred2))

nb_tfidf_pipeline = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

nb_tfidf_pipeline.fit(X_train, y_train)
y_pred = nb_tfidf_pipeline.predict(X_train)
y_pred2 = nb_tfidf_pipeline.predict(X_test)

print(classification_report(y_train, y_pred))
print(classification_report(y_test, y_pred2))


rf_tfidf_pipeline = make_pipeline(
    TfidfVectorizer(min_df=0.01),
    RandomForestClassifier(n_estimators=100, max_depth=5)
)

rf_tfidf_pipeline.fit(X_train, y_train)
y_pred = rf_tfidf_pipeline.predict(X_train)
y_pred2 = rf_tfidf_pipeline.predict(X_test)

print(classification_report(y_train, y_pred))
print(classification_report(y_test, y_pred2))

c_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))
cv_email_matrix = c_vectorizer.fit_transform(X_train)
nmf = NMF(n_components=15)
W = nmf.fit_transform(cv_email_matrix)
H = nmf.components_

feature_names = c_vectorizer.get_feature_names()
num_topics = 15

top_10_topics_and_words = show_topics(nmf, feature_names, num_topics=num_topics)

print(top_10_topics_and_words)