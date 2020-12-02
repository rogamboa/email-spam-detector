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


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()


def rm_non_english_words(text):
    words = set(nltk.corpus.words.words())
    
    text = " ".join(w for w in nltk.wordpunct_tokenize(text)          if w.lower() in words or not w.isalpha() or not len(w) > 10)
    
    return text

def email_num_key_compare(email_path):
    match_pattern = r'[0-9]*$'
    email_num = re.search(match_pattern, email_path)
    return int(email_num.group(0))

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
