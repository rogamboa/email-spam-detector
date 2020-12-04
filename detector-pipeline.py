#!/usr/bin/env python

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

import matplotlib.pyplot as plt

import pickle
import numpy as np
import pandas as pd
import re

plt.style.use('ggplot')

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

def remove_num(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text

def get_word_counts(corpus):
    c_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,1), preprocessor=remove_num)
    cv_email_matrix = c_vectorizer.fit_transform(corpus)
    
    word_list = c_vectorizer.get_feature_names()    
    count_list = np.asarray(cv_email_matrix.sum(axis=0))
    return word_list, count_list

def make_freq_word_bar_plt(dataset_title, corpus, num_words=15, color='red'):
    
    word_list, count_list = get_word_counts(corpus)
    
    top_freq_word_idx = count_list[0].argsort()[:(-15)-1:-1]

    top_words = []
    freq_cnt = []

    for i in top_freq_word_idx:
        top_words.append(word_list[i])
        freq_cnt.append(count_list[0][i])
        
    plt.bar(top_words, freq_cnt, align='center', color=color)
    plt.xticks(top_words, rotation=90)
    plt.ylabel('Frequency Count')
    plt.xlabel('Words')
    plt.title(f'Top 15 Most Frequent Words in {dataset_title}')
    plt.show()


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