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
import unicodedata
import glob
import pandas as pd
from bs4 import BeautifulSoup

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

def get_email_body(email_text):
    
    email_splitted_lines = email_text.splitlines()
    
    for idx, line in enumerate(email_splitted_lines):
        if len(line) == 0:
            first_empty_line = idx
            break
            
    return ''.join(email_splitted_lines[first_empty_line:])

def email_num_key_compare(email_path):
    match_pattern = r'[0-9]*$'
    email_num = re.search(match_pattern, email_path)
    return int(email_num.group(0))

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
    title = '-'.join(dataset_title.split(' '))
    plt.savefig(f'{title}-most-freq-words.png', bbox_inches='tight')
    plt.show()

def create_cv_pipeline(model):  
    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english')
    
    pipeline = make_pipeline(
        vectorizer,
        model
    )
    return pipeline


def create_tdidf_pipeline(model): 
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    
    pipeline = make_pipeline(
        vectorizer,
        model
    )
    return pipeline


if __name__ == "__main__":    
    emails_filepaths = sorted(glob.glob("data/emails/inmail.*"), key=email_num_key_compare)

    col_names =  ['email', 'label'] 
    emails_df = pd.DataFrame(columns = col_names) 

    with open('data/labels') as f:
        labels_lst = [line.rstrip().split(' ')[0] for line in f]

    for path, label in zip(emails_filepaths, labels_lst):
        with open(path, 'rb') as file:
            html = BeautifulSoup(file.read(),"html.parser")
            email_text = html.get_text()

            new_row = {'filename': path[12:], 'email': email_text, 'label': label}
            emails_df = emails_df.append(new_row, ignore_index=True)

    X = emails_df['email']
    y = emails_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

    nb_cv_model = MultinomialNB()
    nb_tfidf_model = MultinomialNB()

    nb_cv_pipeline = create_cv_pipeline(nb_cv_model)
    nb_tfidf_pipeline = create_tdidf_pipeline(nb_tfidf_model)

    rf_cv_model = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=3)
    rf_tfidf_model = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=3)

    rf_cv_pipeline = create_cv_pipeline(rf_cv_model)
    rf_tfidf_pipeline = create_tdidf_pipeline(rf_tfidf_model)

    lr_cv_model = LogisticRegression()
    lr_tfidf_model = LogisticRegression()

    lr_cv_pipeline = create_cv_pipeline(lr_cv_model)
    lf_tfidf_pipeline = create_tdidf_pipeline(lr_tfidf_model)

    pipelines = [nb_cv_pipeline, nb_tfidf_pipeline, rf_cv_pipeline, rf_tfidf_pipeline, lr_cv_pipeline, lf_tfidf_pipeline]

    for pipeline in pipelines:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_train)
        print(classification_report(y_train, y_pred))

    c_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))
    cv_email_matrix = c_vectorizer.fit_transform(X_train)
    nmf = NMF(n_components=15)
    W = nmf.fit_transform(cv_email_matrix)
    H = nmf.components_

    feature_names = c_vectorizer.get_feature_names()
    num_topics = 15

    top_10_topics_and_words = show_topics(nmf, feature_names, num_topics=num_topics)

    print(top_10_topics_and_words)