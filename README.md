# Email Spam Detector

## Purpose

The purpose of this project is to produce a predictive machine learning model that will be able to determine whether an email is spam or "ham" (legitimate email) specifically based on the text content within the email.

## Context

### What is Spam?
Spam email is considered similar to the junk mail one would receive in their physical mail. The majority these spam emails are unsolicited commercial advertising and simply result in clogging up your email inbox. Unlike regular junk mail though, not only is email spam annoying, but may contain potentially dangerous content including:

- links to websites meant to steal personal information or download malware
- a seemingly legitimate file attachment that has malware embedded.

Today, email spam is prevalent as ever. In November 2020, the average daily volume were as follows:

- 210.54 Billion Average Daily Spam Volume (84.84% of total global email traffic)
- 38.16 Billion Average Daily Legitimate Volume (15.16% of total global email traffic) 

Source: https://talosintelligence.com/reputation_center/email_rep#global-volume (Provides most-up-to-date statistics)

## The Dataset

The data is sourced from [here] and contains 75,419 emails in a raw text files with an index file containing the spam or ham labels for each email text file. The email text files included all data found in an email such as metadata, HTML code, MIME information, text and etc.


## Data Cleaning Process

Overview: Data existed as raw email text files which needed to be parsed and loaded into a Pandas Dataframe for further analysis. The steps to create this data were take as follows:

1. Used glob module to produce a list of paths to every email file
2. With the above list, used Beautiful Soup module to parse only text from the raw email and filtered out metadata.
3. Loaded email text data into a dataframe with corresponding email filename and label.

Snippet of dataframe shown below:

|filename | email | label | 
|:-----:|:-----:|:-----:|
| inmail.1 | ----8896484051606557286Content-Type: text/html... | spam |
| inmail.2 | Hi, i've just updated from the gulus and I che... | ham |
| inmail.3 | --F05E057D3F0C38DA4867D386Content-Type: text/p... | spam |
| inmail.4 | Hey Billy, it was really fun going out the oth... | spam |
| inmail.5 | This is a multi-part message in MIME format.--... | spam |

## Exploratory Data Analysis

For exploratory data analysis, since the data existed in a text format, the CountVectorizer module was used to calculate the frequency of terms used in the emails.

Below is the top 15 words used in the email dataset:



## Model Selection and Performance Metrics

### CountVectorizer - MultimodalNB Pipeline

### CountVectorizer - RandomForestClassifier Pipeline

### TfidfVectorizer - MultimodalNB Pipeline

### TfidfVectorizer - RandomForestClassifier Pipeline

