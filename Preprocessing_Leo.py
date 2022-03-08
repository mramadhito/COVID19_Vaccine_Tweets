#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:01:35 2021

@author: leomultiverse2
"""

import pandas as pd
import re
import pickle
from util.my_functions import *
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import os
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sw = set(stopwords.words("English"))
stemmer = PorterStemmer()

data_raw = pd.read_csv("vaccination_all_tweets.csv")
data_raw = pd.DataFrame(data_raw)
data_raw.head()
data_raw.shape

needed_cols = ["id", "user_location", "date", "text", "hashtags","retweets"]
data_raw = data_raw[needed_cols]
data_raw['user_location'] = data_raw['user_location'].astype(str)
data_raw['hashtags'] = data_raw['hashtags'].astype(str)

# Change uder ID to a categorical variable
data_raw.id = data_raw.id.astype('category')

# Exclude the time from date
data_raw.date = pd.to_datetime(data_raw.date).dt.date

# Get text column
texts = data_raw['text']

# Clean text column

def clean_txt(string):
    import re
    cleaned = re.sub('[^A-Za-z0-9]', " ", string).strip().lower()
    return cleaned

data_raw.text = data_raw['text'].apply(clean_txt)

data_raw.user_location = [re.sub('[^A-Za-z0-9]', " ", word).strip().lower() for word in data_raw['user_location']]

data_raw.hashtags = data_raw['hashtags'].apply(clean_txt)

# Remove stopwords in text column

def rem_sw(var):
    my_test = [word for word in var.split() if word not in sw]
    my_test = ' '.join(my_test)
    return my_test

data_raw.text = data_raw['text'].apply(rem_sw)

# Stemming for text column

def stem_fun(var):
    res = [stemmer.stem(word) for word in var.split()]
    res = ' '.join(res)
    return res

data_raw.text = data_raw['text'].apply(stem_fun)

# A large list of words in data_raw['text']
words_in_text = [word for line in data_raw['text'] for word in line.split()]

# Most frequent words top 100
word_ct = Counter(words_in_text).most_common(100)
word_ct_df = pd.DataFrame(word_ct)
word_ct_df.columns = ['word', 'freq']


# Sentiment Analysis
sid = SentimentIntensityAnalyzer()
pol_score = lambda x: sid.polarity_scores(x)
sent_scores = data_raw.text.apply(pol_score)
sent_df = pd.DataFrame(data = list(sent_scores))
sent_df.head()

labelize_fn = lambda x : 'neutral' if x==0 else('positive' if x>0 else 'negative')
sent_df['labels'] = sent_df.compound.apply(labelize_fn)
sent_df.head()


data = data_raw.join(sent_df.labels)

import seaborn as sns
counts_df = data.labels.value_counts().reset_index()
sns.barplot(x = 'index', y = 'labels', data = counts_df)










