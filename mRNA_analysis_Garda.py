# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 14:29:59 2021

@author: garda
"""

import pandas as pd
import numpy as np
import re
import pickle
from my_functions import *
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import os
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

#Remove URLs
data_raw.text = [re.sub(r'http\S+', " ", word).strip().lower() for word in data_raw['text']]

data_raw.user_location = [re.sub('[^A-Za-z0-9]', " ", word).strip().lower() for word in data_raw['user_location']]

data_raw.hashtags = data_raw['hashtags'].apply(clean_txt)

# Remove stopwords in text column
sw = set(stopwords.words("English"))
stemmer = PorterStemmer()

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

#Separate the data into mRNA and non-mRNA

mRNA_data = data_raw[data_raw['text'].str.contains('pfizer|biontech|moderna')]
non_mRNA_data = data_raw[data_raw['text'].str.contains('sinopharm|sinovac|astrazeneca|covaxin|sputnik')]

#Generate word clouds

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
text_mRNA = mRNA_data['text'].str.cat(sep=' ')
wordcloud_mRNA = WordCloud(max_font_size =50, 
                      max_words=100, 
                      background_color="white").generate(text_mRNA)
plt.figure(figsize=(8,8))
plt.imshow(wordcloud_mRNA, interpolation='bilinear')
plt.axis("off")
plt.show()

text_non_mRNA = non_mRNA_data['text'].str.cat(sep=' ')
wordcloud_non_mRNA = WordCloud(max_font_size =50, 
                      max_words=100, 
                      background_color="white").generate(text_non_mRNA)
plt.figure(figsize=(8,8))
plt.imshow(wordcloud_non_mRNA, interpolation='bilinear')
plt.axis("off")
plt.show()

#Compare tf-idf between the groups

from sklearn.feature_extraction.text import TfidfVectorizer

docs_vaccines = [text_mRNA, text_non_mRNA]
tfidf = TfidfVectorizer(ngram_range=(2,3))
terms = tfidf.fit_transform(docs_vaccines)
tfidf_count = pd.DataFrame(terms.toarray(), columns = tfidf.get_feature_names())

tfidf_mRNA = tfidf_count.iloc[0]
tfidf_mRNA = tfidf_mRNA.sort_values(0, ascending=False)
tfidf_mRNA = pd.DataFrame(tfidf_mRNA)
top40_mRNA = tfidf_mRNA.head(40)

tfidf_non_mRNA = tfidf_count.iloc[1]
tfidf_non_mRNA = tfidf_non_mRNA.sort_values(0, ascending=False)
tfidf_non_mRNA = pd.DataFrame(tfidf_non_mRNA)
top40_non_mRNA = tfidf_non_mRNA.head(40)

##Visualize ranking of tfidf

plt.rcParams["figure.figsize"] = (11,8)
plt.barh(top40_mRNA.index, top40_mRNA[0])
plt.gca().invert_yaxis()
plt.xlabel("TF-IDF")
plt.ylabel("Term")
plt.show()

plt.rcParams["figure.figsize"] = (11,8)
plt.barh(top40_non_mRNA.index, top40_non_mRNA[1])
plt.gca().invert_yaxis()
plt.xlabel("TF-IDF")
plt.ylabel("Term")
plt.show()

#Compare sentiments between mRNA and non-mRNA

def sentiment_creator(df):
    analyzer = SentimentIntensityAnalyzer()
    df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in df['text']]
    df['neg'] = [analyzer.polarity_scores(x)['neg'] for x in df['text']]
    df['neu'] = [analyzer.polarity_scores(x)['neu'] for x in df['text']]
    df['pos'] = [analyzer.polarity_scores(x)['pos'] for x in df['text']]
    df['sentiment'] = np.where(df['neg']>df['pos'], 'negative', 
                               np.where(df['pos']>df['neg'], 'positive', 'neutral'))
    return df

mRNA_data = sentiment_creator(mRNA_data)
non_mRNA_data = sentiment_creator(non_mRNA_data)

##Visualize sentiment score

###Histogram of compound scores
plt.hist(mRNA_data['compound'])
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Sentiment scores of mRNA vaccine tweets")
plt.show

plt.hist(non_mRNA_data['compound'])
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Sentiment scores of non-mRNA vaccine tweets")
plt.show

##Pie chart of sentiment by category
mRNA_sentiment_count = pd.DataFrame(mRNA_data['sentiment'].value_counts())
mRNA_sentiment_count['prop'] = mRNA_sentiment_count['sentiment']/sum(mRNA_sentiment_count['sentiment'])

non_mRNA_sentiment_count = pd.DataFrame(non_mRNA_data['sentiment'].value_counts())
non_mRNA_sentiment_count['prop'] = non_mRNA_sentiment_count['sentiment']/sum(non_mRNA_sentiment_count['sentiment'])

fig1, ax1 = plt.subplots()
ax1.pie(mRNA_sentiment_count["prop"], labels=mRNA_sentiment_count.index, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Composition of mRNA tweets by sentiment')
plt.show()

fig2, ax2 = plt.subplots()
ax2.pie(non_mRNA_sentiment_count["prop"], labels=non_mRNA_sentiment_count.index, 
        autopct='%1.1f%%',
        shadow=True, startangle=90)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Composition of non-mRNA tweets by sentiment')
plt.show()

#Compare topics between the groups (LDA)

import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 
                   'use', 'not', 'would', 'say', 'could', 
                   '_', 'be', 'know', 'good', 'go', 'get', 
                   'do', 'done', 'try', 'many', 'some', 'nice', 
                   'thank', 'think', 'see', 'rather', 'easy', 'easily', 
                   'lot', 'lack', 'make', 'want', 'seem', 'run',
                   'need', 'even', 'right', 'line', 'even', 
                   'also', 'may', 'take', 'come'])

def sent_to_words(sentences):
    for sent in sentences:
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  



##MRNA only
data_words = list(sent_to_words(mRNA_data['text']))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# !python3 -m spacy download en  # run in terminal once
def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    nlp.max_length = 1000000000
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

data_ready = process_words(data_words)  # processed Text Data!

# Create Dictionary
id2word = corpora.Dictionary(data_ready)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=4, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)

print(lda_model.print_topics())

##Non-MRNA only
data_words2 = list(sent_to_words(non_mRNA_data['text']))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words2, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words2], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# !python3 -m spacy download en  # run in terminal once
def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    nlp.max_length = 1000000000
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

data_ready2 = process_words(data_words2)  # processed Text Data!

# Create Dictionary
id2word2 = corpora.Dictionary(data_ready2)

# Create Corpus: Term Document Frequency
corpus2 = [id2word.doc2bow(text) for text in data_ready2]

# Build LDA model
lda_model2 = gensim.models.ldamodel.LdaModel(corpus=corpus2,
                                           id2word=id2word2,
                                           num_topics=4, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)

print(lda_model2.print_topics())

##Visualization of LDA
import matplotlib.colors as mcolors
###MRNA
from collections import Counter
topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in data_ready for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=False, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.25); ax.set_ylim(0, 50000)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()

###Non-MRNA
topics2 = lda_model2.show_topics(formatted=False)
data_flat2 = [w for w_list in data_ready2 for w in w_list]
counter2 = Counter(data_flat2)

out = []
for i, topic in topics2:
    for word, weight in topic:
        out.append([word, i , weight, counter2[word]])

df2 = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df2.loc[df2.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df2.loc[df2.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.25); ax.set_ylim(0, 50000)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df2.loc[df2.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()