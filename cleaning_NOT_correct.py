#  CLEANING FROM https://www.kaggle.com/ragnisah/text-data-cleaning-tweets-analysis
# TODO output -> covid_19_tweets_cleaned.csv

import pandas as pd
import numpy as np
import nltk
# TODO nltk.download -> leave it here ?
nltk.download('stopwords') #OR from command line: python -m nltk.downloader stopwords
nltk.download('wordnet')
import string
import re


df = pd.read_csv("covid19_tweets.csv")
# keep only columns "date" and "text"
df.drop(['user_name', 'user_location', 'user_description', 'user_created', 'user_followers', 'user_friends',
         'user_favourites', 'user_verified', 'hashtags', 'source', 'is_retweet'], axis=1, inplace=True)

# TODO choose which functions to use for the cleaning
# TODO the output of this section has to be renamed into "covid19_tweets_cleaned.csv" and put in the project with the original one
# TODO remove "http..."?
# TODO some elements in the final lists are " -> what does it mean?
# TODO other stuff to check in the cleaning -> modify cleaning

# remove punctuation and numbers?
# string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text) # remove numbers
    return text

df['text_punct'] = df['text'].apply(lambda x: remove_punct(x))


# tokenization splits the text string in a list of words (strings)
def tokenization(text):
    text = re.split('\W+', text)
    return text

df['text_tokenized'] = df['text_punct'].apply(lambda x: tokenization(x.lower()))


# remove stopwords
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

df['text_no_stopword'] = df['text_tokenized'].apply(lambda x: remove_stopwords(x))

print(df['text_no_stopword'])


# stemming
ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

df['text_stemmed'] = df['text_no_stopword'].apply(lambda x: stemming(x))


# lemmatization
wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

df['text_lemmatized'] = df['text_no_stopword'].apply(lambda x: lemmatizer(x))

# Everything put together
def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text

# clean dataset
df['text_cleaned'] = df['text'].apply(lambda x: clean_text(x))

# transform clened dataset into a csv file -> "covid19_tweets_cleaned.csv"
df.to_csv(r'/home/veror/PycharmProjects/DataMiningProject/covid19_tweets_cleaned.csv', index=False)