import pandas as pd
import string
import re
from ekphrasis.classes.segmenter import Segmenter
import preprocessor as p
#important libraries for preprocessing using NLTK
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from datetime import datetime


# CLEANING FROM https://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529e
# AND https://www.kaggle.com/ragnisah/text-data-cleaning-tweets-analysis

# STEMMING: Working -> Work ___ it eliminates affixes from a word (stemming is faster)
# LEMMATIZATION: Better -> Good ___ it uses vocabulary and morphological analysis of words to detect the lemma of the word (basic form)


ps = nltk.PorterStemmer()
stopword = nltk.corpus.stopwords.words('english')


# Different functions to have the cleaned text in different possible forms

def clean_text(text):
    '''Given a text (string), it removes punctuation, numbers, it does tokenization, stemming, and removes stopwords,
    the output is a list of tokens'''
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    # TODO uncomment this line to add setemming
    text = [ps.stem(word) for word in tokens if (word not in stopword and word != '')]  # remove stopwords and stemming
    #text = [word for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text

def clean_text_string(text):
    '''Given a text (string), it removes punctuation, numbers, it does tokenization, stemming, and removes stopwords,
    the output is a string of tokens'''
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    # TODO uncomment this line to add setemming
    #text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    text = ""
    #for word in tokens:
    #    if word not in stopword:
    #        text = text + " " + word  # remove stopwords NO stemming
    for word in tokens:
        if (word not in stopword and word != ''):
            text = text + " " + ps.stem(word) # remove stopwords and stemming
    return text

def clean_text_tuple(text):
    '''Given a text (string), it removes punctuation, numbers, it does tokenization, stemming, and removes stopwords,
    the output is a tuple of tokens'''
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    # TODO uncomment this line to add stemming
    text = [ps.stem(word) for word in tokens if (word not in stopword and word != '')]  # remove stopwords and stemming
    #text = [word for word in tokens if word not in stopword]  # remove stopwords not stemming
    text_tuple = tuple(text)
    return text_tuple


def to_date(date_time):
    '''Given a date_time, it returns a date'''
    # split following datetime format in the dataset
    #date_time_obj = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
    # date from datetime object
    #date = date_time_obj.date()
    date = date_time.date()
    return date


def cleaning_fun(df_path, list_column_drop, path_cleaned, path_pickle_cleaned, path_grouped, path_pickle_grouped):
    '''Given a dataset of tweets, it cleans tweets' text and it groups the cleaned tweets by date'''

    # read the starting dataset
    df = pd.read_csv(df_path)

    # keep only columns selected ("date", "text", (and "hashtags"))
    df.drop(list_column_drop, axis=1, inplace=True)
    #df.drop(['user_name', 'user_location', 'user_description', 'user_created', 'user_followers', 'user_friends',
    #         'user_favourites', 'user_verified', 'source', 'is_retweet'], axis=1, inplace=True)

    # for australian dataset
    if df.shape[1] == 2:
        df.columns = ['date', 'text']

    # convert to datetime and drop dates with not correct format
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df[pd.notnull(df['date'])]

    #segmenter using the word statistics from Twitter
    # TO ELIMINATE
    seg_tw = Segmenter(corpus="twitter")

    # CLEANING

    # step 1: preprocess tweets
    # tweet-preprocessor package deals with URLs, mentions, reserved words, emojis, smileys, we keep only hashtags
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)
    df['tweet_preprocessed'] = df['text'].apply(lambda x: p.clean(x))

    # step 2: apply the cleaning functions on the preprocessed tweets and save results into new columns in the dataframe
    df['text_cleaned'] = df['tweet_preprocessed'].apply(lambda x: clean_text(x))
    df['text_cleaned_string'] = df['tweet_preprocessed'].apply(lambda x: clean_text_string(x))
    df['text_cleaned_tuple'] = df['tweet_preprocessed'].apply(lambda x: clean_text_tuple(x))

    # create column containing only the date, without the time
    df['date_only'] = df['date'].apply(lambda x: to_date(x))

    # alternative manual way to remove hashtags
    # df['hashtags'] = df['text'].apply(lambda x: re.findall(r"#(\w+)", x))

    # create the datsets grouped by date -> we only use tuple
    #df1 = pd.DataFrame(df.groupby('date_only')['text_cleaned'].apply(list).reset_index())
    #df2 = pd.DataFrame(df.groupby('date_only')['text_cleaned_string'].apply(list).reset_index())
    df3 = pd.DataFrame(df.groupby('date_only')['text_cleaned_tuple'].apply(list).reset_index())

    # save the datasets created into csv
    # separator is ' '
    df.to_csv(path_cleaned, index=False, sep=' ', line_terminator='\n')
    '''df.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_tweets_cleaned.csv', index=False,
              sep=' ', line_terminator='\n')'''
    df.to_pickle(path_pickle_cleaned)
    #df1.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_group_text.csv', index=False)
    #df2.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_group_string.csv', index=False)
    df3.to_csv(path_grouped, index=False, sep=' ')
    df3.to_pickle(path_pickle_grouped)
    #df3.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_group_tuple.csv', index=False, sep=' ')
    #df3.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_group_tuple_sep_comma.csv', index=False)