# CLEANING FROM https://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529e
# AND https://www.kaggle.com/ragnisah/text-data-cleaning-tweets-analysis

# STEMMING: Working -> Work ___ it eliminates affixes from a word (stemming is faster)
# LEMMATIZATION: Better -> Good ___ it uses vocabulary and morphological analysis of words to detect the lemma of the word (basic form)

import pandas as pd
import string
import re
from ekphrasis.classes.segmenter import Segmenter
import preprocessor as p
#important libraries for preprocessing using NLTK
import nltk
nltk.download
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
import pickle
from datetime import datetime

stopword = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

# read the starting dataset
df = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/auspol2019.csv")
print(df)

# keep only columns "date" and "text" and "hashtags"
df.drop(['id', 'retweet_count', 'favorite_count', 'user_id', 'user_name', 'user_screen_name', 'user_description',
         'user_location', 'user_created_at'], axis=1, inplace=True)
df.columns = ['date', 'text']
print(df.columns)
print("ssssssss", df)

#df = pd.DataFrame({'date':['01/03/1987', '2003', 'Jan-08', '31/01/2010', '2/13/2016'],'value':range(5)})
#pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='drop')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

df = df[pd.notnull(df['date'])]
print("qqqqqqqqq", df)

'''for i in range(len(df['date'])):
    try:
        pd.to_datetime(df.iloc[i]['date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
    except ValueError:
        df.drop(df.index[i])'''

#print(df.iloc[65503])
#print(df)
#df.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/PROVA1.csv', index=False)
#segmenter using the word statistics from Twitter
# TODO PROBABLY TO ELIMINATE
seg_tw = Segmenter(corpus="twitter")

# PROBABLY NOT USEFUL
#forming a separate feature for cleaned tweets
#print(enumerate(df['text']))
#for i,v in enumerate(df['text']):
    #df.loc[v,'text'] = p.clean(i)
    #print(i)
    #print(df.loc[v,'text'])



# CLEANING

# step 1: preprocess tweets
# tweet-preprocessor package deals with URLs, mentions, reserved words, emojis, smileys, we keep only hashtags
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER) # keep hashtags
df['tweet_preprocessed'] = df['text'].apply(lambda x: p.clean(x))


# Different functions to have the cleaned text in different possible forms

# Clean and put the text into a list
def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    # TODO uncomment this line to add setemming
    text = [ps.stem(word) for word in tokens if (word not in stopword and word != '')]  # remove stopwords and stemming
    #text = [word for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text

# Clean and put the text into a string
def clean_text_string(text):
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

# Clean and put the text into a tuple
def clean_text_tuple(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    # TODO uncomment this line to add stemming
    text = [ps.stem(word) for word in tokens if (word not in stopword and word != '')]  # remove stopwords and stemming
    #text = [word for word in tokens if word not in stopword]  # remove stopwords not stemming
    text_tuple = tuple(text)
    return text_tuple

# TODO not used ????????????? used later
def to_date(date_time):
    # split following datetime format in the dataset
    #date_time_obj = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
    # date from datetime object
    date = date_time.date()
    return date



# step 2: apply the cleaning functions on the preprocessed tweets and save results into new columns in the dataframe
df['text_cleaned'] = df['tweet_preprocessed'].apply(lambda x: clean_text(x))
df['text_cleaned_string'] = df['tweet_preprocessed'].apply(lambda x: clean_text_string(x))
df['text_cleaned_tuple'] = df['tweet_preprocessed'].apply(lambda x: clean_text_tuple(x))
# create column containing only the date, without the time
df['date_only'] = df['date'].apply(lambda x: to_date(x))





# list of hashtags added to a new column -> don't know if it is useful -> NO we already have hashtags from the starting dataset
# DECOMMENT/COMMENT THIS LINE TO REMOVE/KEEP HASHTAGS
#df['hashtags'] = df['text'].apply(lambda x: re.findall(r"#(\w+)", x))


# create the datsets grouped by date
df1 = pd.DataFrame(df.groupby('date_only')['text_cleaned'].apply(list).reset_index())
df2 = pd.DataFrame(df.groupby('date_only')['text_cleaned_string'].apply(list).reset_index())
df3 = pd.DataFrame(df.groupby('date_only')['text_cleaned_tuple'].apply(list).reset_index())


# save the datasets created into csv
# SEPARATORE E' ' ' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
df.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_AUSTRALIA_tweets_cleaned.csv', index=False, sep=' ', line_terminator='\n')
df1.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_AUSTRALIA_group_text.csv', index=False)
df2.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_AUSTRALIA_group_string.csv', index=False)
df3.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_AUSTRALIA_group_tuple.csv', index=False, sep=' ')
df3.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_AUSTRALIA_group_tuple_sep_comma.csv', index=False)


