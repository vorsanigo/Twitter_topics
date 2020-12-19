# CLEANING FROM https://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529e
# AND https://www.kaggle.com/ragnisah/text-data-cleaning-tweets-analysis

# TODO NB !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO decide if keeping hashtags in the text -> YES IT HAS TO BE TAKEN !! -> change code to keep hashtags

# TODO output -> covid19_tweets_cleaned_2.csv -> with list-strings as cleaned text
# TODO output -> covid19_tweets_cleaned_string.csv -> with strings as cleaned text

# TODO this second cleaning file (cleaning_1.py) seems better then cleaning.py

# TODO choose if we want to add also stemming (probably better avoid lemmatization)

# STEMMING: Working -> Work ___ it eliminates affixes from a word (stemming is faster)
# LEMMATIZATION: Better -> Good ___ it uses vocabulary and morphological analysis of words to detect the lemma of the word (basic form)

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
import pickle

df = pd.read_csv("/home/veror/PycharmProjects/DataMiningProject/COPY_covid19_tweets .csv")
# keep only columns "date" and "text"
df.drop(['user_name', 'user_location', 'user_description', 'user_created', 'user_followers', 'user_friends',
         'user_favourites', 'user_verified', 'hashtags', 'source', 'is_retweet'], axis=1, inplace=True)

# list of hashtags added to a new column -> don't know if it is useful
df['hashtags'] = df['text'].apply(lambda x: re.findall(r"#(\w+)", x))

#segmenter using the word statistics from Twitter
seg_tw = Segmenter(corpus="twitter")

# PROBABLY NOT USEFUL
#forming a separate feature for cleaned tweets
#print(enumerate(df['text']))
#for i,v in enumerate(df['text']):
    #df.loc[v,'text'] = p.clean(i)
    #print(i)
    #print(df.loc[v,'text'])

# CLEANING
# preprocess tweets
# tweet-preprocessor package deals with URLs, mentions, reserved words, emojis, smileys
df['tweet_preprocessed'] = df['text'].apply(lambda x: p.clean(x))


# CLEANING: step 2 -> remove punctuation, tokenization, stopwords, stemming(?)
# remove punctuation and numbers?
# string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text) # remove numbers
    return text

#df['text_punct'] = df['text'].apply(lambda x: remove_punct(x))


# tokenization splits the text string in a list of words (strings)
def tokenization(text):
    text = re.split('\W+', text)
    return text

#df['text_tokenized'] = df['text_punct'].apply(lambda x: tokenization(x.lower()))


# remove stopwords
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

#df['text_no_stopword'] = df['text_tokenized'].apply(lambda x: remove_stopwords(x))

#print(df['text_no_stopword'])


# stemming
ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

#df['text_stemmed'] = df['text_no_stopword'].apply(lambda x: stemming(x))

'''
# lemmatization
wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

df['text_lemmatized'] = df['text_no_stopword'].apply(lambda x: lemmatizer(x))'''

# Everything put together in a list
def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    # TODO uncomment this line to add setemming
    #text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    text = [word for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text

# Everything put together in a string
def clean_text_string(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    # TODO uncomment this line to add setemming
    #text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    text = ""
    for word in tokens:
        if word not in stopword:
            text = text + " " + word  # remove stopwords and stemming
    return text

# Everything put together for apriori algorithm
def clean_text_tuple(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    # TODO uncomment this line to add setemming
    #text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    text = [word for word in tokens if word not in stopword]  # remove stopwords and stemming
    text_tuple = tuple(text)
    return text_tuple

# clean dataset
# df['text_cleaned'] = df['tweet_preprocessed'].apply(lambda x: clean_text(x))

# create list of tuples for apriori algorithm of python
'''big_list = []
for (idx, row) in df.iterrows():
    big_list.append(clean_text_tuple(row.loc['tweet_preprocessed']))'''

'''file = open('pickle_input_apriori', 'wb')
pickle.dump(big_list, file)
file.close()'''

# transform clened dataset into a csv file -> "covid19_tweets_cleaned.csv"
# df.to_csv(r'/home/veror/PycharmProjects/DataMiningProject/covid19_tweets_cleaned_3.csv', index=False)

df['text_cleaned'] = df['tweet_preprocessed'].apply(lambda x: clean_text_string(x))
df.to_csv(r'/home/veror/PycharmProjects/DataMiningProject/covid19_tweets_cleaned_string.csv', index=False)

# preprocess -> step 2
#
'''def preprocess_data(data):
    # Removes Numbers
    data = data.astype(str).str.replace('\d+', '')
    lower_text = data.str.lower()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    w_tokenizer = TweetTokenizer()

    def lemmatize_text(text):
      return [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize((text))]

    def remove_punctuation(words):
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', (word))
            if new_word != '':
                new_words.append(new_word)
        return new_words

    words = lower_text.apply(lemmatize_text)
    words = words.apply(remove_punctuation)
    return pd.DataFrame(words)

# remove stopwords
stop_words = set(stopwords.words('english'))
df['text_cleaned'] = df['tweet_preprocessed'].apply(lambda x: [item for item in x if item not in stop_words])




# ALTRE COSE DAL SITO, FORSE INUTILI

# segmenter using the word statistics from Twitter
seg_tw = Segmenter(corpus="twitter")
a = []
for i in range(len(df)):
    if df['hashtag'][i] != a
        listToStr1 = ' '.join([str(elem) for elem in df['hashtag'][i]])
        df.loc[i,'Segmented#'] = seg_tw.segment(listToStr1)

df.to_csv(r'/home/veror/PycharmProjects/DataMiningProject/covid19_tweets_cleaned_2.csv', index=False)'''

'''#Frequency of words
fdist = FreqDist(df['Segmented#'])
#WordCloud
wc = WordCloud(width=800, height=400, max_words=50).generate_from_frequencies(fdist)
plt.figure(figsize=(12,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()'''