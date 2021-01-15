# CLEANING FROM https://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529e
# AND https://www.kaggle.com/ragnisah/text-data-cleaning-tweets-analysis

# TODO NB !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# TODO output -> DATASET_covid19_tweets_cleaned_NOhashtags.csv -> with list, string, tuples cleaned text and NO hashtags (in a separated column)
# TODO output -> DATASET_covid19_tweets_cleaned_NOhashtags.csv -> with list, string, tuples cleaned text and YES hashtags

# TODO this second cleaning file (cleaning_1.py) seems better then cleaning.py

# TODO choose if we want stemming (probably better avoid lemmatization), now we used stemming in this code
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
from datetime import datetime




df = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_tweets.csv")
#print(df.iloc[65503])
# keep only columns "date" and "text" and "hashtags"
df.drop(['user_name', 'user_location', 'user_description', 'user_created', 'user_followers', 'user_friends',
         'user_favourites', 'user_verified', 'source', 'is_retweet'], axis=1, inplace=True)
#print(df.iloc[65503])
#print(df)
#df.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/PROVA1.csv', index=False)
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
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER) # keep hashtags
# clean text -> step 1
df['tweet_preprocessed'] = df['text'].apply(lambda x: p.clean(x))
#print(df)
#df.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/PROVA.csv', index=False)
print(df.iloc[65503])



# TODO not used
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
# TODO end not used







# Everything put together in a list
def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    # TODO uncomment this line to add setemming
    text = [ps.stem(word) for word in tokens if (word not in stopword and word != '')]  # remove stopwords and stemming
    #text = [word for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text

# Everything put together in a string
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

# Everything put together for apriori algorithm
def clean_text_tuple(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    # TODO uncomment this line to add stemming
    text = [ps.stem(word) for word in tokens if (word not in stopword and word != '')]  # remove stopwords and stemming
    #text = [word for word in tokens if word not in stopword]  # remove stopwords not stemming
    text_tuple = tuple(text)
    return text_tuple

# TODO not used
def to_date(date_time):
    # split following datetime format in the dataset
    date_time_obj = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
    # date from datetime object
    date = date_time_obj.date()
    return date





# clean text -> step 2
df['text_cleaned'] = df['tweet_preprocessed'].apply(lambda x: clean_text(x))
df['text_cleaned_string'] = df['tweet_preprocessed'].apply(lambda x: clean_text_string(x))
df['text_cleaned_tuple'] = df['tweet_preprocessed'].apply(lambda x: clean_text_tuple(x))
# just date
df['date_only'] = df['date'].apply(lambda x: to_date(x))





# list of hashtags added to a new column -> don't know if it is useful -> NO we already have hashtags from the starting dataset
# DECOMMENT/COMMENT THIS LINE TO REMOVE/KEEP HASHTAGS
#df['hashtags'] = df['text'].apply(lambda x: re.findall(r"#(\w+)", x))


# DATASET GROUPBY DATE
#df1 = pd.DataFrame(df.groupby('date_only')['text_cleaned'].apply(list).reset_index())
#df2 = pd.DataFrame(df.groupby('date_only')['text_cleaned_string'].apply(list).reset_index())
df3 = pd.DataFrame(df.groupby('date_only')['text_cleaned_tuple'].apply(list).reset_index())


# DATASET WITH HASHTAGS
# SEPARATORE E' ' ' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#df.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_tweets_cleaned_PR.csv', index=False, sep=' ', line_terminator='\n')
#df1.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_group_text.csv', index=False)
#df2.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_group_string.csv', index=False)
df3.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_group_tuple.csv', index=False, sep=' ')
df3.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_group_tuple_sep_comma.csv', index=False)




'''def tot_clean_date_string(date_time, text, df):
    cleaned_text = clean_text_string(df[text])
    date = to_date(df[date_time])
    return date, cleaned_text'''

#df1 = df.iloc[:3]






# clean dataset
# df['text_cleaned'] = df['tweet_preprocessed'].apply(lambda x: clean_text(x))

'''file = open('pickle_input_apriori', 'wb')
pickle.dump(big_list, file)
file.close()'''

# LIST OF LISTS FOR APRIORI MLXTEND
# create list of lists for apriori algorithm of python
'''big_list = []
for (idx, row) in df.iterrows():
    big_list.append(clean_text(row.loc['tweet_preprocessed']))
print(big_list)

file = open('pickle_INPUT', 'wb')
pickle.dump(big_list, file)
file.close()'''

# LIST OF TUPLES FOR APRIORI EFFICIENT_APRIORI
# create list of tuples for apriori algorithm of python
'''big_list = []
for (idx, row) in df.iterrows():
    big_list.append(clean_text_tuple(row.loc['tweet_preprocessed']))
print(big_list)

file = open('pickle_INPUT_list_of_tuple', 'wb')
pickle.dump(big_list, file)
file.close()'''



# transform clened dataset into a csv file -> "covid19_tweets_cleaned.csv"
# df.to_csv(r'/home/veror/PycharmProjects/DataMiningProject/covid19_tweets_cleaned_3.csv', index=False)

#df['text_cleaned_list'] = df['tweet_preprocessed'].apply(lambda x: clean_text(x))
#df['text_cleaned_string'] = df['tweet_preprocessed'].apply(lambda x: clean_text_string(x))
#df['text_cleaned_tuple'] = df['tweet_preprocessed'].apply(lambda x: clean_text_tuple(x))
# DATATSET WITH REMOVED HASHTAGS
# df.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_tweets_cleaned.csv', index=False)
# DATASET WITH HASHTAGS
#df.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_tweets_cleaned_YEShashtags_CORRECT.csv', index=False)

# TODO do we need to keep also # and not to cut the words in hashtags?? Or can we consider them as normal terms?
# TODO if they are special we can take them from the column df['hashtags'] (line 38)

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