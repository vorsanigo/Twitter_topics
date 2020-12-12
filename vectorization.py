# VECTORIZATION OF TEXT FOR TF-IDF
# CLUSTERING
# https://pythonprogramminglanguage.com/kmeans-text-clustering/
# http://brandonrose.org/clustering

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pickle
import numpy as np
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.cluster import DBSCAN

# TODO since after the cleaning some rows have text = nan, should we eliminate them?
# TODO maybe we can directly create the list tweets_list eliminating nan instead of doing it later in tweets_no_nan



# VECTORIZATION

df = pd.read_csv('/home/veror/PycharmProjects/DataMiningProject/covid19_tweets_cleaned_string.csv')

#print(df.index.isnull())
#df.loc[df.index.isnull()]
'''x = df['text_cleaned'].isnull().values
x1 = np.array(x)
print(x1)
print(np.argwhere(np.isnan(x1)))'''

# create list of all the cleaned tweets
'''def create_list(df, column):
    tweets = []
    for (idx, row) in df.iterrows():
        tweets.append(row.loc[column])
    return tweets

tweets_list = create_list(df, 'text_cleaned')

file = open('pickle_tweets_list', 'wb')
pickle.dump(tweets_list, file)
file.close()'''

file = open('/home/veror/PycharmProjects/DataMiningProject/pickle_tweets_list', 'rb')
tweets_list = pickle.load(file)
'''for i in tweets_list:
    print(i)
    if type(i) == 'string':
        print('NO')
'''

#x1 = np.array(tweets_list)
#print(type(x1[1]))
#print(df['text_cleaned'].isnull().values.any())


#print(np.argwhere(np.isnan(x1)))
#print(tweets_list)

# remove tweets nan -> dimension of tweets_no_nan < dimension of tweets_list (= dimension of dataset -> ton num of tweets)
tweets_no_nan = [tweet for tweet in tweets_list if str(tweet) != 'nan']

# tweets_list = ['ciao cioa', 're', 'a d erfgjidioo']
#  vectorization
l = [' ciao re ', 'ciao re', ' ', 's', ' r ', ' ', 'quatto re ciao', 'buona doemnixa reale']
# TODO maybe change parameters of TfidfVectorizer ?
vectorizer = TfidfVectorizer()
#tweets_vectors = vectorizer.fit_transform(tweets_no_nan) # TODO maybe rename it "matrix"
tweets_vectors = vectorizer.fit_transform(l)
print(vectorizer.get_feature_names())
print(tweets_vectors.shape)
print(tweets_vectors)




#--------------------------------------------------------------------------------------------

# 1) run K-MEAN clustering algorithm
# TODO how to choose k? And number of items per cluster?
# TODO understand parameters of the model, how to manage them

true_k = 3 # k = number of clusters
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(tweets_vectors)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :2]: # 3 is probably number of feature in the cluster
        print(' %s' % terms[ind]),
    #print


'''print("\n")
print("Prediction")

Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cat is hungry."])
prediction = model.predict(Y)
print(prediction)'''

#-------------------------------------------------------------------------------------------

# 2) HIERARCHICAL clustering

'''linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters'''

#------------------------------------------------------------------------------------------

# DBSCAN clustering

# db = DBSCAN(eps=0.3, min_samples=10).fit(tweets_vectors)




















