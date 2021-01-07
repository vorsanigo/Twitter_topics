import pandas as pd
from efficient_apriori import apriori
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


from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_tweets_cleaned_YEShashtags.csv")
print(df.size)
print(df)

# TODO MLXTEND APRIORI

te = TransactionEncoder()

'''stringA = "['come', 'newstalkbreakfast', 'kick', 'run', 'top', 'stori']"
res = stringA.strip('][').split(', ')
print(res)

stringE = "come you hey friend eee"
li = list(stringE.split(" "))
print("eeeeeeeeeeeeeeeeeeeeeee")
print(li)
'''

'''dataset_list_file = open('pickle_INPUT_list_of_list', 'rb')
dataset_list = pickle.load(dataset_list_file)
print(dataset_list)'''


'''te_ary = te.fit(dataset_list).transform(dataset_list)
df1 = pd.DataFrame(te_ary, columns=te.columns_)
print("-------------")
print(df1)

res = fpgrowth(df1, min_support=0.1, use_colnames=True)
print(res)'''

'''def get_list(str):
  li = list(str.split(" "))
  return li

print(get_list("   c d rrr"))

big_list = []
for (idx, row) in df.iterrows():
    #print(type(row.loc['text_cleaned_string']))
    print(row.loc['text_cleaned_string'])
    row.loc['text_cleaned_string'].split(" ")
    #big_list.append(get_list(row.loc['text_cleaned_string']))
print(big_list)

transactions = [['eggs', 'bacon', 'soup'],
                ['eggs', 'bacon', 'apple', 'pen', 'soup'],
                ['soup', 'bacon', 'banana', 'cucumber', 'vodka', 'soup']]

te_ary = te.fit(transactions).transform(transactions)
df1 = pd.DataFrame(te_ary, columns=te.columns_)
print("-------------")
print(df1)

res = fpgrowth(df1, min_support=0.6, use_colnames=True)
print(res)'''

# TODO END MLXTEND APRIORI




'''def data_generator(filename):
  """
  Data generator, needs to return a generator to be called several times.
  Use this approach if data is too large to fit in memory. If not use a list.
  """
  def data_gen():
    with open(filename) as file:
      for line in file:
        yield tuple(k.strip() for k in line.split(','))

  return data_gen

transactions = data_generator('dataset.csv')
itemsets, rules = apriori(transactions, min_support=0.9, min_confidence=0.6)'''




#apriori_input_file = open('pickle_input_apriori', 'rb')
#apriori_input = pickle.load(apriori_input_file)
#print('labels_lookup_CURRENCY560', apriori_input)

# for experiments
'''stopword = nltk.corpus.stopwords.words('english')

def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    print(1,text_lc)
    print(type(text_lc))
    text_rc = re.sub('[0-9]+', '', text_lc)
    print(2,text_rc)
    print(type(text_rc))
    tokens = re.split('\W+', text_rc)    # tokenization
    print(3,tokens)
    print(type(tokens))
    # TODO uncomment this line to add setemming
    #text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    text = [word for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text

# Everything put together
def clean_text_tuple(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    # TODO uncomment this line to add setemming
    #text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    text = [word for word in tokens if word not in stopword]  # remove stopwords and stemming
    text_tuple = tuple(text)
    return text_tuple

x = clean_text_tuple("ciao and come ti chiamo amico")
print(4,x)
print(type(x))'''


# example for experiments
'''d = {'col1': ["ciao and come ti chiamo amico", "ehllo world and you all"]}

df = pd.DataFrame(data=d)
print(df)

big_list = []
for (idx, row) in df.iterrows():
    print(row.loc['col1'])
    big_list.append(clean_text_tuple(row.loc['col1']))

print("AAA", big_list)'''




# TODO EFFICIENT APRIORI

# Original data
'''transactions = [('eggs', 'bacon', 'soup'),
                ('eggs', 'bacon', 'apple', 'pen', 'soup'),
                ('soup', 'bacon', 'banana', 'cucumber', 'vodka', 'soup')]

# Convert to panas.DataFrame
df = pd.DataFrame(transactions)
print(df)

# first -> identify rules
itemsets, rules = apriori(transactions, min_support=0.2, min_confidence=1)

rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
  print(rule)  # Prints the rule and its confidence, support, lift, ...

print('------------------------------------------------------------------------')

# second -> identify frequent itemsets
itemsets, rules = apriori(transactions, output_transaction_ids=True)
print(itemsets)
# {1: {('bacon',): ItemsetCount(itemset_count=3, members={0, 1, 2}), ...
'''

'''# experiments on "COPY_covid19_tweets_cleaned_2.csv"
df = pd.read_csv("COPY_covid19_tweets_cleaned_2 .csv")
y = df.iloc[23]['text_cleaned']
print(type(y))
print(tuple(y))'''
#df['tuple_text_cleaned'] = df['text_cleaned'].apply(lambda x: print(x))

#transactions_from_df = [tuple(row) for row in df[['text_cleaned']].values]

#print(transactions_from_df[0], transactions_from_df[1], transactions_from_df[2])


# EXPERIMENT APRIORI ON CLEANED DATA

'''file_eff_apriori = open('pickle_INPUT_list_of_tuple', 'rb')
input_eff_apriori = pickle.load(file_eff_apriori)
#print(input_eff_apriori)

#associations = apriori(observations, min_length = 2, min_support = 0.2, min_confidence = 0.2, min_lift = 3)
# first -> identify rules
itemsets, rules = apriori(input_eff_apriori, min_support=0.01, min_confidence=0.5)

rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
  print(rule)  # Prints the rule and its confidence, support, lift, ...
print(itemsets)

print('------------------------------------------------------------------------')

# second -> identify frequent itemsets
itemsets, rules = apriori(input_eff_apriori, output_transaction_ids=True)
print(itemsets)
# {1: {('bacon',): ItemsetCount(itemset_count=3, members={0, 1, 2}), ...'''







'''transactions = [('eggs', 'bacon', 'soup'),
                ('eggs', 'bacon', 'apple', 'pen', 'soup'),
                ('soup', 'bacon', 'banana', 'cucumber', 'vodka', 'soup')]

#associations = apriori(observations, min_length = 2, min_support = 0.2, min_confidence = 0.2, min_lift = 3)
# first -> identify rules
itemsets, rules = apriori(transactions, min_support=0.3, min_confidence=0.5, output_transaction_ids=True)

rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
  print(rule)  # Prints the rule and its confidence, support, lift, ...
print(itemsets)

print('------------------------------------------------------------------------')

# second -> identify frequent itemsets
itemsets, rules = apriori(transactions, output_transaction_ids=True)
print(rules)
print(itemsets)
# {1: {('bacon',): ItemsetCount(itemset_count=3, members={0, 1, 2}), ...'''
# TODO END EFFICIENT APRIORI
