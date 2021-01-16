import pandas as pd
from efficient_apriori import apriori
import pickle
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import time
from collections import Counter
from itertools import combinations
import itertools


# TODO CAPIRE PERCHE' ALCUNI ITEMSETS LI CONTA IN NUMERO != USANDO LE FZ !=
# TODO RAGGRUPPARE PER DATA I TOPIC FREQUENTI COME FATTO PER EFFICIENT APRIORI
# TODO PLOTTARE I TOPIC IMPORTANTI -> COME LI SCEGLIAMO ???????????


# Here we apply the efficient-apriori on the cleaned tweets for each day separately to find, for each day, the frequent
# topics, given byy the frequent itemsets of terms, then we check the frequence of each of them on the total number of days

# read cleaned dataframe -> not useful
df = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_tweets_cleaned_OK.csv", sep=' ')
#print(df)
print("----------------------")

# read dataframe grouped by day
df_grouped = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_group_tuple.csv", sep=' ')
#print(dff)
print("----------------------")

'''df.drop([])
df11 = df_grouped.head()
print(df11)

df2 = df[:1]
print(df2)'''

'''dff1 = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_group_tuple_sep_comma.csv")
#print(dff1)
print("----------------------")'''

'''transactions = pd.eval(df_grouped['text_cleaned_tuple'].values[0])
transactions_string = df_grouped['text_cleaned_tuple'].values[0]
#print(transactions)
print(type(transactions))
print("----------------------")
val1 = dff1['text_cleaned_tuple'].values[0]
print(val1)'''

#-------------------------------------------------------------------------------------------------------------------
# EFFICIENT APRIORI
# TODO CHANGE PARAMETERS, THINK ABOUT ASSOCIATION RULES
def eff_apriori_fun(transactions_string): # , min_sup, min_conf, min_len, min_lift, len_rule
  '''Given a transaction from the dataset (set of cleaned tweets of a single day), it returns a tuple containing the
  itemsets output from efficient-apriori, the list of itemsets, and the list with the support for each itemset
  -> (dict, list, list)'''

  transactions = pd.eval(transactions_string)
  #transactions = transactions_string
  itemsets, rules = apriori(transactions, min_support=0.03, min_confidence=0.5)
  for rule in rules:
    print(rule)
  '''rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
  for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
    print(rule)  # Prints the rule and its confidence, support, lift, ...'''
  #print(itemsets)

  list_itemsets = []
  list_freq = []
  for key in itemsets:
    for el in itemsets[key]:
      list_itemsets.append(el)
      list_freq.append(itemsets[key][el])

  return itemsets, list_itemsets, list_freq
  
# apply apriori to the tweets on each day separately, save the result in the list transactions_topics
'''transactions_topics = []
start_time = time.time()
for i in range(1):
  #print(df_grouped['text_cleaned_tuple'].values[i])
  result = eff_apriori_fun(df_grouped['text_cleaned_tuple'].values[i])
  print(result)
  print("\n")
  transactions_topics.append(result[1])
print('Time to find frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))'''



# apply  apriori on each day
#df_grouped['result_apriori'] = df_grouped['text_cleaned_tuple'].apply(lambda x: apriori_fun(x))

#df11['result_apriori'] = df11['text_cleaned_tuple'].apply(lambda x: apriori_fun(x))

#print(apriori_fun(transactions_string))

#df2['result_apriori'] = df2['text_cleaned_tuple'].apply(lambda x: apriori_fun(x))

#print(df['text_cleaned_tuple'].apply(lambda x: apriori_fun(x)))

#print(df_grouped['text_cleaned_tuple'])





#print("------------------------------------------")
#print(transactions_topics)

'''# save the list transactions_topics into pickel file
file = open('pickle_transactions_topics', 'wb')
pickle.dump(transactions_topics, file)
file.close()'''

# read the list transactions_topics
'''file = open('pickle_transactions_topics', 'rb')
pickle_topics = pickle.load(file)
print(pickle_topics)'''

#print(pickle_topics[17])


# create a dictionary dict_topic_days containing, for each topic, in how many days and in which it appears
'''dict_topic_days = {}
for day in pickle_topics:
  for topic in day:
    if not (topic in dict_topic_days):
      count = 0
      pos = 0
      list_pos = []
      for day in pickle_topics:
        if topic in day:
          count += 1
          list_pos.append(pos)
        pos += 1
      dict_topic_days[topic] = (count, list_pos)'''

'''# save the dictionary dict_topic_days into a pickle file
file1 = open('pickle_topic_freq_days', 'wb')
pickle.dump(dict_topic_days, file1)
file1.close()'''

'''# read the dictionary dict_topic_days
file1 = open('pickle_topic_freq_days', 'rb')
pickle_topic_freq_days = pickle.load(file1)
print(pickle_topic_freq_days)'''

#END EFFICIENT APRIORI
#----------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------
# MLX APRIORI


def mlx_apriori_fun(transactions_string):
  te = TransactionEncoder()

  transactions = pd.eval(transactions_string)

  te_ary = te.fit(transactions).transform(transactions)
  df1 = pd.DataFrame(te_ary, columns=te.columns_)
  print("-------------")
  #print(df1)

  res = fpgrowth(df1, min_support=0.03, use_colnames=True)
  #print(res)

  return(res)


transactions_topics = []
start_time = time.time()
for i in range(25):
  #print(df_grouped['text_cleaned_tuple'].values[i])
  #result = mlx_apriori_fun(df_grouped['text_cleaned_tuple'].values[i])
  result = eff_apriori_fun([
    ['John', 'Mark', 'Jennifer'],
    ['John'],
    ['Joe', 'Mark'],
    ['John', 'Anna', 'Jennifer'],
    ['Jennifer', 'John', 'Mark', 'Mark']
  ])
  print(result)
  print("\n")
  #transactions_topics.append(result[1])
print('Time to find frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))

# END MLX APRIORI
#---------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------
#NAIVE APPROACH
# https://stackoverflow.com/questions/31495507/finding-the-most-frequent-occurrences-of-pairs-in-a-list-of-lists

# HERE: ALL THE POSSIBLE L
def naive_fun(transactions_string):

  transactions = pd.eval(transactions_string)
  #transactions = transactions_string
  print(len(transactions))
  dict_counters = {}
  '''tuples_1 = Counter()
  tuples_2 = Counter()
  tuples_3 = Counter()
  tuples_4 = Counter()'''
  dict_count = {}
  dict_counters[1] = Counter()
  for sub in transactions:
    #if len(transactions) < 2:
      #continue
    sub.sort()
    #for i in range(len(sub)): # TODO così non funzionaaaaa perché ci sono troppe combinazioni
    for i in range(4):
      #print(i)
      if not (i+1 in dict_counters.keys()):
        dict_counters[i+1] = Counter()
      for el in combinations(set(sub), i+1):
        if len(el) == len(set(el)):
          '''if not (el in dict_count.keys()):
            dict_count[el] = 1 / len(transactions)
          else:'''
          dict_counters[i+1][el] += 1 #/ len(transactions)
  for key in dict_counters:
    dict_counters[key] = dict_counters[key].most_common(10)
  return dict_counters


'''start_time = time.time()
for i in range(1):
  #print(df_grouped['text_cleaned_tuple'].values[i])
  result = naive_fun(df_grouped['text_cleaned_tuple'].values[i])
  print(result)
  print("\n")
  #transactions_topics.append(result[1])
print('Time to find frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))'''


# HERE: NOT ALL THE POSSIBLE SUBSETS OF ALL LENGTH
def naive_fun_2(transactions_string):

  transactions = pd.eval(transactions_string)
  #transactions = transactions_string
  print(len(transactions))
  tuples_1 = Counter()
  tuples_2 = Counter()
  tuples_3 = Counter()
  tuples_4 = Counter()
  for sub in transactions:
    sub.sort()
    for singleton in set(sub):
      #if len(singleton) == len(set(singleton)):
      tuples_1[singleton] += 1#/len(transactions)
    for pair in combinations(set(sub), 2):
      if len(pair) == len(set(pair)):
        #print(pair)
        tuples_2[pair] += 1#/len(transactions)
    for triple in combinations(set(sub), 3):
      if len(triple) == len(set(triple)):
        tuples_3[triple] += 1#/len(transactions)
    for tuple4 in combinations(set(sub), 4):
      if len(tuple4) == len(set(tuple4)):
        tuples_4[tuple4] += 1#/len(transactions)
  return(tuples_1.most_common(10), tuples_2.most_common(10), tuples_3.most_common(10), tuples_4.most_common(10))

start_time = time.time()
for i in range(1):
  #print(df_grouped['text_cleaned_tuple'].values[i])
  result = naive_fun_2(df_grouped['text_cleaned_tuple'].values[i])
  print(result)
  print("\n")
  #transactions_topics.append(result[1])
print('Time to find frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))


'''result = naive_fun_2([
    ['John', 'Mark', 'Jennifer'],
    ['John'],
    ['Joe', 'Mark'],
    ['John', 'Anna', 'Jennifer'],
    ['Jennifer', 'John', 'Mark', 'Mark']
  ])'''



























'''itemsets, rules = apriori(pickle_topics, min_support=0.6, min_confidence=0.66)

rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
  print(rule)  # Prints the rule and its confidence, support, lift, ...
print(itemsets)'''

'''transactions = [[('covid',), ('coronaviru',), ('pandem',), ('trump',)], [('covid',), ('coronaviru',), ('death',), ('case',), ('test',)], [('covid',), ('case',)]]

itemsets, rules = apriori(transactions, min_support=0.3, min_confidence=0.66)

rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
  print(rule)  # Prints the rule and its confidence, support, lift, ...
print(itemsets)'''

# TODO EFFICIENT APRIORI

# first -> identify rules
'''itemsets, rules = apriori(transactions, min_support=0.01, min_confidence=0.66)

rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
  print(rule)  # Prints the rule and its confidence, support, lift, ...
print(itemsets)
print('------------------------------------------------------------------------')'''

# second -> identify frequent itemsets
#itemsets, rules = apriori(transactions, output_transaction_ids=False)
#print(itemsets)
# {1: {('bacon',): ItemsetCount(itemset_count=3, members={0, 1, 2}), ...

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
# {1: {('bacon',): ItemsetCount(itemset_count=3, members={0, 1, 2}), ...
'''






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