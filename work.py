import pandas as pd
from efficient_apriori import apriori
import pickle

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

# TODO CHANGE PARAMETERS, THINK ABOUT ASSOCIATION RULES
def apriori_fun(transactions_string): # , min_sup, min_conf, min_len, min_lift, len_rule
  '''Given a transaction from the dataset (set of cleaned tweets of a single day), it returns a tuple containing the
  itemsets output from efficient-apriori, the list of itemsets, and the list with the support for each itemset
  -> (dict, list, list)'''

  transactions = pd.eval(transactions_string)

  itemsets, rules = apriori(transactions, min_support=0.03, min_confidence=0.9)

  rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
  #for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
  #  print(rule)  # Prints the rule and its confidence, support, lift, ...
  #print(itemsets)

  list_itemsets = []
  list_freq = []
  for key in itemsets:
    for el in itemsets[key]:
      list_itemsets.append(el)
      list_freq.append(itemsets[key][el])

  return itemsets, list_itemsets, list_freq


# apply  apriori on each day
#df_grouped['result_apriori'] = df_grouped['text_cleaned_tuple'].apply(lambda x: apriori_fun(x))

#df11['result_apriori'] = df11['text_cleaned_tuple'].apply(lambda x: apriori_fun(x))

#print(apriori_fun(transactions_string))

#df2['result_apriori'] = df2['text_cleaned_tuple'].apply(lambda x: apriori_fun(x))

#print(df['text_cleaned_tuple'].apply(lambda x: apriori_fun(x)))

#print(df_grouped['text_cleaned_tuple'])


# apply apriori to the tweets on each day separately, save the result in the list transactions_topics
transactions_topics = []
for i in range(25):
  #print(df_grouped['text_cleaned_tuple'].values[i])
  result = apriori_fun(df_grouped['text_cleaned_tuple'].values[i])
  print(result)
  print("\n")
  transactions_topics.append(result[1])

print("------------------------------------------")
print(transactions_topics)

# save the list transactions_topics into pickel file
file = open('pickle_transactions_topics', 'wb')
pickle.dump(transactions_topics, file)
file.close()

# read the list transactions_topics
file = open('pickle_transactions_topics', 'rb')
pickle_topics = pickle.load(file)
print(pickle_topics)

#print(pickle_topics[17])



# create a dictionary dict_topic_days containing, for each topic, in how many days and in which it appears
dict_topic_days = {}
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
      dict_topic_days[topic] = (count, list_pos)

# save the dictionary dict_topic_days into a pickle file
file1 = open('pickle_topic_freq_days', 'wb')
pickle.dump(dict_topic_days, file1)
file1.close()

# read the dictionary dict_topic_days
file1 = open('pickle_topic_freq_days', 'rb')
pickle_topic_freq_days = pickle.load(file1)
print(pickle_topic_freq_days)







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