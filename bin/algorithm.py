import pandas as pd
from efficient_apriori import apriori
from collections import Counter
from itertools import combinations
from itertools import dropwhile

#from mlxtend.frequent_patterns import fpgrowth
#from mlxtend.frequent_patterns import apriori as mlxtend_apriori
#from mlxtend.preprocessing import TransactionEncoder

# -*- coding: utf-8 -*-
# Different methods to find popular topics


def eff_apriori_fun(transactions, singleton, num_day): # , min_sup, min_conf, min_len, min_lift, len_rule
  '''Given transactions from the dataset (set of cleaned tweets of a single day) and the indication for singleton
  results, it returns a dictionary containing the frequent topics and their frequency and number of occurrrences,
  obtained by applying efficient_apriori on the transaction -> {topic: (freq, num_occ), ...}'''

  # transactions = pd.eval(transactions_string)
  print("Finding frequent topics on day", num_day)
  print("Number of tweets:", len(transactions))

  itemsets, rules = apriori(transactions, min_support=0.03, min_confidence=0.7) # , output_transaction_ids=True
  #for rule in rules:
    #print(rule.rhs + rule.lhs)
  '''rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
  for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
    print(rule)  # Prints the rule and its confidence, support, lift, ...'''

  dict_topic = {}

  if singleton == 0: # no singletons
    for key in itemsets:
      for el in itemsets[key]:
        if len(el) > 1:
          dict_topic[el] = itemsets[key][el]/len(transactions), itemsets[key][el] # topic: (freq, tot num)
  else: # yes singletons
    for key in itemsets:
      for el in itemsets[key]:
        dict_topic[el] = itemsets[key][el]/len(transactions), itemsets[key][el] # topic: (freq, tot num)'''

  return dict_topic


# In this version we consider the association rule as method to filter the results, so we only consider itemsets of 2
# or more items and we consider only high correlation between the terms
def eff_apriori_rules_fun(transactions, num_day):
  '''Given a transaction from the dataset (set of cleaned tweets of a single day) it returns a dictionary containing
  the frequent topics and their frequency and confidence of the related association rule, obtained by applying
  efficient_apriori on the transaction and considering only sets from which it is possible to derive an association
  rules with confidence above the threshold -> {topic: [freq, confidence], ...}'''

  #transactions = pd.eval(transactions_string)
  print("Finding frequent topics on day", num_day)
  print("Number of tweets:", len(transactions))

  itemsets, rules = apriori(transactions, min_support=0.03, min_confidence=0.7) # , output_transaction_ids=True
  dict_topic = {}

  print("Association rules:\n")
  for rule in rules:
    print(rule)
    itemset = sorted(rule.rhs + rule.lhs)
    dict_topic[tuple(itemset)] = [rule.support, rule.confidence]

  return dict_topic


'''def mlx_apriori_fun(transactions, singleton, num_day):
  #Given a transaction from the dataset (set of cleaned tweets of a single day) and the indication for singleton
  #results, it returns a dictionary containing the frequent topics and their frequency and number of occurrrences,
  #obtained by applying mlx_apriori on the transaction -> {topic: (freq, num_occ), ...}

  # transactions = pd.eval(transactions_string)
  print("Finding frequent topics on day", num_day)
  print("Number of tweets:", len(transactions))

  te = TransactionEncoder()

  te_ary = te.fit(transactions).transform(transactions)
  df1 = pd.DataFrame(te_ary, columns=te.columns_)

  res = fpgrowth(df1, min_support=0.03, use_colnames=True)

  list_itemsets = res['itemsets'].tolist()
  list_support = res['support'].tolist()
  tot_list = list(zip(list_itemsets, list_support))
  dict_topic = {}

  if singleton == 0: # no singletons
    for el in tot_list:
      if len(el) > 1:
        dict_topic[el[0]] = (el[1], el[1]*len(transactions))
  else: # yes singletons
    for el in tot_list:
      dict_topic[el[0]] = (el[1], el[1]*len(transactions))

  return(dict_topic)'''


def naive_fun(transactions, singleton, num_day):
  '''Given a transaction from the dataset (set of cleaned tweets of a single day) and the indication for singleton
  results, it returns a dictionary containing the frequent topics (as top-k) and their frequency and number of
  occurrrences, obtained by computing the possible combinations of terms in a naive way -> {topic: (freq, num_occ), ...}'''

  #transactions = pd.eval(transactions_string)
  print("Finding frequent topics on day", num_day)
  print("Number of tweets:", len(transactions))

  dict_counters = {}

  if singleton == 0: # not singletons
    len_comb = range(1, 4)
  else: # yes singletons
    len_comb = range(4)

  for sub in transactions:
    list(sub).sort()
    #for i in range(len(sub)): no since too many combinations
    for i in len_comb:
      if not (i+1 in dict_counters.keys()):
        dict_counters[i+1] = Counter()
      for el in combinations(set(sub), i+1):
        if len(el) == len(set(el)):
          dict_counters[i+1][tuple(sorted(el))] += 1 #/ len(transactions)

  dict_topic = {}

  for key in dict_counters:
    dict_counters[key] = dict_counters[key].most_common(10)
    for el in dict_counters[key]:
      dict_topic[el[0]] = (el[1] / len(transactions), el[1])

  return dict_topic


def naive_fun_freq(transactions, singleton, num_day):
  '''Given a transaction from the dataset (set of cleaned tweets of a single day) and the indication for singleton
    results, it returns a dictionary containing the frequent topics (considering their freq) and their frequency and
    number of occurrrences, obtained by computing the possible combinations of terms in a naive way ->
    {topic: (freq, num_occ), ...}'''

  #start_time = time.time()
  #transactions = pd.eval(transactions_string)

  print("Finding frequent topics on day", num_day)
  print("Number of tweets:", len(transactions))

  dict_counters = {}

  if singleton == 0: # no singletons
    len_comb = range(1, 4)
  else: # yes singletons
    len_comb = range(4)

  for sub in transactions:
    list(sub).sort()
    for i in len_comb:
      if not (i+1 in dict_counters.keys()):
        dict_counters[i+1] = Counter()
      for el in combinations(set(sub), i+1):
        if len(el) == len(set(el)):
          dict_counters[i+1][tuple(sorted(el))] += 1 / len(transactions)
  #print('Time to find frequent topics:')
  #time_topics = time.time() - start_time
  #print("--- %s seconds ---" % time_topics)

  dict_topic = {}

  # start = time.time()
  for key in dict_counters:
    # start_1 = time.time()
    for k, count in dropwhile(lambda key_count: key_count[1] >= 0.03, dict_counters[key].most_common()):
      del dict_counters[key][k]

    dict_counters[key] = dict_counters[key].most_common()
    #print('TIME 1:')
    #time_topics = time.time() - start_1
    #print("--- %s seconds ---" % time_topics)

    #start_2 = time.time()
    for el in dict_counters[key]:
      dict_topic[el[0]] = (el[1], round(el[1]*len(transactions)))
    #print("time 2")
    #time_2 = time.time() - start_2
    #print("--- %s seconds ---" % time_2)

  #time_topics = time.time() - start
  #print("final time")
  #print("--- %s seconds ---" % time_topics)

  return dict_topic #dict_counters


def apply_fun(name_fun, dimension, column_dataframe, singleton): # column_dataframe = df_grouped['text_cleaned_tuple']
  '''Given a function to find frequent topics, it applies the function on all the transactions in the related column of
  the dataset and returns, for eac day, the frequent topics with their frequency and number of occurrrences ->
  {day0: {topic1_0: (freq, num_occ), topic2_0: (...)}, day1: {topic1_1: (freq, num_occ), ...}, ...}
  '''

  dict_day_topic = {}

  for i in range(dimension):
    if name_fun == "eff_apriori_fun":
      result = eff_apriori_fun(column_dataframe.values[i], singleton, i)
    elif name_fun == "eff_apriori_rules_fun":
      result = eff_apriori_rules_fun(column_dataframe.values[i], i)
    #elif name_fun == "mlx_apriori_fun":
      #result = mlx_apriori_fun(column_dataframe.values[i], singleton, i)
    elif name_fun == "naive_fun_freq":
      result = naive_fun_freq(column_dataframe.values[i], singleton, i)
    else:
      result = naive_fun(column_dataframe.values[i], singleton, i)

    print("RESULT:", result)
    print("\n")

    dict_day_topic["day" + str(i)] = result

  return dict_day_topic


def create_dict_topics(dict_day_topic):
  '''Given a dictionary with days and their frequent topics, it returns a tuple containing a dictionary of topics with
  the days in which they are frequent and the related frequency, a dictionary to transform in dataframe to show results,
  a list with the number of days in which each topic is frequent'''

  dict_topic_day_num = {}
  dict_to_dataframe = {}
  list_count = []

  for day in dict_day_topic:
    for topic in dict_day_topic[day]:
      if not (topic in dict_topic_day_num):
        count = 0
        list_day = []
        list_num = []
        list_freq = []
        list_freq_dataset = []
        for day in dict_day_topic:
          list_day.append(day)
          if topic in dict_day_topic[day].keys():
            count += 1
            list_num.append(dict_day_topic[day][topic][1])
            list_freq.append(dict_day_topic[day][topic][0])
            list_freq_dataset.append(dict_day_topic[day][topic][0])
          else:
            list_num.append("not freq")
            list_freq.append("not freq")
            list_freq_dataset.append("")
        dict_topic_day_num[topic] = (count, list(zip(list_day, list_num, list_freq)))
        dict_to_dataframe[topic] = (list_freq_dataset)
        list_count.append(count)

  return dict_topic_day_num, dict_to_dataframe, list_count






#######################################################################################################################

# NOT USED FUNCTIONS
'''
# LESS CHECKS
def apply_fun(name_fun, dimension, column_dataframe, singleton): # column_dataframe = df_grouped['text_cleaned_tuple']
  #Given a function to find frequent topics, it applies the function on all the transactions in the related column of
  #the dataset and returns, for eac day, the frequent topics with their frequency and number of occurrrences ->
  #{day0: {topic1_0: (freq, num_occ), topic2_0: (...)}, day1: {topic1_1: (freq, num_occ), ...}, ...}

  dict_day_topic = {}
  result = {}
  #print(df_grouped['text_cleaned_tuple'].values[i])
  if name_fun == "eff_apriori_fun":
    for i in range(dimension):
      result = eff_apriori_fun(column_dataframe.values[i], singleton)
      print("RESULT:", result)
      # print("\n")
      dict_day_topic["day" + str(i)] = result
  elif name_fun == "eff_apriori_rules_fun":
    for i in range(dimension):
      result = eff_apriori_rules_fun(column_dataframe.values[i])
      print("RESULT:", result)
      # print("\n")
      dict_day_topic["day" + str(i)] = result
  elif name_fun == "mlx_apriori_fun":
    for i in range(dimension):
      result =mlx_apriori_fun(column_dataframe.values[i], singleton)
      print("RESULT:", result)
      # print("\n")
      dict_day_topic["day" + str(i)] = result
  elif name_fun == "naive_fun_freq":
    for i in range(dimension):
      result = naive_fun_freq(column_dataframe.values[i], singleton)
      print("RESULT:", result)
      # print("\n")
      dict_day_topic["day" + str(i)] = result
  else:
    for i in range(dimension):
      result = naive_fun(column_dataframe.values[i], singleton)
      print("RESULT:", result)
      #print("\n")
      dict_day_topic["day" + str(i)] = result
  #print("\n\n\n")
  print("DICT DAY TOPIC", dict_day_topic)
  return dict_day_topic


def count_itemset(tuple_topic, transactions): # da fare nel day in cui manca la freq
  # Given a tuple A, and a list of tuples L, it returns how many times the tuple A is contained in the tuples of L
  counter = 0
  for tuple in transactions:
    if set(tuple_topic).issubset(tuple):
      counter += 1
  return counter


# VARIANTE PER POTER CONTARE LE FREQUENZE DI TUTTI -> RALLENTA MOLTO !!!!!!!!!
def create_dict_topics_all_freq(dict_day_topic, column_dataframe):
  dict_topic_day_num = {}
  dict_to_dataframe = {}
  list_count = []
  for day in dict_day_topic:
    #print("D", day)
    #print("E", dict_day_topic[day])
    for topic in dict_day_topic[day]:
      #print("Q", topic)
      if not (topic in dict_topic_day_num):
        pos = 0
        count = 0
        list_day = []
        list_num = [] # TODO questa e' list_confidence se si usa eff_apriori_rules_fun
        list_freq = []
        list_flag = []
        #print("EEEEEEE", dict_day_topic[day].keys())
        for day in dict_day_topic:
          list_day.append(day)
          if topic in dict_day_topic[day].keys():
            count += 1
            list_num.append(dict_day_topic[day][topic][1]) # TODO questa e' list_confidence se si usa eff_apriori_rules_fun
            list_freq.append(dict_day_topic[day][topic][0])
            list_flag.append('freq')
            #print("DDDDDDDD", dict_day_topic[day][topic])
          else:
            transactions_string = column_dataframe[pos] #df_grouped['text_cleaned_tuple'][pos]
            transactions = pd.eval(transactions_string)
            num = count_itemset(topic, transactions_string)
            list_num.append(num)
            list_freq.append(num/len(transactions))
            list_flag.append('not_freq')
            #list_num.append("not freq")
            #list_freq.append("not freq")
          pos += 1
        dict_topic_day_num[topic] = (count, list(zip(list_day, list_num, list_freq, list_flag))) # TODO list_num e' list_confidence se si usa eff_apriori_rules_fun
        dict_to_dataframe[topic] = (list_freq)
        list_count.append(count)
        #print("DICT TOPIC DAY NUM", dict_topic_day_num)
  return dict_topic_day_num, dict_to_dataframe, list_count


def eff_apriori_new(transactions, day, singleton):

  itemsets, rules = apriori(transactions, min_support=0.03, min_confidence=0.7) # , output_transaction_ids=True
  
  print(itemsets)
  dict_topic = {}
  if singleton == 0:
    for key in itemsets:
      for el in itemsets[key]:
        if len(el) > 1:
          dict_topic[el] = itemsets[key][el]/len(transactions), itemsets[key][el] # topic: (freq, tot num)
  else:
    for key in itemsets:
      for el in itemsets[key]:
        dict_topic[el] = itemsets[key][el]/len(transactions), itemsets[key][el] # topic: (freq, tot num)

  return (day, dict_topic)


def apply_fun_new(name_fun, dimension, column_dataframe, column_date, singleton): # column_dataframe = df_grouped['text_cleaned_tuple']

  dict_day_topic = {}
  for i in range(dimension):
    #print(df_grouped['text_cleaned_tuple'].values[i])
    if name_fun == "eff_apriori_fun":
      result = eff_apriori_new(column_dataframe.values[i], column_date[i], singleton)
    elif name_fun == "eff_apriori_rules_fun":
      result = eff_apriori_rules_fun(column_dataframe.values[i])
    elif name_fun == "mlx_apriori_fun":
      result = mlx_apriori_fun(column_dataframe.values[i], singleton)
    elif name_fun == "naive_fun_freq":
      result = naive_fun_freq(column_dataframe.values[i], singleton)
    #elif name_fun == "mlx_apriori_new":
      #result = mlx_apriori_new(column_date.values[i], column_dataframe.values[i])
    else:
      result = naive_fun(column_dataframe.values[i], singleton)
    print("RESULT:", result)
    #print("\n")
    dict_day_topic[result[0]] = result[1]
  #print("\n\n\n")
  print("DICT DAY TOPIC", dict_day_topic)
  return dict_day_topic


def create_dict_topics_new(dict_day_topic):
  dict_topic_day_freq = {}
  print("day", dict_day_topic)
  for day in dict_day_topic:
    for topic in dict_day_topic[day]:
      print(topic)
      if not (topic in dict_topic_day_freq):
        dict_topic_day_freq[topic] = []
      dict_topic_day_freq[topic].append((day, dict_day_topic[day][topic]))
  return dict_topic_day_freq
'''

'''start = time.time()
df = pd.read_pickle('data/covid_input')
column_date = df['date_only']
column_dataframe = df['text_cleaned_tuple']
res = apply_fun('eff_apriori_fun', 25, column_dataframe, 0)
res_1 = create_dict_topics(res)
print(res_1)
end = time.time() - start
print("TIME", end)'''

######################################################################################################################
# ALSO SINGLETON
# MLX_APRIORI_FUN with also singleton and computing frequences only of frequent topics
'''start_time = time.time()
res = apply_fun('mlx_apriori_fun', 25, column_dataframe)
print(len(res.keys()))
create_dict_topics_also_singleton_res = create_dict_topics_also_singleton(res)
print("create_dict_topics_also_singleton_res[0]", create_dict_topics_also_singleton_res[0])
print("create_dict_topics_also_singleton_res[1]", create_dict_topics_also_singleton_res[1])
print("create_dict_topics_also_singleton_res[2]", create_dict_topics_also_singleton_res[2])

df_topics = pd.DataFrame.from_dict(create_dict_topics_also_singleton_res[1], orient='index', columns=list_date[:25])
df_topics["Number of occurrences"] = create_dict_topics_also_singleton_res[2]
print(df_topics)
df_topics.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/res_mlx.csv', sep=' ')

print('Time to find frequent itemset')
time_mlx_apriori = time.time() - start_time
print("--- %s seconds ---" % time_mlx_apriori)'''