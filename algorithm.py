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
import matplotlib
import matplotlib.pyplot as plt
from itertools import dropwhile

# TODO input utente: numero di topic restituiti, magari lunghezza topic

# TODO sistemare funzioni mettendo variabili come parametri (es: lunghezza lista delle date)

# TODO sistemare e mettere tutte le cose in ordine

# TODO FARE RUN CON TUTTE LE FREQUENZE PER FARE GRAFICI

# TODO tenere anche singleton? 2 opzioni:
# TODO usare le association rules / modificare codice e tenere solo se non singleton

# TODO considerare le association rules?

# TODO threshold è ok così basso? fare test con anche un poco più alto, ma si perdono cose

# TODO confrontare prestazioni naive e apriori -> minor numero di comparazioni, usa threshold invece di top-k (???), se
# TODO si dovessero guardare tutte le possibili combinazioni con naive necessiterebbe troppo tempo, ci fermiamo a length = 4

# TODO scegliere altri dataset (sottoinsieme? altri?)

# TODO capire perché teniamo anche le parole molto frequenti -> la correlazione con gli altri non è casuale in questo modo?
# TODO oppure no perché anche se frequenti sempre nel periodo sono importanti, dato che se troviamo putin insieme a covid significa che
# TODO il topic riguarda sia covid che putin

# TODO capire se tenere tutte le frequenze per fare i grafici

# TODO scegliere quali grafici fare, quali topic tenere

# TODO creare altri pickle file per fare esperimenti comodi

# TODO fare test utente

# TODO calcolare tempi e prestazioni -> capire quando approccio apriori funziona bene


# TODO CAPIRE PERCHE' ALCUNI ITEMSETS LI CONTA IN NUMERO != USANDO LE FZ != ------> forse ok
# TODO RAGGRUPPARE PER DATA I TOPIC FREQUENTI COME FATTO PER EFFICIENT APRIORI ----> ok manca solo per mlx apriori



# Here we apply the efficient-apriori on the cleaned tweets for each day separately to find, for each day, the frequent
# topics, given by the frequent itemsets of terms, then we check the frequence of each of them on the total number of days

# read cleaned dataframe -> not useful
'''df_1 = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_naive_fun_normal.csv", sep=',')
print(df_1)
x = sorted(list(df_1. iloc[:, 0]))

df_2 = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_naive_fun_normal_1111111111.csv", sep=',')
print(df_2)
y = sorted(list(df_2. iloc[:, 0]))

print("164", x)
print("165", y)

main_list = list(set(x) - set(y))
main_list_1 = list(set(y) - set(x))

print("si 164 no 165", main_list)
print(len(main_list))
print("si 165 no 164", main_list_1)
print((len(main_list_1)))
'''

#print(df)
#print("----------------------")

# read dataframe grouped by day
df_grouped = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_group_tuple.csv", sep=' ')
#print(dff)
print("----------------------")
list_date = df_grouped['date_only'].tolist()

column_dataframe = df_grouped['text_cleaned_tuple']

'''df_grouped = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_AUSTRALIA_group_tuple.csv", sep=' ')
#print(dff)
print("----------------------")
list_date = df_grouped['date_only'].tolist()
print(list_date)
print(len(list_date))
column_dataframe = df_grouped['text_cleaned_tuple']
print(df_grouped.shape[0])'''

# TODO decidere dove metterlo
'''start_time = time.time()
print('Time to find frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))'''

#print(df_grouped['text_cleaned_tuple'][0])
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
  #print(len(transactions))
  #transactions = transactions_string
  itemsets, rules = apriori(transactions, min_support=0.03, min_confidence=0.7) # , output_transaction_ids=True
  #for rule in rules:
    #print(rule)
    #print(rule.rhs + rule.lhs)
  '''rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
  for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
    print(rule)  # Prints the rule and its confidence, support, lift, ...'''
  #print(itemsets)

  '''list_itemsets = []
  list_freq = []
  for key in itemsets:
    for el in itemsets[key]:
      list_itemsets.append(el)
      list_freq.append(itemsets[key][el])'''

  dict_topic = {}
  for key in itemsets:
    for el in itemsets[key]:
      if len(el) > 1: # TODO uncomment if we want itemsets only of length > 1
        dict_topic[el] = itemsets[key][el]/len(transactions), itemsets[key][el] # topic: (freq, tot num)

  return dict_topic

# In this version we consider the association rule as method to filter the results, so we only consider itemsets of 2
# or more items and we consider only high correlation between the terms
# TODO min_confidence: 0.6/0.7/0.8 forse le migliori ??
def eff_apriori_rules_fun(transactions_string): # , min_sup, min_conf, min_len, min_lift, len_rule
  '''Given a transaction from the dataset (set of cleaned tweets of a single day), it returns a tuple containing the
  itemsets output from efficient-apriori, the list of itemsets, and the list with the support for each itemset
  -> (dict, list, list)'''

  transactions = pd.eval(transactions_string)
  print(len(transactions))
  #transactions = transactions_string
  itemsets, rules = apriori(transactions, min_support=0.03, min_confidence=0.7) # , output_transaction_ids=True
  dict_topic = {}
  for rule in rules:
    print(rule)
    itemset = sorted(rule.rhs + rule.lhs)
    dict_topic[tuple(itemset)] = [rule.support, rule.confidence]
  '''rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
  for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
    print(rule)  # Prints the rule and its confidence, support, lift, ...'''
  #print(itemsets)

  '''list_itemsets = []
  list_freq = []
  for key in itemsets:
    for el in itemsets[key]:
      list_itemsets.append(el)
      list_freq.append(itemsets[key][el])'''

  '''dict_topic = {}
  for key in itemsets:
    for el in itemsets[key]:
      dict_topic[el] = itemsets[key][el], itemsets[key][el]/len(transactions) # topic: (tot num, freq)
  '''
  return dict_topic

def mlx_apriori_fun(transactions_string):
  te = TransactionEncoder()

  transactions = pd.eval(transactions_string)

  te_ary = te.fit(transactions).transform(transactions)
  df1 = pd.DataFrame(te_ary, columns=te.columns_)
  print("-------------")
  #print(df1)

  res = fpgrowth(df1, min_support=0.03, use_colnames=True)
  #print(res)

  print("TYPEEEEEEEEEEEEEEEEEE", type(res))
  list_itemsets = res['itemsets'].tolist()
  list_support = res['support'].tolist()
  print(list_support)
  print(list_itemsets)
  tot_list = list(zip(list_itemsets, list_support))
  print(tot_list)
  dict_topic = {}
  for el in tot_list:
    dict_topic[el[0]] = (el[1], el[1]*len(transactions))

  return(dict_topic)

# HERE: ALL THE POSSIBLE L -> only until 4, since more there is a problem with memory -> loop for all the cases
def naive_fun(transactions_string):

  transactions = pd.eval(transactions_string)
  #transactions = transactions_string
  print(len(transactions))
  dict_counters = {}
  #tuples_1 = Counter()
  #tuples_2 = Counter()
  #tuples_3 = Counter()
  #tuples_4 = Counter()
  #dict_counters[1] = Counter()
  for sub in transactions:
    #if len(transactions) < 2:
      #continue
    sub.sort()
    #for i in range(len(sub)): # TODO così non funzionaaaaa perché ci sono troppe combinazioni
    for i in range(1, 4): # put range(1, 4) if we want without singletons
      #print(i)
      if not (i+1 in dict_counters.keys()):
        dict_counters[i+1] = Counter()
      for el in combinations(set(sub), i+1):
        if len(el) == len(set(el)):
          #if not (el in dict_count.keys()):
            #dict_count[el] = 1 / len(transactions)
          #else:
          #print(dict_counters[i+1])
          dict_counters[i+1][tuple(sorted(el))] += 1 #/ len(transactions) #TODO#########################################################
  dict_topic = {}
  for key in dict_counters:
    #print("ECCOLO", dict_counters[key].most_common(20))
    dict_counters[key] = dict_counters[key].most_common(10)
    #c = 0
    for el in dict_counters[key]:
      #c += 1
      #print("EL", el, " ", c)
      dict_topic[el[0]] = (el[1]/len(transactions), el[1])
  #print(dict_topic)
  return dict_topic #dict_counters


def naive_fun_freq(transactions_string):

  transactions = pd.eval(transactions_string)
  #transactions = transactions_string
  print(len(transactions))
  dict_counters = {}
  #tuples_1 = Counter()
  #tuples_2 = Counter()
  #tuples_3 = Counter()
  #tuples_4 = Counter()
  #dict_counters[1] = Counter()
  for sub in transactions:
    #if len(transactions) < 2:
      #continue
    sub.sort()
    #for i in range(len(sub)): # TODO così non funzionaaaaa perché ci sono troppe combinazioni
    for i in range(1, 4): # put range(1, 4) if we want without singletons
      #print(i)
      if not (i+1 in dict_counters.keys()):
        dict_counters[i+1] = Counter()
      for el in combinations(set(sub), i+1):
        if len(el) == len(set(el)):
          #if not (el in dict_count.keys()):
            #dict_count[el] = 1 / len(transactions)
          #else:
          #print(dict_counters[i+1])
          dict_counters[i+1][tuple(sorted(el))] += 1 / len(transactions) #TODO NB: non si può mettere /len(transactions alla fine perché ci derve la freq in riga 357
  dict_topic = {}
  #print(dict_counters)
  #print("-----------------------------------\n\n")
  for key in dict_counters:
    for k, count in dropwhile(lambda key_count: key_count[1] >= 0.03, dict_counters[key].most_common()):
      del dict_counters[key][k]
    #print("ECCOLO", dict_counters[key])
    #print(dict_counters[key])
    dict_counters[key] = dict_counters[key].most_common()
    #c = 0
    for el in dict_counters[key]:
      #print("EL", el, " ", el[1])
      #c += 1
      #print("EL", el, " ", c)
      dict_topic[el[0]] = (el[1], round(el[1]*len(transactions)))
  #print(dict_topic)
  print(dict_counters)
  return dict_topic #dict_counters


def apply_fun(name_fun, dimension, column_dataframe): # column_dataframe = df_grouped['text_cleaned_tuple']
  dict_day_topic = {}
  for i in range(dimension):
    #print(df_grouped['text_cleaned_tuple'].values[i])
    if name_fun == "eff_apriori_fun":
      result = eff_apriori_fun(column_dataframe.values[i])
    elif name_fun == "eff_apriori_rules_fun":
      result = eff_apriori_rules_fun(column_dataframe.values[i])
    elif name_fun == "mlx_apriori_fun":
      result =mlx_apriori_fun(column_dataframe.values[i])
    elif name_fun == "naive_fun_freq":
      result = naive_fun_freq(column_dataframe.values[i])
    else:
      result = naive_fun(column_dataframe.values[i])
    print("RESULT:", result)
    #print("\n")
    dict_day_topic["day" + str(i)] = result
  #print("\n\n\n")
  print("DICT DAY TOPIC", dict_day_topic)
  return dict_day_topic



def count_itemset(tuple_topic, transactions): # da fare nel day in cui manca la freq
  counter = 0
  for tuple in transactions:
    if set(tuple_topic).issubset(tuple):
      counter += 1
  return counter

'''l = pd.eval(df_grouped['text_cleaned_tuple'][0])
count = count_itemset(('covid', 'putin', 'russia'), l)
print("COUNTER", count)'''


# apply apriori to the tweets on each day separately, save the result in the list transactions_topics
'''dict_day_topic = {}
start_time = time.time()
for i in range(10):
  #print(df_grouped['text_cleaned_tuple'].values[i])
  result = eff_apriori_fun(df_grouped['text_cleaned_tuple'].values[i])
  print("RESULT:", result)
  print("\n")
  dict_day_topic["day"+str(i)] = result
print("\n\n\n")
print("DICT DAY TOPIC", dict_day_topic)
print("\n")
print('Time to find frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))
print("\n")'''

# TODO 1.0) da usare con eff_apriori_fun NON calcola freq di quelli non freq
def create_dict_topics_also_singleton(dict_day_topic):
  dict_topic_day_num = {}
  dict_to_dataframe = {}
  list_count = []
  for day in dict_day_topic:
    #print("D", day)
    #print("E", dict_day_topic[day])
    for topic in dict_day_topic[day]:
      #print("Q", topic)
      if not (topic in dict_topic_day_num):
        count = 0
        list_day = []
        list_num = []
        list_freq = []
        list_freq_dataset = []
        #print("EEEEEEE", dict_day_topic[day].keys())
        for day in dict_day_topic:
          list_day.append(day)
          if topic in dict_day_topic[day].keys():
            count += 1
            list_num.append(dict_day_topic[day][topic][1])
            list_freq.append(dict_day_topic[day][topic][0])
            list_freq_dataset.append(dict_day_topic[day][topic][0])
            #print("DDDDDDDD", dict_day_topic[day][topic])
          else:
            list_num.append("not freq")
            list_freq.append("not freq")
            list_freq_dataset.append("")
        dict_topic_day_num[topic] = (count, list(zip(list_day, list_num, list_freq)))
        dict_to_dataframe[topic] = (list_freq_dataset)
        list_count.append(count)
  return dict_topic_day_num, dict_to_dataframe, list_count

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

#######################################################################################################################
#ALSO SINGLETON
# EFF_ARIORI_FUN with also singleton and computing frequences only of frequent topics
'''start_time = time.time()

apply_fun_res_1 = apply_fun('eff_apriori_fun', 11, column_dataframe)
#print("apply_fun_res_1", apply_fun_res_1)
print(len(apply_fun_res_1.keys()))
create_dict_topics_also_singleton_res = create_dict_topics_also_singleton(apply_fun_res_1)
print("create_dict_topics_also_singleton_res[0]", create_dict_topics_also_singleton_res[0])
print("create_dict_topics_also_singleton_res[1]", create_dict_topics_also_singleton_res[1])
print("create_dict_topics_also_singleton_res[2]", create_dict_topics_also_singleton_res[2])

df_topics = pd.DataFrame.from_dict(create_dict_topics_also_singleton_res[1], orient='index', columns=list_date[:11])
df_topics["Number of occurrences"] = create_dict_topics_also_singleton_res[2]
#print(df_topics)

print('Time to find frequent itemset')
time_eff_apriori = time.time() - start_time
print("--- %s seconds ---" % time_eff_apriori)


# save the result dictionary dict_topic_day_num
file1 = open('pickle_eff_apriori_fun_normal_AUSTRALIA', 'wb')
pickle.dump(create_dict_topics_also_singleton_res[0], file1)
file1.close()

# save the dataframe with frequences
df_topics.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/res_eff_apriori_fun_normal_AUSTRALIA.csv')

file1 = open('pickle_time_eff_apriori_fun_normal_AUSTRALIA', 'wb')
pickle.dump(time_eff_apriori, file1)
file1.close()'''

# read the outputs
# read the dictionary dict_topic_day_num
'''file1 = open('/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_NORMAL/pickle_eff_apriori_fun_normal', 'rb')
pickle_eff_apriori = pickle.load(file1)
print(pickle_eff_apriori)

# read the dictionary dict_topic_day_num
file2 = open('/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_NORMAL/pickle_time_eff_apriori_fun_normal', 'rb')
pickle_time_eff_apriori = pickle.load(file2)
print(pickle_time_eff_apriori)

# read the dataset of frequences
df_eff_apriori = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_NORMAL/res_eff_apriori_fun_normal.csv", )
print(df_eff_apriori)'''

#######################################################################################################################
# ALSO SINGLETON
# NAIVE_FUN_FREQ with also singleton and computing frequences only of frequent topics
'''start_time = time.time()

apply_fun_res_3 = apply_fun('naive_fun_freq', 25, column_dataframe)
print(apply_fun_res_3)

create_dict_topics_also_singleton_res = create_dict_topics_also_singleton(apply_fun_res_3)
print("create_dict_topics_also_singleton_res[0]", create_dict_topics_also_singleton_res[0])
print("create_dict_topics_also_singleton_res[1]", create_dict_topics_also_singleton_res[1])
print("create_dict_topics_also_singleton_res[2]", create_dict_topics_also_singleton_res[2])

# save the result dictionary dict_topic_day_num
file1 = open('pickle_naive_fun_freq_normal', 'wb')
pickle.dump(create_dict_topics_also_singleton_res[0], file1)
file1.close()

df_topics = pd.DataFrame.from_dict(create_dict_topics_also_singleton_res[1], orient='index', columns=list_date[:25])
df_topics["Number of occurrences"] = create_dict_topics_also_singleton_res[2]
print(df_topics)

df_topics.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/res_naive_fun_freq_normal.csv')

print('Time to find frequent itemset')
time_naive_freq = time.time() - start_time
print("--- %s seconds ---" % time_naive_freq)

file1 = open('pickle_time_naive_fun_freq_normal', 'wb')
pickle.dump(time_naive_freq, file1)
file1.close()'''

# read the outputs
# read the dictionary dict_topic_day_num
'''file1 = open('/home/veror/PycharmProjects/DataMiningProj_OK/re_NAIVE_FREQ_NORMAL/pickle_naive_fun_freq_normal', 'rb')
pickle_eff_apriori = pickle.load(file1)
print(pickle_eff_apriori)

# read the dictionary dict_topic_day_num
file2 = open('/home/veror/PycharmProjects/DataMiningProj_OK/res_NAIVE_FREQ_NORMAL/pickle_time_naive_fun_freq_normal', 'rb')
pickle_time_eff_apriori = pickle.load(file2)
print(pickle_time_eff_apriori)

# read the dataset of frequences
df_eff_apriori = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_NAIVE_FREQ_NORMAL/res_naive_fun_freq_normal.csv", )
print(df_eff_apriori)'''

#######################################################################################################################
#NOT SINGLETON
# EFF_ARIORI_FUN without singleton and computing frequences only of frequent topics
'''start_time = time.time()

apply_fun_res_1 = apply_fun('eff_apriori_fun', 25, column_dataframe)
#print("apply_fun_res_1", apply_fun_res_1)
print(len(apply_fun_res_1.keys()))
create_dict_topics_also_singleton_res = create_dict_topics_also_singleton(apply_fun_res_1)
print("create_dict_topics_also_singleton_res[0]", create_dict_topics_also_singleton_res[0])
print("create_dict_topics_also_singleton_res[1]", create_dict_topics_also_singleton_res[1])
print("create_dict_topics_also_singleton_res[2]", create_dict_topics_also_singleton_res[2])

df_topics = pd.DataFrame.from_dict(create_dict_topics_also_singleton_res[1], orient='index', columns=list_date[:25])
df_topics["Number of occurrences"] = create_dict_topics_also_singleton_res[2]
#print(df_topics)

print('Time to find frequent itemset')
time_eff_apriori_not_singletons = time.time() - start_time
print("--- %s seconds ---" % time_eff_apriori_not_singletons)'''


# save the result dictionary dict_topic_day_num
'''file1 = open('pickle_eff_apriori_fun_not_singletons', 'wb')
pickle.dump(create_dict_topics_also_singleton_res[0], file1)
file1.close()

# save the dataframe with frequences
df_topics.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/res_eff_apriori_fun_not_singletons.csv')

file1 = open('pickle_time_eff_apriori_fun_not_singletons', 'wb')
pickle.dump(time_eff_apriori_not_singletons, file1)
file1.close()'''

# read the outputs
# read the dictionary dict_topic_day_num
'''file1 = open('/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_NO_SINGLETONS/pickle_eff_apriori_fun_not_singletons', 'rb')
pickle_eff_apriori = pickle.load(file1)
print(pickle_eff_apriori)

# read the dictionary dict_topic_day_num
file2 = open('/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_NO_SINGLETONS/pickle_time_eff_apriori_fun_not_singletons', 'rb')
pickle_time_eff_apriori = pickle.load(file2)
print(pickle_time_eff_apriori)

# read the dataset of frequences
df_eff_apriori = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_NO_SINGLETONS/res_eff_apriori_fun_not_singletons.csv", )
print(df_eff_apriori)'''

#######################################################################################################################
# NOT SINGLETON
# NAIVE_FUN_FREQ without singleton and computing frequences only of frequent topics
'''start_time = time.time()

apply_fun_res_3 = apply_fun('naive_fun_freq', 25, column_dataframe)
print(apply_fun_res_3)

create_dict_topics_also_singleton_res = create_dict_topics_also_singleton(apply_fun_res_3)
print("create_dict_topics_also_singleton_res[0]", create_dict_topics_also_singleton_res[0])
print("create_dict_topics_also_singleton_res[1]", create_dict_topics_also_singleton_res[1])
print("create_dict_topics_also_singleton_res[2]", create_dict_topics_also_singleton_res[2])

# save the result dictionary dict_topic_day_num
file1 = open('pickle_naive_fun_freq_no_singletons', 'wb')
pickle.dump(create_dict_topics_also_singleton_res[0], file1)
file1.close()

df_topics = pd.DataFrame.from_dict(create_dict_topics_also_singleton_res[1], orient='index', columns=list_date[:25])
df_topics["Number of occurrences"] = create_dict_topics_also_singleton_res[2]
print(df_topics)

df_topics.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/res_naive_fun_freq_not_singletons.csv')

print('Time to find frequent itemset')
time_naive_freq = time.time() - start_time
print("--- %s seconds ---" % time_naive_freq)

file1 = open('pickle_time_naive_fun_freq_no_singletons', 'wb')
pickle.dump(time_naive_freq, file1)
file1.close()'''

# read the outputs
# read the dictionary dict_topic_day_num
'''file1 = open('/home/veror/PycharmProjects/DataMiningProj_OK/res_NAIVE_FREQ_NO_SINGLETONS/pickle_naive_fun_freq_no_singletons', 'rb')
pickle_eff_apriori = pickle.load(file1)
print(pickle_eff_apriori)

# read the dictionary dict_topic_day_num
file2 = open('/home/veror/PycharmProjects/DataMiningProj_OK/res_NAIVE_FREQ_NO_SINGLETONS/pickle_time_naive_fun_freq_no_singletons', 'rb')
pickle_time_eff_apriori = pickle.load(file2)
print(pickle_time_eff_apriori)

# read the dataset of frequences
df_eff_apriori = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_NAIVE_FREQ_NO_SINGLETONS/res_naive_fun_freq_not_singletons.csv", )
print(df_eff_apriori)'''

#######################################################################################################################
#NOT SINGLETON, using ASSOCIATION RULES to decide which topics are important
# EFF_ARIORI_RULES_FUN without singleton and computing frequences only of frequent topics

'''start_time = time.time()

apply_fun_res_1 = apply_fun('eff_apriori_rules_fun', 25, column_dataframe)
#print("apply_fun_res_1", apply_fun_res_1)
print(len(apply_fun_res_1.keys()))
create_dict_topics_also_singleton_res = create_dict_topics_also_singleton(apply_fun_res_1)
print("create_dict_topics_also_singleton_res[0]", create_dict_topics_also_singleton_res[0])
print("create_dict_topics_also_singleton_res[1]", create_dict_topics_also_singleton_res[1])
print("create_dict_topics_also_singleton_res[2]", create_dict_topics_also_singleton_res[2])

df_topics = pd.DataFrame.from_dict(create_dict_topics_also_singleton_res[1], orient='index', columns=list_date[:25])
df_topics["Number of occurrences"] = create_dict_topics_also_singleton_res[2]
#print(df_topics)

print('Time to find frequent itemset')
time_eff_apriori_time_rules = time.time() - start_time
print("--- %s seconds ---" % time_eff_apriori_time_rules)'''


# save the result dictionary dict_topic_day_num
'''file1 = open('pickle_eff_apriori_rules', 'wb')
pickle.dump(create_dict_topics_also_singleton_res[0], file1)
file1.close()

# save the dataframe with frequences
df_topics.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/res_eff_apriori_rules.csv')

file1 = open('pickle_time_eff_apriori_rules', 'wb')
pickle.dump(time_eff_apriori_time_rules, file1)
file1.close()'''

# read the outputs
# read the dictionary dict_topic_day_num
'''file1 = open('/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_RULES/pickle_eff_apriori_rules', 'rb')
pickle_eff_apriori = pickle.load(file1)
print(pickle_eff_apriori)

# read the dictionary dict_topic_day_num
file2 = open('/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_RULES/pickle_time_eff_apriori_rules', 'rb')
pickle_time_eff_apriori = pickle.load(file2)
print(pickle_time_eff_apriori)

# read the dataset of frequences
df_eff_apriori = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_RULES/res_eff_apriori_rules.csv", )
print(df_eff_apriori)'''

#######################################################################################################################
# ALSO SINGLETON
# NAIVE_FUN (TOP_K) with also singleton and computing frequences only of frequent topics
'''start_time = time.time()

apply_fun_res_3 = apply_fun('naive_fun', 25, column_dataframe)
print(apply_fun_res_3)

create_dict_topics_also_singleton_res = create_dict_topics_also_singleton(apply_fun_res_3)
print("create_dict_topics_also_singleton_res[0]", create_dict_topics_also_singleton_res[0])
print("create_dict_topics_also_singleton_res[1]", create_dict_topics_also_singleton_res[1])
print("create_dict_topics_also_singleton_res[2]", create_dict_topics_also_singleton_res[2])

# save the result dictionary dict_topic_day_num
file1 = open('pickle_naive_fun_normal', 'wb')
pickle.dump(create_dict_topics_also_singleton_res[0], file1)
file1.close()

df_topics = pd.DataFrame.from_dict(create_dict_topics_also_singleton_res[1], orient='index', columns=list_date[:25])
df_topics["Number of occurrences"] = create_dict_topics_also_singleton_res[2]
print(df_topics)

df_topics.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/res_naive_fun_normal.csv')

print('Time to find frequent itemset')
time_naive = time.time() - start_time
print("--- %s seconds ---" % time_naive)

file1 = open('pickle_time_naive_fun_normal', 'wb')
pickle.dump(time_naive, file1)
file1.close()'''

# read the outputs
# read the dictionary dict_topic_day_num
'''file1 = open('/home/veror/PycharmProjects/DataMiningProj_OK/res_NAIVE_NORMAL/pickle_naive_fun_normal', 'rb')
pickle_eff_apriori = pickle.load(file1)
print(pickle_eff_apriori)

# read the dictionary dict_topic_day_num
file2 = open('/home/veror/PycharmProjects/DataMiningProj_OK/res_NAIVE_NORMAL/pickle_time_naive_fun_normal', 'rb')
pickle_time_eff_apriori = pickle.load(file2)
print(pickle_time_eff_apriori)

# read the dataset of frequences
df_eff_apriori = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_NAIVE_NORMAL/res_naive_fun_normal.csv", )
print(df_eff_apriori)'''

#######################################################################################################################
# NOT SINGLETON
# NAIVE_FUN (TOP_K) without singleton and computing frequences only of frequent topics
'''start_time = time.time()

apply_fun_res_3 = apply_fun('naive_fun', 25, column_dataframe)
print(apply_fun_res_3)

create_dict_topics_also_singleton_res = create_dict_topics_also_singleton(apply_fun_res_3)
print("create_dict_topics_also_singleton_res[0]", create_dict_topics_also_singleton_res[0])
print("create_dict_topics_also_singleton_res[1]", create_dict_topics_also_singleton_res[1])
print("create_dict_topics_also_singleton_res[2]", create_dict_topics_also_singleton_res[2])

# save the result dictionary dict_topic_day_num
file1 = open('pickle_naive_fun_no_singletons', 'wb')
pickle.dump(create_dict_topics_also_singleton_res[0], file1)
file1.close()

df_topics = pd.DataFrame.from_dict(create_dict_topics_also_singleton_res[1], orient='index', columns=list_date[:25])
df_topics["Number of occurrences"] = create_dict_topics_also_singleton_res[2]
print(df_topics)

df_topics.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/res_naive_fun_no_singletons.csv')

print('Time to find frequent itemset')
time_naive_no_singletons = time.time() - start_time
print("--- %s seconds ---" % time_naive_no_singletons)

file1 = open('pickle_time_naive_fun_no_singletons', 'wb')
pickle.dump(time_naive_no_singletons, file1)
file1.close()'''

# read the outputs
# read the dictionary dict_topic_day_num
'''file1 = open('/home/veror/PycharmProjects/DataMiningProj_OK/res_NAIVE_NORMAL_NO_SINGLETONS/pickle_naive_fun_no_singletons', 'rb')
pickle_eff_apriori = pickle.load(file1)
print(pickle_eff_apriori)

# read the dictionary dict_topic_day_num
file2 = open('/home/veror/PycharmProjects/DataMiningProj_OK/res_NAIVE_NORMAL_NO_SINGLETONS/pickle_time_naive_fun_no_singletons', 'rb')
pickle_time_eff_apriori = pickle.load(file2)
print(pickle_time_eff_apriori)

# read the dataset of frequences
df_eff_apriori = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_NAIVE_NORMAL_NO_SINGLETONS/res_naive_fun_no_singletons.csv", )
print(df_eff_apriori)
#print(sorted(df_eff_apriori['Unnamed: 0']))
'''
######################################################################################################################
# TODO QUESTA E' UGUALE A QUELLA SOPRA 1.0
# TODO 1.1) da usare con eff_apriori_rules_fun (considera itemsets freq con dimensione > 1) e NON calcola freq di quelli non freq
def create_dict_topics_tuple(dict_day_topic):
  dict_topic_day_num = {}
  dict_to_dataframe = {}
  list_count = []
  for day in dict_day_topic:
    #print("D", day)
    #print("E", dict_day_topic[day])
    for topic in dict_day_topic[day]:
      #print("Q", topic)
      if not (topic in dict_topic_day_num):
        count = 0
        list_day = []
        list_num = []
        list_freq = []
        list_freq_dataset = []
        #print("EEEEEEE", dict_day_topic[day].keys())
        for day in dict_day_topic:
          list_day.append(day)
          if topic in dict_day_topic[day].keys():
            count += 1
            list_num.append(dict_day_topic[day][topic][1])
            list_freq.append(dict_day_topic[day][topic][0])
            list_freq_dataset.append(dict_day_topic[day][topic][0])
            #print("DDDDDDDD", dict_day_topic[day][topic])
          else:
            list_num.append("not freq")
            list_freq.append("not freq")
            list_freq_dataset.append("")
        dict_topic_day_num[topic] = (count, list(zip(list_day, list_num, list_freq)))
        dict_to_dataframe[topic] = (list_freq_dataset)
        list_count.append(count)
  return dict_topic_day_num, dict_to_dataframe, list_count

# eff_apriori_rules_fun
'''apply_fun_res_2 = apply_fun('eff_apriori_rules_fun', 3, column_dataframe)
print(apply_fun_res_2)'''

'''create_dict_topics_tuple_res = create_dict_topics_tuple(apply_fun_res_2)
print("create_dict_topics_tuple_res[0]", create_dict_topics_tuple_res[0])
print("create_dict_topics_tuple[1]", create_dict_topics_tuple_res[1])
print("create_dict_topics_tuple_res[2]", create_dict_topics_tuple_res[2])

df_topics = pd.DataFrame.from_dict(create_dict_topics_tuple_res[1], orient='index', columns=list_date[:3])
df_topics["Number of occurrences"] = create_dict_topics_tuple_res[2]
print(df_topics)'''


'''print("DICT_TOPIC_DAY_NUM", dict_topic_day_num)

dataframe = pd.DataFrame.from_dict(dict_to_dataframe, orient='index')
print(dataframe)'''
'''x = {'a': [10, [1, 0.2], [2, 0.3]], 'b': [10, [1, 0.2], [2, 0.3]]}
dataframe = pd.DataFrame.from_dict(x, orient='index', columns=['number', 'occurrences, frequence, day0', 'day1'])
print(dataframe)'''

# TODO 2) da usare con eff_apriori_fun/eff_apriori_rules_fun -> calcola freq di quelli non freq
# TODO VARIANTE PER POTER CONTARE LE FREQUENZE DI TUTTI -> RALLENTA MOLTO !!!!!!!!!
# TODO FORSE NON NECESSARIO SE NON CI INTERESSA L'ANDAMENTO NEGLI ALTRI GIORNI
# TODO MAGARI SI PUÒ MIGLIORARE, MA APRIORI FORSE NON RIDA' FREQ DI QUELLI NON FREQUENTI
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
        list_num = [] # TODO questa è list_confidence se si usa eff_apriori_rules_fun
        list_freq = []
        list_flag = []
        #print("EEEEEEE", dict_day_topic[day].keys())
        for day in dict_day_topic:
          list_day.append(day)
          if topic in dict_day_topic[day].keys():
            count += 1
            list_num.append(dict_day_topic[day][topic][1]) # TODO questa è list_confidence se si usa eff_apriori_rules_fun
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
        dict_topic_day_num[topic] = (count, list(zip(list_day, list_num, list_freq, list_flag))) # TODO list_num è list_confidence se si usa eff_apriori_rules_fun
        dict_to_dataframe[topic] = (list_freq)
        list_count.append(count)
        #print("DICT TOPIC DAY NUM", dict_topic_day_num)
  return dict_topic_day_num, dict_to_dataframe, list_count

# with eff_apriori_fun'''
'''create_dict_topics_all_freq_res = create_dict_topics_all_freq(apply_fun_res_1, column_dataframe)
print("create_dict_topics_all_freq_res[0]", create_dict_topics_all_freq_res[0])
print("create_dict_topics_all_freq_res[1]", create_dict_topics_all_freq_res[1])
print("create_dict_topics_all_freq_res[2]", create_dict_topics_all_freq_res[2])

df_topics = pd.DataFrame.from_dict(create_dict_topics_all_freq_res[1], orient='index', columns=list_date[:3])
df_topics["Number of occurrences"] = create_dict_topics_all_freq_res[2]
print(df_topics)'''

# with eff_apriori_rules_fun
'''create_dict_topics_all_freq_res = create_dict_topics_all_freq(apply_fun_res_2, column_dataframe)
print("create_dict_topics_all_freq_res[0]", create_dict_topics_all_freq_res[0])
print("create_dict_topics_all_freq_res[1]", create_dict_topics_all_freq_res[1])
print("create_dict_topics_all_freq_res[2]", create_dict_topics_all_freq_res[2])

df_topics = pd.DataFrame.from_dict(create_dict_topics_all_freq_res[1], orient='index', columns=list_date[:3])
df_topics["Number of occurrences"] = create_dict_topics_all_freq_res[2]
print(df_topics)'''

# TODO example OUTPUT: dict_topic_day_num -> version 1
# 2 days -> day0, day1
# {('covid',): (2, [('day0', 165, 0.559322033898305), ('day1', 10160, 0.6018600793791837)]), ('coronaviru',): (2, [('day0', 25, 0.0847457627118644), ('day1', 1438, 0.08518452698299864)]), ..., ('case', 'covid', 'identifi', 'spread'): (1, [('day0', 'not freq', 'not freq'), ('day1', 122, 0.007227060008293347)])}
# save the dictionary dict_topic_days into a pickle file

'''
file1 = open('pickle_result_EFF_APRIORI_OK_ version_2', 'wb')
pickle.dump(dict_topic_day_num, file1)
file1.close()'''

# read the dictionary dict_topic_days
'''file1 = open('pickle_result_EFF_APRIORI_OK', 'rb')
pickle_eff_apriori = pickle.load(file1)
print(pickle_eff_apriori)'''




'''list_keys_topic = list(pickle_eff_apriori.keys())
items = pickle_eff_apriori.items()
print(items)
for el in pickle_eff_apriori:
  for triple in el[1][1]:
    if triple[1] == 'not freq':'''



'''print(type(pickle_eff_apriori[('covid', 'india')]))

print(type(list_keys_topic[0]))'''


'''x = pickle_eff_apriori[('covid', 'india')]
list_day = []
list_num = []
list_freq = []
for y in x[1]:
  #print(y)
  if y[1] != 'not freq':
    list_day.append(y[0])
    list_num.append(y[1])
    list_freq.append(y[2])'''

'''fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
axes[0].plot(list_day, list_num)
axes[1].plot(list_day, list_freq)
#matplotlib.use('TkAgg')
plt.plot(list_day, list_num)
plt.show()
plt.plot(list_day, list_freq)
#plt.ylim(0, 1)
plt.show()'''
# todo end correct one


# TODO NOT USED ANYMORE
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
#########


#print(set(zip([1,2,3], [4,5,6])))

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

# NOT USED ANYMORE
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
# TODO END NOT USED ANYMORE


#END EFFICIENT APRIORI
#----------------------------------------------------------------------------------------------------------------






#-------------------------------------------------------------------------------------------------------------------
# MLX APRIORI





'''transactions_topics = []
start_time = time.time()
for i in range(25):
  #print(df_grouped['text_cleaned_tuple'].values[i])
  result = mlx_apriori_fun(df_grouped['text_cleaned_tuple'].values[i])
  print(result)
  print("\n")
  #transactions_topics.append(result[1])
print('Time to find frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))'''

'''result = eff_apriori_fun([
    ['John', 'Mark', 'Jennifer'],
    ['John'],
    ['Joe', 'Mark'],
    ['John', 'Anna', 'Jennifer'],
    ['Jennifer', 'John', 'Mark', 'Mark']
  ])'''

# END MLX APRIORI
#---------------------------------------------------------------------------------------------------------------






#---------------------------------------------------------------------------------------------------------------
#NAIVE APPROACH
# https://stackoverflow.com/questions/31495507/finding-the-most-frequent-occurrences-of-pairs-in-a-list-of-lists

# HERE: ALL THE POSSIBLE L -> only until 4, since more there is a problem with memory -> loop for all the cases
def naive_fun(transactions_string):

  transactions = pd.eval(transactions_string)
  #transactions = transactions_string
  print(len(transactions))
  dict_counters = {}
  '''tuples_1 = Counter()
  tuples_2 = Counter()
  tuples_3 = Counter()
  tuples_4 = Counter()'''
  #dict_counters[1] = Counter()
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
          #print(dict_counters[i+1])
          dict_counters[i+1][el] += 1 #/ len(transactions)
  dict_topic = {}
  for key in dict_counters:
    dict_counters[key] = dict_counters[key].most_common(10)
    for el in dict_counters[key]:
      #print("EL", el)
      dict_topic[el[0]] = (el[1], el[1]/len(transactions))
  #print(dict_topic)
  return dict_topic #dict_counters

# todo correct
'''dict_day_topic = {}
start_time = time.time()
for i in range(2):
  #print(df_grouped['text_cleaned_tuple'].values[i])
  result = naive_fun(df_grouped['text_cleaned_tuple'].values[i])
  #print("RESULT:", result)
  #print("\n")
  dict_day_topic["day"+str(i)] = result
print(dict_day_topic)
print('Time to find frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))

dict_topic_day_num = {}
for day in dict_day_topic:
  #print("D", day)
  #print("E", dict_day_topic[day])
  for topic in dict_day_topic[day]:
    #print("Q", topic)
    if not (topic in dict_topic_day_num):
      count = 0
      list_day = []
      list_num = []
      list_freq = []
      #print("EEEEEEE", dict_day_topic[day].keys())
      for day in dict_day_topic:
        list_day.append(day)
        if topic in dict_day_topic[day].keys():
          count += 1
          list_num.append(dict_day_topic[day][topic][0])
          list_freq.append(dict_day_topic[day][topic][1])
          #print("DDDDDDDD", dict_day_topic[day][topic])
        else:
          list_num.append("not freq")
          list_freq.append("not freq")
      dict_topic_day_num[topic] = (count, list(zip(list_day, list_num, list_freq)))
print(dict_topic_day_num)'''

'''file1 = open('pickle_result_NAIVE_APPROACH', 'wb')
pickle.dump(dict_topic_day_num, file1)
file1.close()'''

'''file1 = open('pickle_result_NAIVE_APPROACH', 'rb')
pickle_naive = pickle.load(file1)
print(pickle_naive)'''
# todo end correct

# TODO NOT USED ANYMORE
'''start_time = time.time()
for i in range(1):
  #print(df_grouped['text_cleaned_tuple'].values[i])
  result = naive_fun(df_grouped['text_cleaned_tuple'].values[i])
  print(result)
  print("\n")
  #transactions_topics.append(result[1])
print('Time to find frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))'''
# TODO END NOT USED ANYMORE





# TODO if we want to use this we need to also change a bit as in naive_fun
# HERE: NOT ALL THE POSSIBLE SUBSETS OF ALL LENGTH -> single cases separated
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

'''start_time = time.time()
for i in range(1):
  #print(df_grouped['text_cleaned_tuple'].values[i])
  result = naive_fun_2(df_grouped['text_cleaned_tuple'].values[i])
  print(result)
  print("\n")
  #transactions_topics.append(result[1])
print('Time to find frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))'''


'''result = naive_fun_2([
    ['John', 'Mark', 'Jennifer'],
    ['John'],
    ['Joe', 'Mark'],
    ['John', 'Anna', 'Jennifer'],
    ['Jennifer', 'John', 'Mark', 'Mark']
  ])'''

# END NAIVE APPROACH
# ---------------------------------------------------------------------------------------------------------------

























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