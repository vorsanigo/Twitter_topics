import pandas as pd
from efficient_apriori import apriori


df = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_tweets_cleaned_OK.csv", sep=' ')
#print(df)
print("----------------------")
df_grouped = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_group_tuple.csv", sep=' ')
#print(dff)
print("----------------------")

df11 = df_grouped.head()
print(df11)

df2 = df.iloc[0]
print(df2)

dff1 = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_group_tuple_sep_comma.csv")
#print(dff1)
print("----------------------")

transactions = pd.eval(df_grouped['text_cleaned_tuple'].values[18])
transactions_string = df_grouped['text_cleaned_tuple'].values[18]
#print(transactions)
print(type(transactions))
print("----------------------")
#val1 = dff1['text_cleaned_tuple'].values[0]
#print(val1)

# apriori function
def apriori_fun(transactions_string): # , min_sup, min_conf, min_len, min_lift, len_rule
  transactions = pd.eval(transactions_string)

  itemsets, rules = apriori(transactions, min_support=0.01, min_confidence=0.66)

  rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
  #for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
  #  print(rule)  # Prints the rule and its confidence, support, lift, ...
  #print(itemsets)

  return itemsets, rules # it returns a tuple (dict, list), maybe we can keep only the itemsets, we are not interestd in association rules

# apply  apriori on each day
#df_grouped['result_apriori'] = df_grouped['text_cleaned_tuple'].apply(lambda x: apriori_fun(x))

#df11['result_apriori'] = df11['text_cleaned_tuple'].apply(lambda x: apriori_fun(x))

#print(apriori_fun(transactions_string))



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