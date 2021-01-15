from ast import literal_eval

import pandas as pd
import numpy as np

def literal_eval_special(element):
    print(element)
    return literal_eval(element)

# TODO SEPARATOREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
df = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_tweets_cleaned_OK.csv", sep=' ')
print(df)

df.drop([], axis=1, inplace=True)




'''df1 = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_tweets.csv")
df = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_tweets_cleaned_FINAL_FINAL.csv")
found = df1[df1['date'].str.contains('https://t.co/Krd1NSnZbh… https://t.co/06gPgbGyBj')]
print(found.count())'''
'''#df = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_tweets_cleaned_FINAL_FINAL.csv", converters={'text_cleaned': literal_eval})
#df['text_cleaned'] = df['text_cleaned'].apply(literal_eval)
#df['text_cleaned'] = df['text_cleaned'].apply(lambda x: literal_eval(x))
df.loc[:,'text_cleaned'] = df.loc[:,'text_cleaned'].apply(lambda x: literal_eval_special(x))
#print(df['text_cleaned'].isnull().values.any)
print(df['text_cleaned'])'''

'''print(df.iloc[65503])
print('2020-07-26' in df.values)
df1 = pd.DataFrame(df.groupby('date_only')['text_cleaned'].apply(list).reset_index())
print(df1)
print("ddd")
print(df1.iloc[26])'''

'''print(type(df['date_only'][1]))
print(df1)
print(df1.size)
print(df1.dtypes)
print(list(df1.columns))
print("-----------")
print(df1.iloc[0])
print("------------------")
print(df.iloc[0])
print("--------------------------")
print(df1['date_only'])
print(df1['text_cleaned'])
print("-------------------------------------------------------------------------")
print(df['date_only'])
df1.to_csv(r'/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_groupby_new.csv', index=False)
'''
'''df = pd.DataFrame(
       [
           ("bird", "Falconiformes", 389.0),
           ("bird", "Psittaciformes", 24.0),
           ("mammal", "Carnivora", 80.2),
            ("mammal", "Primates", np.nan),
            ("mammal", "Carnivora", 58),
       ],
       index=["falcon", "parrot", "lion", "monkey", "leopard"],
      columns=("class", "order", "max_speed"),
)
print(df)

print("------------------")
ddd = df.groupby('class')['order'].apply(list)
ddd.apply(print)
d1 = pd.DataFrame(ddd)
print(type(d1))'''
'''grouped = df.groupby("class")
grouped.apply(print)'''
'''
                    
df = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/DATASET_covid19_tweets_cleaned_YEShashtags_FINAL.csv")
df2 = df.iloc[:3]
print(df.dtypes)
print(df2.dtypes)

d = {'date': ['2020-07-25', '2020-07-26', '2020-07-26', '2020-07-25', '2020-07-27', '2020-07-25'], 'text': ["ciao amico", "no no no", "s", "d", "c d r", "e r erre"], 'e': [7, 8, 9, 1, 2, 3]}
df = pd.DataFrame(data=d)
print(df.dtypes)


df1 = df.groupby([df['date']])['e']
#df1 = df.groupby([df['date'].dt.date])['text']
print(df1)'''