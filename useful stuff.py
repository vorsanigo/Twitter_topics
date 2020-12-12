import pickle
from datetime import datetime
import pandas as pd

df = pd.read_csv('/home/veror/PycharmProjects/DataMiningProject/COPY_covid19_tweets .csv')
# STRING -> DATETIME CONVERSION
date_time = "2020-07-25 12:27:21"
date_time_obj = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
# DATE FROM DATETIME OBJECT
date = date_time_obj.date()
print(date)


# USE PICKLE
l = [1,2,3]
file = open('file', 'wb')
pickle.dump(l, file)
file.close()

file = open('file', 'rb')
x = pickle.load(file)
print(x)

r = ['1','5','6']
l = ['1','2','4']
text = ""
for word in l:
    if word not in r:
        text = text + " " + word
print(text)