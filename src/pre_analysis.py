import pandas as pd
import matplotlib.pyplot as plt
import pickle


# Ausiliar functions for preanalysis of datasets and plots


def tweets_day(df_grouped_path, text_col):
    '''Given a dataset with tweets grouped by date, and the text column, it returns a list of dates and a list of the
    number of tweets per day'''
    df_grouped = pd.read_csv(df_grouped_path, sep=' ')
    #print(df_grouped)
    counts = df_grouped[text_col].apply(lambda x: len(pd.eval(x)))
    list_counts = counts.to_list()
    list_date = df_grouped['date_only'].tolist()
    return list_date, list_counts


def plot_date_tweets(list_date_counts, path_figure):
    '''GIven the result of tweets_day, it plots the number of tweets per day in a barplot'''
    # barplot date - number of tweets
    plt.bar(list_date_counts[0], height=list_date_counts[1])
    plt.xticks(list_date_counts[0], rotation='vertical')
    plt.xlabel('Date')
    plt.ylabel('Number of tweets')
    plt.tight_layout()
    plt.savefig(path_figure)
    plt.show()


def plot_freq_topic(list_date, res_df, topic_index, path_figure, title_plot):
    '''Given a popular topic over time and the days in which it is popular, it plots its frequency for each day'''
    list_freq = []
    list_date_ok = []
    for column in list_date:
        el = res_df.iloc[topic_index][column]
        if pd.notnull(el):
            list_freq.append(el)
            list_date_ok.append(column)
    plt.plot(list_date_ok, list_freq, marker='o')
    plt.xticks(list_date_ok, rotation='vertical')
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.title(title_plot, loc='center')
    #b.pyplot.title(label, fontdict=None, loc='center', pad=None, **kwargs)[source]
    plt.tight_layout()
    plt.savefig(path_figure)
    plt.show()


# pickle
'''file1 = open('pickle_covid_date_counts', 'wb')
pickle.dump(list_date_counts, file1)
file1.close()'''

'''file1 = open('/home/veror/PycharmProjects/DataMiningProj_OK/pickle_covid_date_counts', 'rb')
list_date_counts = pickle.load(file1)
print(list_date_counts)'''








