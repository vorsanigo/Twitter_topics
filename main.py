import algorithm
import pandas as pd
import time

# TODO RESULTS SAVED WITH ' ' SEPARATOR IN APRIORI RESULTS ROW 100 OF THIS SCRIPT, BUT THE ONES IN RESULTS HAVE STILL ,
#  -> IF WE RUN LOT_TOPICS WE NEED TO CHANGE SEP IN ROW 5 OF PLOT_TOPICS.PY



# -*- coding: utf-8 -*-
# TODO ESPERIMENTI CON FZ: APRIORI, APRIORI_RULES, NAIVE (FEQ O TOP_KEY?)
# TODO GRAFICO DI QUALCUNO -> OK
# TODO CONSIDERARE QUELLI FREQUENTI SOLO IN UN GIORNO? MAGARI NO/SI -> ARGOMENTARE
# TODO magari togliere parole super frequenti?
# TODO QUALI ALTRI ESPERIMENTI FARE? ALTRO DATASET, SOTTOPORZIONI DI DATASET? ANALISI STATISTICHE?
# TODO COME USARE L'UTENTE?
# TODO REPORT



# TODO per semplicita' lasciare i dataset possibili nella stessa cartella, cosi non necessita il path
# accetta sia .csv che senza perche lo aggiunge lui
def input_user():
    '''Function to get parameters from the user to execute the algorithm to find frequent topics'''

    topics_singletons = 2
    type_algorithm = 3

    dataset = input("Select the dataset where you want to find frequent topics:")
    '''if len(dataset) >= 4 and not (dataset[-4:] == '.csv'):
        dataset = dataset + '.csv'''
    #df = pd.read_csv(dataset, sep=' ')
    #print(df)
    print("\nDATASET ACQUIRED")

    topics_number = int(input("\nSelect how many frequent topics you want are returned, type 0 if you want all the possible ones:"))

    while(type_algorithm != 0 and type_algorithm != 1 and type_algorithm != 2):
        type_algorithm = int(input("\nSelect the number of the algorithm you want to apply:\n0 - baseline algorithm\n1 - apriori-based algorithm\n2 - apriori-based algorithms with association rules\nType:"))

    if type_algorithm != 2:
        while (topics_singletons != 0 and topics_singletons != 1):
            topics_singletons = int(input("\nSelect:\n0 - if you do NOT want topics with only one term\n1 - if you want also topics with only one term\nType:"))

    return {'dataset': dataset, 'topics_number': topics_number, 'topics_singletons': topics_singletons, 'type_algorithm': type_algorithm}
#res = input_user()


def run_algorithm():
    '''Function to run the algorithm to find frequent topics, it return the output as dataframe and it saves it as csv,
    where topics are sorted by the number of days in which they appear on the overall period of time'''

    input_res = input_user()

    df_grouped = pd.read_pickle(input_res['dataset']) # dataset
    column_dataframe = df_grouped['text_cleaned_tuple']
    list_date = df_grouped['date_only'].tolist()
    #print(list_date)
    #print(len(list_date))
    num_date = df_grouped.shape[0]
    #print(num_date)
    #print(num_date[2])
    #if input_res['topics_singletons'] == 1: # yes singletons

    print("\n\nSTART SEARCHING FOR TOPICS\n")
    start_time = time.time()

    if input_res['type_algorithm'] == 0:
        # run naive
        apply_fun_res = algorithm.apply_fun('naive_fun_freq', num_date, column_dataframe,
                                            input_res['topics_singletons'])  # todo scegliere se freq o no
    elif input_res['type_algorithm'] == 1: # apriori yes singletons
        # run eff_apriori
        apply_fun_res = algorithm.apply_fun('eff_apriori_fun', num_date, column_dataframe, input_res['topics_singletons'])
    else:
        # run eff_apriori_rules
        apply_fun_res = algorithm.apply_fun('eff_apriori_rules_fun', num_date, column_dataframe, input_res['topics_singletons'])

    # compute result
    res = algorithm.create_dict_topics(apply_fun_res)
    # create dataframe of results
    df_topics = pd.DataFrame.from_dict(res[1], orient='index', columns=list_date[:num_date])
    df_topics["Number_of_occurrences"] = res[2]
    # sort dataframe of results by descendent frequence of topics
    sorted_df_topics = df_topics.sort_values(by='Number_of_occurrences', ascending=False)
    # cut results according to user's request
    if not (input_res['topics_number'] == 0):
        result = sorted_df_topics.head(input_res['topics_number'])
    else:
        result = sorted_df_topics

    print('\nTime to find frequent topics:\n')
    time_topics = time.time() - start_time
    print("--- %s seconds ---" % time_topics)
    print("\n")

    # save result into folder results
    if input_res['type_algorithm'] == 0:
        result.to_csv('results/algorithm_result_naive.csv', sep=',')
        f = open("results/time_naive.txt", 'w')
        f.write("Time naive: " + str(time_topics))
        f.close()
    elif input_res['type_algorithm'] == 1:
        result.to_csv('results/algorithm_result_apriori.csv', sep=',')
        f = open("results/time_apriori.txt", 'w')
        f.write("Time apriori: " + str(time_topics))
        f.close()
    else:
        result.to_csv('results/algorithm_result_apriori_rules.csv', sep=',')
        f = open("results/time_apriori_rules.txt", 'w')
        f.write("Time apriori rules: " + str(time_topics))
        f.close()
    return result

# RUN PROGRAM
print(run_algorithm())
print("\nTHE OUTPUT IS IN THE FOLDER bin/results")


# SORT BY FREQUENCE
# TODO sistemare separatore csv
'''df = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_NO_SINGLETONS/res_eff_apriori_fun_not_singletons.csv")
print(df.columns)
sorted_df = df.sort_values(by = 'Number of occurrences', ascending = False)
print(sorted_df)
sorted_df.to_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_NO_SINGLETONS/SORTED_PROVA.csv", index=False, sep=' ')
#def algorithm_execution():'''
