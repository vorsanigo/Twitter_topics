import algorithm
import pandas as pd
import time

# TODO per semplicità lasciare i dataset possibili nella stessa cartella, così non necessita il path
# accetta sia .csv che senza perché lo aggiunge lui
def input_user():
    '''Function to get parameters from the user to execute the algorithm to find frequent topics'''
    dataset = input("Select the dataset where you want to find frequent topics:")
    if len(dataset) >= 4 and not (dataset[-4:] == '.csv'):
        dataset = dataset + '.csv'
    df = pd.read_csv(dataset, sep=' ')
    print(df)
    topics_singletons = 2
    type_algorithm = 2
    topics_number = int(input("\nSelect how many frequent topics you want are returned, type 0 if you want all the possible ones:"))
    while(topics_singletons != 0 and topics_singletons != 1):
        topics_singletons = int(input("\nSelect:\n0 - if you do NOT want topics with only one term\n1 - if you want also topics with only one term\nType:"))
    while(type_algorithm != 0 and type_algorithm != 1):
        type_algorithm = int(input("\nSelect the number of the algorithm you want to apply:\n0 - baseline algorithm\n1 - apriori-based algorithm\n2 - apriori-based algorithms with association rules\nType:"))
    return {'dataset': dataset, 'topics_number': topics_number, 'topics_singletons': topics_singletons, 'type_algorithm': type_algorithm}
#res = input_user()

def run_algorithm():
    '''Function to run the algorithm to find frequent topics, it return the output as dataframe and it saves it as csv,
    where topics are sorted by the number of days in which they appear on the overall period of time'''
    input_res = input_user()
    df_grouped = pd.read_csv(input_res['dataset'], sep=' ') # dataset
    column_dataframe = df_grouped['text_cleaned_tuple']
    list_date = df_grouped['date_only'].tolist()
    #print(list_date)
    #print(len(list_date))
    num_date = df_grouped.shape[0]
    #print(num_date)
    #print(num_date[2])
    #if input_res['topics_singletons'] == 1: # yes singletons
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
    if not (input_res['topics_number'] == 0):
        result = sorted_df_topics.head(input_res['topics_number'])
    else:
        result = sorted_df_topics
    print('Time to find frequent topics:')
    time_topics = time.time() - start_time
    print("--- %s seconds ---" % time_topics)
    # cut results according to user's request
    if input_res['type_algorithm'] == 1:
        result.to_csv('/home/veror/PycharmProjects/DataMiningProj_OK/results/algorithm_result_apriori.csv', sep=',')
    else:
        result.to_csv('/home/veror/PycharmProjects/DataMiningProj_OK/results/algorithm_result_naive.csv', sep=' ')
    return result

print(run_algorithm())



# SORT BY FREQUENCE
# TODO sistemare separatore csv
'''df = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_NO_SINGLETONS/res_eff_apriori_fun_not_singletons.csv")
print(df.columns)
sorted_df = df.sort_values(by = 'Number of occurrences', ascending = False)
print(sorted_df)
sorted_df.to_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_NO_SINGLETONS/SORTED_PROVA.csv", index=False, sep=' ')
#def algorithm_execution():'''
