import algorithm
import pandas as pd
import time

# per semplicita' lasciare i dataset possibili nella stessa cartella, cosi non necessita il path
# accetta sia .csv che senza perche lo aggiunge lui
def input_user():
    '''Function to get parameters from the user to execute the algorithm to find frequent topics'''

    topics_singletons = 100
    type_algorithm = 100

    dataset = input("Select the dataset where you want to find frequent topics:")
    '''if len(dataset) >= 4 and not (dataset[-4:] == '.csv'):
        dataset = dataset + '.csv'''

    '''df_grouped = pd.read_pickle(dataset)
    num_date = df_grouped.shape[0]'''

    print("\nDATASET ACQUIRED")

    topics_number = int(input("\nSelect how many frequent topics you want are returned, type 0 if you want all the possible ones:"))

    while(type_algorithm != 1 and type_algorithm != 2 and type_algorithm != 3 and type_algorithm != 4):
        type_algorithm = int(input("\nSelect the number of the algorithm you want to apply:\n1 - apriori-based algorithm\n2 - apriori-based algorithms with association rules\n3 - baseline algorithm frequency\n4 - baseline algorithm top-k\nType:"))

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
    '''max, min = -1, -1
    ans = input("\nDo you want to get only the popular topics in a certain number of days? [y/n]")
    if ans == 'y' or ans == 'Y':
        while ((max < 1 or min < 1)  or (max > num_date or min > num_date)  or max < min):
            max = int(input("Write the maximum number of days in which you want the popular topics appear, remember that the total number of days is " + str(num_date) + ":"))
            min = int(input("Write the minimum number of days in which you want the popular topics appear, remember that the total number of days is " + str(num_date) + ":"))
    '''
    print("\n\nSTART SEARCHING FOR TOPICS\n")
    start_time = time.time()

    if input_res['type_algorithm'] == 1:
        # run eff_apriori
        apply_fun_res = algorithm.apply_fun('eff_apriori_fun', num_date, column_dataframe, input_res['topics_singletons'])
    elif input_res['type_algorithm'] == 2:
        # run eff_apriori_rules
        apply_fun_res = algorithm.apply_fun('eff_apriori_rules_fun', num_date, column_dataframe, input_res['topics_singletons'])
    elif input_res['type_algorithm'] == 3:
        # run naive frequency
        apply_fun_res = algorithm.apply_fun('naive_fun_freq', num_date, column_dataframe,
                                            input_res['topics_singletons'])  # todo scegliere se freq o no
    else:
        # run naive top-k
        apply_fun_res = algorithm.apply_fun('naive_fun', num_date, column_dataframe,
                                            input_res['topics_singletons'])  # todo scegliere se freq o no

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
    if input_res['type_algorithm'] == 1:
        result.to_csv('results/algorithm_result_apriori.csv', sep=',')
        f = open("results/time_apriori.txt", 'w')
        f.write("Time apriori: " + str(time_topics))
        f.close()
    elif input_res['type_algorithm'] == 2:
        result.to_csv('results/algorithm_result_apriori_rules.csv', sep=',')
        f = open("results/time_apriori_rules.txt", 'w')
        f.write("Time apriori rules: " + str(time_topics))
        f.close()
    elif input_res['type_algorithm'] == 3:
        result.to_csv('results/algorithm_result_naive_freq.csv', sep=',')
        f = open("results/time_naive_freq.txt", 'w')
        f.write("Time naive with frequency: " + str(time_topics))
        f.close()
    else:
        result.to_csv('results/algorithm_result_naive_top_k.csv', sep=',')
        f = open("results/time_naive_top_k.txt", 'w')
        f.write("Time naive with top-k: " + str(time_topics))
        f.close()

    print(time_topics)
    '''desired_width = 320

    pd.set_option('display.width', desired_width)

    #np.set_printoption(linewidth=desired_width)

    pd.set_option('display.max_columns', 31)
    print(result)'''
    return result, time_topics

# RUN PROGRAM
#print(run_algorithm())
print("\nTHE OUTPUT IS IN THE FOLDER bin/results")


# SORT BY FREQUENCE
# TODO sistemare separatore csv
'''df = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_NO_SINGLETONS/res_eff_apriori_fun_not_singletons.csv")
print(df.columns)
sorted_df = df.sort_values(by = 'Number of occurrences', ascending = False)
print(sorted_df)
sorted_df.to_csv("/home/veror/PycharmProjects/DataMiningProj_OK/res_EFF_APRIORI_NO_SINGLETONS/SORTED_PROVA.csv", index=False, sep=' ')
#def algorithm_execution():'''
