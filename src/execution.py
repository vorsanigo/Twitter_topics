import algorithm
import pandas as pd
import time
import sys


# Default execution -> results: only topics of at least 2 terms and frequent on at least 2 days


algorithm_type = int(sys.argv[1]) # if 0 -> eff apriori, if 1 -> eff apriori rules
#algorithm_type = 1

if algorithm_type == 0:
    print("\nEfficient apriori approach will be applied\n\n")
else:
    print("Efficient apriori with rules approach will be applied\n\n")

df_grouped = pd.read_pickle('../data/covid_input')
print("DATASET CORRECTLY IMPORTED\n\n")

num_date = df_grouped.shape[0]
column_dataframe = df_grouped['text_cleaned_tuple']
list_date = df_grouped['date_only'].tolist()

print("START SEARCHING FOR TOPICS\n\n")
start_time = time.time()

# apply method
if algorithm_type == 0:
    apply_fun_res = algorithm.apply_fun('eff_apriori_fun', num_date, column_dataframe, 0)  # 0 -> not singletons
else:
    apply_fun_res = algorithm.apply_fun('eff_apriori_rules_fun', num_date, column_dataframe, 0) # 0 -> not singletons

# compute result
res = algorithm.create_dict_topics(apply_fun_res)
# create dataframe of results
df_topics = pd.DataFrame.from_dict(res[1], orient='index', columns=list_date[:num_date])
df_topics["Number_of_occurrences"] = res[2]

# consider only topics that appear at least in 2 days
pruned_df_topics = df_topics[df_topics["Number_of_occurrences"] >= 2]

# sort dataframe of results by descendent frequence of topics
sorted_df_topics = pruned_df_topics.sort_values(by='Number_of_occurrences', ascending=False)

print('\nTime to find frequent topics:\n')
time_topics = time.time() - start_time
print("--- %s seconds ---" % time_topics)
print("\n")

# store result
sorted_df_topics.to_csv('results/result.csv', sep=',')
f = open("results/time.txt", 'w')
f.write("Time: " + str(time_topics))
f.close()
print("\nTHE OUTPUT IS IN THE FOLDER bin/results")

print(sorted_df_topics)