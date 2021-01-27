import pandas as pd
import pre_analysis


# Plot popular topics and their frequency


# covid
df_covid_grouped = pd.read_csv("data/covid_grouped.csv", sep=' ')
covid_apriori_res = pd.read_csv("results/COVID apriori total_no_singletons/algorithm_result_apriori_NO_singleton.csv", sep=',')
list_date = df_covid_grouped['date_only'].tolist()

# (covid, india) -> index = 11
pre_analysis.plot_freq_topic(list_date, covid_apriori_res, 11, 'data/plot_covid_india.png', '(covid, india)')
# ('case', 'covid', 'new') -> index = 8
pre_analysis.plot_freq_topic(list_date, covid_apriori_res, 8, 'data/plot_case_covid_new.png', '(case, covid, new)')
# (covid, vaccin) -> index = 14
pre_analysis.plot_freq_topic(list_date, covid_apriori_res, 14, 'data/plot_covid_vaccin_new.png', '(covid, vaccin)')


# australia
df_australia_grouped = pd.read_csv("data/australia_grouped.csv", sep=' ')
australia_apriori_res = pd.read_csv("/home/veror/PycharmProjects/DataMiningProj_OK/results/RESULTS support 0.015/RESULTS ON REPORT/AUSTRALIA/apriori rules/0.9.csv", sep=',')
list_date = df_australia_grouped['date_only'].tolist()

# (elect, morrison, scott) -> index = 2
pre_analysis.plot_freq_topic(list_date, australia_apriori_res, 2, 'data/plot_australia_elect_morrison_scott.png', '(elect, morrison, scott)')
