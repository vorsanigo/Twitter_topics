import pandas as pd
import pre_analysis

df_covid_grouped = pd.read_csv("data/covid_grouped.csv", sep=' ')
covid_apriori_res = pd.read_csv("results/COVID apriori total_no_singletons/algorithm_result_apriori_NO_singleton.csv", sep=' ')
list_date = df_covid_grouped['date_only'].tolist()
#print(list_date)
#print(covid_apriori_res)

# (covid, india) -> index = 11
pre_analysis.plot_freq_topic(list_date, covid_apriori_res, 11, 'data/plot_covid_india.png', '(covid, india)')
# ('case', 'covid', 'new') -> index = 8
pre_analysis.plot_freq_topic(list_date, covid_apriori_res, 8, 'data/plot_case_covid_new.png', '(case, covid, new)')
# (covid, vaccin) -> index = 14
pre_analysis.plot_freq_topic(list_date, covid_apriori_res, 14, 'data/plot_covid_vaccin_new.png', '(covid, vaccin)')