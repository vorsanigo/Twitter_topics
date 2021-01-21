import cleaning
import pre_analysis
import time

# cleaning dataset covid19
start_time_covid = time.time()
cleaning.cleaning_fun("data/DATASET_covid19_tweets.csv",
                      ['user_name', 'user_location', 'user_description', 'user_created', 'user_followers',
                       'user_friends', 'user_favourites', 'user_verified', 'source', 'is_retweet'],
                    'data/DATASET_COVID_cleaned.csv',
                      'data/covid_grouped.csv')
print('Time to preprocess covid:')
time_preprocessing_covid = time.time() - start_time_covid
print("--- %s seconds ---" % time_preprocessing_covid)

# cleaning dataset Australia
start_time_australia = time.time()
cleaning.cleaning_fun("data/auspol2019.csv", ['id', 'retweet_count', 'favorite_count', 'user_id', 'user_name', 'user_screen_name', 'user_description',
         'user_location', 'user_created_at'], "data/DATASET_AUSTRALIA_cleaned.csv", "data/australia_grouped.csv")
print('Time to preprocess Australia:')
time_preprocessing_australia = time.time() - start_time_australia
print("--- %s seconds ---" % time_preprocessing_australia)

# plot dates - tweets covid19
list_date_counts_covid = pre_analysis.tweets_day("data/covid_grouped.csv", 'text_cleaned_tuple')
pre_analysis.plot_date_tweets(list_date_counts_covid, "data/covid_plot_days_tweets.png")

# plot dates - tweets Australia
list_date_counts_australia = pre_analysis.tweets_day("data/australia_grouped.csv", 'text_cleaned_tuple')
pre_analysis.plot_date_tweets(list_date_counts_australia, "data/australia_plot_days_tweets.png")