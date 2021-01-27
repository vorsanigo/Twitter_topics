# Twitter_topics

The aim of this project is to find popular topics over time in a dataset of tweets. A topic is composed by a set of
terms that appear together in a tweet. Here, in particular, the considered case is the COVID19 dataset.

## Execution

### 1) Prerequisites

1) Installation of `Pyhton 3.8.5` or more
2) It is recommended to create a virtual environment for the project
3) Installation of two libraries:
   1) `pandas 1.2.1`
   2) `efficient-apriori 1.1.1`
  
### 2) Default execution by command line

NB: in this execution the apriori methods (with or without association rules) can be run and the parameters are already set:
1) support threshold = 0.03
2) confidence threshold = 0.7
3) only topics with at least 2 terms are returned
4) only topics frequent in at least 2 days are returned
   
### 3) Execution with execution.py script:
1) Open the terminal and navigate to the `bin` directory, where there is a script named `execution.py`
2) Run the `.py` script: the command on the terminal is `python execution.py arg`
    arg  



Run the `.py` script: in the command prompt the command is the following `python program.py arg`
    - `arg` is the external parameter to be passed to the script via the command line. It represents the number of itemsets we want the solution to return, so it **must** be a number. If you want all possible itemsets, enter 0.

    Example: `python program.py 50` will return the 50 most frequent itemset in time, based on the total number of dates in which the itemset is present.  

### 4) Execution as user with main.py script

NB: in this execution both the baseline and apriori methods can be run, the following parameters are already set:
1) frequency/support threshold = 0.03
2) confidence threshold = 0.7
   
#### 5) Execution:
1) Open the terminal and navigate to the `bin` directory, where there is a script named `main.py`
2) Run the `.py` script: the command on the terminal is `python main.py`
3) When the program asks to select the dataset, write `data/covid_input` (other datasets could be used)
4) Follow the instructions given by the program to set different parameters:
   1) Maximum and minimum number of days in which the returned topics must be popular
   2) Total number of results to be returned
   3) Type of solution method to use to find popular topics
   4) If consider or not topics composed by just one term (singletons)
   


### Input and output

- INPUT: the program takes as input a cleaned dataset where tweets are grouped by date, which is a pickle file with path
  `data/covid_input`. It is possible to use also other cleaned datasets with tweets grouped by date with the following format:
  - it must be a `pickle` file 
  - 2 columns:
      - `date_only` containing the dates in format YY-MM-DD
      - `text_cleaned_tuple` containing the tweets as a list of tuples
- OUTPUT: it is stored in `bin/results` as a `csv` file 

## Structure

### Folders

- `doc` contains the report about the project with the explanation of the solution and the experimental part 
- `src` contains the source code of the project
- `data` contains the datasets
- `bin` contains the files to execute the program

### Datasets

Two datasets have been considered (the second one to support the evaluation of the solution):
1) COVID19 dataset containing tweets between 24-07-2020 and 30-08-2020 about covid
2) Australian election 2019 dataset containing tweets between 10-05-2020 and 20-05-2020

After the cleaning we obtain as input datasets where tweets are grouped by date.

### Solution method

2 methods (each with 2 variants) have been evaluated (the first ones as baseline, the last ones as solution):
1) Baseline:
   - Naive with top-k approach
   - Naive with frequency
2) Apriori approach:
   - Efficient apriori
   - Efficient apriori with association rules

It will be possible to notice that the apriori approach has much better performances.



ELIMINARE
parlare del caso Russia -> row 24 (22) of covid apriori no sigletons

   