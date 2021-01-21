import algorithm
import pandas as pd

# TODO per semplicità lasciare i dataset possibili nella stessa cartella, così non necessita il path
# accetta sia .csv che senza perché lo aggiunge lui
def input_user():
    dataset = input("Select the dataset where you want to find frequent topics:")
    if len(dataset) >= 4 and not (dataset[-4:] == '.csv'):
        dataset = dataset + '.csv'
    df = pd.read_csv(dataset, sep=' ')
    print(df)
    topics_number = int(input("\nSelect how many frequent topics you want are returned, type 0 if you want all the possible ones:"))
    topics_length = int(input("\nSelect:\n1 - if you do NOT want topics with only one term\n2 - otherwise\nType:"))
    type_algorithm = int(input("\nSelect the number of the algorithm you want to apply:\n1 - basic algorithm\n2 - developed algorithm\nType:"))
    return dataset, topics_number, topics_length, type_algorithm
res = input_user()

def run_algorithm():
    input_res = input_user()

#def algorithm_execution():
