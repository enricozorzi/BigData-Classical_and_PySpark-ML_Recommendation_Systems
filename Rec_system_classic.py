import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process
from tqdm import tqdm
from time import sleep
import psutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
import time
from memory_profiler import profile

USER_ID = 35859          # user_id for which we are going to recommend movies
TOP = 20                 # number of movies to recommend


def resource_monitor():
    with tqdm(total=100, desc='cpu%', position=1, leave=False) as cpubar, tqdm(total=100, desc='ram%', position=0, leave=False) as rambar:
        while True:
            rambar.n = psutil.virtual_memory().percent
            cpubar.n = psutil.cpu_percent()
            rambar.refresh()
            cpubar.refresh()
            sleep(0.5)

# Evaluation functions
def precision_treshold(actual,predicted, threshold):
    v = []
    for i, pred in enumerate(predicted):
        if float(pred) <=  int(actual[i])+threshold and float(pred) >= int(actual[i])-threshold:
            v.append(1)
        else:
            v.append(0)

    m = sum(v)/len(v)
    return m

@profile
def run():
    # Load the books dataset
    df_books_name = pd.read_csv("project/BX-Books.csv", on_bad_lines='skip', sep=';',low_memory=False)
    df_books_name = df_books_name.iloc[:, :-3]
    df_books_name = df_books_name.set_index('ISBN')
    df_books_name = df_books_name.rename_axis(None)
    
    # Load the ratings dataset
    df = pd.read_csv("project/BX-Book-Ratings.csv", on_bad_lines='skip', sep=';')
    df = df[df['Book-Rating'] != 0]

    #Remove the books that are not in the books_name dataset
    df = df[df['ISBN'].isin(df_books_name.index)]

    #Cut the dataframe
    cut = 200000
    df = df[:cut]

    # Remove the books with less than 10 ratings
    df_filter = df.groupby('ISBN').filter(lambda x: len(x) >= 2)


    # Unique list of all the users and books
    users = df_filter['User-ID'].unique()
    books = df_filter['ISBN'].unique()

    # Create an empty dataframe
    df_books = pd.DataFrame(0, index=users, columns=books)
   
    # fill the dataframe with the ratings 
    for index, row in df_filter.iterrows():
        df_books.at[row['User-ID'], row['ISBN']] = row['Book-Rating']
    
 
    # Compute item-item similarity
    item_similarity = cosine_similarity(df_books.T)
    

    # Example user's interactions
    user_interactions = df_books.loc[USER_ID]

    # Calculate item scores based on user's interactions and item similarity
    item_scores = user_interactions.dot(item_similarity)

    # Set the scores of the items that the user has already interacted with to 0
    item_scores[user_interactions > 0] = 0

    # Normalize the scores between 0 and 10
    if item_scores.max() != item_scores.min():  # Check to avoid division by zero
        item_scores = (item_scores - item_scores.min()) / (item_scores.max() - item_scores.min()) * 10

    # Sort items by score and recommend the top-n
    recommended_items = np.argsort(item_scores)[::-1][:TOP]
    


    from tabulate import tabulate
    print("\nTop " + str(TOP) + " recommended books for user " + str(USER_ID) + ":")

    table_data = []
    for item in recommended_items:
        Isnb = df_books.columns[item]
        title = df_books_name.loc[Isnb]
        Pred = item_scores[item]
        table_data.append([title['Book-Title'], title['Book-Author'], Isnb, Pred])



    table_headers = ["Book Title", "Book Author", "ISBN","Predicted Rating"]
    print(tabulate(table_data, headers=table_headers))



    book_ratings = 0
    while book_ratings == 0:
        book_id = books[np.random.randint(0, len(books))]

        book_title = df_books_name.loc[book_id]['Book-Title']
        book_ratings = df_books[book_id]
        book_similarity = item_similarity[books == book_id]
        user_ratings = df_books.loc[USER_ID]
        predicted_rating = user_ratings.dot(book_similarity.T)
        predicted_rating = predicted_rating / 100
        book_ratings = book_ratings[USER_ID]
    print("Predicted rating for book '" + book_title + "' for user " + str(USER_ID) + ": " + str(predicted_rating[0]))
    print("Actual rating for book '" + book_title + "' for user " + str(USER_ID) + ": " + str(book_ratings))


    #take all the books that the user has rated
    user_books_ratings = df_books.loc[USER_ID]
    user_books_ratings = user_books_ratings[user_books_ratings > 0]

    #make a prediction for each book
    predictions = []
    for book_id in user_books_ratings.index:
        book_similarity = item_similarity[books == book_id]
        user_ratings = df_books.loc[USER_ID]
        predicted_rating = user_ratings.dot(book_similarity.T)
        predicted_rating = predicted_rating / 100
        book_ratings = user_books_ratings[book_id]
        predictions.append([book_id, np.round(predicted_rating[0]), book_ratings])

    print("Predictions for user " + str(USER_ID) + ":\n")

    table_data = []
    for prediction in predictions:
        book_title = df_books_name.loc[prediction[0]]['Book-Title']
        table_data.append([book_title, prediction[1], prediction[2]])
    table_headers = ["Book Title", "Predicted Rating", "Actual Rating"]

    print(tabulate(table_data, headers=table_headers))

    threshold = 1
    precision = precision_treshold(np.array(predictions)[:,2],np.array(predictions)[:,1], threshold)
    print(f'Precision: {precision} for threshold {threshold}')
    
    


def main():
    # Start the resource monitor in a separate process
    monitor_process = Process(target=resource_monitor)
    monitor_process.daemon = True
    monitor_process.start()

    # Run the main script
    try:
        run()
    finally:
        monitor_process.terminate()

if __name__ == "__main__":
    main()
