import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import random, os
import findspark
findspark.init()
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from multiprocessing import Process
from tqdm import tqdm
from time import sleep
import time
import psutil

def resource_monitor():
    with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:
        while True:
            rambar.n = psutil.virtual_memory().percent
            cpubar.n = psutil.cpu_percent()
            rambar.refresh()
            cpubar.refresh()
            sleep(0.5)

def run():
    # Create Spark session
    spark = SparkSession.builder.appName('rec-sys')\
        .config("spark.python.profile.memory", "true")\
        .config("spark.driver.memory", "15g")\
        .config("spark.executor.memory", "15g")\
        .getOrCreate()

    # Set log level to ERROR
    spark.sparkContext.setLogLevel("ERROR")
    
    # Read data
    ratings_df = spark.read.csv('project/BX-Book-Ratings.csv', sep=';', inferSchema=True, header=True)
    books_df = spark.read.csv('project/BX-Books.csv', sep=';', inferSchema=True, header=True)\
                       .drop('Image-URL-S', 'Image-URL-M', 'Image-URL-L')
    
    # Data preprocessing
    stringToInt = StringIndexer(inputCol='ISBN', outputCol='ISBN_int').fit(ratings_df)
    ratings_dfs = stringToInt.transform(ratings_df)
    ratings_df = ratings_dfs.filter(ratings_dfs['Book-Rating'] != 0)
    
    # Split data
    train_df, test_df = ratings_df.randomSplit([0.8, 0.2])
    
    # Build and train ALS model
    als = ALS(userCol='User-ID', itemCol='ISBN_int', ratingCol='Book-Rating',
              nonnegative=True, coldStartStrategy="drop")
    param_grid = ParamGridBuilder()\
        .addGrid(als.rank, [1])\
        .addGrid(als.maxIter, [20])\
        .addGrid(als.regParam, [0.35])\
        .build()
    evaluator = RegressionEvaluator(metricName='rmse', predictionCol='prediction', labelCol='Book-Rating')
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
    
    # Fit model
    start = time.time()
    print('Training model...')
    model = cv.fit(train_df)
    print('Training time: ', time.time() - start)
    rec_model = model.bestModel
    
    # Make predictions
    predicted_ratings = rec_model.transform(test_df)
    rmse = evaluator.evaluate(predicted_ratings)
    print('Root Mean Square Error: ', rmse)
    
    # Evaluation functions
    def precision(predicted, actual):
        v = [1 if pred == actual[i] else 0 for i, pred in enumerate(predicted)]
        return sum(v) / len(v)
    
    def precision_threshold(predicted, actual, threshold):
        v = [1 if actual[i] - threshold <= pred <= actual[i] + threshold else 0 for i, pred in enumerate(predicted)]
        return sum(v) / len(v)
    
    prediction = np.round(predicted_ratings.select('prediction').toPandas().values.reshape(-1))
    Gt = predicted_ratings.select('Book-Rating').toPandas().values.reshape(-1)
    threshold = 1
    print(f'Precision: {precision(prediction, Gt)}')
    print(f'Precision threshold: {precision_threshold(prediction, Gt, threshold)} with threshold {threshold}')
    
    return spark, ratings_dfs, books_df, rec_model

def recommend_for_user(user_id, n, ratings_dfs, books_df, rec_model):
    ratings_user = ratings_dfs.filter(col('User-ID') == user_id)
    pred_ratings_user = rec_model.transform(ratings_user.filter(col('Book-Rating') == 0))
    recs_user = books_df.join(pred_ratings_user.select(['ISBN', 'prediction']), on='ISBN')
    recs_user = recs_user.sort('prediction', ascending=False).drop('prediction').limit(n)
    return recs_user, pred_ratings_user

def main():
    # Start the resource monitor in a separate process
    monitor_process = Process(target=resource_monitor)
    monitor_process.start()

    # Run the main script
    try:
        spark, ratings_dfs, books_df, rec_model = run()
        recs_user, pred_ratings_user = recommend_for_user(35859, 20, ratings_dfs, books_df, rec_model)
        recs_user.show()
        pred_ratings_user.sort('prediction', ascending=False).show(20)
    finally:
        # Stop the Spark session and resource monitor process
        spark.stop()
        monitor_process.terminate()

if __name__ == "__main__":
    main()
