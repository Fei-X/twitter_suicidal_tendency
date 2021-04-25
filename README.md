This is an applied machine learning project for detecting suicidality in twitter.

### Coding
data_collection.ipynb does the initial data crawling from twitter.com using twint.  
preprocessing.ipynb transforms the data into machine-learning-friendly mode.  
LDA_for_tweet.ipynb performs grid search on LDA.  
real_time_model.ipynb runs real time tweets classification models.  
monitoring_model.ipynb runs classfication models on all features.  
demo.ipynb runs a demonstration for our real-time models.  
evaluate.py has some useful functions in evaluating models.  

### Model folder
LSTM_models and naive_bayes_model stores the models for demo.ipynb.  

### twint
The folder stores the crawling tool to collect all the data.  
