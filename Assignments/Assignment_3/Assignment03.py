#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis 

# In this assignment we are analysing news headlines from Australian newschannel ABC. These are taken from the kaggle source: https://www.kaggle.com/therohk/million-headlines
# 
# Tasks include: 
# 
# Calculate the sentiment score for every headline in the data.
# 
# Create and save a plot of sentiment over time with a 1-week rolling average
# 
# Create and save a plot of sentiment over time with a 1-month rolling average
# 
# Make sure that you have clear values on the x-axis and that you include the following: a plot title; labels for the x and y axes; and a legend for the plot
# 
# Write a short summary (no more than a paragraph) describing what the two plots show. You should mention the following points: 1) What (if any) are the general trends? 2) What (if any) inferences might you draw from them?
# 
# 

# ___Important Note___
# 
# This script is run from the command line using argparse arguments. Instructions for this can be found in the Readme.md
# 
# Parameters: 
# 
# path = str <file-path> #This is where you define your directory in the command line using -d [insert path]
#     
# subset = int <index> #This is where you define whether you want to use a subset of the data by typing -s [sample number]
#     
# __How to use them__
# 
#    Assignment03.py -d <file-path> -s <index>
#     
# __Example__
# 
#     (where you want to run them) (script name.py) -d (insert path to use) -s (insert index number)
#     
#     python3 Assignment03.py -d ../data/abcnews-date-text.csv -s 100000

# In[1]:


#Import the packages to be used 
import os 
import spacy #The main sentiment analysis package to be used 
import pandas as pd #For working with dataframes and reading the data 
from spacytextblob.spacytextblob import SpacyTextBlob
import matplotlib.pyplot as plt 
import argparse


# In[2]:


#load the language model - this defines which dictionary we want to use 
nlp = spacy.load("en_core_web_sm")

type(nlp) #checking we're in the right one 


# 
# __Create function to generate smoothed sentiment plots__
# 
# We need a function that will create a smoothed sentiment score plot using a rolling average across a specified timeframe such as 1 day, 1 week, or 1 month 

# In[3]:


#Our function will be called smoothed_sentiment_plot and it will take: 
#A time_window = the time (in days) the function should roll over
# Sentiment_data = the sentiments pulled from the TextBlob function below 
# defined_time = the time (in days) stated on the plot when it is printed 

def smoothed_sentiment_plot(time_window, sentiment_data ,defined_time ):
    data_smoothed = sentiment_data.sort_index().rolling(time_window).mean()
    #We want to define how our plot should look using matplotlib's plt. functions 
    plt.figure()  #create a figure with... 
    plt.title(f"The sentiment over time using a {defined_time} rolling average") # A title 
    plt.ylabel("Sentiment Score") # And x and y labels 
    plt.xlabel("Date of Headline")
    plt.xticks (rotation=45) #Then tilt the x-labels 45 degrees as they are hard to read without this 
    plt.legend(loc="upper right") #Give the plot a legend in the upper right corner - somewhere it doesn't get in the way
    plt.plot(data_smoothed) #Finally make the plot 
    plt.savefig(os.path.join("sentiment_output", f"{defined_time}_sentiment.png"), bbox_inches='tight') # Save it to "sentiment_output" 
    
    print(f"Figure '{defined_time}_sentiment.png' is saved in current directory") #Give us a message to say it's saved

    


# __Create the main function__

# In[5]:


# main function 
def main():
    
    # tell argparse the arguments you'd like to use - including creating a subset for faster processing 
    parser = argparse.ArgumentParser() 
    parser.add_argument("-d", "--path", required = True, help = "The path to directory of files") #The input option 
    parser.add_argument("-s", "--subset", required = False, help = "The subset option") #The subset option, required is F here
    args = vars(parser.parse_args()) 
    
    # create a data variable with the csv files in the path defined above 
    data = pd.read_csv(args["path"])
    # check whether the subset has been requested - if so, slice the data 
    if args["subset"] is not None:
        slice_index = int(args["subset"])
        data = data[:slice_index]
        
    # create a directory to store the plots (if it doesn't yet exist)
    if not os.path.exists("sentiment_output"):
        os.mkdir("sentiment_output")
        
    # Then call and initialise spacy, TextBlob, and your nlp pipe 
    nlp = spacy.load("en_core_web_sm") #We're using the English small library 
    spacy_text_blob = SpacyTextBlob() #This is taken from the spacy website 
    nlp.add_pipe(spacy_text_blob) 
    
    
    ##Now we're set up, we want to programe the calculation of scores (we'll use batches of 5000)## 
        # message
    print("\nHold on, we're calculating the sentiment scores...")
    
    
    # create an empty list of sentiment scores for every headline, we'll call this sentiment_tracker
    sentiment_tracker = []
    
    # for every headline in data frame (we're looking at docs, not sentences)
    for doc in nlp.pipe(data["headline_text"], batch_size = 5000): 
        # calculate the sentiment of the doc (headline)
        sentiment = doc._.sentiment.polarity
        # append this to sentiment_tracker list
        sentiment_tracker.append(sentiment)
        
    # append the sentiment_tracker list to the dataframe and save as output csv file in sentiment_plots
    data.insert(len(data.columns), "sentiment", sentiment_tracker)
    output_csv_path = os.path.join("sentiment_output", "sentiment_tracker.csv")
    data.to_csv(output_csv_path, index = False)
    
    
        ##Now we can put it together to create rolling mean plots## 
        
    # message
    print("We've calculated the sentiment scores, now we'll generate the plots...")
    
    
    # First, create a sentiment dataframe with date as the index and sentiment scores to calculate means based on date
    df_sentiment = pd.DataFrame(
        {"sentiment": sentiment_scores}, # create a column to hold sentiment scores 
        index = pd.to_datetime(data["publish_date"], format='%Y%m%d', errors='ignore')) # index the date using to_datetime
    
    # apply the smoothing plot function from above, to create and save plots in output
    smoothed_sentiment_plot("7d", "1-week", df_sentiment) # 1-week average
    smoothed_sentiment_plot("30d", "1-month", df_sentiment) # 1 month average
    
    # Print a message to let you know when you're done 
    print("That's you complete - woohoo! The csv file and plots are in output directory.\n ")
    
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()
    


# In[ ]:




