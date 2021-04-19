#!/usr/bin/env python

"""
===========================================================================
Assignment 6: Text classification using Deep Learning (CNN models)  
===========================================================================

This script builds a complex deep learning model by combining deep learning approaches such as word embeddings with CNNs, to create a text classifier. The model is used to classify lines from the popular HBO series "Game of Thrones" into the season which the line belongs to. It uses a kaggle dataset of 23,911 lines taken across the 8 seasons of Game of Thrones. This dataset can be found on kaggle at the following address: https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons.

Combining word embeddings and CNN approaches should enable us to capture local structure in the data which can be generalized across the whole dataset. 

This script takes the following 12 steps: 
    1.   Load and preprocess the data 
    2.   Create the training and validation set 
    3.   Vectorize the data using CountVectorizer()
    4.   Factorize the y-labels 
    5.   Word Embeddings  
    6.   Pad the sentences 
    7:   Define an embedding layer size and create an embedding matrix
    8.   Set the kernal reularization value
    9.   Build the model 
    10.  Compile the model 
    11.  Train and evaluate the model
    12.  Create and save an accuracy plot
    13.  Predictions and classification 


The script has no argparse arguments so it can be run directly from the terminal 
    $ python3 Deep_Learning_Text_Classification.py
    
""" 

"""
=================================
----- Import Depenendencies -----
=================================
"""

# operating system tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy, and gensim
import pandas as pd
import numpy as np
import gensim.downloader

# import classification utilities and functions
import utils.classifier_utils as clf

# Import the machine learning tools from sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

# Import tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2

# Import matplotlib for plots 
import matplotlib.pyplot as plt


"""
=========================
----- Main Function -----
=========================
"""

def main():
    """
    ========================
    Functions through script
    ========================
    """
    
    def plot_history(H, epochs):
    """
    Ultility function for plotting the model history using matplotlib. 
        H:        model history
        epochs:   number of epochs for which the model was trained    
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    def create_embedding_matrix(filepath, word_index, embedding_dim):
    """ 
    A helper function to read in saved GloVe embeddings and create an embedding matrix
    
    filepath: path to GloVe embedding
    word_index: indices from keras Tokenizer
    embedding_dim: dimensions of keras embedding layer
    """
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

    """
    =====================================
    Step 1: Load and preprocess the data 
    =====================================
    """
   
    # Create a pandas dataframe with the Game of Thrones data 
    filepath = "data/Game_of_Thrones_Script.csv"
    df = pd.read_csv(filepath)
    
    # Reduce the dataframe to include only the necessary columns ('Season' and 'Sentence') 
    df = df[["Season", "Sentence"]]
    
    #Get the values from each cell of this dataframe into two lists
    season = df['Season'].values
    sentences = df['Sentence'].values
    
    """
    ==============================================
    Step 2: Create the training and validation set 
    ==============================================
    """
    # We create a test set with the size defined from the command line
    # The random state of 42 makes the train and test sets reproducible 
    X_train, X_test, y_train, y_test = train_test_split(sentences, 
                                                    season, 
                                                    test_size=0.25, 
                                                    random_state=42)
    
    
    """
    ==================================================
    Step 3: Vectorize the data using CountVectorizer() 
    ==================================================
    """
    # Sklearn's count vectorizer creates a numerical vector from the input words
    vectorizer = CountVectorizer()
    
    # We apply this vectorizer to the trainng data to make this into a numerical vector representation 
    X_train_feats = vectorizer.fit_transform(X_train)
    #... then we do it for our test data
    X_test_feats = vectorizer.transform(X_test)
    
    # Finally, we create a list of the feature names. 
    feature_names = vectorizer.get_feature_names()
    
    """
    ==============================
    Step 4: Factorize the y-labels
    ==============================
    """
    # Use panda's factorize function to ensure the y-values are numerical 
    y_train = pd.factorize(y_train)[0]
    y_test = pd.factorize(y_test)[0]
    
    """
    ==============================
    Step 5: Word Embeddings
    ==============================
    """
        """
        Word embeddings tell us more about the word by creating dense representations of the words in a high-dimensional space 
        This encodes more complex linguistic information such as semantic and grammatical meaning to words 
        """
        
    # Initialize a tokenizer with 5000 words (gets the full sentences)
    # Fit this to the x_train data 
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    
    # Then create variables of the tokenized train and test sets 
    X_train_toks = tokenizer.texts_to_sequences(X_train)
    X_test_toks = tokenizer.texts_to_sequences(X_test)
    
    #Correct any possible index issues by adding 1 to the length of the tokenizer 
    vocab_size = len(tokenizer.word_index) + 1  # This helps counter the 0 index in python 

    # Let the user know your text has been tokenized 
    print (f"Your data has been tokenized!") 
    print (f"Your trained data looks like: {X_train[2]} ") 
    print (f"Your tokenized data now looks like: {X_train_toks[2]} ") 
    
    """
    ==========================
    Step 6: Pad the sentences 
    ==========================
    """
        """
        The tokenized output has many differing lengths of the data (as sentence length varies considerably).
        To overcome this, we can pad the sentences to make them of equal length. We do this by standardizing the length 
        of each sentence to a max_length and adding 0s to fill where characters are missing
        
        We'll use a maxlength of 100 (average sentence length is 61 characters) 
        """
    
    # max length for a doc
    maxlen = 100

    # pad training data to maxlen
    X_train_pad = pad_sequences(X_train_toks, 
                            padding='post', # we're adding the sequence post instead of pre
                            maxlen=maxlen)
    
    # pad testing data to maxlen
    X_test_pad = pad_sequences(X_test_toks, 
                           padding='post', 
                           maxlen=maxlen)
    
    
    print("Padding complete. We're about to embed some layers") 
    
    
    """
    =====================================================================
    Step 7: Define an embedding layer size and create an embedding matrix 
    =====================================================================
    """
        """
        The next step is to convert our tokenized numerical representation of the text into a dense, embedded representation. 
        This can be done using tensorflow.keras by defining the number of dimensions (embedding_dim). We'll use 50
        """
    
    # Embedding layer size 
    embedding_dim = 50
    
    #Embedding matrix
    embedding_matrix = create_embedding_matrix('data/glove.6B.50d.txt',
                                           tokenizer.word_index, 
                                           embedding_dim)
    
    """
    ===================================
    Step 8: Set the kernal reularization value   
    ===================================
    """
    
    l2 = L2(0.0001)
     
        
    """
    =======================
    Step 9: Build the model   
    =======================
    """
        """
        This begins with setting a base model
        """
        
    # Create a sequential model 
    model = Sequential()
    
    
        """
        Next, the various layers are added to the model in turn.  
            1. The embedding layer
            2. The convulutional layers  
            3. The GlobalMaxPool layer 
            4. Dense layer of 10 neurons
            5. The prediction layer (of 1 node) 
            
        """
    # 1. Embedding layer (using the embedding_matrix) 
    model.add(Embedding(vocab_size,                  # Vocab size from Tokenizer() (5000) 
                        embedding_dim,               # Embedding input layer size (50) 
                        weights=[embedding_matrix],  # Pretrained embeddings
                        input_length=maxlen,         # Maximum length of the padded sentences (100) 
                        trainable=True))             # The embeddings are trainable
    
    #2. Convulutional layer (using "relu" with a kernal size as defined above in step 8) 
    model.add(Conv1D(128, 5, 
                    activation='relu',
                    kernel_regularizer=l2)) 
    
    #3. Global MaxPool layer (this is used instead of the flatten layer as it can reduce noise and increase signals within data)
    model.add(GlobalMaxPool1D())
    
    #4. Dense layer of 10 neurons (activation = relu) 
    model.add(Dense(10, activation='relu', kernel_regularizer=l2))
    
    #5. Prediction layer (This layer is only 1 node with softmax activation, it approximates logistic regression) 
    model.add(Dense(1, activation='softmax'))
    
    """
    ==========================
    Step 10: Compile the model   
    ==========================
    """
        """
        We're using a categorical loss function because we have 8 categories we are trying to classify between. 
        Optimization algorithm = adam 
        """
    # Compile model 
    model.compile(loss='categorical_crossentropy', 
                  optimizer="adam",
                  metrics=['accuracy'])
    
    # ... and get the summary 
    summary = model.summary()
    
    #print this to the terminal 
    print(summary) 
    
    
    """
    =====================================
    Step 11: Train and evaluate the model    
    =====================================
    """
        """
        Now that the model is built, we want to run (train) it and evaluate how it's performing. We do this in 2 steps: 
            1. Fit the model 
            2. Evaluate the performance 
        """
    
    #Fit the model (we'll run it through 20 epochs) 
    history = model.fit(X_train_pad, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test_pad, y_test),
                    batch_size=10)
    
    #Evaluate the model and print results to the terminal
    #Training data 
    loss, accuracy = model.evaluate(X_train_pad, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    
    #Test data
    loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    
    """
    =========================================
    Step 12: Create and save an accuracy plot     
    =========================================
    """
        """
        We can visualise the performance of the model using an  accuracy plot
        """
    
    #Create plot in a figure called accuracy_plot 
    accuracy_plot = plot_history(history, epochs = 20)
    
    #save this in your output directory 
    plt.savefig(accuracy_plot, "output/DL_accuracy_plot.png")
    
    """
    =========================================
    Step 13: Predictions and classification      
    =========================================
    """
        """
        Finally, we want to test how good our model is with predictions. 
        We'll create predictions and use them to build a classification report. 
        """
        
    #Create predictions 
    predictions = model.predict(X_test_pad, batch_size = 10)
    print(classification_report(y_test, predictions.argmax(axis=1)))
    
    

if __name__=="__main__":
    #execute main function
    main()
