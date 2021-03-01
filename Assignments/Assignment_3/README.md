# Assignment03 - Sentiment Analysis

In this assignment we are analysing news headlines from Australian newschannel ABC. The data to be used can be downloaded from
https://www.kaggle.com/therohk/million-headlines (Please note, the data is 65MB so pretty large!) 

## Running the script 
This script is run from the command line. There will be a few set-up steps for you to get it running correctly. I would recommend using the "subset" option
which can be added directly into the command line as this will be quicker for you. 
The script = Assignment03.py 

## Parameters 
The script will require you to input where you have the script stored and what subset size you'd like to use (index) 
path = str <file-path>    #This is where you define your directory in the command line using -d [insert path]
subset = int <index>      #This is where you define whether you want to use a subset of the data by typing -s [sample number]

How to use them: 
   Assignment03.py -d <file-path> -s <index>
  
 Example: 
     (where you want to run them) (script name.py) -d (insert path to use) -s (insert index number)
    
    python3 Assignment03.py -d ../data/abcnews-date-text.csv -s 100000

# Steps to take 

1. Create the virtual environment
You'll need to create a virtual environment which will allow you to run the script. This will require the requirements.txt file above 
To create the virtual environment you'll need to open your terminal and type the following code: 
```bash
bash create_lang_venv.sh
```
And then activate the environment by typing: 
```bash
$ source sentiment_environment/bin/activate
```
2. Navigate to the directory where you have the files stored
You may have a different directory but an example of this would be
```bash
cd cds-language/assignments
```
3. Specify the parameters and run the directory 
This will involve inserting your path directory after -d  
And also inserting your requested subset size after -s    (NB This will just be a number, see example above) 

I would recommend running the script on only a subset of the data. This can be done with the following command (this will take the first 100,000 headlines) 
```bash
python3 Assignment03.py -d ../data/abcnews-date-text.csv -s 100000
```
4. Check the data 
Providing there are no errors, you should get an output message informing where the output has been saved 

