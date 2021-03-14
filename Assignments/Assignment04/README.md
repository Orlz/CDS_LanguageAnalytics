## Network Analysis (Assignment 04) 

This assignment builds on commandline scripting and network analysis using networkx. 

The script network.py contains functions which will build networks based on entities appearing together in the same documents. It calculates measures of 1) degree, 2) betweenness, and 3) eigenvectors to give an output of network visualisations. 

As the script is written to be executable from the commande line, it can perform network analysis on any input csv of weighted edgelist data, so long as the csv file has a "nodeA" and "nodeB" column.
A file of this nature is provided above called "edges_df.csv". Please feel free to use this as your input_file

You will notice the argparse arguments are helping to define what needs to be input to execute the script from the command line. These are outlined below: 


Parameters (optional)
    input_file: string variable <file-path>
    minimum_edgeweight: integer variable  <minimum_edgeweight>


Usage:
    network.py -i <file-path> -m <minimum_edgeweight>


Example command line:
    $ python3 network.py -i ../data/fake_or_real_news.csv -m 1000
 
 
Default parameters have been set should you not wish to use your own. These are: 
input_file: fake_or_real_news.csv
minimum_edgeweight: 500

## Running the script 
This script is run from the command line. There will be a few set-up steps for you to get it running correctly.  
The script = network.py 


## Parameters 
You have the option to add your own parameters if you would like. These are optional:

-i    (input_file) a csv file containing the columns nodeA, nodeB and weight.
-m    (minimum_edgeweight) a minimum edgeweight limit which filters all edges below this (integer) 

How to use them: 
   network.py -i <input_file> -m <minimum_edgeweight>
  
 Example: 
     (where you want to run them) (script name.py) -i (insert path to file) -m (insert minimum edgeweight number)
    
    python3 network.py -i ../data/edges_df.csv -m 500


# Steps to take 

1. Navigate to the directory where you have the files stored
You may have a different directory but an example of this would be
```bash
cd cds-language/assignments
```

2. Create the virtual environment
You'll need to create a virtual environment which will allow you to run the script. This will require the requirements.txt file above 
To create the virtual environment you'll need to open your terminal and type the following code: 
```bash
bash create_lang_venv.sh
```
And then activate the environment by typing: 
```bash
$ source sentiment_environment/bin/activate
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




    
