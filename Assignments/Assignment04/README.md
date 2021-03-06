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

    input_file: edges_df.csv

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

## 1. Clone the repository 
The easiest way to access the files is to clone the repository from your commend line and move into Assignment04 as outlined below 

```bash
#clone repository into cds-language-orlz
git clone https://github.com/Orlz/CDS_LanguageAnalytics.git cds-language-orlz

#Move into the correct file 
cd cds-language-orlz/Assignments/Assignment04
```

## 2. Create the virtual environment
You'll need to create a virtual environment which will allow you to run the script. This will require the requirements.txt file above 
To create the virtual environment you'll need to open your terminal and type the following code: 

```bash
bash create_lang_venv.sh
```
And then activate the environment by typing: 
```bash
$ source VE_Networks/bin/activate
```

You may need to install networkx and pygraphviz if they do not copy across: 
```bash
pip install networkx pygraphviz
```

## 3. Run the Script 
This can be done in one of two ways: 

(Optional) Specify the parameters and run script    
```bash
python3 network.py -i ../data/YOUR_FILE_NAME.csv -m 100
```

~ OR ~ 

Run the script using the defaults 
```bash
python3 network.py 
```

Providing there are no errors, you should get an output message informing that the graph is saved in "viz" and the dataframe is in "output" 

    
