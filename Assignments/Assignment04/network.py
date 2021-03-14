"""
This script contains functions which will build networks based on entities appearing together in the same documents. It calculates measures of i) degree, 2) betweenness, and 3) eigenvectors to give an output of network visualisations. 

As the script is written to be executable from the commande line, it can perform network analysis on any input csv of weighted edgelist data, so long as the csv file has a "nodeA" and "nodeB" column. 

You will notice the argparse arguments are helping to define what needs to be input to execute the script from the command line. These are outlined below: 

Parameters (optional)
    input_file: string variable <file-path>
    minimum_edgeweight: integer variable  <minimum_edgeweight>

Usage:
    network.py -i <file-path> -m <minimum_edgeweight>

Example command line:
    $ python3 network.py -i ../data/fake_or_real_news.csv -m 1000
    
"""

### Load in your dependency packages ###

# System tools
import os
import argparse

# Data analysis
import pandas as pd
from itertools import combinations 
from tqdm import tqdm

# drawing
import networkx as nx
import matplotlib.pyplot as plt

### Define the main function ### 
"""
This function tells the command line to execute all the code held within when called from the commandline. 
"""
 

def main():
    
    #Construct the argparse arguments with defaults
    #Defaults: input file = "fake_or_real_news.csv",  minimum_edgeweight = 500
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_file", required = False, help = "Path to the input file", default ="../edges_df.csv")
    ap.add_argument("-m", "--minimum_edgeweight", required=False, help="The minimum edge weight of interest", default=500, type=int)
    #We'll then parse the arguments 
    args = vars(ap.parse_args())

    
    #Define the input parameters 
    input_file = args["input_file"]
    minimum_edgeweight = args["minimum_edgeweight"]
    
    #Conduct the network analysis by calling the "network_analysis" class below 
    network_analysis(input_file, minimum_edgeweight)
    

###Building a class structure 
"""
This script has instigated a class structure as part of the bonus challenge. 
We define a class of network_analysis which is run as part of the main() function and conducts all the functions defined within. This is a way of bundling up the code for a more modular approach. 

We have a number of functions bundled into this class: 
1. __init__  ~ This sets up the class with the necessary inputs. It also prints messages to let user know whats happening.
2. output_directory ~ This creates the path for the output to be stored 
3. nx_graph  ~ This creates the network graph using networkx and pygraphviz 
4. centrality_measures  ~ This creates a data frame showing the degree, betweenness, and eigenvector centrality for each node.
"""

class network_analysis:
    
    def __init__(self, input_file, minimum_edgeweight):
        
        #Print a message in the terminal to let the user know that the script is starting to run 
        print("\nHello, I'm getting your network analysis setup. It won't take too long!")

        # load in the csv file
        edgelist = pd.read_csv(input_file)
        
        #Create directories for the output to be saved using self 
        self.output_directory("viz")
        self.output_directory("output") 
        
        # Create a network graph using networkx (nx) and save it using self  
        nx_graph = self.nx_graph(edgelist, minimum_edgeweight)
        
        # Create your centrality measures (betweenness & eigenvectors) and save using self
        centrality_measures = self.centrality_measures(nx_graph)
        
        print("Complete! You can find your graph in viz and your centrality measures dataframe in output.\n")
        
        
    
    def output_directory(self, output_directory_name): 
        """
        Here we create the output directory to store the graph (viz) and dataframe (output) in.
        It first checks the output path to ensure it doesn't already exist 
        Input: output_directory_name
        """
        if not os.path.exists(output_directory_name):
            os.mkdir(output_directory_name)
            
            
    def nx_graph(self, edges_df, minimum_edgeweight):
        """
        This function creates our networkx graph and saves it into "viz" 
        Input: Our csv list of weighted edges with columns "nodeA", "nodeB" and "weight (edges_df) 
        Output: A network graph saved in the viz directory 
        """
        # We take our list of weighted edges(edges_df) and filter using the minimum_edgeweight parameter
        filtered = edges_df[edges_df["weight"] > minimum_edgeweight]
        
        # Then we create a graph object using the nx package
        nx_graph = nx.from_pandas_edgelist(filtered, "nodeA", "nodeB", ["weight"])
        
        #We plot it using pygraphviz 
        pos = nx.nx_agraph.graphviz_layout(nx_graph, prog = "neato")
        nx.draw(nx_graph, pos, with_labels=True, node_size=20, font_size=9)
        
        # Finally, we save this graph into viz 
        plt.savefig("viz/network.png", dpi=300, bbox_inches="tight")
        
        #We return this nx_graph so that we can use it to calculate centrality measures 
        return nx_graph
    
    
    def centrality_measures(self, nx_graph):
        """
        Centrality measures help us to understand more about the relationships between words. 
        This function calculates the 1) degreem 2) betweenness and 3) eigenvector values for each edge 
        This information is saved as a dataframe in the output directory using pandas
        Input: network graph (nx_graph) 
        Output: A csv dataframe 
        """
        
        # Calculate the degree, betweenness, and eigenvector of each edge in the nx_graph 
        degree_value = nx.degree_centrality(nx_graph)
        betweenness_value = nx.betweenness_centrality(nx_graph)
        eigenvector_value = nx.eigenvector_centrality(nx_graph)
        
        # Saving these values into a dataframe (centrality_measures_df) 
        centrality_measures_df = pd.DataFrame({
            'degree':pd.Series(degree_value),
            'betweenness':pd.Series(betweenness_value),
            'eigenvector':pd.Series(eigenvector_value)  
        }).sort_values(['degree', 'betweenness', 'eigenvector'], ascending=False)
        
        # saving the csv file
        centrality_measures_df.to_csv("output/centrality_measures_df.csv") 
    
    
    
    
if __name__ == "__main__":
    main()    
