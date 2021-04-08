## Topic Modelling in Python: World War I Letters 

**The Assignment**

This weeks assignment involved training an LDA model on data selected by onself. The purpose was to extract structured information which could provide insight into the data. This would centre around topic modelling and could look at things such as whether authors would cluster together or how concepts would change over time. 

__The Data__ 

I seleted a dataset of 50 handwritten letters between British or French soldiers and their loved ones during the First World War. I was interested to see what topics would arise throughout these letters, written during one of the most distressing times of their lives. This data is saved in the directory above as WW1_Letters.csv and can also be found in the data folder at the root of the directory. 

20 of the letters have been converted from French to English with the use of Google translate.

__Research Question__ 

What are soldiers and their loved ones talking about in their correspondence through letters during WW1? 
Time has been a little pressed this week but in follow up analysis I hope to dig into this collection further and look at the following points of interest: 
- Do French and soldiers differ in the topics they discuss? 
- Does the sentiment of these letters change over time (as the war goes on?) 
- Finally, I hope to add a section of diary entries and look at whether soldiers talk to themselves differently than in their letters. 

__The Methods__

Topic modelling has been conducted in Python and can be run directly from the terminal. The project used a probabilistic, unsupervised learning method of Latent Dirichlet Allocation to allocate the words in each of the letters into 15 specific topics. 15 topics were used as this was found to be the optimal number of topics in post-hoc tests run on the LDA model. 

Bi-grams and Tri-grams have been used with a minimum count of 3 and threshold of 100 to bind together phrases found commonly together such as city names or phrases. These have been used to create a gensim dictionary and corpus have then been created to use within the lda model.  

The LDA model runs with 100 iterations across 15 topics in turn. The results are formulated into a .txt file which contains the perplexity and coherence score and a html file has been created which allows one to explore through the topics. 


__Topics found __ 

14 topics were created which can be summarised as follows: 
  1. Love and affection 
  2. War and uncertainty 
  3. Greetings and correspondence 
  4. Time uncertainty 
  5. Knowing and leaving 
  6. Shelter and resources 
  7. Rest and trenches 
  8. Army regiments  
  9. Rest and relief 
  10. Suffering 
  11. Abushes and hardships 
  12. Sending and receiving letters 
  13. Life and survival 
  14. Coming and going 

<img width="1113" alt="Screen Shot 2021-04-08 at 9 52 45 am" src="https://user-images.githubusercontent.com/52678852/113988898-36418000-9850-11eb-9fcb-bf4e1beb913e.png">

# Steps to take 

## 1. Clone the repository 
The easiest way to access the files is to clone the repository from your commend line and move into Assignment05 as outlined below 

```bash
#clone repository into cds-language-orlz
git clone https://github.com/Orlz/CDS_LanguageAnalytics.git cds-language-orlz

#Move into the correct file 
cd cds-language-orlz/Assignments/Assignment05
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

## 3. Run the Script 
This can be done from the terminal by navigating to the correct directory as described above and typing: 
    
```bash
python3 WW1_Letters.py
```
There are no argparse arguments for this script yet. 

