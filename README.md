<!-- ### Setting up:

```bash
pip install -r requirements.txt
``` -->

<!-- ### Preparing dataset:

**add the train, val and test csv files to data/ folder** -->

<!-- ### How to run: -->

#### data folder :
the dataset we train our model on.

#### analysis_data folder :
This folder has COVID-19 related tweets from India, that we perform our aspect based analysis on. 
It has two csv files, that contain predictions of our model along with cleaned tweets
1. panacea_india_data.csv: containing all tweets from January to July 4th of 2020
2. panacea_india_data_filt.csv: contains tweets from March 1 of 2020 to July 4th of 2020 (day number:61 to day number:186)

#### aspects folder:
It has two subfolders:
1. raw: it has the raw ABAE output: (7 aspects for Annoyed, Optimistic and Surprised, with 100 support words and their scores for each of the aspects)
2. filtered: it has hand filtered output, where incoherent aspects have been discarded. The remaining aspects have been named, and a few generic, irrelevant support words have been discarded as well. This has been carried out for Annoyed and Optimistic. The final data is saved in json format 

#### word2vec.py
We use this python file to get word2vec models which are required by [ABAE](https://www.comp.nus.edu.sg/~leews/publications/acl17.pdf) to generate the aspects.

#### normalize_tweets.py :
We use the function normalize tweets, for normalizing the tweets, before using word2vec.py and also to generate the "clean_text" field of panacea_india_data_filt.csv

#### For scraping/hydrating (scrape.py) :
```bash
python scrape.py -s True -q [queries] -l [limit on tweets]  
python scrape.py -H True -f [files containing tweets ids]

Note : The -H stands for hydration, and -s for scraping. Restrictions related to coordinates, time intervals, can be modified directly in the script.
```

<!-- #### For training :
```bash
python train.py --exp_name (value) --encoder (value) --data_dir (value) --save_dir (value) --lr (value) --batch_size (value) --save_policy (value) --activation (value) --optim (value) --wd (value) --epochs (value) --seed (value) --use_gpu(to use gpu) --use_empath(to use empath) --l2(to use l2 reg.) --use_scheduler(to use sched) --use_dropout(to use dropout)
``` -->

<!-- #### For generating predictions :
```bash
python generate_predictions.py --gpu_id (gpu_id) --model_name (BERT/ROBERTA) --model_path (path to saved model) --output_path (path to save dir) --data (path to dir containing hydrated csv) --use_empath (y/n) --activation (tanh/bce)
``` -->
#### For plotting graphs (plot_graphs.ipynb) :
It's used to plot the counts of aspects (filtered/annoyed.json and filtered/optimistic.json) for tweets read from panacaea_data_india_filt.csv.
We count the number of occurences of any of the aspect categores for both emotions in chunks of tweets having 4000 tweets in them, and containing the emotion being considered (e.g. for annoyed, each tweet must have annoyed in its predictions). Ocurrence of any of the support words for an aspect of an emotion, contributes 1 to the total count. 
Run all the cells of plot_graphs.ipynb to generate the plots.
<!-- ### Saving scheme:
- final_test=false, stores best validation epoch
- final_test=true, stores test result after training on best validation hp.json -->
