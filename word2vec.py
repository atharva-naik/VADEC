import os
import tqdm
import gensim
import argparse
import pandas as pd
from gensim.models import Word2Vec
from normalize_tweets import normalizeTweet

parser = argparse.ArgumentParser(description="Script to create Word2Vec embeddings from tweet texts")
parser.add_argument("-s", "--size", default=200, help="size/dimension of generated embeddings")
parser.add_argument("-w", "--windows", default=5, help="size of context window used for predicting words")
parser.add_argument("-W", "--workers", default=4, help="number of worker threads used")
parser.add_argument("-c", "--min_count", default=1, help="ignores words that have less number of occurences than this")
parser.add_argument("-d", "--dir", default=None)
args = parser.parse_args()

DIR = args.dir
if DIR:
    files = [os.path.join(DIR, file) for file in os.listdir(DIR)]
elif args.file != None:
    files = [args.file]

for file in files:
    print("file being processed = {}".format(file))
    f = open(file)
    sentences = []
    
    for i, line in enumerate(tqdm.tqdm(f)):
        line = normalizeTweet(line)
        # to make sure that non empty sentences are used (may result in erros otherwise)
        if len(line.split()) > 0:
            sentences.append(line.split())
    f.close()
    # train word2vec model
    model = Word2Vec(sentences, size=args.size, window=args.windows, min_count=args.min_count, workers=args.workers)
    # save the binary word2vec model 
    model.save(f'w2v_{file.split("/")[-1]}_{args.size}.bin')