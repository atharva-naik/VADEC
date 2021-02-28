import os
import json
import time
import twarc
import argparse
from twarc import Twarc
import pandas as pd 
import GetOldTweets3 as got 

# NOTE: GetOldTweets3 hasn't been functional for some time, most likely due to changes by Twitter. 
# It used to work at the time this code was written.

''' This code contains a scraper and a hydrater. The scraper uses an external API GetOldTweets3 to get tweets from queries and other 
constraints, however some limited aspects of the tweets can only be accessed. The hydrater uses a twitter library called twarc, that 
takes in tweet ids of tweets to retrieve all details that can be taken from twitter, if the tweet still exists. '''

parser = argparse.ArgumentParser()
parser.add_argument("-s","--scrape", type=bool, help="run scraper")
parser.add_argument("-H","--hydrate", type=bool, help="hydrate tweet ids")
parser.add_argument("-f","--fname", type=str, help="address of text file of tweet ids")
parser.add_argument("-q","--queries", nargs='+', type=str, help="queries to be searched")
parser.add_argument("-l","--limit", type=int, help="max number of tweets to be extracted for each query")
args = parser.parse_args()

if args.hydrate:
    if not(args.fname):
        raise(FileNotFoundError("please enter a file address for tweet ids"))

#### NOTE: scrape data with getoldtweets3 ####
if args.scrape:
    # max number of tweets to be scraped for each query
    LIM = 10000
    # override with command line argument
    if args.limit:
        LIM = args.limit
    # specific coordinates to be scraped
    coords = [(19.75, 75.71),(22.97, 78.66)]
    # query terms to be searched for
    queries = ['coronavirus','covid','outbreak','sars-cov-2','koronavirus','corona','wuhancoronavirus','lockdown','lock down','wuhanlockdown',
    'kungflu','covid-19','covid19','coronials','coronapocalypse','panicbuy','panicbuying','panic buy','panicbuy','panic shop','panicshopping',
    'panicshop','coronakindness','stayhomechallenge','DontBeASpreader','sheltering in place','shelteringinplace','chinesevirus','chinese virus',
    'quarantinelife','staysafestayhome','stay safe stay home','flattenthecurve','flatten the curve','china virus','chinavirus','quarentinelife','covidiot',
    'epitwitter','saferathome','SocialDistancingNow','Social Distancing','SocialDistancing']
    # over ridden by command line arguments
    if args.queries:
        queries = args.queries
        print(type(queries))
    # "from" and "to" dates for search NOTE ("yyyy-mm-dd" format)
    time_intervals = [("2020-01-01","2020-03-05"),("2020-04-06","2020-06-30")]

    # print params during run
    print("max number of tweets per query {}".format(LIM))
    print("query terms {}".format(queries))
    print("time intervals {}".format(time_intervals))
    print("locations specified {}".format(coords))

    for _from, to in time_intervals:
        for query in queries:
            for x,y in coords:
                tweets_list = []
                # search for tweets with specified conditions
                print("Scraping for {} for dates {} to {}, near coordinates ({},{}) ...".format(query, _from, to, x, y))
                tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query).setMaxTweets(LIM).setSince(_from).setUntil(to).setNear(str(x)+', '+str(y)).setWithin('300km')
                results = got.manager.TweetManager.getTweets(tweetCriteria)
                # convert to list of lists
                for tweet in results:
                    tweets_list.append([tweet.username, tweet.date, tweet.retweets, tweet.favorites, tweet.text, tweet.hashtags, tweet.geo, tweet.id, tweet.permalink])

                # store results as csv
                df = pd.DataFrame(tweets_list, columns=['username', 'date', 'retweets', 'favorites', 'text', 'hashtags', 'geo', 'id', 'permalink']) 
                # drop duplicated text (retweets)
                df = df.drop_duplicates(subset='text')
                # sort by username
                df.sort_values("username",inplace=True)
                try:
                    os.mkdir('data')
                except FileExistsError:
                    pass
                name = "./data/" + query + ".lim="+ str(LIM) + "_since=" + _from + "_until=" + to + "_near=(" + str(x) +','+ str(y) +").csv"
                
                # print result stats for each search criterion 
                print('{} unique tweets scraped from {}'.format(len(df), query))
                print("saving to {}".format(len(df), name))
                df.to_csv(name)

#### scrape data with getoldtweets3 ####

#### NOTE: hydrate tweet ids with twarc ####
if args.hydrate:
    # create the twarc object
    twarc = Twarc()

    # read ids from a text file of ids, with column name "ids"
    
    with open(args.fname,"r") as f:
        ids = f.read().strip().split('\n')
        f.close()
    ids = [int(id_) for id_ in ids]
    tweets = []

    # check input integrity
    print("{} ids were read".format(len(ids)))
    print("hydrating from {} to {} ...".format(ids[0], ids[-1]))

    # create generator of hydrated tweets from list of ids
    results = twarc.hydrate(ids)
    for tweet in results:
        tweet['username'] = tweet['user']['name']
        tweets.append(tweet)

    # convert to dataframe
    df = pd.DataFrame(tweets)
    print("total {} tweets".format(len(tweets)))

    # write the ids of tweets not found to a text file
    if not(df.empty): 
        df = df.drop_duplicates(subset='full_text')
        df.sort_values('username', inplace=True)
        print("results saved to hydrated_tweets.csv")
        df.to_csv('hydrated_tweets.csv')
        not_found = list(set([int(id_) for id_ in  ids]).difference(set(list(df['id']))))
    else: not_found = ids
    print("{} tweets were not found".format(len(not_found)))
    print("their ids are saved to not_found.txt ...")
    with open('not_found.txt',"w") as f:
        for id_ in not_found:
            f.write(str(id_)+'\n')

#### hydrate tweet ids with twarc ####