import glob
from multiprocessing import Pool
import sys
from nltk import TweetTokenizer
import os
import re
import codecs

def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',tweet)
    tweet = re.sub('(\@[^\s]+)','<user>',tweet)
    try:
        tweet = tweet.decode('unicode_escape').encode('ascii','ignore')
    except:
        pass
    return tweet

def tokenize_tweets(filename, dest_folder):
    basename = os.path.basename(filename)
    dest = os.path.join(dest_folder, basename + '.tok')
    print("processing %s" % basename)
    tknzr = TweetTokenizer()
    with codecs.open(dest, 'w', "utf-8") as out_fs:
        with open(filename, 'r', encoding="utf-8") as in_fs:
            for line in in_fs:
                try:
                    language, id, timestamp, username, tweet = line.strip().split('\t')
                except:
                    print("could not parse line.")
                    continue
                if language != 'en':
                    continue
                tweet = tknzr.tokenize(tweet)
                if not 6 < len(tweet) < 110:
                    continue
                tweet = preprocess_tweet(' '.join(tweet))
                out_fs.write(id+'\t'+timestamp+'\t'+username+'\t'+tweet+'\n')

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 tweetTokenize.py <tweets_folder> <dest_folder> <num_process>")
        sys.exit(-1)
    tweets_folder = sys.argv[1]
    dest_folder = sys.argv[2]
    num_process = int(sys.argv[3])
    tweets_filenames = glob.glob(os.path.join(tweets_folder, '*'))
    tweets_filenames = [(f, dest_folder) for f in tweets_filenames]
    if num_process == 1:
        for f, dest_folder in tweets_filenames:
            tokenize_tweets(f, dest_folder)
    else:
        pool = Pool(num_process)
        pool.starmap(tokenize_tweets, tweets_filenames)

if __name__ == "__main__":
    main()
