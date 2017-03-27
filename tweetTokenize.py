#!/usr/bin/python2

import sys
from nltk.tokenize import TweetTokenizer
import re

def tokenize(tknzr, sentence, to_lower=True):
    """Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentence: a string to be tokenized
        - to_lower: lowercasing or not
    """
    sentence = sentence.strip()
    sentence = ' '.join([format_token(x) for x in tknzr.tokenize(sentence)])
    if to_lower:
        sentence = sentence.lower()
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',sentence) #replace urls by <url>
    sentence = re.sub('(\@[^\s]+)','<user>',sentence) #replace @user268 by <user>
    filter(lambda word: ' ' not in word, sentence)
    return sentence

def format_token(token):
    """"""
    if token == '-LRB-':
        token = '('
    elif token == '-RRB-':
        token = ')'
    elif token == '-RSB-':
        token = ']'
    elif token == '-LSB-':
        token = '['
    elif token == '-LCB-':
        token = '{'
    elif token == '-RCB-':
        token = '}'
    return token

def tokenize_sentences(tknzr, sentences, to_lower=True):
    """Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentences: a list of sentences
        - to_lower: lowercasing or not

    """
    
    return [tokenize(tknzr, s, to_lower) for s in sentences]



if __name__ == "__main__":
    
    fileName = sys.argv[1]

    tknzr = TweetTokenizer()

    sentences = []
    with open(fileName, 'r') as fileinput:
       for line in fileinput:
           sentences.append(line)

           
    tknzr = TweetTokenizer()
    tokenized_sentences_NLTK_tweets = tokenize_sentences(tknzr, sentences)

    for sentence in tokenized_sentences_NLTK_tweets:
        print (sentence)