"""
This file contains utility functions for data_preprocessing
"""
import re

import gensim
import nltk
import numpy as np
from nltk.corpus import stopwords


# %%


def str_to_wordlist(tweettext, tweet, remove_stopwords=False):
    str_text = re.sub("[^a-zA-Z]", " ", tweettext)
    words = nltk.word_tokenize(str_text.lower())
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    return (words)


def loadW2vModel():
    # LOAD PRETRAINED MODEL
    global model_GN
    print("Loading w2v model")
    model_GN = gensim.models.KeyedVectors.load_word2vec_format(
        '/home/ifajcik/Work/NLP/Embeddings/google_pretrained_w2v/GoogleNews-vectors-negative300.bin', binary=True)
    print("Done!")


def sumw2v(tweet, avg=True):
    global model_GN
    model = model_GN
    num_features = 300
    temp_rep = np.zeros(num_features)
    wordlist = str_to_wordlist(tweet['text'], tweet, remove_stopwords=False)
    for w in range(len(wordlist)):
        if wordlist[w] in model:
            temp_rep += model[wordlist[w]]
    if avg and len(wordlist) != 0:
        sumw2v = temp_rep / len(wordlist)
    else:
        sumw2v = temp_rep
    return sumw2v


def getW2vCosineSimilarity(words, wordssrc):
    global model_GN
    model = model_GN
    words_in_vocab = []
    for word in words:
        if word in model.wv.vocab:  # change to model.wv.vocab
            words_in_vocab.append(word)
    wordssrc_in_vocab = []
    for word in wordssrc:
        if word in model.wv.vocab:  # change to model.wv.vocab
            wordssrc_in_vocab.append(word)
    if len(words_in_vocab) > 0 and len(wordssrc_in_vocab) > 0:
        # calculates BoW average cosine distance
        return model.n_similarity(words_in_vocab, wordssrc_in_vocab)
    return 0.
