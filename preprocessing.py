import random
import re
import json
import pickle as pkl

from random import shuffle, sample
import numpy as np
import pandas as pd

from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

#loading the json file as corpus
corpus = json.load(open('/Users/rnjsx/OneDrive/Documents/parallel_sentences_en-nl-it.json', 'r'))

#preprocessing : lowering the sentence cases, replacing with "D" and removing all characters thats not letters

def preprocess(corpus):
    for ID in corpus:
        for language in corpus[ID]:
            # Initializing an index to keep track of the position
            index = 0
            for sentence in corpus[ID][language]:
                sentence = sentence.lower()
                sentence = re.sub(r'\d', 'D', sentence)
                sentence = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ\s]', '', sentence)

                # Updating the sentence in the corpus at the current index
                corpus[ID][language][index] = sentence
                index += 1  # Incrementing the index for the next sentence

    return corpus


# Split into training and test sets

train_set_keys, test_set_keys = train_test_split(sorted(corpus.keys()), test_size=0.2, random_state=4242)

# Use dictionary comprehension to build train and test dictionaries from the keys
train_set = {key: corpus[key] for key in train_set_keys}
test_set = {key: corpus[key] for key in test_set_keys}
