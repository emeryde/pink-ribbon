"""
__file__
    config_param.py
__description__
    This file provides global parameter configurations for the project.
__author__
    Dan Emery < emeryde@appstate.edu >
"""

import os


############
## Config ##
############
class ParamConfig:
    def __init__(self,
                 feat_folder,
                 basic_tfidf_ngram_range=(1, 3),
                 basic_tfidf_vocabulary_type="common",
                 cooccurrence_tfidf_ngram_range=(1, 1),
                 cooccurrence_word_exclude_stopword=False,
                 stemmer_type="snowball"):

        self.n_classes = 4

        ## CV params
        self.n_runs = 3
        self.n_folds = 3

        ## path
        self.data_folder = "input"
        self.feat_folder = 'output'
        self.original_data_path = "%s/pink_ribbon_raw.csv" % self.data_folder
        self.processed_data_path = "%s/data_processed.csv." % self.data_folder
        self.sentiments_with_baseline = "%s/sentiments_with_baseline.csv" % self.data_folder
        self.sentiments_final = "%s/sentiments_final.csv" % self.feat_folder

        ## nlp related
        self.basic_tfidf_ngram_range = basic_tfidf_ngram_range
        self.basic_tfidf_vocabulary_type = basic_tfidf_vocabulary_type
        self.cooccurrence_tfidf_ngram_range = cooccurrence_tfidf_ngram_range
        self.cooccurrence_word_exclude_stopword = cooccurrence_word_exclude_stopword
        self.stemmer_type = stemmer_type



## initialize a param config
config = ParamConfig(feat_folder="output",
                     stemmer_type="snowball",
                     cooccurrence_word_exclude_stopword=False)