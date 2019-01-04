from __future__ import print_function
import time

import numpy as np
from six.moves import xrange

from .formatted_logger import formatted_logger
from gensim.corpora import Dictionary

logger = formatted_logger('JointCorpus')

class JointCorpus(object):
    """
    Attributes
    source_corpus_file: str
        file of source corpus
    target_corpus_file: str
        file of target corpus
    ----------

    """
    def __init__(self, source_corpus_file, target_corpus_file, **kwargs):
        self.source_corpus = list()
        self.target_corpus = list()

        self.source_corpus, self.source_dict = self.load_corpus(source_corpus_file)
        self.target_corpus, self.target_dict = self.load_corpus(target_corpus_file)

    def load_corpus(self, corpus_file):
        f = open(corpus_file, "r")
        line_list = f.readlines()
        f.close()

        corpus = list()
        for idx, text in enumerate(line_list):
            # readlines or readline function 都會殘留換行符號\n，故需要替代掉
            temp = text.lower().replace("\n", " ")
            corpus.append(list(filter(None, temp.split(" "))))
        return corpus, Dictionary(corpus)

    def update_doctionary(self, transaltion_pair_file):
        # first initial concept to dictionary
        f = open(transaltion_pair_file)

        updated_source_dict = {}
        updated_target_dict = {}

        # rebuild vocab dict for query JointLDA's topic word distribution
        reconcatenate_dict = []

        for line in f.readlines():
            line = line.rstrip("\n").split(",")
            #######
            # Notice the order here!
            #######
            source_word = line[0].lower()
            target_word = line[1]
            if target_word in self.target_dict.token2id.keys() and source_word in self.source_dict.token2id.keys():
                if target_word not in updated_target_dict.keys() and source_word not in updated_source_dict.keys():
                    updated_target_dict[target_word] = len(updated_target_dict)
                    updated_source_dict[source_word] = len(updated_source_dict)
                    reconcatenate_dict.append((source_word, target_word))
        f.close()

        assert len(updated_target_dict) == len(updated_source_dict) == len(reconcatenate_dict)
        self.n_concept = len(updated_target_dict)

        # then loop corpus to expand dictionary
        for doc in self.source_corpus:
            for word in doc:
                if word not in updated_source_dict:
                    updated_source_dict[word] = len(updated_source_dict)
                    reconcatenate_dict.append(word)

        for doc in self.target_corpus:
            for word in doc:
                if word not in updated_target_dict:
                    updated_target_dict[word] = len(updated_target_dict)
                    reconcatenate_dict.append(word)

        n_s_vocab = len(updated_source_dict)
        n_t_vocab = len(updated_target_dict)
        assert n_s_vocab + n_t_vocab - self.n_concept == len(reconcatenate_dict)
        self.source_dict = updated_source_dict
        self.target_dict = updated_target_dict
        self.reconcatenate_dict = reconcatenate_dict
        self.n_s_vocab = n_s_vocab
        self.n_t_vocab = n_t_vocab
        logger.info("size of concept: {0}, size of source vocab: {1}, size of target vocab: {2}".format(self.n_concept, self.n_s_vocab,
                                                                                                 self.n_t_vocab))

    def convert_raw_corpus_to_trainable_corpus(self):
        """
        convert raw corpus into trainable corpus that each word is represented \
        by idx.

        This function produce self.docs and self.language_flags for JointLDA.
        """
        self.docs = list()
        self.language_flags = list()
        for doc in self.source_corpus:
            self.docs.append([self.source_dict[word] for word in doc])
            self.language_flags.append("S")
        for doc in self.target_corpus:
            self.docs.append([self.target_dict[word] for word in doc])
            self.language_flags.append("T")
        logger.info("Successfully generate idx corpus 'self.docs' and language flags 'self.language_flags'")
