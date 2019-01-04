from __future__ import print_function
import time

import numpy as np
from scipy.special import gammaln
from six.moves import xrange

from .base import BaseJointGibbsParamTopicModel
from .formatted_logger import formatted_logger
from .utils import sampling_from_dist

logger = formatted_logger('JointGibbsLDA')


class JointGibbsLDA(BaseJointGibbsParamTopicModel):
    """
    JointLDA from Jagadeesh Jagarlamudi and Hal Daum´e III

    Reference: Jagarlamudi, J., & Daumé, H. (2010, March). \
    Extracting multilingual topics from unaligned comparable corpora. \
    In European Conference on Information Retrieval (pp. 444-456). \
    Springer, Berlin, Heidelberg.

    Attributes
    ----------
    topic_assignment:
        list of topic assignment for each word token

    """

    def __init__(self, n_doc, n_concept, n_s_vocab, n_t_vocab,
     n_topic, alpha=0.1, beta=0.01, **kwargs):
        super(JointGibbsLDA, self).__init__(n_doc=n_doc, n_concept=n_concept,
         n_s_vocab=n_s_vocab, n_t_vocab=n_t_vocab, n_topic=n_topic,
         alpha=alpha, beta=beta, **kwargs)

    def random_init(self, docs, language_flags):
        """
        Parameters
        ----------
        docs: list, size=n_doc
        language_flags: list, size=n_doc
        """

        # laguage index of each document is observable,
        # hence it should be the same size as the number of documents.
        assert len(docs) == len(language_flags)

        for di in range(len(docs)):
            doc = docs[di]
            lang = language_flags[di]
            topics = np.random.randint(self.n_topic, size=len(doc))
            self.topic_assignment.append(topics)

            for wi in range(len(doc)):
                topic = topics[wi]
                word = doc[wi]
                if lang == "S":
                    self.STW[topic, word] += 1
                    self.sum_ST[topic] += 1
                elif lang == "T":
                    self.TTW[topic, word] += 1
                    self.sum_TT[topic] += 1
                self.DT[di, topic] += 1

    def fit(self, docs, language_flags, max_iter=100):
        """ Gibbs sampling for LDA
        Parameters
        ----------
        docs
        language_flags
        max_iter: int
            maximum number of Gibbs sampling iteration
        """
        self.random_init(docs, language_flags)

        for iteration in xrange(max_iter):
            prev = time.clock()

            for di in xrange(len(docs)):
                doc = docs[di]
                lang = language_flags[di]
                for wi in xrange(len(doc)):
                    word = doc[wi]
                    old_topic = self.topic_assignment[di][wi]

                    self.DT[di, old_topic] -= 1

                    # the following block is used to control z's conditional \
                    # distribution, cause it depends on language index of each \
                    # docuemnt
                    if lang == "S":
                        self.STW[old_topic, word] -= 1
                        self.sum_ST[old_topic] -= 1
                        # compute conditional probability of a topic of current\
                        # word wi
                        prob = (self.STW[:, word] / self.sum_ST) * (self.DT[di, :])

                        new_topic = sampling_from_dist(prob)

                        self.topic_assignment[di][wi] = new_topic
                        self.STW[new_topic, word] += 1
                        self.sum_ST[new_topic] += 1
                    elif lang == "T":
                        self.TTW[old_topic, word] -= 1
                        self.sum_TT[old_topic] -= 1
                        # compute conditional probability of a topic of current\
                        # word wi
                        prob = (self.TTW[:, word] / self.sum_TT) * (self.DT[di, :])

                        new_topic = sampling_from_dist(prob)

                        self.topic_assignment[di][wi] = new_topic
                        self.TTW[new_topic, word] += 1
                        self.sum_TT[new_topic] += 1

                    self.DT[di, new_topic] += 1

            if self.verbose:
                logger.info('[ITER] %d,\telapsed time:%.2f,\tlog_likelihood:%.2f',
                 iteration, time.clock() - prev, self.log_likelihood(docs))
        # in the end concatenate the topic word table
        self.reconcatenate_topic_word_table()

    def log_likelihood(self, docs):
        """
        likelihood function
        """
        ll = len(docs) * gammaln(self.alpha * self.n_topic)
        ll -= len(docs) * self.n_topic * gammaln(self.alpha)
        ll += self.n_topic * gammaln(self.beta * self.n_voca)
        ll -= self.n_topic * self.n_voca * gammaln(self.beta)

        for di in xrange(len(docs)):
            ll += gammaln(self.DT[di, :]).sum() - gammaln(self.DT[di, :].sum())
        for ki in xrange(self.n_topic):
            ll += gammaln(self.STW[ki, :]).sum() - gammaln(self.STW[ki, :].sum())
            ll += gammaln(self.TTW[ki, :]).sum() - gammaln(self.TTW[ki, :].sum())
        return ll

    def reconcatenate_topic_word_table(self):
        """ reconcatenate the topic-word distribution
        The size of self.TW is (n_topic, n_s_vocab + self.n_t_vocab - \
        self.n_concept)
        """
        temp = self.STW[:, :self.n_concept] + self.TTW[:, :self.n_concept]
        self.TW = np.concatenate((temp, self.STW[:, self.n_concept:],
         self.TTW[:, self.n_concept:]), axis=1)
