import numpy as np

class BaseTopicModel(object):
    """
    Attributes
    ----------
    n_doc: int
        the number of total documents in the corpus
    n_voca: int
        the vocabulary size of the corpus
    verbose: boolean
        if True, print each iteration step while inference.
    """
    def __init__(self, n_doc, n_voca, **kwargs):
        self.n_doc = n_doc
        self.n_voca = n_voca
        self.verbose = kwargs.pop('verbose', True)


class BaseGibbsParamTopicModel(BaseTopicModel):
    """ Base class of parametric topic models with Gibbs sampling inference

    Attributes
    ----------
    n_topic: int
        a number of topics to be inferred through the Gibbs sampling
    TW: ndarray, shape (n_voca, n_topic)
        word-topic matrix, keeps the number of assigned word tokens for each word-topic pair
    DT: ndarray, shape (n_doc, n_topic)
        document-topic matrix, keeps the number of assigned word tokens for each document-topic pair
    sum_T: ndarray, shape (n_topic)
        number of word tokens assigned for each topic
    alpha: float
        symmetric parameter of Dirichlet prior for document-topic distribution
    beta: float
        symmetric parameter of Dirichlet prior for topic-word distribution
    """

    def __init__(self, n_doc, n_voca, n_topic, alpha, beta, **kwargs):
        super(BaseGibbsParamTopicModel, self).__init__(n_doc=n_doc, n_voca=n_voca, **kwargs)
        self.n_topic = n_topic
        self.TW = np.zeros([self.n_topic, self.n_voca])
        self.DT = np.zeros([self.n_doc, self.n_topic])
        self.sum_T = np.zeros(self.n_topic)

        self.alpha = alpha
        self.beta = beta

        self.topic_assignment = list()

        self.TW += self.beta
        self.sum_T += self.beta * self.n_voca
        self.DT += self.alpha

class BaseJointGibbsParamTopicModel(object):
    """ Base class of Joint parametric topic models with \
    Gibbs sampling inference

    Attributes
    ----------
    n_doc: int
        the number of total documents in the corpus
    n_concept: int
        the concept size of the both corpus
    n_s_vocab: int
        the vocabulary size of the source corpus
    n_t_vocab: int
        the vocabulary size of the target corpus
    verbose: boolean
        if True, print each iteration step while inference.

    n_topic: int
        a number of topics to be inferred through the Gibbs sampling
    TW: ndarray, shape (n_topic, n_voca)
        word-topic matrix, keeps the number of assigned word tokens for
        each word-topic pair
    STW: ndarray, shape (n_topic, n_s_vocab)
        word-topic matrix for SOURCE language, keeps the number of assigned
        word tokens for each word-topic pair
    TTW: ndarray, shape (n_topic, n_t_vocab)
        word-topic matrix for TARGET language, keeps the number of assigned
        word tokens for each word-topic pair
    DT: ndarray, shape (n_doc, n_topic)
        document-topic matrix, keeps the number of assigned word tokens for
        each document-topic pair
    sum_T: ndarray, shape (n_topic)
        number of word tokens assigned for each topic
    sum_ST: ndarray, shape (n_topic)
        number of word tokens assigned for each topic in SOURCE language
    sum_TT: ndarray, shape (n_topic)
        number of word tokens assigned for each topic in TARGET language
    alpha: float
        symmetric parameter of Dirichlet prior for document-topic distribution
    beta: float
        symmetric parameter of Dirichlet prior for topic-word distribution
    """

    def __init__(self, n_doc, n_concept, n_s_vocab, n_t_vocab, n_topic, alpha,
     beta, **kwargs):
        self.n_doc = n_doc
        self.n_concept = n_concept
        self.n_s_vocab = n_s_vocab
        self.n_t_vocab = n_t_vocab
        self.n_voca = self.n_s_vocab + self.n_t_vocab - self.n_concept
        self.verbose = kwargs.pop('verbose', True)

        self.n_topic = n_topic
        self.TW = np.zeros([self.n_topic, self.n_voca])
        self.STW = np.zeros([self.n_topic, self.n_s_vocab])
        self.TTW = np.zeros([self.n_topic, self.n_t_vocab])
        self.DT = np.zeros([self.n_doc, self.n_topic])
        self.sum_T = np.zeros(self.n_topic)
        self.sum_ST = np.zeros(self.n_topic)
        self.sum_TT = np.zeros(self.n_topic)

        self.alpha = alpha
        self.beta = beta

        self.topic_assignment = list()

        #self.TW += self.beta
        #self.sum_T += self.beta * self.n_voca
        self.STW += self.beta
        self.sum_ST += self.beta * self.n_s_vocab
        self.TTW += self.beta
        self.sum_TT += self.beta * self.n_t_vocab
        self.DT += self.alpha
