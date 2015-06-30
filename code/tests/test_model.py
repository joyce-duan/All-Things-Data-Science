'''
ipython test_scraper.py
'''
import theano
from gensim import corpora, models, similarities
import gensim
import nose.tools as n
import nose
from cStringIO import StringIO
import unicodedata
import inspect
import random
import numpy as np
from gensim import corpora, models, similarities

from configobj import ConfigObj
config = ConfigObj('allds.config')
allds_home = config['allDS_home']

import sys
sys.path.append(allds_home + 'code/model')

# from topic_modeling import
from nmf import run_nmf, get_top_topics_terms
from LDA_topics import LDATopics, run_lda, BaseTopics


def test_run_nmf():
    print 'function: %s ' % inspect.stack()[0][3]
    nmx_max_iter = 6000
    model_name = 'run3_1'
    #func_stemmer = PorterStemmer()
    #func_tokenizer = word_tokenize
    # kw_tfidf = {'max_df': 0.90, 'stop_words': 'english', 'min_df': 10,\
    #            'tokenizer': func_tokenizer, 'ngram_range':(1,3)}
    kw_nmf = {'n_components': 2, 'max_iter': nmx_max_iter}
    X = np.array([random.random() for i in xrange(20)]).reshape((4, 5))
    print X
    W, H, nmf = run_nmf(X, kw_nmf)
    print W
    n.assert_true(len(W), 2)


def test_run_nmf_nokw():
    print '\nfunction: %s ' % inspect.stack()[0][3]
    nmx_max_iter = 6000
    model_name = 'run3_1'
    #func_stemmer = PorterStemmer()
    #func_tokenizer = word_tokenize
    # kw_tfidf = {'max_df': 0.90, 'stop_words': 'english', 'min_df': 10,\
    #            'tokenizer': func_tokenizer, 'ngram_range':(1,3)}
    kw_nmf = {'n_components': 2, 'max_iter': nmx_max_iter}
    X = np.array([random.random() for i in xrange(20)]).reshape((4, 5))
    print X
    W, H, nmf = run_nmf(X)  # , kw_nmf)
    print W
    n.assert_true(len(W), 2)


def test_lda():
    corpus = gensim.corpora.MalletCorpus('android.mallet')
    model = gensim.models.LdaModel(
        corpus, id2word=corpus.id2word, alpha='auto', num_topics=25)
    lda_model, topics_matrix, dictionary = run_lda(
        corpus, min_df=5, num_topics=25)
    n.assert_true(len(topcs_matrix) > 0)


if __name__ == "__main__":
    module_name = sys.modules[__name__].__file__

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s'])
    sys.stdout = old_stdout
    print mystdout.getvalue()
