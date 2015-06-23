'''
to-do:  NMF components = 40 gives 20 topics????
NMF mixing up articles and topics???
'''

from configobj import ConfigObj
config=ConfigObj('allds.config')
#config = ConfigObj('../../allds.config')
print config
allds_home = config.get('allDS_home','/Users/joyceduan/Documents/git/All-Things-Data-Science/')
data_home = allds_home + 'data/'

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from urllib3.util import parse_url

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import re

import sys
sys.path.append(allds_home + 'code/db')
sys.path.append(allds_home +'code/preprocess')
sys.path.append(allds_home  + 'code/model')
sys.path.append(allds_home  + 'code/recommender')
print allds_home + 'code/db'
#print allds_home  + 'code/recommender'
#sys.path.append('../preprocess')
#sys.path.append('../model')


from my_mongo import MyMongo
from ArticleProceser import  clean_articles, fit_tfidf, transform_tfidf, ascii_text
from nmf import run_nmf, get_top_topics_terms
#from recommender import Recommender

import string
import pickle as pickle
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

import unicodedata
from datetime import date, timedelta as tdel

n_top_articles = 5
n_top_terms = 15
root_vectorizer =  '_vectorizer.pkl'
root_topics = '_topics.csv'
root_nmf_model = '_nmf_fitted_model.pkl'

class TopicModel(object):

    def __init__(self,model_name, func_stemmer, func_tokenizer, kw_tfidf = None, kw_nmf = None) :
        '''
        initialize NMF topic modeler
        '''
        self.model_name = model_name
        self.H = None
        self.topic_names = None
        self.vectorizer = None
        self.nmf_model = None

        self.func_tokenizer = func_tokenizer
        self.func_stemmer = func_stemmer

        if kw_tfidf:
            d = {'tokenizer':func_tokenizer}
            self.kw_tfidf = dict(kw_tfidf, **d)
        else:
            self.kw_tfidf = kw_tfidf

        self.kw_nmf = kw_nmf

    def get_topic_names(self):
        return self.topic_names

    def dump_model(self, note = ''):
        '''
        df2: dataframe with all data
        clean_article: function to clean text
        transform_tfidf: 
        vectorizer: fitted vectorizer 
        W: mapping of documents to topics
        H: mapping of terms to topics
        nmf: nmf model
        '''
        model_name = self.model_name
        with open(data_home + model_name + '_readme.txt','w') as out_fh:
            out_fh.write(note)
        with open(data_home + model_name + root_vectorizer ,'w') as out_fh:  
            pickle.dump(self.vectorizer, out_fh)
        with open(data_home + model_name + root_nmf_model ,'w') as out_fh:  
            pickle.dump(self.nmf_model, out_fh)

    def load_model(self):
        '''
        load model from data_home
        - OUTPUT:
            self.vectorizer
            self.nmf_model

            self.topic_names  : manul curation
            self.H: topic x term  # redundant with self.nmf_model?
        '''
        t0 = time.time()

        h_fname = data_home + self.model_name + '_H.pkl'
        vectorizer_fname = data_home + self.model_name + '_vectorizer.pkl'
        topics_fname = data_home + self.model_name + '_topics.csv'
        nmf_model_fname = data_home + self.model_name + '_nmf_fitted_model.pkl'

        try:
            with open(h_fname,'r') as in_fh:
                self.H = pickle.load(in_fh)
        except:
            print '!!!!erroor loading file %s' % h_fname
        
        try:
            with open(vectorizer_fname,'r') as in_fh:
                self.vectorizer = pickle.load(in_fh)
        except:
            print '!!!!erroor loading file %s' % vectorizer_fname            

        try:
            with open(topics_fname,'r') as in_fh:
                t = pd.read_csv(in_fh, sep=',',header=None)
                t.columns=['i','name']
                self.topic_names = t.sort(['i'],ascending=True).values[:,1]
        except:
            print '!!!!error loading file %s' % topics_fname

        try:
            with open(nmf_model_fname,'r') as in_fh:
                self.nmf_model = pickle.load(in_fh)
        except:
            print '!!!!error loading file %s' % topics_fname


        t1 = time.time() # time it
        print "finished in  %4.4fmin in %s " %((t1-t0)/60,'load_model')


    def transform_bodytext2topics(self,docs, flag_print_time = 0):
        '''
        calculate the topic vector for an article document
            - Pre-requisit:  self.vectorizer, self.H
            - INPUT: docs   list of body_text_content 
            - OUTPUT: W   vector n_articles x n_topics
        '''
        #print 'transform_bodytext2topics'
        print flag_print_time
        t0 = time.time()
        vectorized_X, tokenized_articles = transform_tfidf(self.vectorizer, docs,self.func_tokenizer,self.func_stemmer)
        #W = vectorized_X.dot(self.H.T)????
        W = self.nmf_model.transform(vectorized_X)
        t1 = time.time() # time it
        if flag_print_time:
            print "finished in  %4.4fmin for %s " %((t1-t0)/60,'transform body text to topics')

        return W, tokenized_articles, vectorized_X
 
    def fit_featurize(self, df):
        '''
        tokenzie and preprocess df.body_text
        generate tifidf, fitering
            - OUTPUT:  
                self.df
                self.X2
                self.idx_in_model
                self.vectorizer
        '''
        self.df = df

        txt_list = df.body_text
        self.vectorizer, vectorized_X, tokenized_articles = fit_tfidf(txt_list, self.kw_tfidf, self.func_tokenizer, self.func_stemmer)

        df['len'] = [len(t) for t in tokenized_articles]
        df['tokens'] = tokenized_articles

        cond = df['len'] >= 50  # only keeps articles with at least 50 words
        df = df[cond]
        df2 = df
        irows = [i for i in xrange(vectorized_X.shape[0]) if cond[i]]
        self.X2 =vectorized_X[np.ix_(irows)]
        self.idx_in_model = irows

        url_list2 = df2['url']
        #df2.head(2)
        print len(url_list2)
        print df2.shape
        print self.X2.shape

        print df2['uri'].nunique()
        print pd.value_counts(df2['uri'])[:15]
        print 1.0 *pd.value_counts(df2['uri'])[:8].sum()/df2.shape[0]
        print df2.uri.nunique()

        print self.df.shape
        print self.df.columns
        #print self.df.head(2)

    def fit_analyze_nmf(self):
        '''
        fit NMF
        print out stats
        print out summary.html
        print out model.html
        '''
        
        #idx_df = list(self.df.index)

        t0 = time.time() # time it
        self.W, self.H, nmf_model = run_nmf(self.X2 )
        self.nmf_model = nmf_model
        t1 = time.time() # time it
        print "finished in  %4.4fmin for %s " %((t1-t0)/60,'run_nmf')
        print 'w ', self.W.shape, 'h', self.H.shape

        W_t = self.W.T

        print self.W.shape, W_t.shape
        #print type(W_t)
        print len(W_t[0,]), W_t[0,:10]
        print 'range for W: (%.2f - %.2f); range for H: (%.2f - %.2f)' % (np.min(self.W), np.max(self.W), np.min(self.H), np.max(self.H))

        self.topic_terms = get_top_topics_terms(self.vectorizer, self.H, k_top_words=n_top_terms)

        self.print_topic_results_html()
        self.plot_hist_weight_best_topic_per_article()
        cutoff = self.print_summary_stats()
        self.plot_hist_d_to_centroid(min_w = cutoff)

    def get_best_topic_per_article(self):
        '''
        get the best topic for each article 
            - INPUT: self.W
            - OUTPUT: self.best_topic_for_articles n_articles x 2   [i_best_topic, topic_w]
        '''
        best_topics_for_articles = []
        for i_article, topic_w in enumerate(self.W):
            i_best_topic = np.argmax(topic_w)
            best_topics_for_articles.append([i_best_topic, topic_w[i_best_topic]])
        self.best_topics_per_article = np.array(best_topics_for_articles)

    def sorted_topics_for_articles(self, W):
        '''
        return list of sorted topics by relevance
            - INPUT: W n_articles x n_topics, self.topic_names
            - OUPUT: list of list of list [ [(i_topic, topic_name, relevance)]]
        '''
        topic_names = self.topic_names
        #print topic_names

        #topic_names = ['topic # %i' %i  for i in xrange(W.shape[1])]
        sorted_idx_topics = np.argsort(W, axis= 1)
        sorted_topics_all = []
        for i_article in xrange(W.shape[0]):
            #idx_topics_desc = (sorted_idx_topics[i_article,:].getA()).flatten()[::-1]
            idx_topics_desc = (sorted_idx_topics[i_article,:]).flatten()[::-1]
            #print type(idx_topics_desc), idx_topics_desc.shape
            #print idx_topics_desc
            sorted_topics_this = [(i_topic, topic_names[i_topic], W[i_article,i_topic]) for i_topic in idx_topics_desc]
            sorted_topics_all.append(sorted_topics_this)
        return sorted_topics_all

    def plot_hist_weight_best_topic_per_article(self):   
        '''

        '''
        self.get_best_topic_per_article()
        fig = plt.figure(figsize=(8,4))
        plt.hist(self.best_topics_per_article[:,1], bins =50)
        #plt.show()
        fig.savefig(self.model_name + '_hist_best_topic_per_article.png',bbox_inches = 'tight')
        plt.close(fig)

    def print_summary_stats(self):
        '''
        print summary stats of weight of the best topic for each article
            - INPUT: self.best_topics_per_article
            - OUTPUT: weight_20_perncentile (80 percent articles have best topic weight higher tha this)
        '''
        print np.max(self.best_topics_per_article[:,1]), np.min(self.best_topics_per_article[:,1])
        pct = [10, 20,30,40,50,60,70,80,90,100]
        pct_val = np.percentile(self.best_topics_per_article[:,1], pct)
        for i, p in enumerate(pct):
            print '%i: %.3f' % (p, pct_val[i])
        cutoff = pct_val[6]
        print 'cutoff: %.3f' % cutoff

        # number of best articles for each topic above cutoff
        cond_topic = self.best_topics_per_article[:,1] > cutoff
        print pd.value_counts(self.best_topics_per_article[cond_topic,0])
        return pct_val[1]

    def print_topic_results_html(self):
        '''
        print topic modeling results as html file:  model_name.html

        '''
        model_name = self.model_name
        W_t = self.W.T
        topic_terms = self.topic_terms
        model_name = self.model_name
        df2 = self.df

        #model_name, n_top_articles, W_t,topic_terms, n_top_terms 
        with open(model_name+'.html','w') as out_fh:
            out_fh.write('<html>\n<body>')
            for topic_idx, article_w in enumerate(W_t):
                out_fh.write("<h1>Topic #%d: </h1>\n" % topic_idx)

                #    for i, topic in enumerate(topic_terms):
                terms = topic_terms[topic_idx]
                l = sorted(terms.items(), key=lambda x: x[1])[::-1]
                txt_list = []
                for item in l[:n_top_terms]:
                    txt_list.append('%s (%.2f)' % (item[0], item[1]))
                out_fh.write( '<p><strong>top terms: ' + ' '.join(txt_list) +"</strong></p>\n")

                out_fh.write('-----------------------------------<br>\n')
                #print article_w.shape
                idx_article_topn = article_w.argsort()[:-n_top_articles - 1:-1]
                for i, idx in enumerate(idx_article_topn):
                    url = df2.iloc[idx]['url']
                    #title_this = article_dict_all.get(url,'')
                    title_this = df2.iloc[idx]['title']
                    title_this_cleaned = ascii_text(title_this) 
                    if title_this_cleaned == '':
                        title_this_cleaned = url
                    out_fh.write( '<p> ' + str(i) + '. (%.2f))'% article_w[idx] +'</p>\n' )
                    out_fh.write( '<a href="' + url +'" target="_blank"> %s </a>  <br>\n' % title_this_cleaned  )
                    
                    body_text_str = df2.iloc[idx]['body_text'][:400]
                    #body_text_str = body_text_str.encode('utf8')
                    body_cleaned = ascii_text(body_text_str)
                    out_fh.write( body_cleaned +' \n<br>\n')
                    out_fh.write('\n') 
            out_fh.write('</body>\n</html>\n')

    def assign_cluster(self, min_w = 0):
        '''
        assign article to 1 cluster (highest w above min_w)
                -1  if w is too low
            - INPUT: 
                self.best_topics_per_article
            - OUTPUT:
                self.clusters
        '''
        n_articles = self.best_topics_per_article.shape[0]
        clusters = -1 * np.ones(n_articles)
        for i_article in xrange(n_articles):
            if self.best_topics_per_article[i_article,1] > min_w:
                clusters[i_article] = self.best_topics_per_article[i_article,0]
        self.clusters  = np.array([int(c) for c in clusters])
        
    def cal_centroid(self):
        '''
        calculate centroid of each cluster using euclidean distance
        '''
        n_clusters = np.max(self.clusters)
        centroids = []
        for i in xrange(n_clusters + 1):
            cond = self.clusters == i
            arr = self.X2[cond]
            #length = arr.shape[0]
            c = np.mean(arr, axis = 0)
            centroids.append(c)
        self.centroids = np.array(centroids)

    def plot_hist_d_to_centroid(self, min_w = 0):
        '''
        histograms of distance to centroids: overall vs. each cluster
        '''
        self.assign_cluster(min_w)
        self.cal_centroid()
        n_clusters = np.max(self.clusters)
        #fig = plt.figure(figsize=(20,8))
        centroid_overall = np.mean(self.X2, axis = 0)
        sim = linear_kernel(centroid_overall, self.X2)
        max_sim = np.max(sim)
        min_sim = np.min(sim)

        #multiple plot, subplots
        ncols = 3
        nrows = (n_clusters + 1) // ncols + ( ((n_clusters+1) % ncols)>0 )
        fig, ax = plt.subplots(nrows,ncols,figsize=(30,10))   #subplot preferred way
        axs = ax.flatten()
        i_plot = 0
        axs[i_plot].hist(sim.flatten(), alpha=0.2) #, ax=axs[i_plot])
        axs[i_plot].set_xlim(min_sim, max_sim)
        i_plot = i_plot + 1

        for i in xrange(n_clusters + 1):
            cond = self.clusters == i
            arr = self.X2[cond]
            sim = linear_kernel(self.centroids[i], arr)
            axs[i_plot].hist(sim.flatten(), alpha=0.2)#, ax=axs[i_plot])
            axs[i_plot].set_xlim(min_sim, max_sim)
            i_plot = i_plot + 1
        fig.savefig(self.model_name+'_hist_dis_to_centroid.png')
        plt.close(fig)



def plot_mds(data_m, title, distance_metric):
    '''
    takes ~60minuts on 1500 x 6000
        - INPUT: 
            data_m   data matrix
            title:  plot title
            distance_metric: i.e. 'euclidean'
    '''
    X_dist = pdist(data_m, distance_metric)
    X_dist = squareform(X_dist)
    code_for_colors = cluster_assigned
    
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                       dissimilarity="precomputed", n_jobs=1)
    mds_fitted = mds.fit(X_dist).embedding_
    X_dist_transformed = mds.fit_transform(X_dist)
    plt.scatter(X_dist_transformed[:,0], X_dist_transformed[:,1],alpha=0.8,edgecolor='none', c = code_for_colors) 
    plt.title(title)
    plt.show()


def read_articles():
    '''
    ??? should this be moved to preprocess module
    read all articles as dataframe from mongodb collection 'articles'
        - INPUT: None
        - OUTPUT: df.   columns: title, url, uri, body_text, 
    '''
    my_mongo = MyMongo()

    t0 = time.time()
    cur_articles = my_mongo.get_article_body_text(testing=0)

    articles_cleaned = {}
    #print '%d unique articles ' % len(articles_cleaned)
    clean_articles(cur_articles, articles_cleaned)
    print '%d unique articles with body_text' % len(articles_cleaned)

    t1 = time.time() # time it
    print "finished in  %4.4fmin for %s " %((t1-t0)/60,'read/clean articles')

    df = pd.DataFrame([{'url':k,'body_text':v[1]} for k, v in articles_cleaned.items()])

    article_dict,  article_dt =  MyMongo().get_article_attri()
    #article_dict_all = dict(article_dict)
    df['title'] = df['url'].map(lambda x: article_dict.get(x,'Unknown'))
    df['uri'] = df['url'].map(lambda x: parse_url(x).host)

    my_mongo.close()

    return df

def run_model(model_name, kw_tfidf, kw_nmf, func_stemmer, func_tokenizer):
    '''
    run model from read 'body_text' from mongo db to fit and summarize NMF topics
    '''
    topic_model = TopicModel(model_name, kw_tfidf, kw_nmf, func_stemmer, func_tokenizer)
    df = read_data()
    topic_model.featurize(df)
    topic_model.fit_analyze_nmf()
    #pickle.dump(topic_model, open(model_name+'_all.pkl', 'wb'))

def tune_hyper_params():
    '''
    run topic modeling from end to end using various following hyper parameters and save charts and summary results/stats
        n_gram_range:
        stemmer:
        tokenizer:
        min_df: 

    '''
    nmx_max_iter= 6000 # 3000

    '''
    model_name = 'v4'   # 3 snow  
    kw_tfidf = {'max_df': 0.90, 'stop_words': 'english', 'min_df': 10,\
                'tokenizer': None}                
    kw_nmf = {'n_components': 20, 'max_iter': nmx_max_iter}
    func_stemmer = PorterStemmer()
    #func_stemmer = SnowballStemmer('english')
    func_tokenizer = TfidfVectorizer(stop_words = 'english').build_tokenizer()
    run_model(model_name, kw_tfidf, kw_nmf, func_stemmer, func_tokenizer)   
    '''

    # topic_model = TopicModel(model_name, kw_tfidf, kw_nmf, func_stemmer, func_tokenizer)
    # df = read_data()
    # topic_model.featurize(df)
    # topic_model.fit_analyze_nmf()
    # pickle.dump(topic_model, open(model_name+'_all.pkl', 'wb'))


    model_name = 'v5'
    func_stemmer = PorterStemmer()
    func_tokenizer = word_tokenize
    kw_tfidf = {'max_df': 0.90, 'stop_words': 'english', 'min_df': 10,\
                'tokenizer': func_tokenizer, 'ngram_range':(1,3)}                
    kw_nmf = {'n_components': 20, 'max_iter': nmx_max_iter}
    run_model(model_name, kw_tfidf, kw_nmf, func_stemmer, func_tokenizer)


    model_name = 'v6'
    func_stemmer = PorterStemmer()
    func_tokenizer = word_tokenize
    kw_tfidf = {'max_df': 0.90, 'stop_words': 'english', 'min_df': 10,\
                'tokenizer': func_tokenizer, 'ngram_range':(1,3)}                
    kw_nmf = {'n_components': 40, 'max_iter': nmx_max_iter}
    run_model(model_name, kw_tfidf, kw_nmf, func_stemmer, func_tokenizer)


    model_name = 'v7'
    func_stemmer = PorterStemmer()
    func_tokenizer = word_tokenize
    kw_tfidf = { 'stop_words': 'english','ngram_range':(1,5), 'max_features':10000, 'min_df':20, 'max_df':0.9}
    kw_nmf = {'n_components': 20, 'max_iter': nmx_max_iter}
    run_model(model_name, kw_tfidf, kw_nmf, func_stemmer, func_tokenizer)


    model_name = 'v8'
    func_stemmer = SnowballStemmer('english')
    func_tokenizer = word_tokenize
    kw_tfidf = { 'stop_words': 'english','ngram_range':(1,5), 'max_features':10000, 'min_df':20, 'max_df':0.9}
    kw_nmf = {'n_components': 20, 'max_iter': nmx_max_iter}
    run_model(model_name, kw_tfidf, kw_nmf, func_stemmer, func_tokenizer)


    model_name = 'v9'
    kw_tfidf = {'max_df': 0.90, 'stop_words': 'english', 'min_df': 10,\
                'tokenizer': None}                
    kw_nmf = {'n_components': 20, 'max_iter': nmx_max_iter}
    func_stemmer = lambda x: x.lower()
    func_tokenizer = TfidfVectorizer(stop_words = 'english').build_tokenizer()
    run_model(model_name, kw_tfidf, kw_nmf, func_stemmer, func_tokenizer)


def test_run_modeler():
    model_name = 'v2_2'
    func_tokenizer = TfidfVectorizer(stop_words = 'english').build_tokenizer()
    func_stemmer = PorterStemmer()
    topic_model = TopicModel(model_name, func_stemmer, func_tokenizer)
    topic_model.load_model()
 
    df = read_articles()
    W, tokenized_articles, vectorized_X = topic_model.transform_bodytext2topics(df.body_text[0:5],1)

    sorted_topics = topic_model.sorted_topics_for_articles(W) 

if __name__ == '__main__':
    test_run_modeler()

    #tune_hyper_params()

