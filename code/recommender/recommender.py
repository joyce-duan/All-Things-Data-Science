from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import sys

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import codecs
import re
import time
import os
import pickle
#import ipdb

from configobj import ConfigObj
config = ConfigObj('allds.config')
allds_home = config.get('allDS_home', '/Users/joyceduan/Documents/git/All-Things-Data-Science/')
data_home = allds_home + 'data/'

sys.path.append(allds_home+'code/model')
sys.path.append(allds_home+'code/preprocess')

from topic_modeling import TopicModel, read_articles
from ArticleProceser import   ascii_text

class Recommender(object):
    def __init__(self, model_name , func_tokenizer, func_stemmer):
        '''
            - INPUT: 
                model_name   str
                func_tokenizer  tokenizer used in the model_name
                func_stemmer   stemmer used in the model_name
            - Post-condition:
                 self.df_articles, self.W_articles, self.X_articles
        '''
        if model_name == '':
            model_name = 'v2_2'
        self.top_k_recommend = 5
        self.max_rank = 50
        self.top_k_topics = 5
        self.max_cosine_sim_tfidf = 0.5

        self.method = None # content or rating
        self.max_len_body_text = 3000        
        self.model_name = model_name #'v2_2'

        t0 = time.time() # time it

        self.topic_model = load_topic_model(model_name, func_tokenizer, func_stemmer)
        t1 = time.time() # time it
        print "finished in  %4.4f %s " %((t1-t0)/60,'loading model\n')

        # load all articles 
        # these include newest articles, which may not be used in model_name and related H
        t0 = t1
        df_article_fname = data_home + 'df_articles.pkl'
        dict_article_fname = data_home + 'dict_articles.pkl'
        W_article_fname = data_home + self.model_name + 'W_articles.pkl'
        X_article_fname = data_home + self.model_name + 'X_articles.csv'

        if os.path.exists(df_article_fname):
	    print 'found picklet files %s' % df_article_fname
            self.load_articles_from_pickle(df_article_fname, W_article_fname, X_article_fname)
        else:
	    print 'no pickle files %s. read from mongodb' % df_article_fname
            self.df_articles = read_articles()
            self.W_articles, tokenized_articles, self.X_articles = self.topic_model.transform_bodytext2topics(self.df_articles.body_text,1)

            with open(df_article_fname,'w') as out_fh:  
                pickle.dump(self.df_articles, out_fh)
            with open(W_article_fname,'w') as out_fh:  
                pickle.dump(self.W_articles, out_fh)           
            with open(X_article_fname,'w') as out_fh:  
                pickle.dump(self.X_articles, out_fh) 
            with open(dict_article_fname,'w') as out_fh:
                pickle.dump(self.df_articles.to_dict(), out_fh)

        #print topic_model.sorted_topics_for_articles(W_articles[:1,:])
        self.sorted_topics_articles = self.topic_model.sorted_topics_for_articles(self.W_articles)
        t1 = time.time() # time it
        print 'topics for articles:'
        print "finished in  %4.4f min for %s " %((t1-t0)/60,'topics of articles\n')

    def get_topic_names(self):
        return self.topic_model.topic_names

    def load_articles_from_pickle(self, df_article_fname, W_article_fname, X_article_fname):
        ''' 
        with open(df_article_fname, 'r') as in_fh:
            print df_article_fname
            self.df_articles = pickle.load(in_fh)
        ''' 
        with open(dict_aritcle_fname, 'r') as in_fh:
            print dict_article_fname
            dict_articles = pickle.load(in_fh)
            self.df_articles = pd.DataFrame(dict_articles)
        with open(W_article_fname, 'r') as in_fh:
            print W_article_fname
            self.W_articles = pickle.load (in_fh)
        with open(X_article_fname, 'r') as in_fh:
            print X_article_fname
            self.X_articles = pickle.load(in_fh)

    def calculate_recommendations(self, W, test_X2, input_name):
        '''
            - precondisions: all articles to be searched
                self.X_articles
                self.W_articles
                self.df_articles
            - INPUT: 
                W:  1 x n_topics
                test_X2:  1 x n_features
            - OUTPUT: 
                df_recom from get_recommendation_dataframe
        '''
        top_k_recommend = self.top_k_recommend 
        max_rank = self.max_rank
        top_k_topics = self.top_k_topics
        fname = input_name

        # cal simmilarity to all articles
        i_article = 0
        t0 = time.time()
        cosine_similarities = linear_kernel(self.X_articles, test_X2[i_article,:]).flatten()
        cosin_simi_latent_topics = linear_kernel(self.W_articles, W[i_article,:]).flatten()
        cosine_similarities_rank = get_rank(cosine_similarities)
        cosin_simi_latent_topics_rank = get_rank(cosin_simi_latent_topics )
        t1 = time.time() # time it
        print "finished in  %4.4f min for %s " %((t1-t0)/60,'calculate cosine similarity\n')
     
        make_diagnostic_plots(fname, cosine_similarities, cosin_simi_latent_topics,\
           cosine_similarities_rank, cosin_simi_latent_topics_rank )
        #print cosine_similarities[desc_sim_indexes[:20]]

        return self.get_recommendation_dataframe(cosine_similarities, cosin_simi_latent_topics,\
           cosine_similarities_rank, cosin_simi_latent_topics_rank )

    def get_recommendation_dataframe(self, cosine_similarities, cosin_simi_latent_topics,\
           cosine_similarities_rank, cosin_simi_latent_topics_rank):
        '''
        - OUTPUT:
            df_recom data frame each row has the following: 
            r['score'] = dict [cosine_similarities[i], cosine_similarities_rank[i],cosin_simi_latent_topics[i],cosin_simi_latent_topics_rank[i] ]
            r['topics']  dict  [(itpic, topic_name, weight),topic:weight}, {..}]
            r['title'], r['body_text']. r['url'] 
        '''
        desc_sim_indexes = np.argsort(cosine_similarities)[::-1]
        recommed_articles = []
        i_print = 0
        max_cosine_sim_tfidf = 0.5
        if self.max_cosine_sim_tfidf:
            max_cosine_sim_tfidf = self.max_cosine_sim_tfidf

        for i in desc_sim_indexes[:self.max_rank]:
            if cosin_simi_latent_topics_rank[i]< self.max_rank and i_print < self.top_k_recommend:
                r = {}
                url = self.df_articles.iloc[i].url
                r['score'] = {'cosine_sim_tfidf':cosine_similarities[i],'cosine_sim_tfidf_rank':cosine_similarities_rank[i],'cosine_simi_latent_topics':cosin_simi_latent_topics[i],'cosine_sim_latent_topics_rank':cosin_simi_latent_topics_rank[i] }
                r['title'] = self.df_articles.iloc[i].title
                body_cleaned = ascii_text(self.df_articles.iloc[i].body_text[:self.max_len_body_text])
                r['body_text']= body_cleaned
                r['url'] = url
                r['topics'] = self.sorted_topics_articles[i][0:self.top_k_topics] # (itopic, topic_name, weight)
                r['relevance'] = min(100, int(cosine_similarities[i] / self.max_cosine_sim_tfidf * 100))
                recommed_articles.append(r)            
                #print sorted_topics_articles[i][0:2]
                print cosine_similarities[i], cosine_similarities_rank[i], '***'+ self.df_articles.iloc[i].title+'***'
                #print cosin_simi_latent_topics[i],cosin_simi_latent_topics_rank[i]
                #print 
                #body_cleaned = ascii_text(df_articles.iloc[i].body_text[:300])
                #print body_cleaned
                #print 
                i_print = i_print + 1

        df_recom = pd.DataFrame(recommed_articles)
        return df_recom 

    def process_input(self, cleaned_slack, dataset_name = ''):
        '''
            - OUTPUT:
                W  1 x n_topics
                test_X2  tfids
                tokenized_slacks2
                top_features_list : top_n features list of tuple (feature_name, tfidf)
        '''
        #print type(cleaned_slack)
        fname = dataset_name

        W, tokenized_slacks2, test_X2 = self.topic_model.transform_bodytext2topics([cleaned_slack],1)
        print 'topics for input text'
        #print topic_model.sorted_topics_for_articles(W)
        #t1 = time.time() # time it
        #print "finished in  %4.4f min %s " %((t1-t0)/60,'topics of slack message\n')

        # summary of input
        top_n = 50
        print "top %i most frequenct features in input %s" % (top_n, fname)
        sorted_feature_indexes = np.argsort(test_X2, axis=1)
        #print test_X2[desc_feature_indexes[:top_n]]
        features = self.topic_model.vectorizer.get_feature_names()
        i_article = 0
        desc_feature_indexes = sorted_feature_indexes[i_article,:].getA().flatten()[::-1]
        top_features_list = []
        for i in desc_feature_indexes[:top_n] :
            #top_features_list.append('%s (%.2f)' % (features[i], test_X2[i_article,i]))
            top_features_list.append((features[i], test_X2[i_article,i]))
        #print ', '.join(top_features_list)
        return W, tokenized_slacks2, test_X2, top_features_list

def get_rating(good_urls, meh_urls, df2):
    '''
    convert from lists to rating matrix
    create rating_content matrix from list of good and meh urls
        -OUTPUT: rating_content list of list  (i_article, rating_score)
    '''
    score_good = 4
    score_meh = 1
    rating_content = []
    cond1 = df2.url.isin(good_urls)
    #print type(cond1)
    #print cond1[:5]
    #index_good = list(df2[cond1].index)
    ilocs_good = [i for i in xrange(df2.shape[0]) if cond1.iloc[i]]
    #print i_rows_good
    
    cond2 = df2.url.isin(meh_urls)
    #index_meh = list(df2[cond2].index)
    #print index_meh
    ilocs_meh = [i for i in xrange(df2.shape[0]) if cond2.iloc[i]]
    
    for i in ilocs_good:
        rating_content.append((i,score_good))
    for i in ilocs_meh:
         rating_content.append((i,score_meh))
    return np.array(rating_content)
    print get_rating(good_urls, meh_urls, df2)    

def run_article_recom(df2):
    '''
    test run of user rating based article recommendations 
    '''
    good_urls = ['http://www.bloomberg.com/news/articles/2015-06-04/help-wanted-black-belts-in-data', 
            'http://bits.blogs.nytimes.com/2015/04/28/less-noise-but-more-money-in-data-science/'
           , 'http://bits.blogs.nytimes.com/2013/06/19/sizing-up-big-data-broadening-beyond-the-internet/'
           ]
    meh_urls = ['http://bits.blogs.nytimes.com/2013/04/11/education-life-teaching-the-techies/' # too short, no content
    , 'http://www.nytimes.com/2013/04/14/education/edlife/universities-offer-courses-in-a-hot-new-field-data-science.html'  
    ]

    rating_content = get_rating(good_urls, meh_urls, df2)  
    rating_vec = np.zeros(df2.shape[0])
    for row in rating_content:
        rating_vec[row[0]] =  row[1]
    print rating_content
    print [i for i in xrange(len(rating_vec)) if rating_vec[i]>0]

def read_slack_msgs(func_tokenizer, fname ='slack_mod.txt'):
    '''
    read and clean slack message as one article
        - OUTPUT: cleaned_slack string

    ??? need a way to ignore the slack line:  Slackdata
    '''
    data_file = '../../data/'+fname 
    fileObj = codecs.open( data_file, "r", "utf-8" )
    data2 = fileObj.read() # Returns a Unicode string from the UTF-8 bytes in the file
    #data2 = data2.encode('ascii','ignore')
    #data2 = ' '.join([l for l in data2.split('\n') if 'text/html;charset=utf-8' not in l])
    #print u
    '''
    # this does not work due to utf-8 coding
    data_file = '../../data/slack.txt'
    with open(data_file, 'r') as in_fh:
        data = in_fh.read()
    data2 = data.decode(('utf8'))
    '''
    return pre_clean_text(func_tokenizer, data2)
    #test_X, tokenized_slacks = transform_tfidf(vectorizer, [data2])
    #test_X = test_X.getA().flatten()

def pre_clean_text(func_tokenizer, data2):
    '''
    pre-clean text
        INPUT:
            - func_tokenizer  tokenizer function to use
            - text  str
        OUTPUT:
            - text_cleaned  str
    '''
    words_exclude = [ u'text/html', 'pm', 'http' 'io','com', 'github', 'www' ,'config', 'html', 'js' ,'https', 'join', 'repo','chris',u'http', u'__init__']
    cleaned_slack = []
    #ipdb.set_trace()  
    tokenized_slacks = func_tokenizer(data2)
    for w in tokenized_slacks:#[0]:
        if  re.search('[a-zA-Z]',w) and '//' not in w and 'sf_ds' not in w and w not in words_exclude:
            cleaned_slack.append(w)
    return ' '.join(cleaned_slack)

def get_rank(array):
    '''
    get rank for a 1d array descending order
        - INPUT: array 1d
        - OUTPUT:  array 1d
    '''
    order = array.argsort()
    ranks = order.argsort()  
    return len(ranks) - ranks

def load_topic_model(model_name, func_tokenizer, func_stemmer ):
    '''
    load topic model pickel files
    '''
    topic_model = TopicModel(model_name, func_stemmer, func_tokenizer)
    topic_model.load_model()
    return topic_model

def make_recommendation(fname, model_name = 'v2_2'):
    '''
    test run content based recommendation content in fname
    command line make recommendations
        - INPUT: 
            fname str   input file name  ( in folder data/)
        - OUTPUT: 
    '''
    # load model
    t0 = time.time()
    func_tokenizer = TfidfVectorizer(stop_words = 'english').build_tokenizer()
    #model_name = 'v2_2'
    topic_model = load_topic_model(model_name, func_tokenizer )
    t1 = time.time() # time it
    print "finished in  %4.4f %s " %((t1-t0)/60,'loading model\n')

    t0 = t1

    print 'fname: %s' % fname
    #read in input
    cleaned_slack = read_slack_msgs(func_tokenizer, fname=fname)
    #print type(cleaned_slack)
    W, tokenized_slacks2, test_X2 = topic_model.transform_bodytext2topics([cleaned_slack],1)
    print 'topics for slack messages'
    print topic_model.sorted_topics_for_articles(W)
    t1 = time.time() # time it
    print "finished in  %4.4f min %s " %((t1-t0)/60,'topics of slack message\n')

    # load articles
    t0 = t1
    df_articles = read_articles()
    W_articles, tokenized_articles, X_articles = topic_model.transform_bodytext2topics(df_articles.body_text,1)
    #print topic_model.sorted_topics_for_articles(W_articles[:1,:])
    sorted_topics_articles = topic_model.sorted_topics_for_articles(W_articles)
    #print sorted_topics_articles[:1]
    t1 = time.time() # time it
    print '%i articles processed' % df_articles.shape[0]
    print "finished in  %4.4f min for %s " %((t1-t0)/60,'topics of articles\n')
    #test_X2, tokenized_slacks2 = transform_tfidf(vectorizer, [cleaned_slack])
    #test_X2 = test_X2.getA().flatten()

    # summary of input
    top_n = 50
    print "top %i most frequenct features in input %s" % (top_n, fname)
    sorted_feature_indexes = np.argsort(test_X2, axis=1)
    #print test_X2[desc_feature_indexes[:top_n]]
    features = topic_model.vectorizer.get_feature_names()
    i_article = 0
    desc_feature_indexes = sorted_feature_indexes[i_article,:].getA().flatten()[::-1]
    txt_list = []
    for i in desc_feature_indexes[:top_n] :
        txt_list.append('%s (%.2f)' % (features[i], test_X2[i_article,i]))
    print ', '.join(txt_list)

    # cal simmilarity to all articles
    t0 = time.time()
    cosine_similarities = linear_kernel(X_articles, test_X2[i_article,:]).flatten()
    cosin_simi_latent_topics = linear_kernel(W_articles, W[i_article,:]).flatten()
    cosine_similarities_rank = get_rank(cosine_similarities)
    cosin_simi_latent_topics_rank = get_rank(cosin_simi_latent_topics )
    t1 = time.time() # time it
    print "finished in  %4.4f min for %s " %((t1-t0)/60,'calculate cosine similarity\n')
 
    # diagnostic similarity plots
    fig,ax = plt.subplots(1,2,figsize=(10,6))
    ax[0].scatter(cosine_similarities, cosin_simi_latent_topics, alpha = 0.2)
    ax[0].set_title('cosine similarity: tfidf vs. latent topic')

    ax[1].scatter(cosine_similarities_rank, cosin_simi_latent_topics_rank, alpha = 0.2)
    ax[1].set_title('rank cosine similarity: tfidf vs. latent topic')
    #plt.show()
    fig.savefig(fname.replace('.txt','') + '_similarities.png')
    plt.close(fig)

    fig = plt.figure()    
    plt.hist(cosine_similarities, bins=30, alpha = 0.2, label='tfidf')
    plt.hist(cosin_simi_latent_topics, bins=30, alpha = 0.2, label='topics')    
    plt.title('cosine similarity to all articles')
    plt.legend()
    fig.savefig(fname.replace('.txt','') + '_similarity_hist.png')
    #plt.show()

    # recommendations
    print '--------------- recommendations --------------'
    desc_sim_indexes = np.argsort(cosine_similarities)[::-1]
    cosine_similarities[desc_sim_indexes[:20]]
    i_print = 0
    top_k = 5

    for i in desc_sim_indexes[:50]:
        if cosin_simi_latent_topics_rank[i]< 50 and i_print < top_k:
            url = df_articles.iloc[i].url          
            print sorted_topics_articles[i][0:2]
            print cosine_similarities[i], cosine_similarities_rank[i], '***'+ df_articles.iloc[i].title+'***'
            print cosin_simi_latent_topics[i],cosin_simi_latent_topics_rank[i]
            print 
            body_cleaned = ascii_text(df_articles.iloc[i].body_text[:300])
            print body_cleaned
            print 
            i_print = i_print + 1

def make_df_predictions(rating_predict, relevant_items_all, my_rec_engine, df2):
    '''
    user rating based recommendation
    make prediction and the relevant information into dataframe
        - OUTPUT: 
            df_recom: dataframe of predicted rating for top k articles
            relevant_all: evidence for the predition list of relevant items/articles and similarity and user score
                relevant_all index is the same as iloc of df_recom
    '''
    recommed_articles = []
    relevant_all = []

    for i_article in np.argsort(-1.0*rating_predict)[:3]:
        r= {}
        #i_article = 602
        relevant_items = relevant_items_all[i_article]
        sim_vec = my_rec_engine.item_sim_mat[i_article,:].flatten()#, relevant_items] 
        r['score'] = rating_predict[i_article]
        r['title'] = df2.iloc[i_article]['title']
        r['body_text']=df2.iloc[i_article]['body_text']
        r['url'] = df2.iloc[i_article]['url']

        recommed_articles.append(r)
        
        relevent = []
        for i_revelent in relevant_items:
            rele = {}
            rele['rating'] =  rating_vec[i_revelent]
            rele['sim'] = sim_vec[i_revelent]
            rele['title'] = df2.iloc[i_revelent]['title']
            rele['body_text']=df2.iloc[i_revelent]['body_text']
            rele['url'] = df2.iloc[i_revelent]['url']
            rele['i'] = i_revelent
            relevent.append(rele)
        df_relevant = pd.DataFrame(relevent)
        relevant_all.append(df_relevant)
        
    df_recom = pd.DataFrame(recommed_articles)

    print df_recom.head()
    print '-------'
    print relevant_all

    return df_recom, relevant_all 

def process_input(cleaned_slack):
    '''

        - OUTPUT:
            W  1 x n_topics
            test_X2  tfids
            tokenized_slacks2
            txt_list : top_n features
    '''
    #print type(cleaned_slack)
    W, tokenized_slacks2, test_X2 = recommender.topic_model.transform_bodytext2topics([cleaned_slack],1)
    print 'topics for input text'
    #print topic_model.sorted_topics_for_articles(W)
    #t1 = time.time() # time it
    #print "finished in  %4.4f min %s " %((t1-t0)/60,'topics of slack message\n')

    # summary of input
    top_n = 50
    print "top %i most frequenct features in input %s" % (top_n, fname)
    sorted_feature_indexes = np.argsort(test_X2, axis=1)
    #print test_X2[desc_feature_indexes[:top_n]]
    features = topic_model.vectorizer.get_feature_names()
    i_article = 0
    desc_feature_indexes = sorted_feature_indexes[i_article,:].getA().flatten()[::-1]
    txt_list = []
    for i in desc_feature_indexes[:top_n] :
        txt_list.append('%s (%.2f)' % (features[i], test_X2[i_article,i]))
    print ', '.join(txt_list)
    return W, tokenized_slacks2, test_X2, txt_list

def make_diagnostic_plots(fname, cosine_similarities, cosin_simi_latent_topics,\
       cosine_similarities_rank, cosin_simi_latent_topics_rank ):
    '''
    make and save diagnotic plots
        - INPUT: 
            fname:  input file/dataset name
        - OUTPUT: 
    '''
    # diagnostic similarity plots
    fig,ax = plt.subplots(1,2,figsize=(10,6))
    ax[0].scatter(cosine_similarities, cosin_simi_latent_topics, alpha = 0.2)
    ax[0].set_title('cosine similarity: tfidf vs. latent topic')

    ax[1].scatter(cosine_similarities_rank, cosin_simi_latent_topics_rank, alpha = 0.2)
    ax[1].set_title('rank cosine similarity: tfidf vs. latent topic')
    #plt.show()
    fig.savefig(fname.replace('.txt','') + '_similarities.png')
    plt.close(fig)

    fig = plt.figure()    
    plt.hist(cosine_similarities, bins=30, alpha = 0.2, label='tfidf')
    plt.hist(cosin_simi_latent_topics, bins=30, alpha = 0.2, label='topics')    
    plt.title('cosine similarity to all articles')
    plt.legend()
    fig.savefig(fname.replace('.txt','') + '_similarity_hist.png')
    #plt.show()

if __name__ == '__main__':

    func_tokenizer = TfidfVectorizer(stop_words = 'english').build_tokenizer()
    func_stemmer = PorterStemmer()

    if len(sys.argv)>1:
        fname = sys.argv[1]
    else:
        fname = 'test_input.txt' # '06-10.txt'  folder: /Users/joyceduan/documents/git/All-Things-Data-Science/data

    # load model
    t0 = time.time()

    model_name = 'v2_2'
    recommender = Recommender(model_name, func_tokenizer, func_stemmer)

    #read in input
    #cleaned_slack = read_slack_msgs(func_tokenizer, fname=fname)
    cleaned_slack = 'spark hadoop big data'
    W, tokenized_slacks2, test_X2, top_features_list = recommender.process_input(cleaned_slack)
    sorted_topics = recommender.topic_model.sorted_topics_for_articles(W)

    print 'input name: %s' % fname

    # recommendations
    print '--------------- recommendations --------------'
    df_recom = recommender.calculate_recommendations(W, test_X2, fname)

    print sorted_topics

