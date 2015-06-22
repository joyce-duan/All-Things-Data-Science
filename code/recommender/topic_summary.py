
'''
modifed from tech news trending analysis for now
'''

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from urllib3.util import parse_url

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from configobj import ConfigObj
config = ConfigObj('../../allds.config')
allds_home = config.get('allDS_home','/Users/joyceduan/Documents/git/All-Things-Data-Science/')
data_home = allds_home + 'data/'

import sys
sys.path.append(allds_home + 'code/db')
sys.path.append(allds_home +'code/preprocess')
sys.path.append(allds_home  + 'code/model')
sys.path.append(allds_home  + 'code/recommender')
#print allds_home + 'code/db'
#print allds_home  + 'code/recommender'
from recommender import Recommender
from mongo import MyMongo

import time
import pandas as pd
import numpy as np
import datetime

class TopicSummarizer(object):
    def __init__(self, df, W, sorted_topics, topic_names):
        self.df = df
        self.W = W
        self.sorted_topics = sorted_topics
        self.topic_names = topic_names

        self.add_features()
        self.assign_topics()

    def add_features(self):
        '''
        Adds columns to df for csv file output.
            - INPUT:  df['dt']
        '''
        # add columns needed for analysis
        self.df['pub_date'] = pd.to_datetime(self.df['dt'])
        self.df['pub_year'] = self.df.pub_date.map(lambda x: datetime.date.isocalendar(x)[0])
        self.df['pub_month'] = self.df.pub_date.map(lambda x: x.month)

        t = []
        for d in self.df.pub_date:
            try:
               t1 =  datetime.datetime(d.year,d.month,1)
               t.append(t1.strftime("%Y-%m-%d"))
            except:  # d is Null/missing
                #print d  
                t.append('')
        self.df['pub_month_date_str'] = t

    def assign_topics(self):
        #self.df['topic_sorted'] = self.df['topic'].map(lambda x : d[x])
        self.df['topic_sorted'] = [ t[0][0] for t in self.sorted_topics]
        self.df['topic_name'] = [t[0][1] for t in self.sorted_topics]
    
    def articles_num_by_month_dict(self): 
        #articles per month
        dg = self.df[['pub_month_date_str','topic_sorted']].groupby('pub_month_date_str')
        return dg.size().to_dict()

    def articles_per_month(self, outfile):
        d = self.articles_num_by_month_dict()
        f = open(outfile,'w')
        f.write("date,n_articles\n")

        keylist = sorted(d.keys())
        for key in keylist:
            if key == '':
                pass
            else:
                f.write(key+','+str(d[key])+'\n')
        f.close()

    def articles_topic(self, outfile): #pct articles per topic
        n_articles = self.df.shape[0]
        dg = self.df[['topic_sorted','url']].groupby('topic_sorted')
        print dg
        x = sorted(dg.groups.keys())
        y = [ len(dg.groups[i]) for i in x]
        idx_sorted = np.argsort(y)[::-1]

        f = open(outfile,'w')
        f.write("itopic,narticles,i,topic_name\n")
        for i_topic_display, idx  in enumerate(idx_sorted): #,val in enumerate(x):
            f.write(str(i_topic_display)+','+str(y[idx])+','+str(idx)+ ','+ self.topic_names[idx]+'\n')
        f.close()


    def articles_bytopic_permonth(self, outfile): #example articles per topic per week
        '''
        written to csv file which is used by web app
        i_sorted_topic,n_articles,fraction,month_date
        '''
        n_articles = self.df.shape[0]
        dg = self.df[['topic_sorted','url']].groupby('topic_sorted')
        x = sorted(dg.groups.keys())
        y = [ len(dg.groups[i]) for i in x]
        idx_sorted = np.argsort(y)[::-1]
        df_monthly = self.articles_num_by_month_dict()
        print type(df_monthly)
        print zip(df_monthly)[:3]

        #columns to be written in csv file
        cond_has_date = self.df['pub_month_date_str'] != ''
        data = []
        for i_topic_display, i_topic_orig in enumerate(idx_sorted):
            cond = self.df['topic_sorted'] == i_topic_orig
            d_thistopic = self.df[cond & cond_has_date][['pub_month_date_str','topic_sorted']]\
            .groupby('pub_month_date_str').size().to_dict()
            d_thistopic_all = [ (i_topic_display, m, d_thistopic.get(m,0), self.topic_names[i_topic_orig]) \
            for m in df_monthly]
            data.extend(d_thistopic_all)
        df = pd.DataFrame(data)
        print 'df:', df.shape
        print df.head(5)
        df.columns = ['topic','month_date','n_articles','topicname']
        df_subset = df[['month_date','n_articles']]
        fraction = [  100.0 * df.iloc[i]['n_articles'] / df_monthly.get(df.iloc[i]['month_date']) \
            for i in xrange(df.shape[0])]
        #df['fraction'] = df_subset.apply(lambda x: 100.0 * x[1]/df_monthly.get(x[0],100000))
        df['fraction'] = fraction
        df = df.sort(['topic','month_date'])
        df2 = df[df['month_date'] != '']
        df2.to_csv(outfile, index=False)

    def write_data(self, outfile1, outfile2, outfile3):
        #self.articles_topic(outfile1)
        #self.articles_per_month(outfile2)
        self.articles_bytopic_permonth(outfile3)

if __name__ == '__main__':
    '''
    use the trained model v2_2 to get topics for all articles 
    '''
    model_name = 'v2_2'
    func_tokenizer = TfidfVectorizer(stop_words = 'english').build_tokenizer()
    func_stemmer = PorterStemmer()

    # load model
    t0 = time.time()
    recommender = Recommender(model_name, func_tokenizer, func_stemmer)

    article_dict,  article_dt =  MyMongo().get_article_attri()
    recommender.df_articles['dt'] = recommender.df_articles['url'].map(lambda x: article_dt.get(x, None))

    topic_summary = TopicSummarizer(recommender.df_articles, \
        recommender.W_articles, recommender.sorted_topics_articles, recommender.get_topic_names() )

    topic_summary.write_data('../../webapp/static/articles_per_topic.csv',
                        '../../webapp/static/articles_per_month.csv',
                        '../../webapp/static/data.csv')

