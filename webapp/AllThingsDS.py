from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import sys

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from flask import Flask
from flask import render_template
from flask import request
import datetime
import pandas as pd
#from build_model import read_model, transform_tfidf, model_pkl_fname, vectorizer_pkl_fname 
import pickle
import time

sys.path.append('../code/model')
sys.path.append('../code/preprocess')
sys.path.append('../code/recommender')

from topic_modeling import TopicModel, read_articles
from ArticleProceser import   ascii_text
from recommender import Recommender, get_rank, make_diagnostic_plots, pre_clean_text

from configobj import ConfigObj
config = ConfigObj('../allds.config')
allds_home = config['allDS_home']
data_home = allds_home + 'data/'
dummy_result_pkl = 'dummy_results.pkl'

'''
ToDo: ????
                1. unicode character: copy and paste did not work
2. make the form larger
3. what happens in '\n' was entered in the form??


input:  


'''

app = Flask(__name__)

'''
to start this
python the_app.py
'''

my_title = '''<html>
        <head>
            <meta charset="utf-8">
            <title>Article Recommender </title>
        </head>
        '''

# OUR HOME PAGE
#============================================
@app.route('/')
#  http://localhost:6969/
def index():
    p_acronym = 'All Things DS'
    pname_full = 'All Things Data Science '
    return render_template('index.html', pname_full = pname_full, p_acronym = p_acronym)

# Form page to submit text
#============================================
# create page with a form on it
@app.route('/form')
#  http://localhost:6969/form
def cool_form():
    action = '/recommender_by_content'
    return render_template('form.html', action = action)

# My  app
#==============================================
# create the page the form goes to

@app.route('/recommender_by_content', methods=['POST'] )


def recommender_by_content():
    username = 'Joyce'
    data = request.form['user_input']
    input_name = 'your_input'
    #results_dict = get_recom_from_input(username, input_name, data)
    # for testing
    results_dict = dummy_recom(username, input_name, data)
    return render_template('recom_by_content.html', **results_dict )

def get_recom_from_input(username, input_name, data):
    model_name = 'v2_2'
    fname = input_name

    #df_recom =  pd.DataFrame(dict(body_text=[1,2,3]), index=['a','b','c'])
    relevant_all = None

    func_tokenizer = TfidfVectorizer(stop_words = 'english').build_tokenizer()
    func_stemmer = PorterStemmer()

    # load model
    t0 = time.time()
    recommender = Recommender(model_name, func_tokenizer, func_stemmer)

    #read in input
    cleaned_slack = pre_clean_text(func_tokenizer, data)

    W, tokenized_slacks2, test_X2, top_features_list = recommender.process_input(cleaned_slack)
    sorted_topics = recommender.topic_model.sorted_topics_for_articles(W)

    print 'input name: %s' % input_name
    # recommendations
    print '--------------- recommendations --------------'
    df_recom = recommender.calculate_recommendations(W, test_X2, fname)
    print sorted_topics
    t1 = time.time()
    print "finished in  %4.4f min %s " %((t1-t0)/60,'finished all processing\n')

    results_dict = dict(username = username,\
    input_data = data, input_name = input_name, sorted_topics = sorted_topics, \
    idx = range(df_recom.shape[0]), \
        df_recom = df_recom, relevant_all=relevant_all)
    with open(dummy_result_pkl,'w') as out_fh:  
        pickle.dump(results_dict, out_fh)
    return results_dict

def dummy_recom(username, input_name, data):
    with open(dummy_result_pkl, 'r') as in_fh:
        results_dict = pickle.load(in_fh)
    return results_dict

def recommender_by_contentold():
    username = 'Joyce'
    model_name = 'v2_2'

    data = request.form['user_input']
    input_name = 'your_input'

    fname = input_name

    #df_recom =  pd.DataFrame(dict(body_text=[1,2,3]), index=['a','b','c'])
    relevant_all = None

    func_tokenizer = TfidfVectorizer(stop_words = 'english').build_tokenizer()
    func_stemmer = PorterStemmer()

    # load model
    t0 = time.time()
    recommender = Recommender(model_name, func_tokenizer, func_stemmer)

    #read in input
    cleaned_slack = pre_clean_text(func_tokenizer, data)

    W, tokenized_slacks2, test_X2, top_features_list = recommender.process_input(cleaned_slack)
    sorted_topics = recommender.topic_model.sorted_topics_for_articles(W)

    print 'input name: %s' % input_name
    # recommendations
    print '--------------- recommendations --------------'
    df_recom = recommender.calculate_recommendations(W, test_X2, fname)
    print sorted_topics
    t1 = time.time()
    print "finished in  %4.4f min %s " %((t1-t0)/60,'finished all processing\n')

    return render_template('recom_by_content.html', username = username,\
        input_data = data, input_name = input_name, sorted_topics = sorted_topics, \
        idx = range(df_recom.shape[0]), \
    df_recom = df_recom, relevant_all=relevant_all)

@app.route('/recommender_by_ratings')

def recommender_by_ratings():
    username = 'Joyce'
    with open('../code/eda/df_recom.pkl','r') as f:
        df_recom = pickle.load(f)
    with open('../code/eda/relevant_all.pkl','r') as f:
        relevant_all = pickle.load(f)
    return render_template('recom.html', username = username, idx = range(df_recom.shape[0]), \
        df_recom = df_recom, relevant_all=relevant_all)

@app.rout('/topics_trends')
def topics_trends():
    """
    explore topic trends
    """
    return flask.render_template("topic_trends.html")
    #return flask.render_template("basic-carousel.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
