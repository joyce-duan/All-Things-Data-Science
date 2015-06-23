
# coding: utf-8

# In[161]:

from pymongo import MongoClient
import string
import json 
import pickle as pkl
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from urllib3.util import parse_url
import re
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel

get_ipython().magic(u'matplotlib inline')

pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
plt.rcParams['figure.figsize'] = (15, 5)


# In[111]:

import sys
sys.path.append('../db')
from my_mongo import MyMongo


# ## read in articles from mongo db, clean, merge, tfidf

# In[3]:

def clean_articles(cur_articles, articles_cleaned):
    articles = list(cur_articles)
    print '%d articles found' % len(articles)
    counter = 0
    for a in articles:
    #for a in cur_articles:
        url = a['url']
        body_text = a['body_text']
        l = len(body_text)
        if l > articles_cleaned.get(url,[0])[0]:
            articles_cleaned[url] = [len(body_text),body_text]


# In[4]:

my_mongo = MyMongo()

articles_cleaned = {}
print '%d unique articles ' % len(articles_cleaned)

cur_articles = my_mongo.get_article_body_text(testing=0)
clean_articles(cur_articles, articles_cleaned)
print '%d unique articles ' % len(articles_cleaned)


# In[5]:

my_mongo_nytimes = MyMongo(dbname  = 'nytimes')
cur_articles = my_mongo_nytimes.get_article_body_text(testing=0)
clean_articles(cur_articles, articles_cleaned)
print '%d unique articles ' % len(articles_cleaned)


# In[6]:

df = pd.DataFrame([{'url':k,'body_text':v[1]} for k, v in articles_cleaned.items()])


# In[7]:

df.head(3)


# In[297]:


article_dict =  MyMongo().get_article_attri()
article_dict2 =  MyMongo(dbname  = 'nytimes').get_article_attri()
article_dict_all = dict(article_dict, **article_dict2)


# In[303]:

print article_dict2.items()[0][1].encode('ascii', 'ignore')
print article_dict2.items()[0][1].encode('utf8', 'ignore')


# In[144]:

print df2.shape
print pd.value_counts(df2['uri'])[:8]
print 1.0 *pd.value_counts(df2['uri'])[:8].sum()/df2.shape[0]
print df2.uri.nunique()


# In[128]:

url = 'http://example.com/random/folder/path.html'
print parse_url(url)


# In[134]:

parse_url('http://google.com/mail/').host

#Url(scheme='http', host='google.com', port=None, path='/')


# In[172]:

df2['uri'] = df2['url'].map(lambda x: parse_url(x).host)
#df2.head(2)
pd.value_counts(df2['uri'])[:10]



# In[8]:

from sklearn.decomposition import NMF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[9]:

def tokenize(doc):
        '''
        Tokenize and stem/lemmatize the document.
            - INPUT: string
            - OUTPUT: list of strings

        '''
        snowball = SnowballStemmer('english')
        #print type(doc), type([snowball.stem(word) for word in word_tokenize(doc.lower())])
        #print len(doc), len([snowball.stem(word) for word in word_tokenize(doc.lower())])
        words = []
        for word in word_tokenize(doc.strip().lower()):
            try:
                # .decode('utf-8'
                t = word.encode('ascii', 'ignore')
                words.append(t)                
            except:
                 t = ''
        return [snowball.stem(word) for word in words]

def fit_tfidf(docs):
    '''
        - INPUT: X. list of string
        - OUPUT: list of list
    '''
    tokenized_articles = [tokenize(doc) for doc in docs]
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 5000)
    documents = [' '.join(article) for article in tokenized_articles]
    vectorizer =  vectorizer.fit(documents)

    vectors_doc = vectorizer.transform(documents).todense()
    return vectorizer, vectors_doc, tokenized_articles

def transform_tfidf(vectorizer, docs):
    '''
    transform text to tfidf
        - INPUT:  list of string :  (string / text)
        - OUTOUT: list of float
    '''
    tokenized_articles = [tokenize(doc) for doc in docs]
    documents = [' '.join(article) for article in tokenized_articles]
    return vectorizer.transform(documents).todense(), tokenized_articles


# In[10]:

def run_nmf(X, n_topics=20, **kwargs):
    '''
    NMF on the TF-IDF feature matrix to create a topic model.
        - INPUT:  
            X: 2d numpy array 
            n_topics: int 
            **kwargs: NMF parameters
        - OUTPUT: 
             W (Article-Topic matrix):2d numpy array
            H (Topic-Term matrix): 2d numpy array 
    '''
    nmf = NMF(n_components=n_topics, **kwargs)
    W = nmf.fit_transform(X)
    H = nmf.components_
    return W, H, nmf


# In[27]:

txt_list = df.body_text
vectorizer, vectorized_X, tokenized_articles = fit_tfidf(txt_list)
df['len'] = [len(t) for t in tokenized_articles]
df['tokens'] = tokenized_articles


# ## Run NMF for topic modeling

# In[12]:

#vectorized_X, tokenized_articles = transform_tfidf(vectorizer, txt_list)


# In[42]:

cond = df['len'] >= 50  # only keeps articles with at least 50 words
df2 = df[cond]
irows = [i for i in xrange(vectorized_X.shape[0]) if cond[i]]
X2 =vectorized_X[np.ix_(irows)]
url_list2 = df2['url']


# In[352]:

idx_df = list(df2.index)


# In[304]:

df2['title'] = df2['url'].map(lambda x: article_dict_all.get(x,'Unknown'))


# In[45]:

W, H = run_nmf(X2 )


# In[75]:

print 'w ', W.shape, 'h', H.shape


# In[83]:

W_t = (W.T)
print W.shape, W_t.shape
print type(W_t)
print len(W_t[0,]), W_t[0]


# In[114]:

W_t = W.T
n_top_articles = 5
n_top_terms = 5
for topic_idx, article_w in enumerate(W_t):
    print("Topic #%d:" % topic_idx)
    
    #    for i, topic in enumerate(topic_terms):
    terms = topic_terms[topic_idx]
    l = sorted(terms.items(), key=lambda x: x[1])[::-1]
    txt_list = []
    for item in l[:n_top_terms]:
        txt_list.append('%s (%.2f)' % (item[0], item[1]))
    print 'top terms: ',', '.join(txt_list)
    print '-------'
    #print article_w.shape
    idx_article_topn = article_w.argsort()[:-n_top_articles - 1:-1]
    for i, idx in enumerate(idx_article_topn):
        url = df2.iloc[idx]['url']
        print '**** ', i,'. (%.2f))'% article_w[idx], article_dict_all.get(url,'')
        print url
        print df2.iloc[idx]['body_text'][:400]
        print 
    print 


# #### get the most relevant topic for each article 

# In[239]:

best_topics_for_articles = []
for i_article, topic_w in enumerate(W):
    i_best_topic = np.argmax(topic_w)
    best_topics_for_articles.append([i_best_topic, topic_w[i_best_topic]])
best_topics_for_articles = np.array(best_topics_for_articles)
fig = plt.figure(figsize=(8,4))
plt.hist(best_topics_for_articles[:,1], bins =50)
plt.show()


# In[244]:

cond_topic = best_topics_for_articles[:,1] > 0.33
pd.value_counts(best_topics_for_articles[cond_topic,0])


# In[105]:

n = W_t.shape[0]
ncols = 4
nrows = n // ncols + ( (n % ncols)>0 )
fig, ax = plt.subplots(nrows, ncols, figsize = (20,14))
axs = ax.flatten()
for i, article_w in enumerate(W_t):
    axs[i].hist(article_w, bins =20)
    axs[i].set_title(str(i))
plt.show()
    


# In[120]:

p =  [100,99,98,97,96,95,90,85,80,75,50]
print zip(p, np.percentile(W_t.flatten(), p))
fig = plt.figure(figsize=(8,5))
plt.hist(W_t.flatten(), bins=50)
plt.show()


# In[72]:

def get_top_topics_terms(vectorizer, H, k_top_words=50):
    '''
    get the k_top_words for each topic
        - INPUT:  
        - OUTPUT: dict - topics_dicts (most important terms for each topic)
    '''
    topic_terms = []
    n_topics = H.shape[0]

    for i in xrange(n_topics):
        k, v = zip(*sorted(zip(vectorizer.get_feature_names(), H[i]),
                           key=lambda x: x[1])[:-k_top_words:-1])
        val_arr = np.array(v)
        norms = val_arr / np.sum(val_arr)
        topic_terms.append(dict(zip(k, norms * 100)))
    return topic_terms


def print_topic_terms(topic_terms):
    for i, topic in enumerate(topic_terms):
        l = sorted(topic.items(), key=lambda x: x[1])[::-1]
        print "Topic #" + str(i)
        txt_list = []
        for item in l:
            txt_list.append('%s (%.2f)' % (item[0], item[1]))
        print '.'.join(txt_list)


# ## article recommendations based on slack chats

# In[149]:

get_ipython().system(u' ls ../../data')


# In[158]:

data_file = '../../data/slack.txt'
with open(data_file, 'r') as in_fh:
    data = in_fh.read()


# In[ ]:

data2 = data.decode(('utf8'))


# In[ ]:




# In[163]:

#data2 = data2.encode('ascii', 'ignore')
test_X, tokenized_slacks = transform_tfidf(vectorizer, [data2])
test_X = test_X.getA().flatten()


# In[223]:

stop_words_fix = [ u'text/html', 'pm', 'http' 'io','com', 'github', 'www' ,'config', 'html', 'js' ,'https', 'join', 'repo','chris',u'http']
cleaned_slack = []
for w in tokenized_slacks[0]:
    if  re.search('[a-zA-Z]',w) and '//' not in w and 'sf_ds' not in w:
        if w not in stop_words_fix:
            cleaned_slack.append(w)
print cleaned_slack


# In[224]:

cleaned_slack = ' '.join(cleaned_slack)


# In[225]:

test_X2, tokenized_slacks2 = transform_tfidf(vectorizer, [cleaned_slack])
test_X2 = test_X2.getA().flatten()


# #### most common words in 06-04 slack

# In[229]:

top_n = 100
desc_feature_indexes = list(np.argsort(test_X2)[::-1])
#print test_X2[desc_feature_indexes[:top_n]]
features = vectorizer.get_feature_names()
txt_list = []
for i in desc_feature_indexes[:top_n] :
    txt_list.append('%s (%.2f)' % (features[i], test_X2[i]))
print ', '.join(txt_list)


# #### most relevant articles

# In[231]:

cosine_similarities = linear_kernel(vectorized_X, test_X).flatten()


# In[232]:

plt.hist(cosine_similarities,bins=30)
plt.show()


# In[234]:

desc_sim_indexes = np.argsort(cosine_similarities)[::-1]
cosine_similarities[desc_sim_indexes[:5]]
for i in desc_sim_indexes[:5]:
    url = df2.iloc[i].url
    print cosine_similarities[i], '***'+article_dict_all.get(url,' ')+'***'
    print url
    print df2.iloc[i].body_text[:300]
    print 


# ### recommend articles based on known user rating

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[160]:

tokenized_slacks[0][:10]


# In[ ]:

n_topics = 20
#n_top_words = 5
t0 = time.time()
print("Fitting the NMF model with") # n_samples=%d and n_features=%d..."
    #  % ())
nmf = NMF(n_components=n_topics, random_state=1).fit(X2)


# In[68]:

print("done in %0.3fs." % (time.time() - t0))
n_top_words = 20
feature_names = vectorizer.get_feature_names()

for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print


# In[73]:

topic_terms = get_top_topics_terms(vectorizer, H)
print_topic_terms(topic_terms)


# In[35]:

t[np.ix_(irows, icols)]
X2 = vectorized_X[list(cond)]
len(vectorized_X), len(X2)


# In[14]:

df['len'].describe()


# In[19]:

df['len_capped'] = df['len'].map(lambda x: x if x < 5000 else 5000 )


# In[23]:

df[['len','len_capped']].hist( bins = 50)
plt.show()


# In[26]:

cuts_min = [30,50,100]
for cut in cuts_min:
    print cut, df[df['len']<cut].shape
cuts_max = [4000,5000]
for cut in cuts_max:
    print cut, df[df['len']>cut].shape


# In[ ]:

len_min = 50


# In[15]:

'''
regex = re.compile('<.+?>|[^a-zA-Z]')
porter = PorterStemmer()
stop_words_lst = stop_words = stopwords.words('english')

def tokenize(text):
    clean_txt = regex.sub(' ', text) #remove any characters that are not english characters.
    clean_txt = clean_txt.lower()
    #  punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])???
    stemmed =  [porter.stem(w.strip(string.punctuation)) for w in re.split(' +', clean_txt) \
            if w not in stop_words_lst]
    return [w for w in stemmed if w != '']  
'''


# In[293]:

df2.head()


# In[254]:

url = 
df2[df2.url == url]


# In[257]:

good_urls = ['http://www.bloomberg.com/news/articles/2015-06-04/help-wanted-black-belts-in-data', 
        'http://bits.blogs.nytimes.com/2015/04/28/less-noise-but-more-money-in-data-science/'
       , 'http://bits.blogs.nytimes.com/2013/06/19/sizing-up-big-data-broadening-beyond-the-internet/'
       ]
meh_urls = ['http://bits.blogs.nytimes.com/2013/04/11/education-life-teaching-the-techies/' # too short, no content
, 'http://www.nytimes.com/2013/04/14/education/edlife/universities-offer-courses-in-a-hot-new-field-data-science.html'  
]


# In[363]:

def get_rating(good_urls, meh_urls, df2):
    '''
    generate rating_content from list of good and meh urls
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


# In[362]:

print df2.iloc[220]['url']


# In[279]:

sys.path.append('../recommender')
from ItemItem import *


# In[392]:

rating_content = get_rating(good_urls, meh_urls, df2)  
rating_vec = np.zeros(df2.shape[0])
for row in rating_content:
    rating_vec[row[0]] =  row[1]
print rating_content
print [i for i in xrange(len(rating_vec)) if rating_vec[i]>0]


# In[263]:

print X2.shape, df2.shape


# In[339]:

def format_article(i, df2):
    n_chars = 500
    s = '***** %s ****' % df2.iloc[i]['title']
    s = s + '\n' + df2.iloc[i]['url']
    s = s + '\n\n' + df2.iloc[i]['body_text'][:n_chars]
    return s

def print_articles(idx, df2, scores):
    '''
    print list of articles and associated scores
    '''
    for i in idx:
        str_article = format_article(i, df2)
        print '%d %.2f %s ' % (i, scores[i], str_article)
        print 

def print_articles_2scores(idx, df2, sim, rating):
    '''
    print list of articles and associated scores
    '''
    for i in idx:
        str_article = format_article(i, df2)
        print '%d sim: %.2f rating: %.2f %s ' % (i, sim[i],rating[i], str_article)
        print 

        


# In[268]:

my_rec_engine = ItemItemRec(neighborhood_size=75)
my_rec_engine.fit_item_features(X2)


# In[393]:

rating_predict, relevant_items_all = my_rec_engine.pred_one_user_from_rating (rating_content, report_run_time=False)
    


# In[ ]:

rating_weighted = []
for i_article in xrange(len(rating_predict)):
    sim_vec = [my_rec_engine.item_sim_mat[i_article,j] 
               for j in  relevant_items_all[i_article]] 
    sim_max = np.max(sim_vec)
    rating_weighted.append(rating_predict[i_article] * sim_avg)
rating_weighted = np.array(rating_weighted)
rating_weighted[np.argsort(-1.0*rating_weighted)[:5]]


# In[394]:

for i_article in np.argsort(-1.0*rating_predict)[:3]:
    #i_article = 602
    relevant_items = relevant_items_all[i_article]
    #ratings_sim = [rating_dict[i] for i in relevant_items]
    sim_vec = my_rec_engine.item_sim_mat[i_article,:].flatten()#, relevant_items] 
    print 'recommend:'
    print_articles([i_article],df2, rating_predict)
    print '---- based on:'
    print_articles_2scores(relevant_items, df2, sim_vec, rating_vec)


# In[397]:

pkl.dump(df_recom,open('df_recom.pkl','w'))
pkl.dump(relevant_all,open('relevant_all.pkl','w'))


# In[380]:

with open('relevant_all.pkl','r') as f:
    test = pkl.load(f)
type(test)


# In[390]:

df = test[0]
print type(df)
for index, row in df.iterrows():
    print row


# In[396]:

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


# In[307]:

n_top = 10
idx_top = np.argsort(-1.0*rating_predict)[:n_top]
print_articles(idx_top, df2, rating_predict)


# In[313]:

print sum(rating_predict>3.9999), 1.0 * sum(rating_predict>3.99)/len(rating_predict)


# In[311]:

len(rating_predict)


# In[ ]:



