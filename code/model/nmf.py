from sklearn.decomposition import NMF 
import numpy as np
import pickle

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

def assign_cluster(best_topics_for_articles, min_w = 0):
    n_articles = best_topics_for_articles.shape[0]
    clusters = -1 * np.ones(n_articles)
    for i_article in xrange(n_articles):
        if best_topics_for_articles[i_article,1] > min_w:
            clusters[i_article] = best_topics_for_articles[i_article,0]
    return clusters
        
def pickle_topic_model(model_name,df,clean_articles,transform_tfidf, vectorizer,nmf_model,W, H, note=''):
    '''
        df2: dataframe with all data
        clean_article: function to clean text
        transform_tfidf: 
        vectorizer: fitted vectorizer 
        W: mapping of documents to topics
        H: mapping of terms to topics
        nmf: nmf model
    '''
    with open(model_name + '_readme.txt','w') as out_fh:
        out_fh.write(note)
    with open(model_name + '_df.pkl','w') as out_fh:  
        pickle.dump(df, out_fh)
    with open(model_name + '_func_clean.pkl','w') as out_fh: 
        pickle.dump(clean_articles,out_fh)
    with open(model_name + '_func_transform.pkl','w') as out_fh: 
        pickle.dump(transform_tfidf,out_fh)
    with open(model_name + '_vectorizer.pkl','w') as out_fh: 
        pickle.dump(vectorizer,out_fh)
    with open(model_name + '_nmf_fitted_model.pkl','w') as out_fh: 
        pickle.dump(nmf_model,out_fh)
    with open(model_name + '_W.pkl','w') as out_fh: 
        pickle.dump(W,out_fh)        
    with open(model_name + '_H.pkl','w') as out_fh: 
        pickle.dump(H,out_fh)             

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




