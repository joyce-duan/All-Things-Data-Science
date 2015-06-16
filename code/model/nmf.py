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




