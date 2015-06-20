
'''
depreciated.  use LDA_topics
'''
import theano
from gensim import corpora, models, similarities 
import numpy as np
# tutorial: https://radimrehurek.com/gensim/tut1.html


n_top_articles = 5
n_top_terms = 15

class BaseTopics(object):
    def __init__(self, model_name = 'tmp'):
		'''
		initialize NMF topic modeler
		'''
		self.model_name = model_name
		self.lda_model = None
		self.dictionary = None
		self.W = None
		self.H = None
		self.df = None
		self.best_topics_per_article = None

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

    def assign_cluster(self, min_w = 0):
    	'''
    	assign the best cluster to each article
    		- INPUT:  self.best_topics_per_article
    		- OUTPUT: self.clusters  (-1:  no topic found with weighth > min_w)
    	'''
		n_articles = self.best_topics_per_article.shape[0]
		clusters = -1 * np.ones(n_articles)
		for i_article in xrange(n_articles):
		    if self.best_topics_per_article[i_article,1] > min_w:
		        clusters[i_article] = self.best_topics_per_article[i_article,0]
		self.clusters  = np.array([int(c) for c in clusters])

	def plot_hist_weight_best_topic_per_article(self):   
		'''
		plot historgram of weight of the best topic for all article
			- INPUT: see self.get_best_topic_per_article()
			- OUTPUT: 
		'''
		self.get_best_topic_per_article()
		fig = plt.figure(figsize=(8,4))
		plt.hist(self.best_topics_per_article[:,1], bins =50)
		#plt.show()
		fig.savefig(self.model_name + '_hist_best_topic_per_article.png',bbox_inches = 'tight')
		plt.close(fig)

    def print_topic_results_html(self):
    	'''
    	print the results of topic modeling in html file: for each topic, n_top_terms  and n_top_articles 
    		- INPUT: self.model_name, self.topic_terms, self.W, self.df
    		- OUTPUT: filename model_name+'.html'
    	'''
        model_name = self.model_name
        W_t = self.W.T
        topic_terms = self.topic_terms
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
                    out_fh.write( '<a href=' + url +' > %s </a>  <br>\n' % title_this_cleaned  )
                    
                    body_text_str = df2.iloc[idx]['body_text'][:400]
                    #body_text_str = body_text_str.encode('utf8')
                    body_cleaned = ascii_text(body_text_str)
                    out_fh.write( body_cleaned +' \n<br>\n')
                    out_fh.write('\n') 
            out_fh.write('</body>\n</html>\n')

	    def cal_centroid(self):
	    	'''
	    	calculate the centroid for each cluster
	    		- INPUT: self.cluster, self.X2 (tifidf)
	    		- OUTPUT: self.centroids
	    	'''
	        n_clusters = np.max(self.clusters)
	        centroids = []
	        for i in xrange(n_clusters + 1):
	            cond = self.clusters == i
	            arr = self.X2[cond]
	            length = arr.shape[0]
	            c = np.mean(arr, axis = 0)
	            centroids.append(c)
	        self.centroids = np.array(centroids)

	    def plot_hist_d_to_centroid(self, min_w = 0):
	    	'''
	    	plot histogram of distance to centroid, overall vs. per cluster
	    		- INPUT: self.X2
	    	'''
	        self.assign_cluster(min_w)
	        self.cal_centroid()
	        n_clusters = np.max(self.clusters)
	        #fig = plt.figure(figsize=(20,8))


	        #multiple plot, subplots
	        ncols = 3
	        nrows = (n_clusters + 1) // ncols + ( ((n_clusters+1) % ncols)>0 )
	        fig, ax = plt.subplots(nrows,ncols,figsize=(30,10))   #subplot preferred way
	        axs = ax.flatten()

	        centroid_overall = np.mean(self.X2, axis = 0)
	        sim = linear_kernel(centroid_overall, self.X2)
	        max_sim = np.max(sim)
	        min_sim = np.min(sim)

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

#create a Gensim dictionary from the texts
class LDATopics(BaseTopics):

    def __init__(self, model_name):
		'''
		initialize NMF topic modeler
		'''
		self.model_name = model_name
		self.lda_model = None
		self.dictionary = None

	def set_models(self, num_topics=20, num_words=15, kw_dict):
		'''
		hack to set the models after running lda
		'''
		for k, v in kw_dict:
			if k == 'data':
				self.df = v
				self.n_docs = self.df.shape[0]
			if k == 'lda_model':
				self.lda_model = v
				self.lda_topic_matrix = lda_model.show_topics(formatted=False,num_topics=num_topics, num_words=num_words )
				self.n_terms = num_words
				self.n_top_terms = num_words
				self.H = self.set_H(self.lda_topic_matrix)
			if k == 'dictionary':
				self.dictionary = v
			#if k == 'lda_topics_matrix':
			#	self.lda_topic_matrix = topics_matrix

	def set_H(self, lda_topic_matrix):
		self.H = np.zeros((self.n_topics, self.n_terms))
		self.word_features = []
		self.word_feature_map = {}
		topic_words = lda_topics_matrix[:,:,1]
		weights = topics_matrix[:,:,0]
		for i_topic , words in enumerate(topic_words):
			for j_word in xrange(len(words)):
				word = words[j_word]
				if word in self.word_feature_map:
					i_feature = self.word_feature_map[word]
				else:
					self.word_features.append(word)
					i_feature = len(self.word_features) - 1
					self.word_feature_map[word] = i_feature
			self.H[i_topic, i_feature] = float(weights[i_topic, j_word])

    def set_W(self):
     	'''
     	calculate W doc x topic (weight of article for each topic)
     	'''
     	self.W = np.zeros((self.n_docs, self.n_topics))
     	for i_article in xrange(df.shape[0]):
     		w_this = self.lda_model[self.dictionary.doc2bow(df.tokens.values[i_article])]
     		for i_topic, w_topic in w_this:
     			self.W[i_article,i_topic] = w_topic

	def set_top_topics_terms(self):
	    '''
	    get the k_top_words for each topic
	        - INPUT:  self.H, self.word_features
	        - OUTPUT: self.topic_terms  list of dictionary 
	                    each element: for article i, the dictonary of the weigths for top terms topics_dicts   
	                    	word:weight  (k_top_words most important terms for each topic)
	    '''
	    k_top_words = self.n_top_terms
	    self.topic_terms = []
	    n_topics = H.shape[0]

	    for i in xrange(n_topics):
	        k_term, v = zip(*sorted(zip(self.word_features, H[i]),
	                           key=lambda x: x[1])[:-k_top_words:-1])
	        val_arr = np.array(v)
	        #norms = val_arr / np.sum(val_arr)
	        #topic_terms.append(dict(zip(k, norms * 100)))
	        self.topic_terms.append(dict(zip(k_term, val_arr * 100)))


def run_lda(texts, min_df = 5, 	num_topics = 20):
# texts: list of list of words
	#num_topics = 20
	#print texts
	dictionary = corpora.Dictionary(texts)

	#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
	dictionary.filter_extremes(no_below=min_df, no_above=0.99)

	#convert the dictionary to a bag of words corpus for reference
	corpus = [dictionary.doc2bow(text) for text in texts]
	#print corpus

	lda_model = models.LdaModel(corpus, num_topics=num_topics, 
	                            id2word=dictionary, 
	                            update_every=5, 
	                            chunksize=10000, 
	                            passes=100)
	print '%d topics ' % num_topics
	print 'show topics:\n'
	print lda_model.show_topics()

	topics_matrix = lda_model.show_topics(formatted=False, num_words=20)
	topics_matrix = np.array(topics_matrix)

	
	topic_words = topics_matrix[:,:,1]
	weights = topics_matrix[:,:,0]
	print 
	for i , words in enumerate(topic_words):
		'''
		float_w = []
		for w in weights[i]:
			try:
				w = '%.2f' % float(w)
			else:
				w = ''
		'''
		print 'topic #%d:' % i + ', '. join ([str(words[j]) + '( '+ weights[i,j][:4] + ')' for j in xrange(len(words))])
	    	
	return lda_model, topics_matrix, dictionary



