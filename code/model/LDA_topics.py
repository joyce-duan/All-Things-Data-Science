
# tutorial: https://radimrehurek.com/gensim/tut1.html
'''
from LdaTopics import BaseTopics, LDATopics, run_lda

run_lda:  90 minutes for 1200 articles 

'''

import theano
from gensim import corpora, models, similarities 
import numpy as np
import sets
import sys
import matplotlib.pyplot as plt
sys.path.append('../preprocess')
from ArticleProceser import ascii_text
from sklearn.metrics.pairwise import linear_kernel

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

	def set_top_topics_terms(self):
		'''
		get the k_top_words for each topic
			- INPUT:  self.H, self.top_word_features
			- OUTPUT: self.topic_terms  list of dictionary 
				each element: for article i, the dictonary of the weigths for top terms topics_dicts   
				word:weight  (k_top_words most important terms for each topic)
		'''
		print 'set_top_topics_terms'
		k_top_words = self.n_top_terms
		self.topic_terms = []
		n_topics = self.H.shape[0]

		for i in xrange(n_topics):

			v = self.H[i].flatten()
			#print type(self.H), type(v), v.shape, type(self.top_word_features)
			idx_sorted_topk= np.argsort(-1.0 * v)[:k_top_words]

			#print idx_sorted_topk
			#print self.top_word_features[idx_sorted_topk]
			#print v[idx_sorted_topk]

			#k_term, v = zip(*sorted(zip(self.top_word_features, self.H[i].flatten()),\
			#                   key=lambda x: x[1])[:-k_top_words:-1])
			#val_arr = np.array(v)
			#norms = val_arr / np.sum(val_arr)
			#topic_terms.append(dict(zip(k, norms * 100)))
			terms_this_topic = zip(self.top_word_features[idx_sorted_topk],v[idx_sorted_topk]*100 )
			#print dict(terms_this_topic)
			self.topic_terms.append(dict(terms_this_topic))


	def get_summary_stats(self):
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
		cond_topic = selfbest_topics_per_article[:,1] > cutoff
		print pd.value_counts(self.best_topics_per_articles[cond_topic,0])
		return pct_val[1]

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

		print 'print_topic_results_html'

		#model_name, n_top_articles, W_t,topic_terms, n_top_terms 
		with open(model_name+'.html','w') as out_fh:
			out_fh.write('<html>\n<body>')
			for topic_idx, article_w in enumerate(W_t):
				out_fh.write("<h1>Topic #%d: </h1>\n" % topic_idx)

				#    for i, topic in enumerate(topic_terms):
				terms = topic_terms[topic_idx]


				l = sorted(terms.items(), key=lambda x: x[1])[::-1]

				print "terms: ", terms
				print "l: ", l

				txt_list = []
				for item in l[:n_top_terms]:
					txt_list.append('%s (%.4f)' % (item[0], item[1]))
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

	def set_X2(self, X2):
		'''
		hack to set the tfidf matrix
		'''
		self.X2 = X2

	def cal_centroid(self):
		'''
		calculate the centroid for each cluster
		- INPUT: self.clusters, self.X2 (tifidf)
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
		print 'sim shape: %s  X shape: %s centroid_overall shape: %s' % (sim.shape, self.X2.shape, centroid_overall.shape)
		print 'min %.2f max %.2f ' % (min_sim, max_sim)
		print sorted(sim.flatten(), reverse = True)[:5]
		print sorted(centroid_overall.getA().flatten(), reverse=True)[:5]

		max_sim = 1
		min_sim = 0

		i_plot = 0
		axs[i_plot].hist(sim.flatten(), alpha=0.2) #, ax=axs[i_plot])
		axs[i_plot].set_xlim(min_sim, max_sim)
		i_plot = i_plot + 1

		for i in xrange(n_clusters + 1):
			cond = self.clusters == i
			arr = self.X2[cond]
			sim = linear_kernel(self.centroids[i], arr)
			print 'sim shape: %s  arr shape: %s  centroid shape: %s' % (sim.shape, arr.shape, self.centroids[i].shape)
			print sorted(sim.flatten(), reverse = True)[:5]
			print sorted(self.centroids[i].flatten(), reverse=True)[:5]
			axs[i_plot].hist(sim.flatten(), alpha=0.2)#, ax=axs[i_plot])
			axs[i_plot].set_xlim(min_sim, max_sim)
			i_plot = i_plot + 1

		plt.show()
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

	def set_models(self, kw_dict, num_topics=20, num_words=15 ):
		'''
		hack to set the models after running lda
		'''
		for k, v in kw_dict.items():
			if k == 'data':
				self.df = v
				self.n_docs = self.df.shape[0]
			if k == 'lda_model':
				self.lda_model = v
				self.lda_topic_matrix = np.array(self.lda_model.show_topics(formatted=False,num_topics=num_topics, num_words=num_words))
				self.n_terms = num_words
				self.n_top_terms = num_words
				self.n_topics = num_topics
				self.set_H(self.lda_topic_matrix)
			if k == 'dictionary':
				self.dictionary = v
			#if k == 'lda_topics_matrix':
			#	self.lda_topic_matrix = topics_matrix

	def set_H(self, lda_topic_matrix):
		'''
			INPUT:  
				- 
			OUTPUT:
				- self.top_word_features
				- self.top_word_feature_map
				- self.H
		'''
		print 'set_H'
		print lda_topic_matrix.shape, type(lda_topic_matrix)
		topic_words = lda_topic_matrix[:,:,1]
		weights = lda_topic_matrix[:,:,0]

		self.top_word_features = np.array(list(set(topic_words.flatten())))#[]  #words used in at least one topic
		self.top_word_feature_map = dict(zip(self.top_word_features, xrange(len(self.top_word_features))))

		self.H = np.zeros((self.n_topics, len(self.top_word_features)))

		for i_topic , words in enumerate(topic_words):
			for j_word in xrange(len(words)):
				word = words[j_word]
				if word in self.top_word_feature_map:
					i_feature = self.top_word_feature_map[word]
				else:
					self.top_word_features.append(word)
					i_feature = len(self.top_word_features) - 1
					self.top_word_feature_map[word] = i_feature
				#print 'H shape: %s weights shape: %s i_topic: %i  i_feature: %i  j_word: %i' % (self.H.shape, weights.shape, i_topic, i_feature, j_word)
				self.H[i_topic, i_feature] = float(weights[i_topic, j_word])

	def set_X2(self, X2):
		'''
		hack to set the tfidf matrix
		'''
		self.X2 = X2

	def set_W(self):
		'''
		calculate W: doc x topic (weight of article for each topic)
		'''
		self.W = np.zeros((self.n_docs, self.n_topics))
		for i_article in xrange(self.df.shape[0]):
			w_this = self.lda_model[self.dictionary.doc2bow(self.df.tokens.values[i_article])]
			for i_topic, w_topic in w_this:
				self.W[i_article,i_topic] = w_topic

	def get_summary_stats(self):
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

def load_model_print_stats():
	t0 = time.time() # time it
	min_df = 5
	dictionary = corpora.Dictionary(docs)
	dictionary.filter_extremes(no_below= min_df, no_above=0.99)

	lda_topic_model = LDATopics('test_lda')
	lda_kw_dict = {'data':df,'lda_model':lda_model, 'dictionary':dictionary}
	lda_topic_model.set_models(lda_kw_dict, num_topics=20, num_words=15 )
	lda_topic_model.set_W()
	#lda_topic_model.set_H()

	lda_topic_model.set_top_topics_terms()
	lda_topic_model.print_topic_results_html()
	lda_topic_model.plot_hist_weight_best_topic_per_article()

	cutoff = lda_topic_model.get_summary_stats()

	lda_topic_model.set_X2(X2)
	lda_topic_model.plot_hist_d_to_centroid(min_w = 0.3)

def test_run():
	model_name = 'test_lda'
	num_topics  = 20
	t0 = time.time() # time it
	my_stop_words = TfidfVectorizer(stop_words = 'english').get_stop_words()
	print '%i stop words ' % len(my_stop_words)

	docs = []
	for doc in df.tokens.values:
		docs.append([w for w in doc if w not in my_stop_words])

	lda_model, topics_matrix, dictionary = run_lda(docs, min_df = 5, num_topics = num_topic)
	t1 = time.time() # time it
	print "finish in  %4.4fmin for %s " %((t1-t0)/60,'LDA')

	lda_topic_model = LDATopics(model_name)

	lda_kw_dict = {'data':df,'lda_model':lda_model, 'dictionary':dictionary}
	lda_topic_model.set_models(lda_kw_dict, num_topics=20, num_words=15 )
	lda_topic_model.set_W()
	lda_topic_model.set_H()

	lda_topic_model.set_top_topics_terms()

	lda_topic_model.print_topic_results_html()
	lda_topic_model.plot_hist_weight_best_topic_per_article()

	cutoff = lda_topic_model.get_summary_stats()




