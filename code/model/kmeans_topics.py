from sklearn.cluster import KMeans
from LdaTopics import BaseTopics
import numpy as np

class KmeansTopics(BaseTopics):
	def __init__(self, model_name = 'tmp'):
		
		BaseTopics.__init__(self, model_name)
		#BaseTopics.__init__(model_name)

	def set_models(self, centroids, clusters, features, df, num_topics=20, num_words=15 ):
		self.df = df
		self.n_docs = df.shape[0]
		self.n_topics = num_topics

		self.set_W(clusters)
		self.set_H(centroids,features)

	def set_W(self, clusters):
		'''
		calculate W: doc x topic (weight of article for each topic)
		'''
		w_in_cluster = 1
		self.W = np.zeros((self.n_docs, self.n_topics))
		for i_article in xrange(self.df.shape[0]):
			cluster_id = clusters[i_article]
			self.W[i_article,cluster_id] = 1

	def set_H(self, centroids, features):
		'''
			INPUT:  
				- 
			OUTPUT:
				- self.top_word_features
				- 
				- self.H
		'''
		print 'set_H'
		self.top_word_features = np.array(features)
		self.n_top_terms = len(features)
		#self.top_word_feature_map = dict(zip(self.top_word_features, xrange(len(self.top_word_features))))

		self.H = np.zeros((self.n_topics, len(self.top_word_features)))
		for i_centroid , centroid in enumerate(centroids):
				#print 'H shape: %s weights shape: %s i_topic: %i  i_feature: %i  j_word: %i' % (self.H.shape, weights.shape, i_topic, i_feature, j_word)
				self.H[i_centroid:] = centroid

def test_run():
	run_k_means()
	t0 = time.time() # time it

	kmeans_topic_model = KmeansTopics('test_kmeans')
	kmeans_topic_model.set_X2(X2)
	kmeans_topic_model.set_models(centroids, clusters, features, df, num_topics=20, num_words=15 )

	kmeans_topic_model.set_top_topics_terms()
	kmeans_topic_model.print_topic_results_html()
	kmeans_topic_model.plot_hist_weight_best_topic_per_article()

	#cutoff = lda_topic_model.get_summary_stats()

	kmeans_topic_model.set_X2(X2)
	kmeans_topic_model.plot_hist_d_to_centroid(min_w = 0.0)

def run_k_means():

	num_clusters = 20
	n_terms = 15

	t0 = time.time()
	km = KMeans(n_clusters=num_clusters)
	km = km.fit(X2)
	t1 = time.time() # time it
	print "finish in  %4.4fmin for %s " %((t1-t0)/60,'k means')

	clusters = km.labels_.tolist()
	print pd.value_counts(clusters)

	print("Top terms per cluster:")
	print()
	#sort cluster centers by proximity to centroid
	order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
	features = vectorizer.get_feature_names()

	for i in range(num_clusters):
	    print("Cluster %d words:" % i)
	    
	    for ind in order_centroids[i, :n_terms]: #replace 6 with n words per cluster
	        #print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
	        print (' %s' % features[ind].encode('utf-8','ignore')), km.cluster_centers_[i, ind]
	    print() #add whitespace
	    print() #add whitespace
	    
	    print("Cluster %d articles:" % i)
	    #for title in frame.ix[i]['title'].values.tolist():
	    #    print(' %s,' % title, end='')
	    print() #add whitespace
	    print() #add whitespace
	    
	print()
	print()