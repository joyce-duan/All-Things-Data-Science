
import theano
from gensim import corpora, models, similarities 
import numpy as np
# tutorial: https://radimrehurek.com/gensim/tut1.html

#create a Gensim dictionary from the texts

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
	    	
	return lda_model, topics_matrix



