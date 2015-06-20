

'''
build simple text classifier
'''
import pandas as pd
import numpy as np
import scipy as sp
import pickle
import time

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#TfidfTransformer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
import random 


sys.path.append('../code/model')
sys.path.append('../code/preprocess')
sys.path.append('../code/recommender')
from topic_modeling import TopicModel, read_articles
from ArticleProceser import   ascii_text
from recommender import Recommender, get_rank, make_diagnostic_plots

#model_pkl_fname = 'data/mnb_model.pkl'
#vectorizer_pkl_fname = 'data/vectorizer.pkl'

def tokenize(doc):
        '''
        INPUT: string
        OUTPUT: list of strings
        Tokenize and stem/lemmatize the document.
        '''
        snowball = SnowballStemmer('english')
        #print type(doc), type([snowball.stem(word) for word in word_tokenize(doc.lower())])
        #print len(doc), len([snowball.stem(word) for word in word_tokenize(doc.lower())])
        return [snowball.stem(word) for word in word_tokenize(doc.decode('utf-8').strip().lower())]

def read_data():
	fname = 'data/articles.csv'
	df = pd.read_csv(fname)
	return df.iloc[:50000]  # for Testing

def fit_tfidf(X):
	'''
		- INPUT: X. list of string
		- OUPUT: list of list
	'''
	tokenized_articles = [tokenize(doc) for doc in X]

	vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 5000)
	documents = [' '.join(article) for article in tokenized_articles]
	vectorizer =  vectorizer.fit(documents)

	vectors_doc = vectorizer.transform(documents)
	return vectorizer, vectors_doc

def transform_tfidf(vectorizer, X):
	'''
	transform text to tfidf
		- INPUT:  list of string :  (string / text)
		- OUTOUT: list of float
	'''
	tokenized_articles = [tokenize(doc) for doc in X]
	documents = [' '.join(article) for article in tokenized_articles]
	return vectorizer.transform(documents)

def dump_model(clf, vectorizer):

	fobj_out  = open (model_pkl_fname ,'w')
	pickle.dump(clf,fobj_out)
	fobj_out.close()

	fobj_out  = open (vectorizer_pkl_fname ,'w')
	pickle.dump(vectorizer, fobj_out)
	fobj_out.close()	 

def read_model():
	clf = pickle.load(open(model_pkl_fname))
	vectorizer = pickle.load(open(vectorizer_pkl_fname))
	return vectorizer, clf

'''
0.0185min for read data
(10001, 21)
 5.8091min for tfidf
['U.S.']
 0.0022min for fit and predict
 0.0297min for dump model
 0.0183min for read pickle
2920 actual: U.S. predict: U.S.
 0.0001min for predict
'''
if __name__ == '__main__':
	#print 'do something'

	t0 = time.time()
	df = read_data()
	t1 = time.time() # time it
	print " %4.4f min for %s " %((t1-t0)/60,'read data')
	print df.shape

	t0 = t1
	xname = 'body'
	yname = 'section_name'
	X = df[xname]
	y = df[yname]
	vectorizer, vectorized_X = fit_tfidf(X)
	t1 = time.time() # time it
	print " %4.4f min for %s " %((t1-t0)/60,'tfidf')

	t0 = t1
	clf = MultinomialNB()
	clf.fit(vectorized_X, y)
	print clf.predict(vectorized_X[5])
	t1 = time.time() # time it
	print " %4.4f min for %s " %((t1-t0)/60,'fit and predict')

	t0 = t1
	dump_model(clf, vectorizer)
	t1 = time.time() # time it
	print " %4.4f min for %s " %((t1-t0)/60,'dump model')

	t0 = t1
	vectorizer2, clf2 = read_model()
	t1 = time.time() # time it
	print " %4.4f min for %s " %((t1-t0)/60,'read pickle')

	t0 = t1
	i = random.randint(1, len(X))
	x_test = X[i]
	x_vectorized = transform_tfidf(vectorizer2, [x_test])
	y_pred = clf2.predict(x_vectorized[0])[0]
	print '%i actual: %s predict: %s \n %s' % (i, y[i], y_pred, x_test.decode('utf-8'))
	t1 = time.time() # time it
	print " %4.4f min for %s " %((t1-t0)/60,'predict ')



