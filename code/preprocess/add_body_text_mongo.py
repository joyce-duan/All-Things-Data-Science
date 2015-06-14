'''
extract main text from html and added in mongodb 
collection articles 
field 'body_text'

to-do:
is it necessary to try other extraction methods and why?
'''

import sys
sys.path.append('../db')
from mongo import MyMongo
from httplib import BadStatusLine
from boilerpipe.extract import Extractor

if __name__ == '__main__':

	my_mongo = MyMongo()
	query = {'raw_html':{'$exists':1}, 'body_text':{'$exists':0}}
	cur_articles = my_mongo.get_articles(query = query)

	articles = list(cur_articles)
	print '%d articles to be processed. ' % (len(articles))
	for a in articles:
		try:
			extractor = Extractor(extractor='ArticleExtractor', html=a['raw_html'])
			extracted_text = extractor.getText()
			l = extracted_text.split('\n')
			a_id = a['_id']
			my_mongo.update_record('articles', a_id, 'body_text', extracted_text)
			#print(extracted_text)
	    # do something with page
		except BadStatusLine:
			print("could not fetch %s" % urls)
	my_mongo.close()