'''
MyMongo class manages collections to 2 mongo db:  'atds' and 'nytimes'

articles:
	raw_html, url, body_text, 
links: 
	url, (web_url for 'nytimes')
'''

nytimes_dbname = 'nytimes'
atds_dbname = 'atds'
links_collection_name = 'links'
articles_collection_name = 'articles'

import sys, os # these you almost always have...

import requests
from pymongo import MongoClient
from bs4 import BeautifulSoup
import time
from pymongo.errors import DuplicateKeyError, CollectionInvalid
import re

def WHERE( back = 0 ):
    frame = sys._getframe( back + 1 )
    return "%s/%s %s()" % ( os.path.basename( frame.f_code.co_filename ),
                        frame.f_lineno, frame.f_code.co_name )

class MyMongo(object):
	'''
	base class to handle connection to mongodb
	'''
	def __init__(self, dbname = atds_dbname, links = links_collection_name,\
	articles = articles_collection_name ):
		self.dbname = dbname
		self.links_collection_name = links_collection_name
		self.articles_collection_name =articles_collection_name
		self.link_url_field_name = 'url'
		if self.dbname == 'nytimes':
			self.link_url_field_name = 'web_url'
		print WHERE(1)
		print 'connected to database %s, collection name: %s ' % (self.dbname, self.links_collection_name)

		self.client = MongoClient()
		self.db = self.client[self.dbname]

		# Initiate Table if not existing
		self.links_collection = self.db[self.links_collection_name]
		self.articles_collection = self.db[self.articles_collection_name]

	def get_one_article(self):
		'''
		get all the unique url, raw_html
		'''
		#db.articles.findOne({'raw_html':{$exists:true}},{'url':1,'raw_html':1, _id:0})
		query = {'raw_html':{'$exists':1}}
		article = self.articles_collection.find_one(query)
		return article

	def get_articles(self, query=None, testing=0):
		'''
		get all the articles with raw_html; return 5 articles if testing
			- INPUT: int (1 if is testing, 0 not testing - get all records)
			- OUTPUT: cursor to the articles
		'''
		if query == None:
			query = {'raw_html':{'$exists':1}}
		if testing:
			articles = self.articles_collection.find(query, limit = 5)
		else:
			articles = self.articles_collection.find(query)
		return articles

	def update_record(self, collection_name, uid, field_name, val):
		'''
		update the row of record_id uid, set value of field_name = val
		'''
		#print self.dbname, self.links_collection_name, self.articles_collection_name

		if collection_name == 'articles':
			collection = self.articles_collection
		elif collection_name == 'links':
			collection = self.links_collection
		else:
			print "no collection found %s" % collection_name
			return None
		#print uid, field_name
		collection.update({'_id': uid}, {'$set': {field_name: val}})

	def get_article_attri(self, testing = 0):
		'''
		get the attributes of article by cleaned_url
			- INPUT:
			- OUTPUT:  dictionary: article_dict[url] = (title)
		'''

		query = {self.link_url_field_name :{'$exists':1}, 'title':{'$exists':1}}
		projection = {"_id": 0, 'title': 1, self.link_url_field_name :1, 'dt':1}
		if testing:
			links = self.links_collection.find(query, projection, limit = 25)
		else:
			links = self.links_collection.find(query, projection)
		article_dict = {}
		article_dt = {}
		for link in list(links):
			url = link[self.link_url_field_name]
			if 'https://web.archive.orgitem?id' in url: #=1515
				pass
			elif 'web.archive.org' in url:
				#print url
				url = '/'.join(re.split('\d{8}',url)[1].split('/')[1:])
				url = url.strip()
			article_dict[url] = link['title']
			try: 
				t = link['dt']
			except:
				t = ''
			article_dt[url] = link.get('dt', t)
		return article_dict, article_dt



	def get_article_body_text(self, testing=0):
		'''
		get the url and body text
			- OUTPUT: cursor
			??? projection not working????
		'''
		query = {'body_text':{'$exists':1}, 'url':{'$exists':1}}
		projection = {'body_text': 1, 'url':1}
		if testing:
			articles = self.articles_collection.find(query, projection = projection, limit = 25)
		else:
			articles = self.articles_collection.find(query, projection = projection)
		return articles		

	def close(self):
		self.client.close()
'''
class NytimesMongo(object):

def get_mongodb_collections(dbname ):#, links_collection_name, articles_collection_name):
	client = MongoClient()

	# Initiate Database
'''



if __name__ == '__main__':
	my_mongo = MyMongo()
	#print my_mongo.get_one_article()
	print my_mongo.get_article_attri(testing =1)
	my_mongo.close()
