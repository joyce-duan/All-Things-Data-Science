

'''
get url from collection links in mongodb atds
check if html file is available 
if not  get html files, insert in collection add_article
update 
	'has_html' = 1 for all records of this url in collection links
	'triedhtml' = 1 for all records in collection "links"

request error code:
	401 unaurthorized
	403  forbidden
	404 not found

to-do:
1. uri vsl response code error; exclude those with too many errors?
2. wget seemed to be work with some of these??
'''
from get_links import get_mongodb_collections

link_query = {'gothtml':{'$exists': 0}, 'org':{'$ne': 'datatau'}}
link_url_field_default = 'url'

import os
import datetime as dt
import re
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import time


def update_scraped(links_collection, articles_collection, urls_no_content, url_field_in_links = 'url'):
	'''
	go through the list of urls, remove those already has content in collection articles
	update the entry in collection links
	'''
	new_urls = []
	for url in urls_no_content:
		cursor = articles_collection.find({'url': url}).limit(1)
		if cursor.count() > 0:
			links_collection.update({url_field_in_links: url}, {'$set': {'gothtml': 1}}, multi = True)
		else:
			new_urls.append(url)
	return new_urls

def get_articles(links_collection, articles_collection, link_url_field = link_url_field_default, query = link_query ):
	'''
	get distinct url from collection links in mongodb atds with no gothtml
	request the url, insert in collection add_article
		update triedhtml=1 for all urls
		update gothtml = True for all records of this url in collection links
	'''

	counter = 0
	counter_inserted = 0

	#  db.links.find({'gothtml':{$exists:0},org:{$ne:'datatau'}}).count()
	#query = {'gothtml':{'$exists': 0}, 'org':{'$ne': 'datatau'}}

	#query = {'gothtml':{'$exists': 0}, 'linksource':'kdnuggetmonthly'}
	urls_no_content = links_collection.find(query).distinct(link_url_field)
	print '%i urls found to get content' % (len(urls_no_content))

	urls_no_content = update_scraped(links_collection, articles_collection, urls_no_content, link_url_field)

	print '%i urls found to get content after updating' % (len(urls_no_content))
	t0 = time.time()

	for url in urls_no_content[1:]:#: #[:5]:  # for Testing

		counter += 1
		url = url.strip()

		if  'https://web.archive.orgitem?i' in url or url[-4:] == '.pdf':
			pass
		else:
			print 'url: %s' % url
			if 'web.archive.org' in url:
				url_new = '/'.join(re.split('\d{8}',url)[1].split('/')[1:])
				response = single_query(url_new)		
				if response:
					#soup = BeautifulSoup(response.text, 'html.parser')
					try:
						print 'try insert %s  %d' % (url_new, len(response.text))
						articles_collection.insert( {'url':url, 'raw_html':response.text} )
						counter_inserted += 1
						#print response.text
					except DuplicateKeyError:
						print 'duplicate keys'
					links_collection.update({'url': url}, {'$set': {'gothtml': 1}}, multi = True)

				else:
					response = single_query(url)
					if response:
						#soup = BeautifulSoup(response.text, 'html.parser')
						try:
							print 'try insert %s  %d' % (url, len(response.text))
							articles_collection.insert( {'url':url, 'raw_html':response.text} )
							counter_inserted += 1
							#print response.text
						except DuplicateKeyError:
							print 'duplicate keys'
						links_collection.update({'url': url}, {'$set': {'gothtml': 1}}, multi = True)

		links_collection.update({'url': url}, {'$set': {'triedhtml': 1}}, multi = True)
		if counter % 10 == 0:
			print '%i urls processed, %i records inserted' % (counter, counter_inserted)
		#if counter_inserted % 50 == 0:  # for Testing
			t1 = time.time() # time it
			print "%i urls inserted %4.4f min " %(counter_inserted, (t1-t0)/60)
	
		time.sleep(2)
	print '%i urls processed, %i records inserted' % (counter, counter_inserted)

def single_query(url):
	#response = None
	try:
		response = requests.get(url)
		if response.status_code != 200:
			print 'WARNING', url,' ', response.status_code
		else:
			return response  # 
	except:
		print 'invalid url:',  url




if __name__ == '__main__':
	mongo_client = MongoClient()
	links_collection, articles_collection = get_mongodb_collections(mongo_client)

	#q =  {'gothtml':{'$exists': 0}, 'org':{'$ne': 'datatau'}, 'linksource':'datatau'}

	# db.links.find({gothtml:{$exists:0},org:{$ne:'datatau'},triedhtml:{$exists:0}}).count()
	#db.links.distinct('url',{gothtml:{$exists:0},org:{$ne:'datatau'},triedhtml:{$exists:0}})
	q =  {'gothtml':{'$exists': 0}, 'org':{'$ne': 'datatau'}, 'triedhtml':{'$exists':0}, 'org':{'$ne':'datatau.com'}}
	get_articles(links_collection, articles_collection, link_url_field = link_url_field_default, query = q)
