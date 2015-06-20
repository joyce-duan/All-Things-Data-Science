'''
hacks of copy and paste as html file all the links to articles
extract the url and tilte and insert into mongo db atds links collection.

'''
linksource = 'feedly'
attributes = ['url','title', 'linksource']

'''
there were imported from get_links.py
	# Initiate Database all things data science
	db = client['atds']
	links = db['links']
	articles = db['articles']
'''
import os
import datetime as dt
import re
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from get_links import get_mongodb_collections

def get_links(fname):
	links = []
	f = open(fname, 'r')
	#print f
	soup = BeautifulSoup(f, 'html.parser')
	for a in soup.find_all('a'):
		links.append([('url',a['href']), ('title',a.text)])
	return links

def insert_links(links_collection, links, linksource=linksource):#use Scraper.py. this depreciated, subsource, linksourceversion):
	'''
	use Scraper.py instead
	base version of insert links attributes = ['url','title', 'linksource']
	insert datatau entries into mongodb links collection

	'''
	d = {'linksource':linksource
	}

	for l in links:   
		d_this = dict(l)
		d_allinfo = dict(d, **d_this)

		print d_allinfo

		try:
			links_collection.insert(d_allinfo)
		except DuplicateKeyError:
			print "Duplicate keys"

if __name__ =='__main__':

	dir_name = '/Users/joyceduan/data/'
	filename= 'feedly/feedly.htm'
	links = get_links(dir_name+filename)

	mongo_client = MongoClient()
	links_collection, articles_collection = get_mongodb_collections(mongo_client)

	insert_links(links_collection, links)


