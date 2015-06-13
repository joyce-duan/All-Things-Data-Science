
'''
hacks of gettng articles from data science weekly

'''


linksource = 'dsweekly'
attributes = ['url','title', 'linksource']
link_url_field_default = 'url'

import os
import datetime as dt
import re
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from get_links import get_mongodb_collections
from my_html_links_scraper import insert_links
from get_article import get_articles

def extract_urls(fname, dir_name):  
	'''
	get url and associated metrics

		- OUTPUT: list of tuple (url,title)
	'''
	links = []
	f = open(dir_name + fname, 'r')
	#print f
	soup = BeautifulSoup(f, 'html.parser')

	title_entries = soup.select('ul li')
	i = 0
	while i < len(title_entries): # change to 4 for  Testing
		l = title_entries[i]
		a_list = l.find_all('a')
		for a in a_list:
			if a['href'][0] != '/':
				links.append([('url',a['href']), ('title',a.text)])
		i = i + 1
	f.close()

	return links

if __name__ == '__main__':

	dir_name = '/Users/joyceduan/data/datascienceweekly/'

	mongo_client = MongoClient()
	links_collection, articles_collection = get_mongodb_collections(mongo_client)

	'''
	fnames = os.listdir(dir_name)
	for fname in fnames:
		#fname = '1.html'
		links = extract_urls(fname, dir_name)
		insert_links(links_collection, links, linksource = linksource)
	'''
	link_query = {'gothtml':{'$exists': 0}, 'linksource':linksource}
	get_articles(links_collection, articles_collection, link_url_field = link_url_field_default, query = link_query)


