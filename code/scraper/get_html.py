'''
links
	linksource: datatau
	linksubsource:  news
	linksourceversion: 0606
	url:                              title
	org:  codeneuro.org
	title: CodeNeuro Jupyter Notebooks
	date_added: 19 hours agao
	datatau_points: 6
	submittedby:                      subtext

articles 
    url
    html

read in list of url from datatau
get the html page 
save in mongodb 
'''
import os
import datetime as dt
import re
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

class Datatau_article(object):
	attributes = ['url','title','org','dt','internalurl']

	def __init__(self, url, title, org, dt=None, internal_url = None):
		self.title = title
		self.url = url
		self.org = org
		self.dt = dt
		self.internalurl = internal_url

	def set_dt(self, dt):
		self.dt = dt

	def set_internal_url(self, internal_url):
		self.internalurl = internal_url

	def get_attri(self, attri):
		if attri == 'title':
			return self.title
		if attri == 'url':
			return self.url
		if attri == 'org':
			return self.org
		if attri == 'dt':
			return self.dt
		if attri == 'internalurl':
			return self.internalurl
		else:
			return None

	def print_me(self):
		for a in Datatau_article.attributes:
			print a, ": ", self.get_attri(a)

def get_mongodb_collections( client):
	'''
	create if does not exists; else return handler
	'''
	# Initiate Database all things data science
	db = client['atds']
	# Initiate collection if not existing

	links = db['links']
	#print list(li.find())
	articles = db['articles']
	return links, articles

def insert_one_doc(doc_dict, tab):
	try:
		tab.insert(doc_dict)
	except DuplicateKeyError:
		print "Duplicate keys"

def extract_urls(fname, dir_name):  
	'''
	get url and associated metrics

		- OUTPUT: list of tuple [(url, title)]
	'''
	links = []
	f = open(dir_name + fname, 'r')
	#print f
	soup = BeautifulSoup(f, 'html.parser')
	p = re.compile('</a> ([0-9]+) day')

	title_entries = soup.select('td.title')
	i = 0
	while i < len(title_entries): # change to 4 for  Testing
	#for i in xrange(len(title_entries)):
		cnt = title_entries[i].text
		if cnt != 'More':
			try:
				title_txt = title_entries[i+1].text.split('(')
				title = title_txt[0]
				org = 'datatau'
				if len(title_txt) >1:
					org = title_txt[1].replace(')','')
				urls_inthis = title_entries[i+1].select('a')
				url = urls_inthis[0]['href']
				datatau_entry = Datatau_article(url, title, org)
				links.append(datatau_entry)
				#print '\ncreated an entry'
				#datatau_entry.print_me()

			except KeyError:
				print '%s error in title and url' % (cnt)
				print title_entries[i:i+2]
		i = i + 2

	subtext_entries = soup.select('td.subtext')

	dt_submitted = None
	for i, subtext in enumerate(subtext_entries): # [:2]):  # Testing
		internal_url = subtext.select('a')[1]['href']
		m = re.search(p, str(subtext))
		if m:
			ndays = int(m.group(1))
			dt_submitted = (dt.datetime.now() - dt.timedelta(ndays)).strftime('%Y-%m-%d')
		else:
			if 'a> hour' in str(subtext):
				dt_submitted = (dt.datetime.now())
		#print i
		links[i].set_internal_url(internal_url)
		links[i].set_dt(dt_submitted)
	f.close()

	return links

def insert_links(links_collection, links, subsource, linksourceversion):
	'''
	insert datatau entries into mongodb links collection

	'''
	d = {'linksource':'datatau',
	'linksubsource':subsource,
	'linksourceversion':linksourceversion
	}
	attributes = Datatau_article.attributes

	for l in links:   
		t = [(a, l.get_attri(a)) for a in attributes if l.get_attri(a)]
		d_this = dict(t)
		d_allinfo = dict(d, **d_this)

		#print d_allinfo

		try:
			links_collection.insert(d_allinfo)
		except DuplicateKeyError:
			print "Duplicate keys"


if __name__ == '__main__':

	dir_datatau = '/Users/joyceduan/data/datatau/0606/new/'
	linksourceversion = '0606'
	subsource = 'new'

	mongo_client = MongoClient()
	links_collection, articles_collection = get_mongodb_collections(mongo_client)

	fnames = os.listdir(dir_datatau)
	for fname in fnames:
		#fname = '1.html'
		links = extract_urls(fname, dir_datatau)
		insert_links(links_collection, links, subsource, linksourceversion)


