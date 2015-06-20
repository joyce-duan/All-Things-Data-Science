
#



linksource = 'gradient'
attributes = ['url','title', 'linksource','dt']

import os
import datetime as dt
import re
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from get_links import get_mongodb_collections
#from my_html_links_scraper import insert_links
from Scraper import insert_links
from Scraper import singel_query_raw_html_all_methods
import datetime
import time
import numpy as np

exclude_list = '''
us4.camp
eepurl
facebook.com
twitter.com
youtube.com
google.com
eventbrite.com
meetup.com
mailto:
.jobs/
/jobs.
/jobs/
/job/
jobs.
/jobs
/career/
/careers/
careers.
glassdoor.com
jobs.com
jobvite
galvanizeu.com
taleo.net
jobdetail
?job=
/galvanize.
/careers/
http://www.linkedin.com/directory/
http://www.linkedin.com/legal/
https://help.linkedin.com/app/a
http://www.linkedin.com/redir/
https://www.linkedin.com/uas/login?
http://www.linkedin.com/static?
https://www.linkedin.com/in/
https://www.linkedin.com/pulse/channel/
https://www.linkedin.com/channels/api/follows
http://www.linkedin.com/company/linkedin/careers
http://www.linkedin.com/advertising?
http://www.linkedin.com/about-us
http://www.linkedin.com/company/linkedin/careers?
http://www.linkedin.com/advertising?
http://www.linkedin.com/mobile?
http://www.linkedin.com/in/updates?
http://www.linkedin.com/today/post/whoToFollow?
http://www.linkedin.com/pulse/channel/big_data
'''.split()

def to_exclude(url):
	in_exclusion_list = [t.lower() in url.lower() for t in exclude_list]
	#print in_exclusion_list
	return [t for t in in_exclusion_list if t] #np.logical_or(in_exclusion_list)

def extract_urls(fname, dir_name):
	links = []
	f = open(dir_name + fname, 'r')
	#print f
	soup = BeautifulSoup(f, 'html.parser')

	#title_entries = soup.select('ul.three_ul li')
	a_list = soup.find_all('a')
	print len(a_list), fname
	for a in a_list:
		#a = a_list[0]
		#if not to_exclude(a['href']):
		#print a
		url = ''
		try:
			url = a['href']
			if to_exclude(url) or '/' not in a['href'] or '.' not in a['href'] or \
			url == 'http://www.linkedin.com/' or url == 'http://www.linkedin.com':
				#print 'exclude %s' % a['href']
				pass
			else:
				links.append([('url',a['href']), ('title',a.text)])
		except:
			pass
	f.close()
	return links

def get_links_from_dir():
	mongo_client = MongoClient()
	links_collection, articles_collection = get_mongodb_collections(mongo_client)
	dir_name = '/Users/joyceduan/data/gradient/'
	fnames = os.listdir(dir_name)
	print fnames
	for fname in fnames: # [0:1]:  for testing

		links = extract_urls(fname, dir_name)
		parts = fname.split('_')
		
		d = parts[2].split('.')[0]
		y = parts[0]
		m = parts[1]
		dt_submit = datetime.datetime(int(y),int(m), int(d)).strftime('%Y-%m-%d')
		#print dt_submit
		#print [link[0][1] for link in links]
		insert_links(links_collection, links, linksource=linksource, dt_submit=dt_submit)


def get_links_from_dir_linkedin():
	mongo_client = MongoClient()
	links_collection, articles_collection = get_mongodb_collections(mongo_client)
	dir_name = '/Users/joyceduan/data/linkedin/'
	fnames = os.listdir(dir_name)
	print fnames
	for fname in fnames: # [0:1]:  for testing

		links = extract_urls(fname, dir_name)
		
		dt_submit = None
		#print dt_submit
		#print '\n'.join([link[0][1] for link in links])
		insert_links(links_collection, links, linksource='linkedin', dt_submit=dt_submit)

if __name__ =='__main__':
	get_links_from_dir_linkedin()
