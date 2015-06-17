'''
get wayback snapshot of datatau, extract url for articles and store in mongdb 

    list of snapshots from:
    	https://web.archive.org/web/20140210061922*/http://www.datatau.com/
to-do:

 https://web.archive.orgitem?id=6927 is not handled correctly
'''


from Scraper import single_query
from Datatau import insert_links, extract_urls
import sys
import datetime
from Scraper import single_query
from bs4 import BeautifulSoup

sys.path.append('../db')
from mongo import MyMongo

prefix_url = 'https://web.archive.org'

def scrape_one_page_wayback(url, my_mongo):
	'''
	url = 'https://web.archive.org/web/20140210061922/http://www.datatau.com/'
	scrape_one_page_wayback(url, my_mongo)
	'''
	prefix_url = 'https://web.archive.org'

	dt_str = url.split('/web/')[1]
	dt_ref = datetime.datetime.strptime(dt_str[:8],'%Y%m%d')

	response = single_query(url)

	links = extract_urls(response.text, prefix_url)
	#print links


	insert_links(my_mongo, links, linksourceversion='', subsource='', dt_ref = dt_ref)

def get_datatau_links(fname):
	'''
	get list of ursl for datatau snapshot
		- INPUT: fname
		- OUTPUT urls - list of string
	'''
	urls = []
	with open(fname, 'r') as f_handle:
		links_text = f_handle.read()
	soup = BeautifulSoup(links_text, 'html.parser')
	for a in soup.find_all('a'):
		urls.append(prefix_url+a['href'])
	return urls


if __name__ == '__main__':
	#url = 'https://web.archive.org/web/20140210061922/http://www.datatau.com/'
	my_mongo = MyMongo()
	urls = get_datatau_links('/Users/joyceduan/data/wayback/links_datatau.html')
	for url in urls[1:]:
		print url
		scrape_one_page_wayback(url, my_mongo)
	my_mongo.close()