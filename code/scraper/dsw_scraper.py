
'''
hacks of gettng articles from data science weekly

'''


linksource = 'dsweekly'
attributes = ['url', 'title', 'linksource']  # dt
link_url_field_default = 'url'

import os
import datetime
import re
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from get_links import get_mongodb_collections
from Scraper import insert_links  # use this version
from get_article import get_articles
from Scraper import single_query


def extract_urls(fname, dir_name):
    '''
    get url and associated metrics

            - OUTPUT: list of tuple (url,title)
    '''
    links = []
    f = open(dir_name + fname, 'r')
    # print f
    soup = BeautifulSoup(f, 'html.parser')

    title_entries = soup.select('ul li')
    i = 0
    while i < len(title_entries):  # change to 4 for  Testing
        l = title_entries[i]
        a_list = l.find_all('a')
        for a in a_list:
            if a['href'][0] != '/':
                links.append([('url', a['href']), ('title', a.text)])
        i = i + 1
    f.close()

    return links


def exctract_urls_from_text(response_text):
    links = []
    soup = BeautifulSoup(response_text, 'html.parser')

    title_entries = soup.select('ul li')
    i = 0
    while i < len(title_entries):  # change to 4 for  Testing
        l = title_entries[i]
        a_list = l.find_all('a')
        for a in a_list:
            if a['href'][0] != '/':
                links.append([('url', a['href']), ('title', a.text)])
        i = i + 1

    return links

mons = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
dict_m = dict(zip(mons, xrange(1, len(mons) + 1)))


def get_links_to_weekly():
    urls_weekly = []
    dates = []
    fname = '/Users/joyceduan/data/datascienceweekly/' + \
        'ds_weekly_archive.html'
    with open(fname, 'r') as f:
        # print f
        soup = BeautifulSoup(f, 'html.parser')
    urls_all = soup.select('a')
    for url in urls_all:
        href = url['href']
        txt = url.text
        print txt
        if 'issue' in txt.lower() and txt[:len('Data Science Weekly Newsletter')] == 'Data Science Weekly Newsletter':
            urls_weekly.append(href)

            str_dt = txt.split('(')[1].split()

            m = dict_m[str_dt[0].lower()]
            d = int(str_dt[1][:-1])
            dt_submitted = ''
            try:
                dt_submitted = datetime.datetime(
                    int(str_dt[2][:4]), m, d).strftime('%Y-%m-%d')
            except:
                pass
            dates.append(dt_submitted)
    return urls_weekly, dates

if __name__ == '__main__':

    dir_name = '/Users/joyceduan/data/datascienceweekly/'

    mongo_client = MongoClient()
    links_collection, articles_collection = get_mongodb_collections(
        mongo_client)

    urls_weekly, dates = get_links_to_weekly()
    url_prefix = 'http://www.datascienceweekly.org'
    print urls_weekly
    print dates

    for i, url in enumerate(urls_weekly[1:]):
        print i, url, dates[i]
        response = single_query(url_prefix + url)
        if response:
            links = exctract_urls_from_text(response.text)
            insert_links(
                links_collection, links, linksource=linksource, dt_submit=dates[i])
        # except:
        #	print '!!error %s' % url_prefix+url

    '''
	fnames = os.listdir(dir_name)
	for fname in fnames:
		#fname = '1.html'
		links = extract_urls(fname, dir_name)
		insert_links(links_collection, links, linksource = linksource)
	'''
    link_query = {'gothtml': {'$exists': 0},
                  'linksource': linksource, 'triedhtml': {'$exists': 0}}
    get_articles(links_collection, articles_collection,
                 link_url_field=link_url_field_default, query=link_query)
