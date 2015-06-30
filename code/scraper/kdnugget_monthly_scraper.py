
'''
hacks of gettng articles from kdnugget monthly newsletter

'''


linksource = 'kdnuggetmonthly'
attributes = ['url', 'title', 'linksource']


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


def extract_urls(fname, dir_name):
    '''
    get url and associated metrics

            - OUTPUT: list of Datatau_article instances
    '''
    links = []
    f = open(dir_name + fname, 'r')
    # print f
    soup = BeautifulSoup(f, 'html.parser')

    title_entries = soup.select('ul.three_ul li')
    i = 0
    while i < len(title_entries):  # change to 4 for  Testing
        l = title_entries[i]
        a_list = l.find_all('a')
        a = a_list[0]
        links.append([('url', a['href']), ('title', a.text)])
        i = i + 1
    f.close()

    return links


def extract_urls_from_text(text):
    '''
    get url and associated metrics

            - OUTPUT: list of Datatau_article instances
    '''
    links = []
    #f = open(dir_name + fname, 'r')
    # print f
    soup = BeautifulSoup(text, 'html.parser')

    title_entries = soup.select('ul.three_ul li')
    i = 0
    while i < len(title_entries):  # change to 4 for  Testing
        l = title_entries[i]
        a_list = l.find_all('a')
        a = a_list[0]
        links.append([('url', a['href']), ('title', a.text)])
        i = i + 1
    # f.close()

    return links


def get_links_from_dir():
    mongo_client = MongoClient()
    links_collection, articles_collection = get_mongodb_collections(
        mongo_client)
    dir_name = '/Users/joyceduan/data/kdnugget/'
    fnames = os.listdir(dir_name)
    for fname in fnames:
        #fname = '1.html'
        links = extract_urls(fname, dir_name)
        insert_links(links_collection, links, linksource=linksource)

if __name__ == '__main__':
    mongo_client = MongoClient()
    links_collection, articles_collection = get_mongodb_collections(
        mongo_client)
    sections = [
        'opinions-interviews.html', 'meetings.html', 'publications.html', 'news-features.html'
    ]
    for y in [2014, 2015]:
        m_max = 13
        if y == 2015:
            m_max = 7
        for m in xrange(1, m_max):
            dt_submit = datetime.datetime(y, m, 1).strftime('%Y-%m-%d')
            for section in sections:
                str_m = "%02d" % m
                url_newsletter = 'http://www.kdnuggets.com/%s/%s/%s' % (
                    str(y), str_m, section)
                print url_newsletter

                raw_html = singel_query_raw_html_all_methods(url_newsletter)
                links = extract_urls_from_text(raw_html)
                print links
                #insert_links(links_collection, links, linksource = linksource)
                insert_links(
                    links_collection, links, linksource=linksource, dt_submit=dt_submit)
                time.sleep(2)
