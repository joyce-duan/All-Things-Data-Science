'''
base class to extract all the visible texts from html files; ArticleProceser.py generally works better than this one.
'''
from get_links import get_mongodb_collections
import math
import random

import os
import datetime as dt
import re
from bs4 import BeautifulSoup
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError


name_article_html = 'raw_html'
name_url = 'url'
name_textcontent = 'textcontent'


def list_get(l, i):
    return l[i] if i < len(l) else ''


def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf8'))):
        return False
    return True


def quick_content(raw_html):
    '''
    content section not excluded yet
    '''
    soup = BeautifulSoup(raw_html)
    texts = soup.findAll(text=True)

    min_char = 6
    visible_texts = filter(visible, texts)
    v_txt = [re.sub('[^0-9a-zA-Z]+', ' ', v) for v in visible_texts]
    v_txt = [re.sub(' +', ' ', v) for v in visible_texts]
    # print v_txt[:3]
    # print [len(v_txt[i]) for i in xrange(10)]
    i_content_start = 0
    for i in xrange(len(v_txt)):
        if len(v_txt[i]) > 5 * 10 and (len(list_get(v_txt, i + 1)) > 3 * 5 or len(list_get(v_txt, i + 2)) > 3 * 5):
            i_content_start = i
            break
    # print i
    # for j, v in enumerate (v_txt[:i]):
    #    print j,": ", v

    # print '\n'
    # print v_txt[i]
    j = 0
    content = ''
    for c in v_txt[i_content_start:]:
        if re.search('[0-9a-zA-Z]', c):
            content = content + c
        if len(c) == 1:
            content = content + c
    return content


class Processer(object):

    '''
    process articles in mongodb
    '''

    def __init__(self):
        self.mongo_client = MongoClient()
        links_collection, articles_collection = get_mongodb_collections(
            self.mongo_client)
        self.articles_collection = articles_collection
        self.content_extract_func = quick_content

    def sample_and_print(self, k=2):
        '''
        random sample k articles, and save as html file
        '''
        # query = { ': 'OK' };
        n = self.articles_collection.count()
        r = int(math.floor(random.random() * n))
        random_article = self.articles_collection.find().limit(1).skip(r)

        print random_article[0][name_url]
        f = open('test.html', 'w')
        f.write(random_article[0][name_article_html].encode('utf8') + '\n')
        f.close

    def add_content(self):
        '''
        go through all articles w/o textcontent
        add in textcontent
        '''
        query = {name_textcontent: {'$exists': 0},
                 name_article_html: {'$exists': 1}}
        articles = list(self.articles_collection.find(query))
        for a in articles:
            article_id = a['_id']
            print article_id, a['url']
            str_content = self.content_extract_func(a[name_article_html])
            self.articles_collection.update(
                {'_id': article_id}, {'$set': {name_textcontent: str_content}})

if __name__ == '__main__':
    p = Processer()
    # p.sample_and_print()
    p.add_content()
