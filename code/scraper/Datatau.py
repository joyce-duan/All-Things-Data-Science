'''
Datatau_list class:
   attributes of article meta data extracted from datatau
'''


import os
import datetime as dt
import re
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from Scraper import single_query


class Datatau_list(object):
    attributes = ['url', 'title', 'org', 'dt', 'internalurl']

    def __init__(self, url, title, org, dt=None, internal_url=None):
        self.title = title
        self.url = url
        self.org = org
        self.dt = dt  # submission date
        self.internalurl = internal_url  # data tau internal link

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
        for a in Datatau_list.attributes:
            print a, ": ", self.get_attri(a)


def extract_urls(html_handle, prefix_url=''):
    links = []
    # print f
    soup = BeautifulSoup(html_handle, 'html.parser')
    p = re.compile('</a> ([0-9]+) day')

    title_entries = soup.select('td.title')
    i = 0
    while i < len(title_entries):  # change to 4 for  Testing
        # for i in xrange(len(title_entries)):
        cnt = title_entries[i].text
        if cnt != 'More':
            try:
                #title_txt = title_entries[i+1].text.split('(')
                # title may contains '('
                title_txt = title_entries[i + 1].text.rsplit('(', 1)
                title = title_txt[0]
                org = 'datatau'
                if len(title_txt) > 1:
                    org = title_txt[1].replace(')', '')
                urls_inthis = title_entries[i + 1].select('a')
                url = prefix_url + urls_inthis[0]['href']
                if url[:5] == 'item?':
                    url = None
                    org = 'datatau.com'
                datatau_entry = Datatau_list(url, title, org)
                links.append(datatau_entry)
                # print '\ncreated an entry'
                # datatau_entry.print_me()

            except KeyError:
                print '%s error in title and url' % (cnt)
                print title_entries[i:i + 2]
        i = i + 2

    subtext_entries = soup.select('td.subtext')

    dt_submitted = None
    for i, subtext in enumerate(subtext_entries):  # [:2]):  # Testing
        internal_url = subtext.select('a')[1]['href']
        m = re.search(p, str(subtext))
        if m:
            ndays = int(m.group(1))
            dt_submitted = (
                dt.datetime.now() - dt.timedelta(ndays)).strftime('%Y-%m-%d')
        else:
            if 'a> hour' in str(subtext):
                dt_submitted = (dt.datetime.now())
        # print i
        links[i].set_internal_url(internal_url)
        links[i].set_dt(dt_submitted)

    print '%d links extracted' % len(links)
    return links


def insert_links(my_mongo, links, linksourceversion='', subsource='', dt_ref=None):
    '''
    insert datatau entries into mongodb links collection

    linksourceversion = '0608'
    subsource = 'new'
    '''
    d = {'linksource': 'datatau',
         'linksubsource': subsource,
         'linksourceversion': linksourceversion
         }
    attributes = Datatau_list.attributes

    counter = 0

    if dt_ref == None:
        pass
    else:
        '''
        update dt_submitted
        '''
        for l in links:
            # if l.get_attri('dt') == None:
            l.set_dt(dt_ref)
            # else:

    for l in links:
        t = [(a, l.get_attri(a)) for a in attributes if l.get_attri(a)]
        d_this = dict(t)
        d_allinfo = dict(d, **d_this)

        l.print_me()
        print '---'
        # print d_allinfo

        try:
            my_mongo.links_collection.insert(d_allinfo)
            counter = counter + 1
        except DuplicateKeyError:
            print "Duplicate keys"
    print '%d links added.' % counter
