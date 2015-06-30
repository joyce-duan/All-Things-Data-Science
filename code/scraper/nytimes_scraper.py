
dbname = 'nytimes'
links_collection_name = 'links'
articles_collection_name = 'articles'

end_dt = '20150612'

api_link = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
org = 'nytimes'
base_url = 'http://api.nytimes.com/svc/search/v2/articlesearch.json?'
# url_field_name
link_url_field_default = 'web_url'

link_query_default = {'gothtml': {'$exists': 0}}
dt_field = '&begin_date=20130101&end_date=20150610'
#query = 'q=data+scientist'

'''
to-do:
1. make into class
2. query input as command line????
3. for article meta data, if search results have more than 100 pages, roll back date and redo
4. 
'''

from get_article import update_scraped, get_articles, single_query

'''
payload = {'api-key': '12882678342bcec0ac82fcc4f99d5e15:9:72049955', 
'begin_date':'20130101', 'end_date':end_dt, 'sort':'newest'}
'''

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
	gothtml      got html content using request

articles 
    url
    raw_html
    textcontent


'''
import requests
from pymongo import MongoClient
from bs4 import BeautifulSoup
import time
from pymongo.errors import DuplicateKeyError, CollectionInvalid
# def get_search_api_link(query, base_url = base_url, dt_field = dt_field):


def get_db_collections(links_collection_name, articles_collection_name):
    client = MongoClient()
    # Initiate Database
    db = client[dbname]
    # Initiate Table
    links_collection = db[links_collection_name]
    articles_collection = db[articles_collection_name]
    return links_collection, articles_collection


def insert_one_article(article_dict, links_collection):
    '''
    insert meta info about one article into the links collection
    '''
    try:
        links_collection.insert(article_dict)
    except DuplicateKeyError:
        print 'DUPS!'


def call_api(url, base_payload, query, p=0):
    '''
    update the payload and get call search api, return page p results
    '''
    payload = {}
    payload.update(base_payload)
    payload['q'] = query
    payload['page'] = p
    # print payload
    return single_query_wpayload(url, payload)


def single_query_wpayload(link, payload):
    '''
    make api request for one query
    '''
    response = requests.get(link, params=payload)

    if response.status_code != 200:
        print 'WARNING', response.status_code
        return None
    else:
        return response.json()


def get_articles_meta(links_collection, api_search_query):
    '''
    insert all results of the query to links_collection
    '''
    link = api_link
    ipage = 1
    base_payload = {'api-key': '12882678342bcec0ac82fcc4f99d5e15:9:72049955',
                    'begin_date': '20130101', 'end_date': end_dt, 'sort': 'newest'}  # 'page':ipage}

    html_json = call_api(link, base_payload, api_search_query, p=0)
    hits = html_json['response']['meta']['hits']
    total_pages = (hits / 10) + 1
    total_pages = min(10, total_pages)
    print ' %d hits, %d pages to get' % (hits, total_pages)

    for j in html_json['response']['docs']:
        j['org'] = org
        j['query'] = api_search_query
        insert_one_article(j, links_collection)

    if total_pages > 1:
        for ipage in xrange(1, total_pages + 1):
            # print list(tab.find())
            html_json = call_api(link, base_payload, api_search_query, p=ipage)
            for j in html_json['response']['docs']:
                insert_one_article(j, links_collection)
            # if ipage % 10 == 0:
            print 'page %i' % (ipage)

    print len(list(links_collection.find())), 'links'

# def get_articles_nytimes( links_collection, articles_collection, query =
# query):


def get_articles_nytimes(links_collection, articles_collection, link_url_field=link_url_field_default, query=link_query_default):
    '''
    get the links that missing contents
    get article and insert to article collections
    update links_collection
    '''

    url_field_name = link_url_field

    counter = 0
    counter_inserted = 0
    # need to dedupe???
    # links_no_content = list(links_collection.find({'gothtml': {$exists:
    # false}} ))

    #  db.links.find({'gothtml':{$exists:0}}).count()
    #query = {'gothtml':{'$exists': 0}}
    urls_no_content = links_collection.find(query).distinct(url_field_name)
    print '%i urls found to get content' % (len(urls_no_content))

    urls_no_content = update_scraped(links_collection, articles_collection, urls_no_content,
                                     url_field_in_links=url_field_name)
    print '%i urls found to get content after updating ' % (len(urls_no_content))
    t0 = time.time()

    for url in urls_no_content:  # [:5]:  # for Testing
        # for i, article in enumerate(links_collection.find()):

        counter += 1
        response = single_query(url)
        if response:
            #soup = BeautifulSoup(response.txt, 'html.parser')
            try:
                articles_collection.insert(
                    {'url': url, 'raw_html': response.text})
                #articles_collection.update({'_id': article['_id']}, {'$set': {'raw_html': str(soup)}})

                soup = BeautifulSoup(response.text, 'html.parser')
                content = soup.select('.story-body-text')
                body = ''
                for para in content:
                    body += para.text + ' '

                articles_collection.update(
                    {'url': url}, {'$set': {'body_text': body}}, multi=True)
                counter_inserted += 1
            except DuplicateKeyError:
                print 'duplicate keys'

        links_collection.update(
            {'url': url}, {'$set': {'gothtml': 1}}, multi=True)

        if counter % 5 == 0:
            print '%i urls processed, %i records inserted' % (counter, counter_inserted)
        if counter_inserted % 5 == 0:  # for Testing
            t1 = time.time()  # time it
            print "%i urls  %4.4f min " % (counter_inserted, (t1 - t0) / 60)

        time.sleep(2)
    print '%i urls processed, %i records inserted' % (counter, counter_inserted)


def add_title_to_links(links_collection):
    '''
    add filed 'title' to collection links in db nytimes
    '''

    field_name = 'title'
    query = {field_name: {'$exists': 0}}
    docs_no_title = links_collection.find(query)

    # print '%d docs to update' % len(docs_no_titles)
    for doc in docs_no_title:
        #url = doc[link_url_field_default]
        doc_id = doc['_id']
        title = doc['headline'].get("print_headline", '')
        if len(title) < 5:
            title = doc['headline'].get("main", '')
        links_collection.update({'_id': doc_id}, {'$set': {field_name: title}})


def add_url_to_links(links_collection):
    '''
    added field url to all links documents
    '''

    field_name = 'url'
    query = {field_name: {'$exists': 0}}
    docs_no_url = links_collection.find(query)

    # print '%d docs to update' % len(docs_no_url)
    for doc in docs_no_url:
        url = doc[link_url_field_default]
        doc_id = doc['_id']
        links_collection.update({'_id': doc_id}, {'$set': {field_name: url}})


if __name__ == '__main__':
    # query = 'data+scientists' #2
    # query = 'predictive+analytics' #1
    # query = '"data scientist"' #3

    # these need actual articles
    # query = '"data scientists"' #4
    # query = '"data mining"' #5
    # query = '"deep learning"' #6
    # query = '"data science"' #7
    # query = '"predictive analytics"'  #8
    query = '"big data"+analysis'

    links_collection, articles_collection = get_db_collections(
        links_collection_name, articles_collection_name)

    print 'query: %s' % (query)
    #get_articles_meta(links_collection, query)

    link_query = {'gothtml': {'$exists': 0}}  # , 'linksource':linksource}
    #get_articles(links_collection, articles_collection, link_url_field = link_url_field_default, query = link_query)
    get_articles_nytimes(links_collection, articles_collection,
                         link_url_field=link_url_field_default, query=link_query)
