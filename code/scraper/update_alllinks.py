from get_links import get_mongodb_collections
link_url_field_default = 'url'

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

if __name__ == '__main__':

    mongo_client = MongoClient()
    links_collection, articles_collection = get_mongodb_collections(
        mongo_client)

    link_url_field = link_url_field_default

    urls = links_collection.find().distinct(link_url_field)
    print '%i urls found ' % (len(urls))

    field_name = 'triedhtml'
    for url in urls:
        links_collection.update(
            {link_url_field: url}, {'$set': {field_name: 1}}, multi=True)
