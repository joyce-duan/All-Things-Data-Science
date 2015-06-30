
# coding: utf-8

# In[4]:

from httplib import BadStatusLine
#import extraction
from boilerpipe.extract import Extractor
import feedparser


# In[ ]:

from boilerpipe.extract import Extractor

URL = 'http://radar.oreilly.com/2010/07/louvre-industrial-age-henry-ford.html'

extractor = Extractor(extractor='ArticleExtractor', url=URL)

print extractor.getText()


# In[ ]:

# used by popart:
# https://github.com/mickaellegal/Popart/blob/master/Code/database.py
from httplib import BadStatusLine
from boilerpipe.extract import Extractor
try:
    extractor = Extractor(extractor='ArticleExtractor', url=URL)
    extracted_text = extractor.getText()
    print(extracted_text)
# do something with page
except BadStatusLine:
    print("could not fetch %s" % urls)


# In[ ]:

import requests
import bs4
from bs4 import BeautifulSoup
import urllib2
import re

#url = 'http://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/'
url = 'http://notebooks.codeneuro.org/'
html = urllib2.urlopen(url).read()
soup = BeautifulSoup(html)
texts = soup.findAll(text=True)


def list_get(l, i):
    return l[i] if i < len(l) else ''


def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf8'))):
        return False
    return True

min_char = 6
visible_texts = filter(visible, texts)
v_txt = [re.sub('[^0-9a-zA-Z]+', ' ', v) for v in visible_texts]
v_txt = [re.sub(' +', ' ', v) for v in visible_texts]
# print v_txt[:3]
# print [len(v_txt[i]) for i in xrange(10)]
for i in xrange(len(v_txt)):
    if len(v_txt[i]) > 5 * 10 and (len(list_get(v_txt, i + 1)) > 3 * 5 or len(list_get(v_txt, i + 2)) > 3 * 5):
        break
# print i
# for j, v in enumerate (v_txt[:i]):
#    print j,": ", v

# print '\n'
# print v_txt[i]
j = 0
content = ''
for c in v_txt[i:]:
    if re.search('[0-9a-zA-Z]', c):
        content = content + c
    if len(c) == 1:
        content = content + c
print content


# In[ ]:


topic_names = ['topic # %i' % i for i in xrange(W.shape[1])]
sorted_idx_topics = np.argsort(W, axis=1)
sorted_topics_all = []
for i_article in xrange(W.shape[0]):
    idx_topics_desc = (sorted_idx_topics[i_article, :].getA()).flatten()[::-1]
    # print type(idx_topics_desc), idx_topics_desc.shape
    # print idx_topics_desc
    sorted_topics_this = [
        (i_topic, topic_names[i_topic], W[i_article, i_topic]) for i_topic in idx_topics_desc]
    sorted_topics_all.append(sorted_topics_this)
print sorted_topics_all


import mechanize
from mechanize import Browser

url = 'http://blog.treasuredata.com/blog/2015/04/24/python-for-aspiring-data-nerds/'
# create a new instance of Browser class
br = Browser()
# Ignore robots.txt. Do not do this without thought and consideration.
br.set_handle_robots(False)
br.set_handle_equiv(False)
#cj = cookielib.LWPCookieJar()
# br.set_cookiejar(cj)
br.set_handle_equiv(True)
br.set_handle_redirect(True)
br.set_handle_robots(False)
br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)
br.addheaders = [
    ('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
# fetch the stackoverflow.com url
# br.open(url)

# now we need to get the search form, so we get the first form
# in page, which is the search form, in position 0
# br.select_form(nr=0)
# now we set value of the search field with name property as 'q'.
#br.form['q'] = 'python'
# submit the request
#response1 = br.submit()

#response = br.open(url)
# print response.read()      # the text of the page
raw_html = br.response()  # get the response again
# print response1.read()     # can apply lxml.html.fromstring()

# print the response, by calling the read() method
#raw_html = response1.read()
print raw_html[:300]


url = 'http://blog.treasuredata.com/blog/2015/04/24/python-for-aspiring-data-nerds/'

#url = 'http://www.pythonforbeginners.com/cheatsheet/python-mechanize-cheat-sheet'

br = mechanize.Browser()
# br.set_all_readonly(False)    # allow everything to be written to
br.set_handle_robots(False)   # ignore robots
br.set_handle_refresh(False)  # can sometimes hang without this
# [('User-agent', 'Firefox')]
br.addheaders = [('User-agent', 'Firefox')]
response = br.open(url)
print response.read()      # the text of the page
response1 = br.response()  # get the response again
print response1.read()     # can apply lxml.html.fromstring()


#import urllib.request
import requests
url = 'http://blog.treasuredata.com/blog/2015/04/24/python-for-aspiring-data-nerds/'
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

#url = "http://en.wikipedia.org/wiki/List_of_TCP_and_UDP_port_numbers"
headers = {'User-Agent': user_agent, }

request = request(url, None, headers)  # The assembled request
response = urllib.request.urlopen(request)
data = response.read()  # The data u need


url = 'http://blog.treasuredata.com/blog/2015/04/24/python-for-aspiring-data-nerds/'
#agent  = 'DataWrangling/1.1 (http://zipfianacademy.com; '
#agent += 'class@zipfianacademy.com)'

agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'


headers = {'user_agent': agent}

# Wrap in a try-except to prevent a maxTry connection error from erroring
# out the program. Return None if there are any issues.
r = requests.get(url, headers=headers)

# Just in case there was a normal error returned. Pass back None.
# if r.status_code != 200: return None

# Otherwise return a soupified object containing the url text encoded in
# utf-8. Will toss back errors on some pages without the encoding in place.
# return BeautifulSoup(r.text.encode('utf-8'))
