
# coding: utf-8

# In[4]:

from httplib import BadStatusLine
#import extraction
from boilerpipe.extract import Extractor
import feedparser


# In[ ]:

from boilerpipe.extract import Extractor

URL='http://radar.oreilly.com/2010/07/louvre-industrial-age-henry-ford.html'

extractor = Extractor(extractor='ArticleExtractor', url=URL)

print extractor.getText()


# In[ ]:

# used by popart:  https://github.com/mickaellegal/Popart/blob/master/Code/database.py
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
url ='http://notebooks.codeneuro.org/'
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
v_txt = [ re.sub('[^0-9a-zA-Z]+', ' ', v) for v in visible_texts]
v_txt = [ re.sub(' +', ' ', v) for v in visible_texts]
#print v_txt[:3]
#print [len(v_txt[i]) for i in xrange(10)]
for i in xrange(len(v_txt)):
    if len(v_txt[i]) > 5*10 and (len(list_get(v_txt, i+1)) > 3*5 or len(list_get(v_txt, i+2)) > 3*5):
        break
#print i
#for j, v in enumerate (v_txt[:i]):
#    print j,": ", v

#print '\n'
#print v_txt[i]
j = 0
content = ''
for c in v_txt[i:]:
    if re.search('[0-9a-zA-Z]', c): 
        content  = content + c
    if len(c)==1:
        content = content + c 
print content


# In[ ]:



