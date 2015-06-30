'''
ipython test_scraper.py
'''

import nose.tools as n
import nose
from cStringIO import StringIO
import unicodedata
import inspect

from configobj import ConfigObj
config = ConfigObj('allds.config')
allds_home = config['allDS_home']

import sys
sys.path.append(allds_home + 'code/scraper')
print allds_home + 'code/scraper'

from Scraper import singel_query_raw_html_all_methods


def test_get_content():
    print '\nfunction: %s ' % inspect.stack()[0][3]

    url = 'http://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/'
    out_fname = 'data/test.html'
    min_length = 100
    raw_html = singel_query_raw_html_all_methods(url)
    print 'length: %i' % len(raw_html)
    print 'html written out to file  %s' % out_fname
    with open(out_fname, 'w') as in_fh:
        #in_fh.write(unicodedata.normalize('NFKD', raw_html).encode('ascii','ignore'))
        in_fh.write(raw_html.encode('ascii', 'ignore'))
    n.assert_greater(len(raw_html), min_length)
    print ''


def test_2():
    print '\nfunction: %s ' % inspect.stack()[0][3]
    print ''

if __name__ == "__main__":
    module_name = sys.modules[__name__].__file__

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s'])
    sys.stdout = old_stdout
    print mystdout.getvalue()
