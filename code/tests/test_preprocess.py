import nose.tools as n
import nose
from cStringIO import StringIO  
import unicodedata

from configobj import ConfigObj
config = ConfigObj('allds.config')
allds_home = config['allDS_home']

import sys
sys.path.append(allds_home  + 'code/preprocess')

from ArticleProceser import html_to_bodytext
#from my_mongo import MyMongo
#from httplib import BadStatusLine
#from boilerpipe.extract import Extractor

def test_boilerpipe():
	in_fname = 'data/test.html'
	min_length = 100
	#url = 'http://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/'
	with open(in_fname, 'r') as in_fh:
		raw_html = in_fh.read()

	raw_html = raw_html.decode('utf8', 'ignore')
	body_text = html_to_bodytext(raw_html)

	t2 = body_text.encode('ascii','ignore')
	print t2[:400]
	n.assert_greater(len(t2), min_length  )

if __name__=="__main__":
    module_name = sys.modules[__name__].__file__

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s'])
    sys.stdout = old_stdout
    print mystdout.getvalue()