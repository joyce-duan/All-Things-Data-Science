'''
test module db: mongodb related class and functions
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
sys.path.append(allds_home  + 'code/db')
print allds_home  + 'code/db'

from my_mongo import  MyMongo

def test_get_article_attri():
	print '\nfunction: %s ' % inspect.stack()[0][3]

	my_mongo = MyMongo(dbname = 'nytimes')
	#print my_mongo.get_one_article()
	article_atrri, article_dt = my_mongo.get_article_attri(testing =1)
	print '%i items retrieved' % (len(article_atrri))
	print 'title:'
	print zip(article_atrri)[:5]
	print 'publication date'
	print zip(article_dt)[:5]

	my_mongo.close()
	n.assert_greater(len(article_atrri), 2)

if __name__=="__main__":
    module_name = sys.modules[__name__].__file__

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s'])
    sys.stdout = old_stdout
    print mystdout.getvalue()