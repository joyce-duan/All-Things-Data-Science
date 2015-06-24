
import requests

def singel_query_raw_html_all_methods(url):
	'''
	use all the methods to get raw_html

		- INPUT: url string
		- OUTPUT: raw_html  (in unicdoe); '' if none found
	'''
	response = single_query(url)
	
	raw_html = ''
	if response:
		if response.status_code == 403:
			#try:
			raw_html = single_query_browser(url)
			#except:
			#	pass
		else:
			if response.status_code == 200:
				raw_html = response.text
	try:
		raw_html = raw_html.decode('utf8')
	except:
		pass
	return raw_html

def single_query(url):
	'''
	basic requests to get content of a url

		INPUT:  
		- url   string
		OUTPUT:
		- response object
	'''
	#response = None
	try:
		response = requests.get(url)
		if response.status_code != 200:
			print 'WARNING', url,' ', response.status_code
			return response
		else:
			return response  # 
	except:
		print 'invalid url:',  url

def single_query_browser(url):
	'''
	as plan b, use mechanize to get html content of an url
	if single_query gave response.status_code 403

	INPUT:  
		- url   string
	OUTPUT:
		- raw_html   string
	'''
	#url = 'http://blog.treasuredata.com/blog/2015/04/24/python-for-aspiring-data-nerds/'

	br = mechanize.Browser()
	#br.set_all_readonly(False)    # allow everything to be written to
	br.set_handle_robots(False)   # ignore robots
	br.set_handle_refresh(False)  # can sometimes hang without this
	br.addheaders =  [('User-agent', 'Firefox')]           # [('User-agent', 'Firefox')]
	#try:
	response = br.open(url)
	#print response.read()      # the text of the page
	response1 = br.response()  # get the response again
	raw_html = response1.read()     # can apply lxml.html.fromstring()

	return raw_html
	#print raw_html[:300]

def insert_links(links_collection, links, linksource='', dt_submit=None):# use this, subsource, linksourceversion):
	'''
	base version of insert links attributes = ['url','title', 'linksource']
	insert datatau entries into mongodb links collection

	'''
	d = {'linksource':linksource
	}
	if dt_submit:
		d = {'linksource':linksource, 'dt':dt_submit
		}		

	for l in links:
		d_this = dict(l)
		d_allinfo = dict(d, **d_this)

		print d_allinfo

		try:
			links_collection.insert(d_allinfo)
		except DuplicateKeyError:
			print "Duplicate keys"

def single_query_v0(url):  #depreciated
	'''
	first version; depreciated; use singe_query instead
	get raw html of url

		- INPUT: string url
		- OUTPUT:  requests.models.Response
	'''
	#response = None
	try:
		response = requests.get(url)
		if response.status_code != 200:
			print 'WARNING', url,' ', response.status_code
		else:
			return response  # 
	except:
		print 'invalid url:  url'


