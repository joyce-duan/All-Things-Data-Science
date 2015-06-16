
import requests

def single_query(url):
	'''
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


