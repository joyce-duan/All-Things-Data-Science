'''
feature generation from body_text
'''
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from httplib import BadStatusLine
from boilerpipe.extract import Extractor

def clean_articles(cur_articles, articles_cleaned):
    '''
    read articles from mongodb and save in dictionary 
            exclude *.pdf, youtube
            standardize url (strip prefix for wayback machine)

        - INPUT:  
            cur_articles: cursor_articles, mongo query results
            articles_cleaned:  {}  
        - OUTPUT: 
            none:  updated articles_cleaned: 
            articles_cleaned[url] = [len(body_text),body_text]
    '''
    articles = list(cur_articles)
    print '%d articles found' % len(articles)
    counter = 0
    for a in articles:
    #for a in cur_articles:
        url = a['url']
        url = url.strip()
        # exclude 
        if url[-4:] == '.pdf' or 'https://www.youtube.com/watch?' in url :
            pass
        else:
            if 'web.archive.org' in url:
                    url = '/'.join(re.split('\d{8}',url)[1].split('/')[1:])
                    url = url.strip()
            body_text = a['body_text']
            l = len(body_text)
            # for duplicate urls, use the one with longer length
            if l > articles_cleaned.get(url,[0])[0]:
                articles_cleaned[url] = [len(body_text),body_text]

def tokenize(doc, func_tokenizer, func_stemmer):
        '''
        Tokenize and stem/lemmatize the document.
            - INPUT: string
            - OUTPUT: list of strings
        '''

        #sklearn_tokenizer = TfidfVectorizer(stop_words = 'english').build_tokenizer()

        #snowball = SnowballStemmer('english')
        #print type(doc), type([snowball.stem(word) for word in word_tokenize(doc.lower())])
        #print len(doc), len([snowball.stem(word) for word in word_tokenize(doc.lower())])
        words = []
        func = func_tokenizer
        if func_tokenizer == None:
            func = TfidfVectorizer(stop_words = 'english').build_tokenizer()
        #for word in word_tokenize(doc.strip().lower()):
        for word in func(doc.strip().lower()):
            try:
                # .decode('utf-8'
                t = word.encode('ascii', 'ignore')

                if re.search('[a-zA-Z]+',t):
                    words.append(t)                
            except:
                 t = ''
        stemmed_word = [func_stemmer.stem(word) for word in words ]
        return [w for w in stemmed_word if len(w) >=3 ]


def fit_tfidf(docs, kw_tfidf, func_tokenizer, func_stemmer):
    '''
        - INPUT: X. list of string
        - OUPUT: list of list
    '''
    
    tokenized_articles = [tokenize(doc, func_tokenizer, func_stemmer) for doc in docs]


    #vectorizer = TfidfVectorizer(stop_words = 'english', min_df= 0.01) #v1 max_features = 5000)
    #vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range= (1,3), min_df= 5)   #v2

    vectorizer = TfidfVectorizer(**kw_tfidf)   #v2
   
    documents = [' '.join(article) for article in tokenized_articles]
    vectorizer =  vectorizer.fit(documents)

    #vectors_doc = vectorizer.transform(documents).todense()
    ectors_doc = vectorizer.transform(documents)
    return vectorizer, vectors_doc, tokenized_articles

def transform_tfidf(vectorizer, docs, func_tokenizer,func_stemmer):
    '''
    transform text to tfidf
        - INPUT:  list of string :  (string / text)
        - OUTOUT: list of float
    '''
    tokenized_articles = [tokenize(doc, func_tokenizer, func_stemmer) for doc in docs]
    documents = [' '.join(article) for article in tokenized_articles]
    #return vectorizer.transform(documents).todense(), tokenized_articles
    return vectorizer.transform(documents), tokenized_articles

def format_article(i, df2):
    n_chars = 500
    s = '***** %s ****' % df2.iloc[i]['title']
    s = s + '\n' + df2.iloc[i]['url']
    s = s + '\n\n' + df2.iloc[i]['body_text'][:n_chars]
    return s

def print_articles(idx, df2, scores):
    '''
    print list of articles and associated scores
    '''
    for i in idx:
        str_article = format_article(i, df2)
        print '%d %.2f %s ' % (i, scores[i], str_article)
        print 

def print_articles_2scores(idx, df2, sim, rating):
    '''
    print list of articles and associated scores
    '''
    for i in idx:
        str_article = format_article(i, df2)
        print '%d sim: %.2f rating: %.2f %s ' % (i, sim[i],rating[i], str_article)
        print 

def ascii_text(text):
    '''
    convert to ascii
    '''
    words = text.split()
    cleaned_words = []
    for w in words:
        try:
            t = w.encode('ascii', 'ignore')
        except:
             t = ''
        cleaned_words.append(t)
    return ' '.join(cleaned_words)

def html_to_bodytext(raw_html):
    '''
    extract body text from raw html content
        - INPUT: 
            raw_html    str
        - OUTPUT:
            extracted_text    str
    '''
    try:
        extractor = Extractor(extractor='ArticleExtractor', html=raw_html)
        extracted_text = extractor.getText()
        #l = extracted_text.split('\n')
    # do something with page
    except BadStatusLine:
        print("could not extract body_text ")
    return extracted_text

