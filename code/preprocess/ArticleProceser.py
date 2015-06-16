def clean_articles(cur_articles, articles_cleaned):
    articles = list(cur_articles)
    print '%d articles found' % len(articles)
    counter = 0
    for a in articles:
    #for a in cur_articles:
        url = a['url']
        body_text = a['body_text']
        l = len(body_text)
        if l > articles_cleaned.get(url,[0])[0]:
            articles_cleaned[url] = [len(body_text),body_text]

def tokenize(doc):
        '''
        Tokenize and stem/lemmatize the document.
            - INPUT: string
            - OUTPUT: list of strings

        '''
        snowball = SnowballStemmer('english')
        #print type(doc), type([snowball.stem(word) for word in word_tokenize(doc.lower())])
        #print len(doc), len([snowball.stem(word) for word in word_tokenize(doc.lower())])
        words = []
        for word in word_tokenize(doc.strip().lower()):
            try:
                # .decode('utf-8'
                t = word.encode('ascii', 'ignore')
                words.append(t)                
            except:
                 t = ''
        return [snowball.stem(word) for word in words]


def fit_tfidf(docs):
    '''
        - INPUT: X. list of string
        - OUPUT: list of list
    '''
    tokenized_articles = [tokenize(doc) for doc in docs]
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 5000)
    documents = [' '.join(article) for article in tokenized_articles]
    vectorizer =  vectorizer.fit(documents)

    vectors_doc = vectorizer.transform(documents).todense()
    return vectorizer, vectors_doc, tokenized_articles

def transform_tfidf(vectorizer, docs):
    '''
    transform text to tfidf
        - INPUT:  list of string :  (string / text)
        - OUTOUT: list of float
    '''
    tokenized_articles = [tokenize(doc) for doc in docs]
    documents = [' '.join(article) for article in tokenized_articles]
    return vectorizer.transform(documents).todense(), tokenized_articles




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




