
def get_rating(good_urls, meh_urls, df2):
    '''
    generate rating_content from list of good and meh urls
        -OUTPUT: rating_content list of list  (i_article, rating_score)
    '''
    score_good = 4
    score_meh = 1
    rating_content = []
    cond1 = df2.url.isin(good_urls)
    #print type(cond1)
    #print cond1[:5]
    #index_good = list(df2[cond1].index)
    ilocs_good = [i for i in xrange(df2.shape[0]) if cond1.iloc[i]]
    #print i_rows_good
    
    cond2 = df2.url.isin(meh_urls)
    #index_meh = list(df2[cond2].index)
    #print index_meh
    ilocs_meh = [i for i in xrange(df2.shape[0]) if cond2.iloc[i]]
    
    for i in ilocs_good:
        rating_content.append((i,score_good))
    for i in ilocs_meh:
         rating_content.append((i,score_meh))
    return np.array(rating_content)
print get_rating(good_urls, meh_urls, df2)    


def run_article_recom():
    '''
    test run of article recommendations
    '''
 
    good_urls = ['http://www.bloomberg.com/news/articles/2015-06-04/help-wanted-black-belts-in-data', 
            'http://bits.blogs.nytimes.com/2015/04/28/less-noise-but-more-money-in-data-science/'
           , 'http://bits.blogs.nytimes.com/2013/06/19/sizing-up-big-data-broadening-beyond-the-internet/'
           ]
    meh_urls = ['http://bits.blogs.nytimes.com/2013/04/11/education-life-teaching-the-techies/' # too short, no content
    , 'http://www.nytimes.com/2013/04/14/education/edlife/universities-offer-courses-in-a-hot-new-field-data-science.html'  
    ]

    rating_content = get_rating(good_urls, meh_urls, df2)  
    rating_vec = np.zeros(df2.shape[0])
    for row in rating_content:
        rating_vec[row[0]] =  row[1]
    print rating_content
    print [i for i in xrange(len(rating_vec)) if rating_vec[i]>0]
