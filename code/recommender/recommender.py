
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



  
def get_df_recommender