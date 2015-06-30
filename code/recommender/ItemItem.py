'''
ItemItem simiarity based collaborative filtering usein cosine_similairty of TFIDF 

modified from example code of recommendation system module zipfian dsi program 
'''

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from time import time
from sklearn.metrics.pairwise import linear_kernel
import pickle

class ItemItemRec(object):
    '''
    Item based collaborative filtering
    '''

    def __init__(self, neighborhood_size):
        self.neighborhood_size = neighborhood_size

    def fit_item_features(self, item_features_mat):
        '''
        set neighborhood for items using cosine_similarity of item_feature matrix
            - INPUT: 
                item_features_mat: n_items x n_item_features
        '''
        self.item_features_mat = item_features_mat
        self.n_items= item_features_mat.shape[0]
        self.n_item_features  = item_features_mat.shape[1]
        # ref: http://scikit-learn.org/stable/modules/metrics.html
        self.item_sim_mat = linear_kernel(item_features_mat, item_features_mat)
        #self.item_sim_mat = cosine_similarity(self.ratings_mat.T)
        self._set_neighborhoods()

    def fit(self, ratings_mat):
        '''
        set neighborhood for each time, store rating data and calculate cosine similarity 
            -INPUT:  ratings_mat  ratings_matrix
            - 
        '''
        self.ratings_mat = ratings_mat
        self.n_users = ratings_mat.shape[0]
        self.n_items = ratings_mat.shape[1]
        self.item_sim_mat = cosine_similarity(self.ratings_mat.T)
        self._set_neighborhoods()

    def _set_neighborhoods(self):
        '''
        set items neighborhood 
        '''
        least_to_most_sim_indexes = np.argsort(self.item_sim_mat, 1)
        self.neighborhoods = least_to_most_sim_indexes[:, -self.neighborhood_size:]

    def pred_one_user_from_rating (self, rating_content, report_run_time=False):
        '''
        make prediction for a user based on rating input
            - INPUT: 
                rating_content: 2d array    item_id, rating
            - OUTPUT: 

        '''
        start_time = time()
        #items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        items_rated_by_this_user = rating_content[:,0]
        ratings_vec= sparse.lil_matrix((1, self.n_items ))
        for row in rating_content:
            ratings_vec[0,row[0]] =  row[1]
        # Just initializing so we have somewhere to put rating preds
        out = np.zeros( self.n_items)
        relevant_items_all = []  # n_items 
        for item_to_rate in range(self.n_items):
            relevant_items = np.intersect1d(self.neighborhoods[item_to_rate],
                                            items_rated_by_this_user,
                                            assume_unique=True)  # assume_unique speeds up intersection op
            '''
            out[item_to_rate] = self.ratings_mat[user_id, relevant_items] * \
                self.item_sim_mat[item_to_rate, relevant_items] / \
                self.item_sim_mat[item_to_rate, relevant_items].sum()
            '''
            out[item_to_rate] = ratings_vec[0, relevant_items] * \
                self.item_sim_mat[item_to_rate, relevant_items] / \
                self.item_sim_mat[item_to_rate, relevant_items].sum()

            relevant_items_all.append(relevant_items)
        if report_run_time:
            print("Execution time: %f seconds" % (time()-start_time))
        cleaned_out = np.nan_to_num(out)
        return cleaned_out , relevant_items_all     

    def pred_one_user(self, user_id, report_run_time=False):
        '''
        make prediction for one user
        '''
        start_time = time()
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        # Just initializing so we have somewhere to put rating preds
        out = np.zeros(self.n_items)
        for item_to_rate in range(self.n_items):
            relevant_items = np.intersect1d(self.neighborhoods[item_to_rate],
                                            items_rated_by_this_user,
                                            assume_unique=True)  # assume_unique speeds up intersection op
            out[item_to_rate] = self.ratings_mat[user_id, relevant_items] * \
                self.item_sim_mat[item_to_rate, relevant_items] / \
                self.item_sim_mat[item_to_rate, relevant_items].sum()
        if report_run_time:
            print("Execution time: %f seconds" % (time()-start_time))
        cleaned_out = np.nan_to_num(out)
        return cleaned_out

    def pred_all_users(self, report_run_time=False):
        '''
        make predictions for all the users
        '''
        start_time = time()
        all_ratings = [
            self.pred_one_user(user_id) for user_id in range(self.n_users)]
        if report_run_time:
            print("Execution time: %f seconds" % (time()-start_time))
        return np.array(all_ratings)

    def top_n_recs(self, user_id, n):
        '''
        top n recommendations for user_id
            - INPUT:  
                user_id: int  (the index?)
                n: int top n recommendations
            - OUTPUT: 
                list of n itmes unrated by user_id
        '''
        pred_ratings = self.pred_one_user(user_id)
        item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                        if item not in items_rated_by_this_user]
        return unrated_items_by_pred_rating[-n:]

def get_ratings_data():
    '''
    read in user rating data (tab delmited file)
        - OUTPUT:
            ratings_contents: 
            ratings_as_mat:  matrix (n_users, n_movies).  with rating as the element.
    '''
    ratings_contents = pd.read_table("data/u.data",
                                     names=["user", "movie", "rating", "timestamp"])
    highest_user_id = ratings_contents.user.max()
    highest_movie_id = ratings_contents.movie.max()
    ratings_as_mat = sparse.lil_matrix((highest_user_id, highest_movie_id))
    for _, row in ratings_contents.iterrows():
            # subtract 1 from id's due to match 0 indexing
        ratings_as_mat[row.user-1, row.movie-1] = row.rating
    return ratings_contents, ratings_as_mat

def get_item_feature_data():
    '''
    read in features for the item
    '''
    pass

def make_df_predictions(rating_predict, relevant_items_all, my_rec_engine, df2):
    '''
    make prediction and the relevant information into dataframe
        - OUTPUT: 
            df_recom: dataframe of predicted rating for top k articles
            relevant_all: evidence for the predition list of relevant items/articles and similarity and user score
                relevant_all index is the same as iloc of df_recom
    rating_predict, relevant_items_all = my_rec_engine.pred_one_user_from_rating (rating_content, report_run_time=False)
    '''
    recommed_articles = []
    relevant_all = []

    for i_article in np.argsort(-1.0*rating_predict)[:3]:
        r= {}
        #i_article = 602
        relevant_items = relevant_items_all[i_article]
        sim_vec = my_rec_engine.item_sim_mat[i_article,:].flatten()#, relevant_items] 
        r['score'] = rating_predict[i_article]
        r['title'] = df2.iloc[i_article]['title']
        r['body_text']=df2.iloc[i_article]['body_text']
        r['url'] = df2.iloc[i_article]['url']

        recommed_articles.append(r)
        
        relevent = []
        for i_revelent in relevant_items:
            rele = {}
            rele['rating'] =  rating_vec[i_revelent]
            rele['sim'] = sim_vec[i_revelent]
            rele['title'] = df2.iloc[i_revelent]['title']
            rele['body_text']=df2.iloc[i_revelent]['body_text']
            rele['url'] = df2.iloc[i_revelent]['url']
            rele['i'] = i_revelent
            relevent.append(rele)
        df_relevant = pd.DataFrame(relevent)
        relevant_all.append(df_relevant)
        
    df_recom = pd.DataFrame(recommed_articles)

    print df_recom.head()
    print '-------'
    print relevant_all

    return df_recom, relevant_all 

def dump_pickle_df_prediction(df_recom, relevant_all):
    pickle.dump(df_recom,open('df_recom.pkl','w'))
    pickle.dump(relevant_all,open('relevant_all.pkl','w'))

def run_movie_recom():
    ratings_data_contents, ratings_mat = get_ratings_data()
    my_rec_engine = ItemItemRec(neighborhood_size=75)
    my_rec_engine.fit(ratings_mat)
    user_1_preds = my_rec_engine.pred_one_user(user_id=1, report_run_time=True)

    # Show predicted ratings for user #1 on first 100 items
    print(user_1_preds[:100])
    print(my_rec_engine.top_n_recs(2, 20))


if __name__ == "__main__":

    run_movie_rating()
