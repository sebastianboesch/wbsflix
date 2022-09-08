import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

links = pd.read_csv('ml-latest-small/links.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')


st.set_page_config(page_title='WBSFLIX group 4', page_icon="random", layout="wide", initial_sidebar_state="auto", menu_items=None)

st.title("WBSFLIX")
st.sidebar.write('menu selector')


# input

pop_thres = 20 # predefined
user_id = st.sidebar.number_input("userId")
# user_id = 509
item_id = st.sidebar.number_input("itemId")
# item_id = 7076
# n = st.sidebar.number_input("how many movies per row?")
# n = '6'
# n = input("how many movies to display?")
n = st.sidebar.slider("how many movies to display?", min_value=1, max_value=10, value=3, step=1, format=None, key='n-movies', disabled=False)


# popularity-based (n, pop_thres)

popularity = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
popularity['rating_count'] = ratings.groupby('movieId')['rating'].count()
best_movies = movies.join(popularity).dropna(subset=['rating_count'])
best_movies.loc[best_movies['rating_count'] >= pop_thres].sort_values(['rating'], ascending=False)
top_n_pop = best_movies.head(int(n))

# print(top_n_pop['title'])
st.dataframe(top_n_pop['title'])



# item-based (n, item_id)

places_crosstab = pd.pivot_table(data=ratings, values='rating', index='userId', columns='movieId')
top_popular_placeID = item_id
Tortas_ratings = places_crosstab[top_popular_placeID]
Tortas_ratings[Tortas_ratings>=0] # exclude NaNs
similar_to_Tortas = places_crosstab.corrwith(Tortas_ratings)
corr_Tortas = pd.DataFrame(similar_to_Tortas, columns=['PearsonR'])
corr_Tortas.dropna(inplace=True)
rating = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
rating['rating_count'] = ratings.groupby('movieId')['rating'].count()
Tortas_corr_summary = corr_Tortas.join(rating['rating_count'])
Tortas_corr_summary.drop(top_popular_placeID, inplace=True) # drop Tortas Locas itself
top_n_item = Tortas_corr_summary[Tortas_corr_summary['rating_count']>=10].sort_values('PearsonR', ascending=False).head(int(n))
places =  movies[['movieId', 'title']]
top_n_item = top_n_item.merge(places, left_index=True, right_on="movieId")

# print(top_n_item['title'])
if (item_id != 0):
    st.dataframe(top_n_item['title'])



# user-based (n, user_id)

users_items = pd.pivot_table(data=ratings, values='rating', index='userId', columns='movieId')
users_items.fillna(0, inplace=True)
user_similarities = pd.DataFrame(cosine_similarity(users_items), columns=users_items.index, index=users_items.index)
weights = (user_similarities.query("userId!=@user_id")[user_id] / sum(user_similarities.query("userId!=@user_id")[user_id]))
not_visited_restaurants = users_items.loc[users_items.index!=user_id, users_items.loc[user_id,:]==0]
not_visited_restaurants.T
weighted_averages = pd.DataFrame(not_visited_restaurants.T.dot(weights), columns=["predicted_rating"])
recommendations = weighted_averages.merge(movies, left_index=True, right_on="movieId")
top_n_user = recommendations.sort_values("predicted_rating", ascending=False).head(int(n))

# print(top_n_user['title'])
if (user_id != 0):
    st.dataframe(top_n_user['title'])