import streamlit as st

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests

# content-based
with open('tfidf_df.pkl', 'rb') as f:
    tfidf_df = pickle.load(f)

def rec_content_based(list_of_games_enjoyed):
    games_enjoyed_df = tfidf_df.reindex(list_of_games_enjoyed)
    user_prof = games_enjoyed_df.mean()

    tfidf_subset_df = tfidf_df.drop(list_of_games_enjoyed, axis=0)

    similarity_array = cosine_similarity(user_prof.values.reshape(1, -1), tfidf_subset_df)
    similarity_df = pd.DataFrame(similarity_array.T, index=tfidf_subset_df.index, columns=["similarity_score"])
    sorted_similarity_df = similarity_df.sort_values(by="similarity_score", ascending=False)

    print(sorted_similarity_df.head())

with open('game_df.pkl', 'rb') as f:
    game_df = pickle.load(f)

game_fr = game_df['Title'].values

st.image('https://tedis.telkom.design/assets/download_logo/logo-gameqoo.png', width=150)
st.title('Recommendation System')

list_of_games_enjoyed = st.multiselect('Select your favorite game', game_fr)

if st.button('Find similar games'):
    games_enjoyed_df = tfidf_df.reindex(list_of_games_enjoyed)
    user_prof = games_enjoyed_df.mean()

    tfidf_subset_df = tfidf_df.drop(list_of_games_enjoyed, axis=0)

    similarity_array = cosine_similarity(user_prof.values.reshape(1, -1), tfidf_subset_df)
    similarity_df = pd.DataFrame(similarity_array.T, index=tfidf_subset_df.index, columns=["similarity_score"])
    sorted_similarity_df = similarity_df.sort_values(by="similarity_score", ascending=False)
    print(sorted_similarity_df.head())

    games_to_recommend = sorted_similarity_df.head().index   

    for count, game in enumerate(games_to_recommend):
        try:
            st.subheader(game)
            cos_sim = format(sorted_similarity_df.head().values[count][0], '.4f')
            text = 'Cosine Similarity = ' + cos_sim
            st.caption(text)
            st.image(game_df.loc[game_df['Title'] == game].CoverWebUrl.values[0])
        except:
            st.caption('Image is Unavailable :(')