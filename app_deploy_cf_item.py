import streamlit as st

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests

# item-to-item collaborative filtering
with open('item_item_sim_matrix.pkl', 'rb') as f:
    item_item_sim_matrix = pickle.load(f)

def rec_collaborative_filtering_by_item(game):
    top_5_similar_items = list(
        item_item_sim_matrix\
            .loc[game]\
            .sort_values(ascending=False)\
            .iloc[:5]\
        .index
    )
    print(top_5_similar_items)

    return top_5_similar_items

with open('game_df.pkl', 'rb') as f:
    game_df = pickle.load(f)

game_fr = game_df['Title'].values

st.image('https://tedis.telkom.design/assets/download_logo/logo-gameqoo.png', width=150)
st.title('Recommendation System')

game_enjoyed = st.selectbox(
'Select your favorite game',
game_fr)

if st.button('Find similar games'):
    similar_game = rec_collaborative_filtering_by_item(game_enjoyed)

    for count, game in enumerate(similar_game):
        try:
            st.subheader(game)
            st.image(game_df.loc[game_df['Title'] == game].CoverWebUrl.values[0])
        except:
            st.caption('Image is Unavailable :(')
