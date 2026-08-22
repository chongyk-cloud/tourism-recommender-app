import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Tourism Recommender", layout="wide")

# Load all saved artifacts
@st.cache_resource
def load_artifacts():
    pred_cf = np.load('pred_cf_matrix.npy')
    pred_content = np.load('pred_content_matrix.npy')
    pred_nn = np.load('pred_nn_matrix.npy')
    hybrid = np.load('hybrid_matrix.npy')

    with open('user_ids.pkl', 'rb') as f:
        user_ids = pickle.load(f)
    with open('item_ids.pkl', 'rb') as f:
        item_ids = pickle.load(f)
    with open('idx_to_item.pkl', 'rb') as f:
        idx_to_item = pickle.load(f)
    with open('user_to_idx.pkl', 'rb') as f:
        user_to_idx = pickle.load(f)
    with open('train_seen.pkl', 'rb') as f:
        train_seen = pickle.load(f)

    attr_meta = pd.read_csv('attraction_metadata.csv')
    return pred_cf, pred_content, pred_nn, hybrid, user_ids, item_ids, idx_to_item, user_to_idx, train_seen, attr_meta

# Load data
try:
    pred_cf, pred_content, pred_nn, hybrid, user_ids, item_ids, idx_to_item, user_to_idx, train_seen, attr_meta = load_artifacts()

    # Recommendation function
    def recommend_for_user(user_id, score_matrix, top_n=5):
        if user_id not in user_to_idx:
            return None, None
        user_idx = user_to_idx[user_id]
        scores = score_matrix[user_idx].copy()
        seen = train_seen.get(user_idx, set())
        # Mask seen items
        for i in seen:
            scores[i] = -np.inf
        # Get top indices
        top_indices = np.argsort(scores)[::-1][:top_n]
        # Convert indices to attraction names and scores
        recommendations = [(idx_to_item[i], scores[i]) for i in top_indices]
        return recommendations, seen

    # Streamlit UI
    st.title("🗺️ Personalized Tourism Recommender")
    st.markdown("Enter a tourist ID to get top attraction recommendations based on their past ratings.")

    # Input
    tourist_id = st.text_input("Tourist ID", value="605")
    if tourist_id:
        try:
            tourist_id_int = int(tourist_id)
        except ValueError:
            st.error("Please enter a valid integer Tourist ID.")
            st.stop()

        # Check if tourist exists
        if tourist_id_int not in user_to_idx:
            st.warning(f"Tourist ID {tourist_id_int} not found in the system. Showing popular attractions instead.")
            st.info("Displaying a sample of popular attractions (not personalised).")
            st.stop()
        else:
            # Get recommendations using the hybrid model
            recommendations, seen = recommend_for_user(tourist_id_int, hybrid, top_n=5)

            # Display past ratings
            st.subheader("⭐ Your Previous Ratings")
            st.info("Past ratings are shown in the notebook; for the app, you can load a CSV with user ratings.")

            # Display recommendations
            st.subheader("🎯 Top 5 Recommended Attractions")
            cols = st.columns(5)
            for i, (name, score) in enumerate(recommendations):
                with cols[i]:
                    st.markdown(f"**{name}**")
                    st.caption(f"Score: {score:.3f}")
                    meta = attr_meta[attr_meta['attraction_name'] == name].iloc[0]
                    st.caption(f"Category: {meta['attraction_category']}")
                    st.caption(f"Level: {meta['attraction_level']}")

except Exception as e:
    st.error(f"An error occurred while loading the app data: {e}")
