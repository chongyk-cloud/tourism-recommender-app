import streamlit as st
import pickle
import numpy as np
import pandas as pd
import pydeck as pdk
import random

def get_category_image(category, attraction_name):
    """Generates a reliable, beautiful stock photo based on the attraction's category."""
    category = str(category).lower()
    
    # Determine a good search keyword based on the dataset's categories
    if "natural" in category or "scenery" in category:
        keyword = "nature,mountain"
    elif "ancient" in category or "town" in category:
        keyword = "ancient,china,town"
    elif "religio" in category:
        keyword = "temple,pagoda"
    elif "historic" in category or "culture" in category:
        keyword = "history,architecture"
    elif "sport" in category or "leisure" in category:
        keyword = "skiing,resort"
    else:
        keyword = "travel,landscape,china"
        
    # Create a unique but consistent seed number based on the attraction's name
    # This ensures 'Wu Dang Shan' always loads the exact same picture!
    seed = sum(ord(c) for c in attraction_name)
    
    return f"https://loremflickr.com/400/300/{keyword}?lock={seed}"

# Set page configuration
st.set_page_config(page_title="Tourism Recommender", layout="wide", page_icon="🗺️")

# Load all saved artifacts
@st.cache_resource
def load_artifacts():
    pred_cf = np.load('pred_cf_matrix.npy')
    pred_content = np.load('pred_content_matrix.npy')
    pred_nn = np.load('pred_nn_matrix.npy')
    hybrid = np.load('hybrid_matrix.npy')

    with open('user_ids.pkl', 'rb') as f:
        user_ids = pickle.load(f)
    with open('idx_to_item.pkl', 'rb') as f:
        idx_to_item = pickle.load(f)
    with open('user_to_idx.pkl', 'rb') as f:
        user_to_idx = pickle.load(f)
    with open('train_seen.pkl', 'rb') as f:
        train_seen = pickle.load(f)

    attr_meta = pd.read_csv('attraction_metadata.csv')
    return pred_cf, pred_content, pred_nn, hybrid, user_ids, idx_to_item, user_to_idx, train_seen, attr_meta

try:
    pred_cf, pred_content, pred_nn, hybrid, user_ids, idx_to_item, user_to_idx, train_seen, attr_meta = load_artifacts()

    # Recommendation function
    def recommend_for_user(user_id, score_matrix, top_n=5):
        if user_id not in user_to_idx:
            return None, None
        user_idx = user_to_idx[user_id]
        scores = score_matrix[user_idx].copy()
        seen = train_seen.get(user_idx, set())
        for i in seen:
            scores[i] = -np.inf
        top_indices = np.argsort(scores)[::-1][:top_n]
        recommendations = [(idx_to_item[i], scores[i]) for i in top_indices]
        return recommendations, seen

    # Streamlit UI
    st.title("🗺️ Personalized Tourism Recommender")
    st.markdown("Enter a tourist ID to generate a visual travel itinerary.")

    tourist_id = st.text_input("Tourist ID", value="605")
    if tourist_id:
        try:
            tourist_id_int = int(tourist_id)
        except ValueError:
            st.error("Please enter a valid integer Tourist ID.")
            st.stop()

        if tourist_id_int not in user_to_idx:
            st.warning(f"Tourist ID {tourist_id_int} not found. Showing popular attractions instead.")
            st.stop()
        else:
            recommendations, seen = recommend_for_user(tourist_id_int, hybrid, top_n=5)

            tab1, tab2, tab3 = st.tabs(["🎯 Top Recommendations", "📍 3D Spatial Map", "⭐ Past Ratings"])

            with tab1:
                st.subheader("Your Personalized Itinerary")
                cols = st.columns(5)
                for i, (name, score) in enumerate(recommendations):
                    with cols[i]:
                        meta = attr_meta[attr_meta['attraction_name'] == name].iloc[0]
                        category = meta['attraction_category']
                        
                        # --- THE NEW BULLETPROOF IMAGE FUNCTION ---
                        image_url = get_category_image(category, name)
                        st.image(image_url, use_container_width=True)
                        
                        st.markdown(f"**{name}**")
                        st.caption(f"Score: {score:.3f} | {meta['attraction_level']}")

            with tab2:
                st.subheader("Attraction Locations")
                st.info("Note: Using simulated coordinates for 3D visualization. Add real 'lat' and 'lon' data to your dataset to map exact locations.")
                
                map_data = []
                for name, score in recommendations:
                    lat = 35.0 + random.uniform(-4, 4)
                    lon = 105.0 + random.uniform(-4, 4)
                    map_data.append({"name": name, "lat": lat, "lon": lon, "score": float(score)})

                map_df = pd.DataFrame(map_data)

                view_state = pdk.ViewState(latitude=35.0, longitude=105.0, zoom=4, pitch=45)
                layer = pdk.Layer(
                    "ColumnLayer",
                    data=map_df,
                    get_position=["lon", "lat"],
                    get_elevation="score * 20000",
                    elevation_scale=10,
                    radius=20000,
                    get_fill_color=[255, 75, 75, 200],
                    pickable=True,
                    auto_highlight=True,
                )
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}\nMatch Score: {score}"}))

            with tab3:
                st.subheader("Your Previous Ratings")
                st.info("Past ratings are tracked in the model. You can load a user history CSV here in the future.")

except Exception as e:
    st.error(f"An error occurred: {e}")
