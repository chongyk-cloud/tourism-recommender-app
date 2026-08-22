import streamlit as st
import pickle
import numpy as np
import pandas as pd
import pydeck as pdk
import random
from duckduckgo_search import DDGS

# --- DuckDuckGo Image Fetcher ---
@st.cache_data(show_spinner=False)
def get_attraction_photo(attraction_name):
    """Dynamically fetches a real image from the web using DuckDuckGo."""
    try:
        query = f"{attraction_name} attraction China"
        results = DDGS().images(query, max_results=1)
        if results:
            return results[0]['image']
    except Exception as e:
        print(f"Image search failed for {attraction_name}: {e}")
        pass
        
    return f"https://placehold.co/400x300/e0e0e0/000000?text={attraction_name.replace(' ', '+')}"

# --- Page Configuration ---
st.set_page_config(page_title="Tourism Recommender", layout="wide", page_icon="🗺️")

# --- Load Artifacts ---
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

    # Pre-index metadata for fast lookups
    meta_lookup = attr_meta.set_index('attraction_name').to_dict('index')

    # Recommendation function with metadata filtering
    def recommend_for_user(user_id, score_matrix, top_n=5, selected_level="All"):
        if user_id not in user_to_idx:
            return None, None
            
        user_idx = user_to_idx[user_id]
        scores = score_matrix[user_idx].copy()
        seen = train_seen.get(user_idx, set())
        
        # Mask out previously seen attractions
        for i in seen:
            scores[i] = -np.inf
            
        sorted_indices = np.argsort(scores)[::-1]
        recommendations = []

        for idx in sorted_indices:
            if scores[idx] == -np.inf:
                break
                
            name = idx_to_item[idx]
            item_info = meta_lookup.get(name, {})
            item_level = str(item_info.get('attraction_level', 'Unknown'))

            # Apply Level Filter
            if selected_level != "All" and item_level != selected_level:
                continue

            recommendations.append((name, scores[idx], item_level))
            
            if len(recommendations) >= top_n:
                break

        return recommendations, seen

    # --- UI Layout ---
    st.title("🗺️ Personalized Tourism Recommender")
    st.markdown("Customize your travel requirements and generate a personalized itinerary.")

    # --- Sidebar Controls / Filters ---
    st.sidebar.header("⚙️ Recommendation Filters")

    # 1. Model Selection
    model_options = {
        "Hybrid Model (Recommended)": hybrid,
        "Collaborative Filtering": pred_cf,
        "Content-Based": pred_content,
        "Neural Network": pred_nn
    }
    selected_model_name = st.sidebar.selectbox("Recommendation Algorithm", list(model_options.keys()))
    active_matrix = model_options[selected_model_name]

    # 2. Attraction Level Dropdown Filter
    available_levels = ["All"]
    if 'attraction_level' in attr_meta.columns:
        unique_levels = sorted([str(lvl) for lvl in attr_meta['attraction_level'].dropna().unique()])
        available_levels.extend(unique_levels)
        
    selected_level = st.sidebar.selectbox("Attraction Rating / Level", options=available_levels)

    # 3. Top-N Count Slider
    top_n = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

    # --- Main Input ---
    tourist_id = st.text_input("Tourist ID", value="605")

    if tourist_id:
        try:
            tourist_id_int = int(tourist_id)
        except ValueError:
            st.error("Please enter a valid integer Tourist ID.")
            st.stop()

        if tourist_id_int not in user_to_idx:
            st.warning(f"Tourist ID {tourist_id_int} not found in the dataset.")
            st.stop()
        else:
            recommendations, seen = recommend_for_user(
                user_id=tourist_id_int,
                score_matrix=active_matrix,
                top_n=top_n,
                selected_level=selected_level
            )

            if not recommendations:
                st.warning(f"No unseen attractions found matching the criteria: **Level = {selected_level}**.")
                st.stop()

            # --- Tabs Section ---
            tab1, tab2, tab3 = st.tabs(["🎯 Top Recommendations", "📍 3D Spatial Map", "⭐ Past Ratings"])

            with tab1:
                st.subheader("Your Personalized Itinerary")
                cols = st.columns(len(recommendations))
                
                for i, (name, score, level) in enumerate(recommendations):
                    with cols[i]:
                        image_url = get_attraction_photo(name)
                        st.image(image_url, use_container_width=True)
                        st.markdown(f"**{name}**")
                        st.caption(f"Score: {score:.3f} | {level}")

            with tab2:
                st.subheader("Attraction Locations")
                st.info("Note: Using simulated coordinates for 3D visualization. Add real 'lat' and 'lon' data to your dataset to map exact locations.")
                
                map_data = []
                for name, score, _ in recommendations:
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
                st.info(f"User has interacted with {len(seen)} attractions in the training set.")

except Exception as e:
    st.error(f"An error occurred: {e}")
