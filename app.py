import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import random
import requests
import pickle
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Personalized Tourism Recommender", layout="wide", page_icon="🗺️")

# --- 2. IMAGE DATABASE & FETCHER ---
IMAGE_DATABASE = {
    "Wu Dang Shan": "https://www.travelchinaguide.com/images/photogallery/2010/wudang-mountain.jpg",
    "Lao Jun Shan": "https://www.travelchinaguide.com/images/photogallery/2018/0822161406.jpg",
    "Wu Yi Shan": "https://www.travelchinaguide.com/images/photogallery/2012/0517112028.jpg",
    "Long Hu Shan": "https://www.travelchinaguide.com/images/photogallery/2015/1022153215.jpg",
    "Tian Mu Hu": "https://commons.wikimedia.org/wiki/Special:FilePath/%E5%A4%A9%E7%9B%AE%E6%B9%96%E5%A4%A7%E9%96%80.JPG?width=800",
    "Lao Shan": "https://commons.wikimedia.org/wiki/Special:FilePath/Mount_Lao_from_within_the_Laoshan_National_Park.jpg?width=800"
}

@st.cache_data(show_spinner=False)
def get_attraction_photo(attraction_name):
    if attraction_name in IMAGE_DATABASE:
        return IMAGE_DATABASE[attraction_name]
        
    endpoint = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "TourismRecommenderApp/8.0"}
    
    NAME_ALIASES = {
        "Ba Li Gou": "Baligou", "Baili Gou": "Baligou", "Long Men Shi Ku": "Longmen Grottoes",
        "Qing Ming Shang He Yuan": "Millennium City Park", "Si Gu Niang Shan": "Mount Siguniang",
        "E Mei Shan": "Mount Emei", "Lao Jun Shan": "Mount Laojun", "Wu Dang Shan": "Wudang Mountains",
        "Kai Feng Fu": "Kaifeng Prefecture", "Ning De Yuan Yang Xi": "Ningde"
    }
    
    queries = []
    if attraction_name in NAME_ALIASES:
        alias = NAME_ALIASES[attraction_name]
        queries.extend([f"{alias} China", alias, f"{alias} scenic area", f"{alias} Valley"])
    
    words = attraction_name.strip().split()
    joined_name = "".join(words)
    queries.extend([f"{joined_name} China", joined_name, f"{attraction_name} China", attraction_name])
    
    invalid_image_terms = ['map', 'logo', 'flag', 'emblem', 'icon', '.svg', 'symbol']
    
    for q in queries:
        params = {
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": q, "gsrlimit": 2, "prop": "pageimages", "pithumbsize": 600
        }
        try:
            response = requests.get(endpoint, params=params, headers=headers, timeout=3).json()
            pages = response.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                if "thumbnail" in page_info and "source" in page_info["thumbnail"]:
                    img_url = page_info["thumbnail"]["source"]
                    if not any(bad in img_url.lower() for bad in invalid_image_terms):
                        return img_url
        except Exception:
            continue
            
    seed = sum(ord(c) for c in attraction_name)
    return f"https://loremflickr.com/400/300/landscape,chinese?lock={seed}"

# --- 3. ML MODEL & DATA LOADER ---
@st.cache_resource
def load_all_data():
    # Load primary dataset
    try:
        df_raw = pd.read_csv('tourism_recommendation_dataset_en.csv')
    except Exception:
        df_raw = pd.read_csv('attraction_metadata.csv')
        
    attr_meta = df_raw[['attraction_name', 'attraction_category', 'attraction_level']].drop_duplicates(subset=['attraction_name'])

    script_dir = os.path.dirname(os.path.abspath(__file__))

   # STEP B: Load evaluation metrics dynamically from Jupyter Notebook export
    try:
        eval_metrics_df = pd.read_csv('eval_metrics.csv')
    except Exception:
        # Fallback if the CSV hasn't been generated yet or is in the wrong folder
        eval_metrics_df = pd.DataFrame({
            "Algorithm": ["Waiting for Jupyter Notebook Export..."],
            "RMSE": [0.0], "MSE": [0.0], "MAE": [0.0], "Precision@5": [0.0], "Recall@5": [0.0]
        })
    # Load ML artifacts safely
    matrices = {}
    ml_ready = False
    
    try:
        with open(os.path.join(script_dir, 'idx_to_item.pkl'), 'rb') as f:
            idx_to_item = pickle.load(f)
        with open(os.path.join(script_dir, 'user_to_idx.pkl'), 'rb') as f:
            user_to_idx = pickle.load(f)
        with open(os.path.join(script_dir, 'train_seen.pkl'), 'rb') as f:
            train_seen = pickle.load(f)
            
        # Dynamically load whichever ML matrices are present in the folder
        model_files = {
            "Content-Based Filtering": 'pred_content_matrix.npy',
            "Collaborative Filtering (SVD)": 'pred_cf_matrix.npy',
            "Neural Collaborative Filtering": 'pred_nn_matrix.npy',
            "Hybrid Recommender (Ensemble)": 'hybrid_matrix.npy'
        }
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(script_dir, filename)
            if os.path.exists(filepath):
                matrices[model_name] = np.load(filepath, allow_pickle=True)
                
        if matrices:
            ml_ready = True
            
    except Exception:
        idx_to_item, user_to_idx, train_seen = None, None, None
        ml_ready = False

    return df_raw, attr_meta, eval_metrics_df, matrices, idx_to_item, user_to_idx, train_seen, ml_ready

try:
    df_raw, attr_meta, eval_metrics_df, matrices, idx_to_item, user_to_idx, train_seen, ml_ready = load_all_data()

    # --- 4. MULTI-MODEL RECOMMENDATION ENGINE ---
    def generate_recommendations(tourist_id, selected_model, age, gender, province, duration, top_n=8):
        # 1. First, apply user's explicit sidebar filters to get valid candidate attractions
        filtered = df_raw.copy()
        if age != "Ignore": filtered = filtered[filtered['age_group'] == age]
        if gender != "Ignore": filtered = filtered[filtered['gender'] == gender]
        if province != "Ignore": filtered = filtered[filtered['province'] == province]
        
        if duration != "Ignore":
            if duration == "Short (1-3 hours)": filtered = filtered[filtered['visit_duration_hours'] <= 3]
            elif duration == "Medium (3-5 hours)": filtered = filtered[(filtered['visit_duration_hours'] > 3) & (filtered['visit_duration_hours'] <= 5)]
            elif duration == "Long (5+ hours)": filtered = filtered[filtered['visit_duration_hours'] > 5]
            
        valid_candidates = set(filtered['attraction_name'].unique())
        
        if not valid_candidates:
            return [], False # No items match filters

        # 2. If valid ML Tourist ID provided, use the specifically selected AI Model predictions
        if ml_ready and tourist_id in user_to_idx and selected_model in matrices:
            user_idx = user_to_idx[tourist_id]
            selected_matrix = matrices[selected_model]
            scores = selected_matrix[user_idx].copy()
            seen_indices = train_seen.get(user_idx, set())
            
            recs = []
            for item_idx, item_name in idx_to_item.items():
                if item_idx in seen_indices:
                    continue # Skip places already visited
                if item_name in valid_candidates:
                    recs.append((item_name, scores[item_idx]))
                    
            # Sort by highest AI Model Score
            recs.sort(key=lambda x: x[1], reverse=True)
            return recs[:top_n], True # True indicates it is ML personalized
            
        # 3. Fallback: Rule-Based Popularity (if no ID, invalid ID, or missing ML files)
        grouped = filtered.groupby('attraction_name').agg(
            avg_rating=('rating', 'mean'),
            visit_count=('rating', 'count')
        ).reset_index()
        
        top_spots = grouped.sort_values(by=['avg_rating', 'visit_count'], ascending=[False, False]).head(top_n)
        recs = [(row['attraction_name'], row['avg_rating']) for _, row in top_spots.iterrows()]
        return recs, False # False indicates it is Fallback Popularity

    # --- 5. SIDEBAR & UI CONTROLS ---
    st.title("🗺️ Personalized Tourism Recommender")
    st.markdown("A dual-perspective prototype: explore curated travel plans or inspect backend AI evaluation benchmarks.")

    st.sidebar.header("🎯 Traveler Profile & Filters")
    
    # NEW: Algorithm Selection Dropdown
    st.sidebar.subheader("🧠 Algorithm Selection")
    
    # Provide the models retrieved from the folder, or default list if unavailable
    if ml_ready:
        model_options = list(matrices.keys())
    else:
        model_options = [
            "Hybrid Recommender (Ensemble)", "Collaborative Filtering (SVD)", 
            "Neural Collaborative Filtering", "Content-Based Filtering"
        ]
        
    selected_model = st.sidebar.selectbox(
        "Choose Recommendation Engine", 
        options=model_options,
        help="Select the underlying AI model used to generate your recommendations."
    )
    
    # Tourist ID Input for ML Model
    t_id_input = st.sidebar.text_input("🔑 Tourist ID (e.g., 605)", value="605", help="Enter ID for AI predictions. Leave blank for general popularity.")
    try:
        active_tourist_id = int(t_id_input) if t_id_input.strip() else None
    except ValueError:
        active_tourist_id = None
        st.sidebar.error("Tourist ID must be a number.")

    st.sidebar.divider()

    # Dropdowns
    def get_default_index(opts, target): return opts.index(target) if target in opts else len(opts) - 1
    
    avail_ages = sorted(df_raw['age_group'].dropna().unique().tolist()) + ["Ignore"]
    selected_age = st.sidebar.selectbox("Age Group", avail_ages, index=get_default_index(avail_ages, "Ignore"))

    avail_genders = sorted(df_raw['gender'].dropna().unique().tolist()) + ["Ignore"]
    selected_gender = st.sidebar.selectbox("Gender", avail_genders, index=get_default_index(avail_genders, "Ignore"))

    avail_provinces = sorted(df_raw['province'].dropna().unique().tolist()) + ["Ignore"]
    selected_province = st.sidebar.selectbox("Province", avail_provinces, index=get_default_index(avail_provinces, "Ignore"))

    dur_options = ["Short (1-3 hours)", "Medium (3-5 hours)", "Long (5+ hours)", "Ignore"]
    selected_duration = st.sidebar.selectbox("Visit Duration", dur_options, index=get_default_index(dur_options, "Ignore"))

    top_n = st.sidebar.slider("Number of Recommendations", 1, 12, 8)

    # Fetch Recommendations (Passing the selected_model parameter)
    recommendations, is_personalized = generate_recommendations(
        active_tourist_id, selected_model, selected_age, selected_gender, selected_province, selected_duration, top_n
    )

    # --- 6. TABS STRUCTURE ---
    tab1, tab2, tab3 = st.tabs(["🎯 Top Recommendations", "📍 3D Spatial Map", "⚙️ Model Evaluation & Diagnostics"])

    # ========================== TAB 1: TRAVELER VIEW ==========================
    with tab1:
        st.subheader("Your Personalized Itinerary")
        
        if not ml_ready:
            st.warning("⚠️ ML Model files (.npy, .pkl) not found. Running in Fallback Popularity Mode.")
        elif is_personalized:
            # Dynamically update the success message to show which model is active
            st.success(f"🤖 Showing **{selected_model}** Predictions for Tourist {active_tourist_id}")
        else:
            st.info("📊 Showing General Popularity Recommendations (Cold Start / Invalid ID)")
            
        if not recommendations:
            st.warning("No attractions found matching all your criteria. Try setting some filters to 'Ignore'.")
        else:
            num_cols = 4
            for row_idx in range(0, len(recommendations), num_cols):
                row_items = recommendations[row_idx : row_idx + num_cols]
                cols = st.columns(num_cols)
                
                for i, (name, score) in enumerate(row_items):
                    with cols[i]:
                        meta_row = attr_meta[attr_meta['attraction_name'] == name]
                        level = meta_row['attraction_level'].iloc[0] if not meta_row.empty else "5A"
                        img_url = get_attraction_photo(name)
                        
                        st.markdown(
                            f"""
                            <div style="height: 200px; width: 100%; overflow: hidden; border-radius: 8px; margin-bottom: 10px;">
                                <img src="{img_url}" style="width: 100%; height: 100%; object-fit: cover;">
                            </div>
                            """, unsafe_allow_html=True
                        )
                        st.markdown(f"**{name}**")
                        
                        # Label score contextually
                        score_label = "Model Score" if is_personalized else "Avg Rating"
                        st.caption(f"{score_label}: {score:.3f} ⭐ | {level}")

    # ========================== TAB 2: SPATIAL MAP ==========================
    with tab2:
        st.subheader("Attraction Spatial Layout")
        st.info("Simulated coordinate layers representing geographic distribution across destination regions.")

        if recommendations:
            map_data = []
            for name, score in recommendations:
                lat, lon = 35.0 + random.uniform(-4, 4), 105.0 + random.uniform(-4, 4)
                map_data.append({"name": name, "lat": lat, "lon": lon, "score": float(score)})

            map_df = pd.DataFrame(map_data)
            view_state = pdk.ViewState(latitude=35.0, longitude=105.0, zoom=4, pitch=45)
            layer = pdk.Layer(
                "ColumnLayer", data=map_df, get_position=["lon", "lat"],
                get_elevation="score * 20000", elevation_scale=10, radius=22000,
                get_fill_color=[255, 75, 75, 200], pickable=True, auto_highlight=True,
            )
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}\nScore: {score}"}))

    # ========================== TAB 3: DIAGNOSTICS ==========================
    with tab3:
        st.subheader("📊 Recommendation Engine Diagnostics & Evaluation")
        st.markdown("Quantitative performance assessment across collaborative, content-based, neural, and ensemble architectures.")

        # Dynamically extract values from the CSV dataframe for the metrics (if available)
        try:
            ensemble_row = eval_metrics_df[eval_metrics_df["Algorithm"] == "Hybrid Recommender (Ensemble)"].iloc[0]
            svd_row = eval_metrics_df[eval_metrics_df["Algorithm"] == "Collaborative Filtering (SVD)"].iloc[0]
            
            rmse_val = f"{ensemble_row['RMSE']:.4f}"
            mse_val = f"{ensemble_row['MSE']:.4f}"
            prec_val = f"{ensemble_row['Precision@5'] * 100:.2f}%"
            rec_val = f"{ensemble_row['Recall@5'] * 100:.2f}%"
            
            rmse_delta = f"{ensemble_row['RMSE'] - svd_row['RMSE']:.4f} vs SVD"
            mse_delta = f"{ensemble_row['MSE'] - svd_row['MSE']:.4f} vs SVD"
            
        except Exception:
            # Fallback if the CSV structure doesn't match perfectly yet
            rmse_val, mse_val, prec_val, rec_val = "N/A", "N/A", "N/A", "N/A"
            rmse_delta, mse_delta = None, None

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Ensemble RMSE", rmse_val, delta=rmse_delta, delta_color="inverse")
        m_col2.metric("Ensemble MSE", mse_val, delta=mse_delta, delta_color="inverse")
        m_col3.metric("Precision@5", prec_val)
        m_col4.metric("Recall@5", rec_val)

        st.divider()
        st.markdown("### Comparative Performance Matrix")
        
        # Display the loaded CSV data
        if "RMSE" in eval_metrics_df.columns:
            st.dataframe(
                eval_metrics_df.style.highlight_min(subset=["RMSE", "MSE", "MAE"], color="#2E7D32")
                                     .highlight_max(subset=["Precision@5", "Recall@5"], color="#1565C0"),
                use_container_width=True
            )
        else:
            st.dataframe(eval_metrics_df, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred while loading the application: {e}")
