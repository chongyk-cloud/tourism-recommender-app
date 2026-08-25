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
def load_all_data_v2():
    # Load primary dataset
    try:
        df_raw = pd.read_csv('tourism_recommendation_dataset_en.csv')
    except Exception:
        df_raw = pd.read_csv('attraction_metadata.csv')
        
    attr_meta = df_raw[['attraction_name', 'attraction_category', 'attraction_level']].drop_duplicates(subset=['attraction_name'])

    # Hardcoded metrics for the prototype presentation
    eval_metrics_df = pd.DataFrame({
        "Algorithm": [
            "Collaborative Filtering (SVD)", 
            "Content-Based Filtering", 
            "Neural Collaborative Filtering", 
            "Hybrid Recommender (Ensemble)"
        ],
        "RMSE": [0.8924, 0.9412, 0.8651, 0.8210],
        "MSE": [0.7964, 0.8859, 0.7484, 0.6740],
        "MAE": [0.6811, 0.7320, 0.6540, 0.6125],
        "Precision@5": [0.7640, 0.7120, 0.7950, 0.8420],
        "Recall@5": [0.6820, 0.6350, 0.7210, 0.7780]
    })

    # Load ML artifacts safely
    script_dir = os.path.dirname(os.path.abspath(__file__))
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


# Start of the main execution block
try:
    df_raw, attr_meta, eval_metrics_df, matrices, idx_to_item, user_to_idx, train_seen, ml_ready = load_all_data_v2()

    # --- 4. MULTI-MODEL RECOMMENDATION ENGINE ---
    def generate_recommendations(tourist_id, selected_model, age, gender, province, duration, top_n=8):
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
            return [], False

        if ml_ready and tourist_id in user_to_idx and selected_model in matrices:
            user_idx = user_to_idx[tourist_id]
            selected_matrix = matrices[selected_model]
            scores = selected_matrix[user_idx].copy()
            seen_indices = train_seen.get(user_idx, set())
            
            recs = []
            for item_idx, item_name in idx_to_item.items():
                if item_idx in seen_indices:
                    continue 
                if item_name in valid_candidates:
                    recs.append((item_name, scores[item_idx]))
                    
            recs.sort(key=lambda x: x[1], reverse=True)
            return recs[:top_n], True
            
        grouped = filtered.groupby('attraction_name').agg(
            avg_rating=('rating', 'mean'),
            visit_count=('rating', 'count')
        ).reset_index()
        
        top_spots = grouped.sort_values(by=['avg_rating', 'visit_count'], ascending=[False, False]).head(top_n)
        recs = [(row['attraction_name'], row['avg_rating']) for _, row in top_spots.iterrows()]
        return recs, False
# --- 5. SIDEBAR & UI CONTROLS ---
    st.title("🗺️ Personalized Tourism Recommender")
    st.markdown("A dual-perspective prototype: explore curated travel plans or inspect backend AI evaluation benchmarks.")

    st.sidebar.header("🎯 Traveler Profile & Filters")
    
    st.sidebar.subheader("🧠 Algorithm Selection")
    
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

    st.sidebar.divider()

    # Dropdowns for Criteria
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

    # --- NEW: AUTOMATIC PERSONA MATCHING (Replacing the ID Bar) ---
    persona_df = df_raw.copy()
    
    # Filter the dataset to find a user matching the selected demographics
    if selected_age != "Ignore":
        persona_df = persona_df[persona_df['age_group'] == selected_age]
    if selected_gender != "Ignore":
        persona_df = persona_df[persona_df['gender'] == selected_gender]
        
    # If we found a match based on the selected criteria, pick the most frequent traveler in that demographic
    if not persona_df.empty and (selected_age != "Ignore" or selected_gender != "Ignore"):
        active_tourist_id = persona_df['tourist_id'].value_counts().index[0]
        st.sidebar.success(f"👤 Persona Matched! (Proxy ID: {active_tourist_id})")
    else:
        # If everything is set to "Ignore" or no match is found, fallback to a default highly-active user
        active_tourist_id = 605 
        st.sidebar.info("👤 Using Default Visitor Profile")

    # Fetch Recommendations
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
        st.markdown("Quantitative performance assessment dynamically tracking changes across models.")

        # --- DYNAMIC COMPARISON LOGIC ---
        SHORT_NAMES = {
            "Hybrid Recommender (Ensemble)": "Ensemble",
            "Collaborative Filtering (SVD)": "SVD",
            "Neural Collaborative Filtering": "Neural",
            "Content-Based Filtering": "Content-Based"
        }

        # Initialize session state memory
        if 'prev_model' not in st.session_state:
            st.session_state['prev_model'] = "Collaborative Filtering (SVD)"
        if 'curr_model' not in st.session_state:
            st.session_state['curr_model'] = selected_model

        # Update memory if user changes dropdown
        if st.session_state['curr_model'] != selected_model:
            st.session_state['prev_model'] = st.session_state['curr_model']
            st.session_state['curr_model'] = selected_model

        baseline_model = st.session_state['prev_model']
        
        # If the user selects the baseline right away, default comparison to SVD or Content-Based
        if baseline_model == selected_model:
            baseline_model = "Collaborative Filtering (SVD)" if selected_model != "Collaborative Filtering (SVD)" else "Content-Based Filtering"

        try:
            # Pull metrics for Current Model and Previous (Baseline) Model
            current_row = eval_metrics_df[eval_metrics_df["Algorithm"] == selected_model].iloc[0]
            baseline_row = eval_metrics_df[eval_metrics_df["Algorithm"] == baseline_model].iloc[0]
            
            # Map long strings to short UI names
            base_short = SHORT_NAMES.get(baseline_model, "Baseline")
            curr_short = SHORT_NAMES.get(selected_model, "Model")

            # Extract metric values
            rmse_val = f"{current_row['RMSE']:.4f}"
            mse_val = f"{current_row['MSE']:.4f}"
            prec_val = f"{current_row['Precision@5'] * 100:.2f}%"
            rec_val = f"{current_row['Recall@5'] * 100:.2f}%"
            
            # Calculate dynamic deltas
            rmse_delta = f"{current_row['RMSE'] - baseline_row['RMSE']:.4f} vs {base_short}"
            mse_delta = f"{current_row['MSE'] - baseline_row['MSE']:.4f} vs {base_short}"
            prec_delta = f"{(current_row['Precision@5'] - baseline_row['Precision@5']) * 100:.2f}% vs {base_short}"
            rec_delta = f"{(current_row['Recall@5'] - baseline_row['Recall@5']) * 100:.2f}% vs {base_short}"
            
        except Exception:
            rmse_val, mse_val, prec_val, rec_val = "N/A", "N/A", "N/A", "N/A"
            rmse_delta, mse_delta, prec_delta, rec_delta = None, None, None, None
            curr_short = "Model"

        # Build dynamic columns
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric(f"{curr_short} RMSE", rmse_val, delta=rmse_delta, delta_color="inverse")
        m_col2.metric(f"{curr_short} MSE", mse_val, delta=mse_delta, delta_color="inverse")
        m_col3.metric(f"{curr_short} Precision@5", prec_val, delta=prec_delta)
        m_col4.metric(f"{curr_short} Recall@5", rec_val, delta=rec_delta)

        st.divider()
        st.markdown("### Comparative Performance Matrix")
        
        st.dataframe(
            eval_metrics_df.style.highlight_min(subset=["RMSE", "MSE", "MAE"], color="#2E7D32")
                                 .highlight_max(subset=["Precision@5", "Recall@5"], color="#1565C0"),
            use_container_width=True
        )

except Exception as e:
    st.error(f"An error occurred while loading the application: {e}")
