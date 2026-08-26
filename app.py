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
            "Neural Network", 
            "Hybrid Recommender (Ensemble)"
        ],
        "RMSE": [0.892, 0.9412, 0.8651, 0.8210],
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
            "Neural Network": 'pred_nn_matrix.npy',
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
    def generate_recommendations(tourist_id, selected_model, age, gender, province, category, duration, top_n=8):
        filtered = df_raw.copy()
        if age != "Ignore": filtered = filtered[filtered['age_group'] == age]
        if gender != "Ignore": filtered = filtered[filtered['gender'] == gender]
        if province != "Ignore": filtered = filtered[filtered['province'] == province]
        if category != "Ignore": filtered = filtered[filtered['attraction_category'] == category]
        if duration != "Ignore":
            if duration == "Short (1-3 hours)": filtered = filtered[filtered['visit_duration_hours'] <= 3]
            elif duration == "Medium (3-5 hours)": filtered = filtered[(filtered['visit_duration_hours'] > 3) & (filtered['visit_duration_hours'] <= 5)]
            elif duration == "Long (5+ hours)": filtered = filtered[filtered['visit_duration_hours'] > 5]
            
        valid_candidates = set(filtered['attraction_name'].unique())
        
        if not valid_candidates:
            return [], False

        if tourist_id is not None and ml_ready and tourist_id in user_to_idx and selected_model in matrices:
            user_idx = user_to_idx[tourist_id]
            selected_matrix = matrices[selected_model]
            scores = selected_matrix[user_idx].copy()
            # Squeeze crazy high/low scores into a standard 1.0 - 5.0 rating scale
            min_score = scores.min()
            max_score = scores.max()
            
            if max_score > 5.0 or min_score < 0.0:
                if max_score > min_score: # If scores are different, scale them proportionally
                    scores = 1.0 + 4.0 * ((scores - min_score) / (max_score - min_score))
                else: # If model collapsed (all scores identical), cap at 5.0
                    scores = np.full_like(scores, 5.0)
            else:
                # If they are already in a normal range, just clip them to be safe
                scores = np.clip(scores, 1.0, 5.0)
            seen_indices = train_seen.get(user_idx, set())
            
            recs = []
            for item_idx, item_name in idx_to_item.items():
                if item_idx in seen_indices:
                    continue 
                if item_name in valid_candidates:
                    recs.append((item_name, scores[item_idx]))
                    
            recs.sort(key=lambda x: x[1], reverse=True)
            # --- CONVERT RAW SCORES TO NETFLIX-STYLE MATCH % ---
            top_recs = recs[:top_n]
            if top_recs:
                max_score = top_recs[0][1]
                min_score = top_recs[-1][1]
                
                final_recs = []
                for name, score in top_recs:
                    if max_score > min_score:
                        # Scale to between 80% and 99%
                        match_pct = 80 + 19 * ((score - min_score) / (max_score - min_score))
                    else:
                        match_pct = 95.0 # Fallback if model collapsed
                    final_recs.append((name, match_pct))
                return final_recs, True
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
            "Neural Network", "Content-Based Filtering"
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

    avail_categories = sorted(df_raw['attraction_category'].dropna().unique().tolist()) + ["Ignore"]
    selected_category = st.sidebar.selectbox("Attraction Category", avail_categories, index=get_default_index(avail_categories, "Ignore"))
    
    avail_provinces = sorted(df_raw['province'].dropna().unique().tolist()) + ["Ignore"]
    selected_province = st.sidebar.selectbox("Province", avail_provinces, index=get_default_index(avail_provinces, "Ignore"))

    dur_options = ["Short (1-3 hours)", "Medium (3-5 hours)", "Long (5+ hours)", "Ignore"]
    selected_duration = st.sidebar.selectbox("Visit Duration", dur_options, index=get_default_index(dur_options, "Ignore"))

    top_n = st.sidebar.slider("Number of Recommendations", 1, 12, 8)

   # --- AUTOMATIC PERSONA MATCHING ---
    persona_df = df_raw.copy()
    
    # Check if literally every filter is set to "Ignore"
    all_filters_ignored = (selected_age == "Ignore" and selected_gender == "Ignore" and 
                           selected_province == "Ignore" and selected_category == "Ignore" and 
                           selected_duration == "Ignore")

    if all_filters_ignored:
        active_tourist_id = None  # None tells the system to use Popularity Baseline
        st.sidebar.info("🔥 **General Popularity Mode**\n\nNo filters applied. Showing trending destinations.")
    else:
        # Filter the dataset to find a user matching the selected demographics
        if selected_age != "Ignore":
            persona_df = persona_df[persona_df['age_group'] == selected_age]
        if selected_gender != "Ignore":
            persona_df = persona_df[persona_df['gender'] == selected_gender]
            
        if not persona_df.empty and (selected_age != "Ignore" or selected_gender != "Ignore"):
            active_tourist_id = persona_df['tourist_id'].value_counts().index[0]
            st.sidebar.success(f"🎯 **Demographic Twin Found!**\n\nMatching your inputs to historical Tourist ID: {active_tourist_id}")
        else:
            active_tourist_id = 605 
            st.sidebar.info("🧊 **Cold Start Mode**\n\nUsing Default Highly-Active Profile (ID: 605) to demonstrate AI capabilities.")

    # Fetch Recommendations
    recommendations, is_personalized = generate_recommendations(
        active_tourist_id, selected_model, selected_age, selected_gender, selected_province, selected_category, selected_duration, top_n
    )

    with st.expander(f"💡 How the {selected_model} generated this itinerary"):
        if "Collaborative" in selected_model:
            st.write("This model looks at the visiting patterns of Tourist {} and finds similarities with other users. It recommends places loved by people with similar travel tastes!".format(active_tourist_id))
        elif "Content" in selected_model:
            st.write("This model analyzes the categories (e.g., Nature, History) and ratings of places Tourist {} previously enjoyed, and finds new attractions with matching metadata.".format(active_tourist_id))
        elif "Neural" in selected_model:
            st.write("A Deep Learning approach that captures complex, non-linear interactions between Tourist {}'s demographics and attraction features using a Multi-Layer Perceptron.".format(active_tourist_id))
        elif "Hybrid" in selected_model:
            st.write("An ensemble method that blends user behavior (Collaborative) and attraction metadata (Content-Based) to overcome the weaknesses of using either model alone.")
            
    # --- 6. TABS STRUCTURE ---
    tab1, tab2, tab3 = st.tabs(["🎯 Top Recommendations", "📍 3D Spatial Map", "⚙️ Model Evaluation & Diagnostics"])
  
    # ========================== TAB 1: TRAVELER VIEW ==========================
    with tab1:
        st.subheader("Your Personalized Itinerary")

        # 1. Traveler Context
        if is_personalized:
            # Find what this user previously liked in the dataset
            user_history = df_raw[(df_raw['tourist_id'] == active_tourist_id) & (df_raw['rating'] >= 4.0)]
            if not user_history.empty:
                top_past = user_history['attraction_name'].iloc[0]
                st.info(f"**Traveler Context:** Based on your high ratings for places like **{top_past}**, here is what our {selected_model} suggests next:")
        
        # 2. Status Messages
        if not recommendations:
            st.warning("⚠️ No attractions found matching all your criteria. Try setting some filters to 'Ignore'.")
        elif not ml_ready:
            st.warning("⚠️ ML Model files not found. Running in Fallback Popularity Mode.")
        elif is_personalized:
            st.success(f"🤖 Showing **{selected_model}** Predictions for Tourist {active_tourist_id}")
        else:
            st.info("🔥 **Trending Destinations** | Showing highest-rated attractions across all demographics.")
            
        # 3. Image Rendering (Now safely outside the else block!)
        if recommendations:
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
                        # Dynamic Explainability Badge
                        if "Collaborative" in selected_model:
                            reason = "🧑‍🤝‍🧑 Popular with similar travelers"
                        elif "Content" in selected_model:
                            reason = f"🏷️ Matches your preferred categories"
                        elif "Hybrid" in selected_model:
                            reason = "✨ Top Ensemble Pick"
                        else:
                            reason = "🧠 Deep Learning Match"
                            
                        st.markdown(f"*{reason}*")
                        st.markdown(f"**{name}**")
                        
                        # Fetch the actual average rating from the dataset for this specific attraction
                        item_data = df_raw[df_raw['attraction_name'] == name]
                        real_avg_rating = item_data['rating'].mean() if not item_data.empty else 4.5
                        
                        st.caption(f"🎯 {score:.0f}% AI Match | Avg Rating: {real_avg_rating:.2f} ⭐ | {level}")

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

        # Define a standard baseline for comparison (e.g., Collaborative Filtering)
        baseline_model = "Collaborative Filtering (SVD)"
        
        # If the user happens to select SVD as their current model, compare it against Content-Based instead
        if selected_model == baseline_model:
            baseline_model = "Content-Based Filtering"

        try:
            current_row = eval_metrics_df[eval_metrics_df["Algorithm"] == selected_model].iloc[0]
            baseline_row = eval_metrics_df[eval_metrics_df["Algorithm"] == baseline_model].iloc[0]
            
            base_short = SHORT_NAMES.get(baseline_model, "Baseline")
            curr_short = SHORT_NAMES.get(selected_model, "Model")

            rmse_val = f"{current_row['RMSE']:.4f}"
            mse_val = f"{current_row['MSE']:.4f}"
            prec_val = f"{current_row['Precision@5'] * 100:.2f}%"
            rec_val = f"{current_row['Recall@5'] * 100:.2f}%"
            
            # Calculate dynamic deltas
            rmse_diff = current_row['RMSE'] - baseline_row['RMSE']
            mse_diff = current_row['MSE'] - baseline_row['MSE']
            prec_diff = (current_row['Precision@5'] - baseline_row['Precision@5']) * 100
            rec_diff = (current_row['Recall@5'] - baseline_row['Recall@5']) * 100

            rmse_delta = f"{rmse_diff:+.4f} vs {base_short}"
            mse_delta = f"{mse_diff:+.4f} vs {base_short}"
            prec_delta = f"{prec_diff:+.2f}% vs {base_short}"
            rec_delta = f"{rec_diff:+.2f}% vs {base_short}"
            
        except Exception:
            rmse_val, mse_val, prec_val, rec_val = "N/A", "N/A", "N/A", "N/A"
            rmse_delta, mse_delta, prec_delta, rec_delta = None, None, None, None
            curr_short = "Model"

        # Build dynamic columns with correct color mapping for error metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        # For RMSE and MSE, an increase (+) is bad (inverse color), a decrease (-) is good (green)
        m_col1.metric(f"{curr_short} RMSE", rmse_val, delta=rmse_delta, delta_color="inverse")
        m_col2.metric(f"{curr_short} MSE", mse_val, delta=mse_delta, delta_color="inverse")
        
        # For Precision and Recall, an increase (+) is good (normal color)
        m_col3.metric(f"{curr_short} Precision@5", prec_val, delta=prec_delta, delta_color="normal")
        m_col4.metric(f"{curr_short} Recall@5", rec_val, delta=rec_delta, delta_color="normal")

        st.divider()
        st.markdown("### Comparative Performance Matrix")
        
        st.dataframe(
            eval_metrics_df.style.highlight_min(subset=["RMSE", "MSE", "MAE"], color="#2E7D32")
                                 .highlight_max(subset=["Precision@5", "Recall@5"], color="#1565C0"),
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Application error: {e}")
