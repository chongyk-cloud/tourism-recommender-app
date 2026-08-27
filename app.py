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
    try:
        df_raw = pd.read_csv('tourism_recommendation_dataset_en.csv')
    except Exception:
        df_raw = pd.read_csv('attraction_metadata.csv')
        
    attr_meta = pd.read_csv('attraction_metadata_filled.csv')

    eval_metrics_df = pd.DataFrame({
        "Algorithm": [
            "Collaborative Filtering (SVD)", 
            "Content-Based Filtering", 
            "Neural Network", 
            "Hybrid Recommender (Ensemble)"
        ],
        "Precision@5": [0.0045, 0.0043, 0.0053, 0.0049],
        "Recall@5": [0.0121, 0.0117, 0.0139, 0.0138],
        "F1@5": [0.0064, 0.0062, 0.0075, 0.0071],
        "HR@5": [0.0222, 0.0217, 0.0261, 0.0246],
        "NDCG@5": [0.0082, 0.0082, 0.0097, 0.0089],
        "RMSE": [0.2872, 0.3939, 0.3090, 0.3312],
        "MAE": [0.2449, 0.3212, 0.2587, 0.2751],
        "Accuracy": [0.8955, 0.8895, 0.8895, 0.8963],
        "Class F1-Score": [0.9436, 0.9415, 0.9400, 0.9452]
    })

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

# Start of main execution block
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
            min_score = scores.min()
            max_score = scores.max()
            
            if max_score > 5.0 or min_score < 0.0:
                if max_score > min_score:
                    scores = 1.0 + 4.0 * ((scores - min_score) / (max_score - min_score))
                else:
                    scores = np.full_like(scores, 5.0)
            else:
                scores = np.clip(scores, 1.0, 5.0)
            seen_indices = train_seen.get(user_idx, set())
            
            recs = []
            for item_idx, item_name in idx_to_item.items():
                if item_idx in seen_indices:
                    continue 
                if item_name in valid_candidates:
                    recs.append((item_name, scores[item_idx]))
                    
            recs.sort(key=lambda x: x[1], reverse=True)
            top_recs = recs[:top_n]
            if top_recs:
                max_score = top_recs[0][1]
                min_score = top_recs[-1][1]
                
                final_recs = []
                for name, score in top_recs:
                    if max_score > min_score:
                        match_pct = 80 + 19 * ((score - min_score) / (max_score - min_score))
                    else:
                        match_pct = 95.0
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

    # --- 6. TABS STRUCTURE (Main added to the front) ---
    tab_main, tab1, tab2, tab3 = st.tabs(["🏠 Main", "🎯 Top Recommendations", "📍 3D Spatial Map", "⚙️ Model Evaluation & Diagnostics"])

    # ========================== TAB MAIN: WELCOME & HERO ==========================
    with tab_main:
        # Hero Banner
        st.markdown("""
            <div style="background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url(https://en.wikipedia.org/wiki/Great_Wall_of_China#/media/File:The_Great_Wall_of_China_at_Jinshanling-edit.jpg); background-size: cover; background-position: center; padding: 60px 40px; border-radius: 12px; color: white; text-align: center; margin-bottom: 30px;">
                <h1 style="font-size: 3em; margin-bottom: 10px; font-weight: 800;">Discover Your Next Adventure in China.</h1>
                <p style="font-size: 1.2em; max-width: 700px; margin: 0 auto; line-height: 1.6;">
                    Immerse yourself in five thousand years of magnificent history, breathtaking landscapes, architectural marvels, and vibrant cultures. China offers a journey like no other place on Earth.
                </p>
            </div>
        """, unsafe_allow_html=True)
    # ========================== TAB 1: TRAVELER VIEW ==========================
    # --- AUTOMATIC PERSONA MATCHING ---
    persona_df = df_raw.copy()
    all_filters_ignored = (selected_age == "Ignore" and selected_gender == "Ignore" and 
                           selected_province == "Ignore" and selected_category == "Ignore" and 
                           selected_duration == "Ignore")

    if all_filters_ignored:
        active_tourist_id = None
        st.sidebar.info("🔥 **General Popularity Mode**\n\nNo filters applied. Showing trending destinations.")
    else:
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

    with tab1:
        st.subheader("Your Personalized Itinerary")

        if is_personalized:
            user_history = df_raw[(df_raw['tourist_id'] == active_tourist_id) & (df_raw['rating'] >= 4.0)]
            if not user_history.empty:
                top_past = user_history['attraction_name'].iloc[0]
                st.info(f"**Traveler Context:** Based on your high ratings for places like **{top_past}**, here is what our {selected_model} suggests next:")
        
        if not recommendations:
            st.warning("⚠️ No attractions found matching all your criteria. Try setting some filters to 'Ignore'.")
        elif not ml_ready:
            st.warning("⚠️ ML Model files not found. Running in Fallback Popularity Mode.")
        elif is_personalized:
            st.success(f"🤖 Showing **{selected_model}** Predictions for Tourist {active_tourist_id}")
        else:
            st.info("🔥 **Trending Destinations** | Showing highest-rated attractions across all demographics.")
            
        if recommendations:
            num_cols = 4
            for row_idx in range(0, len(recommendations), num_cols):
                row_items = recommendations[row_idx : row_idx + num_cols]
                cols = st.columns(num_cols)
                
                for i, (name, score) in enumerate(row_items):
                    with cols[i]:
                        meta_row = attr_meta[attr_meta['attraction_name'] == name]
                        level = meta_row['attraction_level'].iloc[0] if not meta_row.empty and not pd.isna(meta_row['attraction_level'].iloc[0]) else "5A"
                        img_url = get_attraction_photo(name)
                        
                        st.markdown(
                            f"""
                            <div style="height: 200px; width: 100%; overflow: hidden; border-radius: 8px; margin-bottom: 10px;">
                                <img src="{img_url}" style="width: 100%; height: 100%; object-fit: cover;">
                            </div>
                            """, unsafe_allow_html=True
                        )
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
                        
                        item_data = df_raw[df_raw['attraction_name'] == name]
                        real_avg_rating = item_data['rating'].mean() if not item_data.empty else 4.5
                        
                        st.caption(f"🎯 {score:.0f}% AI Match | Avg Rating: {real_avg_rating:.2f} ⭐ | {level}")

    # ========================== TAB 2: SPATIAL MAP ==========================
    with tab2:
        st.subheader("📍 3D Journey & Spatial Layout")
        st.info("Interactive routing from your origin point to recommended destinations.")

        PROVINCE_COORDS = {
            "Beijing": [116.4074, 39.9042], "Shanghai": [121.4737, 31.2304],
            "Guangdong": [113.2644, 23.1291], "Shandong": [117.1201, 36.6512],
            "Zhejiang": [120.1551, 30.2741], "Jiangsu": [118.7969, 32.0603],
            "Sichuan": [104.0648, 30.6586], "Henan": [113.6253, 34.7466],
            "Default": [108.9398, 34.3416]
        }
        
        origin_lon, origin_lat = PROVINCE_COORDS.get(selected_province, PROVINCE_COORDS["Default"])
        origin_name = selected_province if selected_province != "Ignore" else "Default Hub"

        if recommendations:
            map_data = []
            
            for name, score in recommendations:
                meta_row = attr_meta[attr_meta['attraction_name'] == name]
                if not meta_row.empty:
                    raw_lat = meta_row['latitude'].iloc[0]
                    raw_lon = meta_row['longitude'].iloc[0]
                    
                    if pd.isna(raw_lat) or pd.isna(raw_lon):
                        continue
                        
                    lat = float(raw_lat)
                    lon = float(raw_lon)
                    
                    color = [46, 204, 113, 220] if score > 90 else [241, 196, 15, 220]
                    
                    R = 6371.0
                    lat1, lon1, lat2, lon2 = map(np.radians, [origin_lat, origin_lon, lat, lon])
                    dlon = lon2 - lon1
                    dlat = lat2 - lat1
                    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                    distance_km = R * 2 * np.arcsin(np.sqrt(a))
                    
                    safe_distance = int(distance_km) if not np.isnan(distance_km) else 0
                    
                    map_data.append({
                        "name": name, "lat": lat, "lon": lon, "score": float(score),
                        "color": color, "origin_lat": origin_lat, "origin_lon": origin_lon,
                        "distance": safe_distance
                    })

            if map_data:
                map_df = pd.DataFrame(map_data)
                avg_lat = (map_df["lat"].mean() + origin_lat) / 2
                avg_lon = (map_df["lon"].mean() + origin_lon) / 2
                
                view_state = pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=4.5, pitch=50, bearing=-10)
                
                scatter_layer = pdk.Layer(
                    "ScatterplotLayer", data=map_df, get_position=["lon", "lat"],
                    get_radius=8000, get_fill_color="color", pickable=False, 
                )
                
                column_layer = pdk.Layer(
                    "ColumnLayer", data=map_df, get_position=["lon", "lat"],
                    get_elevation="score * 1200", elevation_scale=10, radius=3500,
                    get_fill_color="color", pickable=True, auto_highlight=True,
                )
                
                arc_layer = pdk.Layer(
                    "ArcLayer", data=map_df,
                    get_source_position=["origin_lon", "origin_lat"],
                    get_target_position=["lon", "lat"],
                    get_source_color=[33, 150, 243, 160], 
                    get_target_color="color",
                    get_width=3,
                    tilt=15
                )
                
                custom_tooltip = {
                    "html": "<b>{name}</b><br/>🎯 AI Match: {score}%<br/>📏 Distance: {distance} km from " + origin_name,
                    "style": {"backgroundColor": "#1E1E1E", "color": "white", "border": "1px solid #4682B4", "borderRadius": "5px"}
                }
                
                st.pydeck_chart(pdk.Deck(
                    map_provider="carto", map_style="dark",
                    layers=[scatter_layer, arc_layer, column_layer], 
                    initial_view_state=view_state, tooltip=custom_tooltip
                ))
                
                st.markdown("### 🚗 Start Your Journey")
                nav_cols = st.columns(4)
                for i, row in enumerate(map_data):
                    with nav_cols[i % 4]:
                        nav_link = f"https://www.google.com/maps/dir/?api=1&destination={row['lat']},{row['lon']}"
                        st.markdown(f"**[{row['name']}]({nav_link})** <br> <span style='font-size:0.8em; color:gray;'>({row['distance']} km away)</span>", unsafe_allow_html=True)
            else:
                st.warning("Coordinate data not found for these specific recommendations.")

    # ========================== TAB 3: DIAGNOSTICS ==========================
    with tab3:
        st.subheader("📊 Recommendation Engine Diagnostics & Evaluation")
        st.markdown("Quantitative performance assessment dynamically tracking changes across models.")

        SHORT_NAMES = {
            "Hybrid Recommender (Ensemble)": "Ensemble",
            "Collaborative Filtering (SVD)": "SVD",
            "Neural Network": "Neural",
            "Content-Based Filtering": "Content-Based"
        }

        baseline_model = "Collaborative Filtering (SVD)"
        if selected_model == baseline_model:
            baseline_model = "Content-Based Filtering"

        try:
            current_row = eval_metrics_df[eval_metrics_df["Algorithm"] == selected_model].iloc[0]
            baseline_row = eval_metrics_df[eval_metrics_df["Algorithm"] == baseline_model].iloc[0]
            
            base_short = SHORT_NAMES.get(baseline_model, "Baseline")
            curr_short = SHORT_NAMES.get(selected_model, "Model")

            prec_val = f"{current_row['Precision@5'] * 100:.2f}%"
            rec_val = f"{current_row['Recall@5'] * 100:.2f}%"
            f1_val = f"{current_row['F1@5'] * 100:.2f}%"
            ndcg_val = f"{current_row['NDCG@5']:.4f}"
            
            rmse_val = f"{current_row['RMSE']:.4f}"
            mae_val = f"{current_row['MAE']:.4f}"
            acc_val = f"{current_row['Accuracy'] * 100:.2f}%"
            clf_f1_val = f"{current_row['Class F1-Score'] * 100:.2f}%"
            
            prec_delta = f"{(current_row['Precision@5'] - baseline_row['Precision@5']) * 100:+.2f}% vs {base_short}"
            rec_delta = f"{(current_row['Recall@5'] - baseline_row['Recall@5']) * 100:+.2f}% vs {base_short}"
            f1_delta = f"{(current_row['F1@5'] - baseline_row['F1@5']) * 100:+.2f}% vs {base_short}"
            ndcg_delta = f"{current_row['NDCG@5'] - baseline_row['NDCG@5']:+.4f} vs {base_short}"
            
            rmse_delta = f"{current_row['RMSE'] - baseline_row['RMSE']:+.4f} vs {base_short}"
            mae_delta = f"{current_row['MAE'] - baseline_row['MAE']:+.4f} vs {base_short}"
            acc_delta = f"{(current_row['Accuracy'] - baseline_row['Accuracy']) * 100:+.2f}% vs {base_short}"
            clf_f1_delta = f"{(current_row['Class F1-Score'] - baseline_row['Class F1-Score']) * 100:+.2f}% vs {base_short}"
            
        except Exception:
            prec_val = rec_val = f1_val = ndcg_val = rmse_val = mae_val = acc_val = clf_f1_val = "N/A"
            prec_delta = rec_delta = f1_delta = ndcg_delta = rmse_delta = mae_delta = acc_delta = clf_f1_delta = None
            curr_short = "Model"

        st.divider()
        
        st.markdown("### 🏆 Top-N Ranking Performance")
        r1_col1, r1_col2, r1_col3, r1_col4 = st.columns(4)
        
        r1_col1.metric(f"{curr_short} Precision@5", prec_val, delta=prec_delta, delta_color="normal")
        r1_col2.metric(f"{curr_short} Recall@5", rec_val, delta=rec_delta, delta_color="normal")
        r1_col3.metric(f"{curr_short} F1@5", f1_val, delta=f1_delta, delta_color="normal")
        r1_col4.metric(f"{curr_short} NDCG@5", ndcg_val, delta=ndcg_delta, delta_color="normal")

        st.dataframe(
            eval_metrics_df[["Algorithm", "Precision@5", "Recall@5", "F1@5", "HR@5", "NDCG@5"]]
            .style.highlight_max(subset=["Precision@5", "Recall@5", "F1@5", "HR@5", "NDCG@5"], color="#1565C0"),
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 🎯 Rating Prediction & Classification")
        r2_col1, r2_col2, r2_col3, r2_col4 = st.columns(4)
        
        r2_col1.metric(f"{curr_short} RMSE", rmse_val, delta=rmse_delta, delta_color="inverse")
        r2_col2.metric(f"{curr_short} MAE", mae_val, delta=mae_delta, delta_color="inverse")
        r2_col3.metric(f"{curr_short} Accuracy", acc_val, delta=acc_delta, delta_color="normal")
        r2_col4.metric(f"{curr_short} Class F1-Score", clf_f1_val, delta=clf_f1_delta, delta_color="normal")

        st.dataframe(
            eval_metrics_df[["Algorithm", "RMSE", "MAE", "Accuracy", "Class F1-Score"]]
            .style.highlight_min(subset=["RMSE", "MAE"], color="#2E7D32")
            .highlight_max(subset=["Accuracy", "Class F1-Score"], color="#1565C0"),
            use_container_width=True
        )

except Exception as e:
    st.error(f"Application error: {e}")
