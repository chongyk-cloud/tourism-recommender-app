import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import random
import requests
import pickle
import os
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns

selected_age = "Ignore"
selected_category = "Ignore"
selected_province = "Ignore"
selected_duration = "Ignore"
selected_season = "Ignore"
spend_range = (0, 1000)
min_rating = 3.0
top_n = 8
active_tourist_id = None
is_personalized = False
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
    st.session_state.is_personalized = False
    st.session_state.active_tourist_id = None

# --- 1. PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(page_title="Personalized Tourism Recommender", layout="wide", page_icon="🗺️")

if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

st.markdown("""
    <div style="top: 15px; left: 80px; font-weight: bold; color: black; font-size: 5rem; z-index: 9999999;">
        Duolingo
    </div>
""", unsafe_allow_html=True)

# Custom CSS to match the image aesthetics and center the tabs
st.markdown("""
<style>
/* Center top navigation tabs */
div[data-testid="stTabs"] > div[data-baseweb="tab-list"] {
    justify-content: center;
    width: 100%;
    gap: 15px;
    margin-bottom: 20px;
}
div[data-baseweb="tab"] {
    font-weight: 800 !important;
    font-size: 1.15rem !important;
    color: #444;
}

/* Hero Banner Styling */
.hero-banner {
    position: relative;
    background: linear-gradient(rgba(0, 0, 0, 0.45), rgba(0, 0, 0, 0.45)), url('https://images.unsplash.com/photo-1508804185872-d7badad00f7d?q=80&w=1600&auto=format&fit=crop');
    background-size: cover;
    background-position: center;
    border-radius: 16px;
    padding: 80px 50px;
    margin-bottom: 25px;
    color: white;
    height: 65vh;
    min-height: 550px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}

.hero-top {
    max-width: 650px;
}

.hero-title {
    font-size: 4.5rem !important;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 25px;
    color: white;
}

.hero-btn {
    background-color: #0078D4;
    color: white;
    border: none;
    padding: 14px 28px;
    font-size: 1.1rem;
    font-weight: bold;
    border-radius: 30px;
    cursor: pointer;
    transition: background-color 0.3s ease;
    display: inline-flex;
    align-items: center;
    gap: 10px;
}

.hero-btn:hover {
    background-color: #005A9E;
}

.hero-bottom {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    width: 100%;
}

.hero-pills {
    display: flex;
    gap: 10px;
}

.pill {
    border: 1px solid rgba(255,255,255,0.6);
    padding: 8px 18px;
    border-radius: 20px;
    font-size: 0.9rem;
    backdrop-filter: blur(5px);
}

.hero-desc {
    max-width: 500px;
    text-align: right;
    font-size: 1.05rem;
    line-height: 1.6;
    color: rgba(255,255,255,0.95);
}

/* Trending Destination Hover Cards */
.dest-card {
    position: relative;
    height: 280px;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    margin-bottom: 20px;
}

.dest-card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.3s ease;
}

.dest-card:hover img {
    transform: scale(1.08);
}

.dest-overlay {
    position: absolute;
    inset: 0;
    background: linear-gradient(to top, rgba(0, 0, 0, 0.9) 0%, rgba(0, 0, 0, 0.3) 60%, transparent 100%);
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 15px;
    color: white;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.dest-card:hover .dest-overlay {
    opacity: 1;
}

.dest-title a {
    color: white !important;
    font-weight: bold;
    font-size: 1.05rem;
    text-decoration: none;
}

.dest-details {
    font-size: 0.85rem;
    margin-top: 6px;
    line-height: 1.4;
    color: #e0e0e0;
}

/* Why Choose TravelAI section */
.why-choose-wrapper {
    border: 1px solid #e5e5e5;
    border-radius: 16px;
    padding: 1px 20px 8px 20px;
    background: #fafafa;
    margin-bottom: 20px;
}
.why-choose-heading {
    margin: 0 0 18px 0;
    color: #111111;
    font-weight: 800;
    font-size: 1.2rem;
}
.why-choose-card {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    flex: 1;
    min-width: 180px;
}
.why-choose-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: #ffffff;
    border: 1px solid #eaeaea;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
}
.why-choose-title {
    font-weight: 700;
    font-size: 0.95rem;
    color: #111111;
    margin-bottom: 2px;
}
.why-choose-sub {
    font-size: 0.82rem;
    color: #666666;
    line-height: 1.3;
}
</style>
""", unsafe_allow_html=True)

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
        attraction_spend_map = (
            df_raw.groupby('attraction_name')['spend_amount'].mean().to_dict()
            if not df_raw.empty and 'spend_amount' in df_raw.columns else {}
        )
    except Exception:
        df_raw = pd.DataFrame()
        attraction_spend_map = {}
    try:
        attr_meta = pd.read_csv('attraction_metadata_filled.csv')
    except Exception:
        attr_meta = pd.DataFrame()

    eval_metrics_df = pd.DataFrame({
        "Algorithm": ["Collaborative Filtering (SVD)", "Content-Based Filtering", "Neural Network", "Hybrid Recommender (Ensemble)"],
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
    matrices, ml_ready = {}, False
    
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

    return df_raw, attr_meta, eval_metrics_df, matrices, idx_to_item, user_to_idx, train_seen, ml_ready, attraction_spend_map

# Start main execution block
try:
    df_raw, attr_meta, eval_metrics_df, matrices, idx_to_item, user_to_idx, train_seen, ml_ready, attraction_spend_map = load_all_data_v2()

    # --- 4. MULTI-MODEL RECOMMENDATION ENGINE ---
    def generate_recommendations(tourist_id, selected_model, age, province, category, duration,
                                 spend_range, min_rating, season, top_n=8):
        filtered = df_raw.copy()
        if age != "Ignore": filtered = filtered[filtered['age_group'] == age]
        if province != "Ignore": filtered = filtered[filtered['province'] == province]
        if category != "Ignore": filtered = filtered[filtered['attraction_category'] == category]
        if duration != "Ignore":
            if duration == "Short (1-3 hours)": filtered = filtered[filtered['visit_duration_hours'] <= 3]
            elif duration == "Medium (3-5 hours)": filtered = filtered[(filtered['visit_duration_hours'] > 3) & (filtered['visit_duration_hours'] <= 5)]
            elif duration == "Long (5+ hours)": filtered = filtered[filtered['visit_duration_hours'] > 5]
        if season != "Ignore": filtered = filtered[filtered['season'] == season]
        # Spend amount filter (range)
        if spend_range is not None:
            min_spend, max_spend = spend_range
            attraction_avg_spend = df_raw.groupby('attraction_name')['spend_amount'].mean()
            valid_spend_attractions = attraction_avg_spend[
                (attraction_avg_spend >= min_spend) & (attraction_avg_spend <= max_spend)
            ].index
            filtered = filtered[filtered['attraction_name'].isin(valid_spend_attractions)]
        # Minimum rating filter
        if min_rating is not None:
            filtered = filtered[filtered['rating'] >= min_rating]
            
        valid_candidates = set(filtered['attraction_name'].unique())
        
        if not valid_candidates:
            return [], False

        if tourist_id is not None and ml_ready and tourist_id in user_to_idx and selected_model in matrices:
            user_idx = user_to_idx[tourist_id]
            selected_matrix = matrices[selected_model]
            scores = selected_matrix[user_idx].copy()
            min_score, max_score = scores.min(), scores.max()
            
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
                if item_idx in seen_indices: continue 
                if item_name in valid_candidates:
                    recs.append((item_name, scores[item_idx]))
                    
            recs.sort(key=lambda x: x[1], reverse=True)
            top_recs = recs[:top_n]
            if top_recs:
                max_score, min_score = top_recs[0][1], top_recs[-1][1]
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
            avg_rating=('rating', 'mean'), visit_count=('rating', 'count')
        ).reset_index()
        
        top_spots = grouped.sort_values(by=['avg_rating', 'visit_count'], ascending=[False, False]).head(top_n)
        recs = [(row['attraction_name'], row['avg_rating']) for _, row in top_spots.iterrows()]
        return recs, False

    # =========================================================================
    # ========================== SIDEBAR (only algorithm + mode info) ==========
    # =========================================================================
    st.sidebar.header("🎯 Traveler Profile & Filters")
    st.sidebar.subheader("🧠 Algorithm Selection")
    
    if ml_ready:
        model_options = list(matrices.keys())
    else:
        model_options = ["Hybrid Recommender (Ensemble)", "Collaborative Filtering (SVD)", "Neural Network", "Content-Based Filtering"]
        
    selected_model = st.sidebar.selectbox("Choose Recommendation Engine", options=model_options)
    st.sidebar.divider()

    # We'll compute the persona info after defining main filters later, but we'll keep the sidebar info box.
    # For now, we'll create placeholder for active_tourist_id and update after filters.

    # =========================================================================
    # ========================== CUSTOM NAVIGATION =============================
    # =========================================================================
    page_to_col = {"Home": 1, "Recommendations": 2, "Diagnostics": 3}
    active_col = page_to_col.get(st.session_state.active_page, 1)
    
    st.markdown(f"""
    <style>
    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stButton"] button {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        text-transform: uppercase !important;
        font-size: 0.9rem !important;
        letter-spacing: 1.5px;
        color: #555 !important;
        font-weight: 500 !important;
        border-radius: 0 !important;
        padding-bottom: 6px !important;
        transition: color 0.3s ease;
    }}
    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stButton"] button:hover {{
        color: #000 !important;
    }}
    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"]:nth-child({active_col}) div[data-testid="stButton"] button p {{
        font-weight: 800 !important;
        color: #111 !important;
    }}
    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"]:nth-child({active_col}) div[data-testid="stButton"] button {{
        border-bottom: 2px solid #C4A47C !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button("Home", use_container_width=True):
            st.session_state.active_page = "Home"
            st.rerun()
    with nav_col2:
        if st.button("Recommendations", use_container_width=True):
            st.session_state.active_page = "Recommendations"
            st.rerun()
    with nav_col3:
        if st.button("Diagnostics", use_container_width=True):
            st.session_state.active_page = "Diagnostics"
            st.rerun()
            
    st.markdown('<hr style="border: none; border-bottom: 1px solid #eaeaea; margin-top: 5px; margin-bottom: 25px;">', unsafe_allow_html=True)
    

    # =========================================================================
    # ========================== PAGE CONTENT =================================
    # =========================================================================
    # ========================== TAB 1: HOME ==================================
    if st.session_state.active_page == "Home":
        st.markdown("""
            <div class="hero-banner">
                <div class="hero-top">
                    <h1 class="hero-title">Discover your next<br>adventure in China.</h1>
                </div>
                <div class="hero-bottom">
                    <div class="hero-pills">
                        <span class="pill">Mountain</span>
                        <span class="pill">History</span>
                        <span class="pill">Nature</span>
                        <span class="pill">Culture</span>
                    </div>
                    <p class="hero-desc">
                        Unforgettable experiences are just a click away, waiting for you to discover. Whether you're dreaming of vibrant cities, rich historical landmarks, or serene mountain retreats across China, we've got the perfect escape tailored just for you.
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Streamlit button for "Start Explore"
        with st.container():
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:   # <-- changed from col_btn1 to col_btn2
                if st.button("Start Explore ➔", key="start_explore_btn", use_container_width=True):
                    st.session_state.active_page = "Recommendations"
                    st.rerun()

        # Trending destinations 
        st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; margin-top: 20px;">
                <h3 style="margin: 0; padding: 0;">Trending Destinations China</h3>
                <a href="#" style="color: #333; text-decoration: underline; font-size: 1rem; font-weight: 500; cursor: pointer;">
                    See all
                </a>
            </div>
        """, unsafe_allow_html=True)
        
        try:
            if 'rating' in df_raw.columns:
                top_grouped = df_raw.groupby('attraction_name').agg(
                    avg_rating=('rating', 'mean'),
                    visit_count=('rating', 'count')
                ).reset_index()
                top_5_df = top_grouped.sort_values(by=['avg_rating', 'visit_count'], ascending=[False, False]).head(5)
                top_5_data = top_5_df.to_dict('records')
            else:
                top_5_data = [{'attraction_name': name, 'avg_rating': 4.8} for name in attr_meta['attraction_name'].head(5)]
        except Exception:
            top_5_data = [{'attraction_name': name, 'avg_rating': 4.8} for name in attr_meta['attraction_name'].head(5)]
        
        card_cols = st.columns(5)
        for idx, item in enumerate(top_5_data[:5]):
            name = item['attraction_name']
            avg_rating = item.get('avg_rating', 4.8)
            
            with card_cols[idx]:
                meta_row = attr_meta[attr_meta['attraction_name'] == name]
                category = meta_row['attraction_category'].iloc[0] if not meta_row.empty and not pd.isna(meta_row['attraction_category'].iloc[0]) else "Cultural Landmark"
        
                lat = float(meta_row['latitude'].iloc[0]) if not meta_row.empty and not pd.isna(meta_row['latitude'].iloc[0]) else 35.0
                lon = float(meta_row['longitude'].iloc[0]) if not meta_row.empty and not pd.isna(meta_row['longitude'].iloc[0]) else 105.0
        
                img_url = get_attraction_photo(name)
                seed = sum(ord(c) for c in name)
                est_spend = f"¥{150 + (seed % 200)} ($22–$50)"
                nav_link = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}"
        
                card_html = f"""
                    <div class="dest-card">
                        <img src="{img_url}" alt="{name}">
                        <div class="dest-overlay">
                            <div class="dest-title">
                                <a href="{nav_link}" target="_blank">{name} ↗</a>
                            </div>
                            <div class="dest-details">
                                ⭐ Rating: {avg_rating:.1f} / 5.0<br>
                                📂 Category: {category}<br>
                                💰 Est. Spend: {est_spend}
                            </div>
                        </div>
                    </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
        # ========== Why Choose TravelAI ==========
        st.markdown("<br>", unsafe_allow_html=True)

        why_items = [
                ("👍", "Personalized", "Recommendations just for you"),
                ("🗺️", "Smart & Accurate", "AI-powered insights"),
                ("💰", "Budget Friendly", "Find the best value"),
                ("❤️", "Plan with Confidence", "Explore with ease"),
            ]
    
        cards_html = "".join(
                f'<div class="why-choose-card">'
                f'<div class="why-choose-icon">{icon}</div>'
                f'<div>'
                f'<div class="why-choose-title">{title}</div>'
                f'<div class="why-choose-sub">{sub}</div>'
                f'</div>'
                f'</div>'
                for icon, title, sub in why_items
            )
    
        why_choose_html = (
                '<div class="why-choose-wrapper">'
                '<h3 class="why-choose-heading">Why Choose Duolingo?</h3>'
                '<div style="display:flex; gap:24px; flex-wrap:wrap;">'
                + cards_html +
                '</div>'
                '</div>'
            )
    
        st.markdown(why_choose_html, unsafe_allow_html=True)
    # Tab 2
    elif st.session_state.active_page == "Recommendations":
        # ========== Banner ==========
        st.markdown("""
            <div style="
                background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), 
                            url('https://2021-2025.state.gov/wp-content/uploads/2023/07/shutterstock_245773270v2-768x512.jpg');
                background-size: cover;
                background-position: center;
                border-radius: 16px;
                padding: 60px 40px;
                margin-bottom: 30px;
                color: white;
                text-align: center;
            ">
                <h1 style="font-size: 3rem; font-weight: 700; margin-bottom: 10px;">
                    Find Your Perfect Travel Destinations
                </h1>
                <p style="font-size: 1.2rem; opacity: 0.9; max-width: 700px; margin: 0 auto;">
                    Personalized recommendations based on your preferences, budget and travel style.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # ========== Filters ==========
        st.markdown("""
            <style>
            .filter-title {
                color: #000000;
                font-size: 1.4rem;
                font-weight: 800;
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 15px;
            }
            div[data-testid="column"]:last-child div[data-testid="stButton"] button {
                background: linear-gradient(to right, #0078D4, #004A87) !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                font-weight: bold !important;
                width: 100% !important;
                height: 42px !important;
                margin-top: 28px !important;
            }
            </style>
            <div class="filter-title">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line>
                    <line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line>
                    <line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line>
                    <line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line>
                    <line x1="17" y1="16" x2="23" y2="16"></line>
                </svg>
                Tell Us About Your Travel Preferences
            </div>
        """, unsafe_allow_html=True)

        with st.container():
            def get_default_index(opts, target="Ignore"):
                return opts.index(target) if target in opts else len(opts)-1

            avail_ages = sorted(df_raw['age_group'].dropna().unique().tolist()) + ["Ignore"] if not df_raw.empty else ["Ignore"]
            avail_categories = sorted(df_raw['attraction_category'].dropna().unique().tolist()) + ["Ignore"] if not df_raw.empty else ["Ignore"]
            avail_provinces = sorted(df_raw['province'].dropna().unique().tolist()) + ["Ignore"] if not df_raw.empty else ["Ignore"]
            dur_options = ["Short (1-3 hours)", "Medium (3-5 hours)", "Long (5+ hours)", "Ignore"]
            
            season_mapping = {"Chun Ji": "Spring", "Summer": "Summer", "Autumn": "Autumn", "Winter": "Winter"}
            avail_seasons = sorted(df_raw['season'].dropna().unique().tolist()) if not df_raw.empty else []
            season_display = [season_mapping.get(s, s) for s in avail_seasons] + ["Ignore"]
            season_values = avail_seasons + ["Ignore"]

            r1_col1, r1_col2, r1_col3, r1_col4, r1_col5 = st.columns(5)
            with r1_col1:
                selected_age = st.selectbox("Age Group", avail_ages, index=get_default_index(avail_ages))
            with r1_col2:
                selected_category = st.selectbox("Attraction Category", avail_categories, index=get_default_index(avail_categories))
            with r1_col3:
                selected_province = st.selectbox("Location", avail_provinces, index=get_default_index(avail_provinces))
            with r1_col4:
                min_rating = st.slider("Minimum Rating", 3.0, 5.0, 3.0, 0.1)
            with r1_col5:
                if not df_raw.empty:
                    min_spend = int(df_raw['spend_amount'].min())
                    max_spend = int(df_raw['spend_amount'].max())
                else:
                    min_spend, max_spend = 0, 1000
                spend_range = st.slider("Budget (¥)", min_spend, max_spend, (min_spend, max_spend))

            # ROW 2 
            r2_col1, r2_col2, r2_col3 = st.columns(3)
            with r2_col1:
                selected_duration = st.selectbox("Trip Duration", dur_options, index=get_default_index(dur_options))
            with r2_col2:
                idx_season = season_values.index("Ignore") if "Ignore" in season_values else len(season_values)-1
                selected_season_display = st.selectbox("Preferred Season", season_display, index=idx_season)
                selected_season = {v: k for k, v in season_mapping.items()}.get(selected_season_display, selected_season_display)
            with r2_col3:
                top_n = st.slider("Number of Recommendations", 1, 12, 8)

        st.markdown("<br><hr style='border: none; border-bottom: 1px solid #eaeaea;'><br>", unsafe_allow_html=True)
        
        # ========== Persona Matching & Recommendation Generation ==========
        # This now runs automatically every time a filter is changed
        persona_df = df_raw.copy()
        all_filters_ignored = (
            selected_age == "Ignore" and selected_province == "Ignore" and
            selected_category == "Ignore" and selected_duration == "Ignore" and
            selected_season == "Ignore"
        )
    
        if all_filters_ignored:
            active_id = None
            st.sidebar.info("🔥 **General Popularity Mode**\n\nNo filters applied. Showing trending destinations.")
        else:
            if selected_age != "Ignore":
                persona_df = persona_df[persona_df['age_group'] == selected_age]
            if not persona_df.empty and selected_age != "Ignore":
                active_id = persona_df['tourist_id'].value_counts().index[0]
                st.sidebar.success(f"🎯 **Demographic Twin Found!**\n\nMatching to Tourist ID: {active_id}")
            else:
                active_id = 605
                st.sidebar.info("🧊 **Cold Start Mode**\n\nUsing Default Highly-Active Profile (ID: 605).")
    
        recs, personalized = generate_recommendations(
            active_id, selected_model, selected_age, selected_province, 
            selected_category, selected_duration, spend_range, min_rating, selected_season, top_n
        )
        
        # Save to session state so the Map tab can access it
        st.session_state.recommendations = recs
        st.session_state.is_personalized = personalized
        st.session_state.active_tourist_id = active_id
        
        # ========== Display Itinerary ==========
        st.subheader("Your Personalized Itinerary")
    
        if st.session_state.is_personalized and not df_raw.empty:
            user_history = df_raw[(df_raw['tourist_id'] == st.session_state.active_tourist_id) & (df_raw['rating'] >= 4.0)]
            if len(user_history) > 0:
                top_past = user_history['attraction_name'].iloc[0]
                st.info(f"**Traveler Context:** Based on your high ratings for places like **{top_past}**, here is what our {selected_model} suggests next:")
                
        main_col, side_col = st.columns([3, 1])

        with main_col:
            if not st.session_state.recommendations:
                st.warning("⚠️ No attractions found matching all your criteria. Try setting some filters to 'Ignore'.")
            elif not ml_ready:
                st.warning("⚠️ ML Model files not found. Running in Fallback Popularity Mode.")
            elif st.session_state.is_personalized:
                st.success(f"🤖 Showing **{selected_model}** Predictions for Tourist {st.session_state.active_tourist_id}")
            else:
                st.info("🔥 **Trending Destinations** | Showing highest-rated attractions across all demographics.")
        
            if st.session_state.recommendations:
                num_cols = 3   # drop from 4 to 3 since the panel now takes some width
                for row_idx in range(0, len(st.session_state.recommendations), num_cols):
                    row_items = st.session_state.recommendations[row_idx : row_idx + num_cols]
                    cols = st.columns(num_cols)
                    for i, (name, score) in enumerate(row_items):
                        with cols[i]:
                            meta_row = attr_meta[attr_meta['attraction_name'] == name] if not attr_meta.empty else pd.DataFrame()
                            category = meta_row['attraction_category'].iloc[0] if not meta_row.empty and not pd.isna(meta_row['attraction_category'].iloc[0]) else "Cultural Landmark"
                            level = meta_row['attraction_level'].iloc[0] if not meta_row.empty else "5A"
                            
                            lat = float(meta_row['latitude'].iloc[0]) if not meta_row.empty and not pd.isna(meta_row['latitude'].iloc[0]) else 35.0
                            lon = float(meta_row['longitude'].iloc[0]) if not meta_row.empty and not pd.isna(meta_row['longitude'].iloc[0]) else 105.0
                            
                            img_url = get_attraction_photo(name)
                            seed = sum(ord(c) for c in name)
                            avg_spend = attraction_spend_map.get(name)
                            if avg_spend is not None:
                                est_spend = f"¥{avg_spend:.0f}"
                            else:
                                seed = sum(ord(c) for c in name)
                                est_spend = f"¥{150 + (seed % 200)} (est.)"   # fallback only if no real data exists
                            nav_link = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}"
                            
                            item_data = df_raw[df_raw['attraction_name'] == name] if not df_raw.empty else pd.DataFrame()
                            real_avg_rating = item_data['rating'].mean() if not item_data.empty else 4.5
                            
                            if "Collaborative" in selected_model: reason = "🧑‍🤝‍🧑 Popular with similar travelers"
                            elif "Content" in selected_model: reason = "🏷️ Matches your preferred categories"
                            elif "Hybrid" in selected_model: reason = "✨ Top Ensemble Pick"
                            else: reason = "🧠 Deep Learning Match"
                            
                            # 1. The Hover Card (Image with name and details inside)
                            card_html = f"""
                            <div class="dest-card">
                                <img src="{img_url}" alt="{name}">
                                <div class="dest-overlay">
                                    <div class="dest-title">
                                        <a href="{nav_link}" target="_blank">{name} ↗</a>
                                    </div>
                                    <div class="dest-details">
                                        ⭐ Rating: {real_avg_rating:.1f} / 5.0 ({level})<br>
                                        📂 Category: {category}<br>
                                        💰 Est. Spend: {est_spend}
                                    </div>
                                </div>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
                            
                            # 2. The Text Below (Reason and Caption kept, Name removed)
                            st.markdown(f"*{reason}*")
                            st.caption(f"🎯 {score:.0f}% AI Match | Avg Rating: {real_avg_rating:.2f} ⭐ | {level}")
                            pass
        
        
            
        # --- Monochrome info-panel CSS (matches white background) ---
        st.markdown("""
        <style>
        .info-card {
            background: #ffffff;
            border: 1px solid #e5e5e5;
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 18px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .info-card h4 {
            color: #111111;
            font-weight: 800;
            font-size: 1.05rem;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .info-card p, .info-card li {
            color: #333333;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .info-card ul {
            list-style: none;
            padding-left: 0;
            margin: 0;
        }
        .info-card li {
            margin-bottom: 6px;
        }
        .info-card li::before {
            content: "✓";
            color: #111111;
            font-weight: bold;
            margin-right: 8px;
        }
        .cost-bar-track {
            background: #eeeeee;
            border-radius: 8px;
            height: 8px;
            margin: 10px 0 6px 0;
            overflow: hidden;
        }
        .cost-bar-fill {
            background: #111111;
            height: 100%;
            border-radius: 8px;
        }
        .itinerary-day {
            border-left: 2px solid #111111;
            padding-left: 12px;
            margin-bottom: 10px;
        }
        .itinerary-day b { color: #111111; }
        .itinerary-day span { color: #555555; font-size: 0.85rem; }
        </style>
        """, unsafe_allow_html=True)
        
        
        def render_info_panel(recommendations, spend_range, top_n, attraction_spend_map, selected_model):
            # --- Card 1: How We Recommend (dynamic based on selected model) ---
            MODEL_INFO = {
                "Hybrid Recommender (Ensemble)": {
                    "icon": "✨",
                    "desc": "Our AI blends multiple models together to find the best attractions for you, based on:",
                    "points": ["Your personal preferences", "Your budget", "Trip duration", "Popular user reviews"],
                },
                "Content-Based Filtering": {
                    "icon": "🏷️",
                    "desc": "This engine matches you to attractions with similar characteristics to places you've rated highly, based on:",
                    "points": ["Attraction category & theme", "Location and season fit", "Your budget", "Attraction attributes"],
                },
                "Collaborative Filtering (SVD)": {
                    "icon": "🧑‍🤝‍🧑",
                    "desc": "This engine finds travelers similar to you and recommends what they enjoyed, based on:",
                    "points": ["Ratings from similar travelers", "Your travel history", "Shared demographic patterns", "Popular user reviews"],
                },
                "Neural Network": {
                    "icon": "🧠",
                    "desc": "This engine uses a deep learning model trained on traveler behavior to predict what you'll enjoy, based on:",
                    "points": ["Learned traveler patterns", "Your personal preferences", "Your budget", "Trip duration"],
                },
            }
        
            info = MODEL_INFO.get(selected_model, MODEL_INFO["Hybrid Recommender (Ensemble)"])
            points_html = "".join(f"<li>{p}</li>" for p in info["points"])
        
            st.markdown(f"""
                <div class="info-card">
                    <h4>{info['icon']} How We Recommend</h4>
                    <p>{info['desc']}</p>
                    <ul>{points_html}</ul>
                </div>
            """, unsafe_allow_html=True)

            # --- Card 2: Estimated Trip Cost ---
            if recommendations:
                est_total = sum(
                    attraction_spend_map.get(name, 150 + (sum(ord(c) for c in name) % 200))
                    for name, _ in recommendations
                )
            else:
                est_total = 0
        
            budget_cap = spend_range[1] if spend_range else 1000
            fill_pct = min(100, int((est_total / budget_cap) * 100)) if budget_cap else 0
            within_budget = est_total <= budget_cap
        
            st.markdown(f"""
                <div class="info-card">
                    <h4>💰 Estimated Trip Cost</h4>
                    <p style="font-size:1.6rem; font-weight:800; color:#111111; margin-bottom:2px;">
                        ¥{est_total:,.2f} <span style="font-size:0.9rem; font-weight:500; color:#666;">({len(recommendations)} stops)</span>
                    </p>
                    <div class="cost-bar-track"><div class="cost-bar-fill" style="width:{fill_pct}%;"></div></div>
                    <p style="font-size:0.85rem; color:{'#111' if within_budget else '#a33'};">
                        {'✓ Within your budget' if within_budget else '⚠ Above your budget'}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
            # --- Card 3: Suggested Itinerary (first 3 recs as a mini day plan) ---
            if recommendations:
                days_html = ""
                for i, (name, score) in enumerate(recommendations[:3], start=1):
                    days_html += (
                        f'<div class="itinerary-day">'
                        f'<b>Day {i}</b><br><span>{name}</span>'
                        f'</div>'
                    )
                st.markdown(f"""<div class="info-card"><h4>📅 Suggested Itinerary</h4>{days_html}</div>""", unsafe_allow_html=True)
    
        with side_col:
            render_info_panel(st.session_state.recommendations, spend_range, top_n, attraction_spend_map, selected_model)

        # ========== Spatial Map (moved here from its own tab) ==========
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("3D Journey & Spatial Layout")
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

        if st.session_state.recommendations and not attr_meta.empty:
            map_data = []
            for name, score in st.session_state.recommendations:
                meta_row = attr_meta[attr_meta['attraction_name'] == name]
                if not meta_row.empty:
                    raw_lat, raw_lon = meta_row['latitude'].iloc[0], meta_row['longitude'].iloc[0]
                    if pd.isna(raw_lat) or pd.isna(raw_lon): continue

                    lat, lon = float(raw_lat), float(raw_lon)
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
                    get_target_color="color", get_width=3, tilt=15
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
            else:
                st.warning("Coordinate data not found for these specific recommendations.")
        
    # ========================== TAB 4: DIAGNOSTICS ==========================
    elif st.session_state.active_page == "Diagnostics":
        st.subheader("Recommendation Engine Diagnostics & Evaluation")
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
        st.markdown("""
        <style>
        .model-card {
            border: 1px solid #e5e5e5;
            border-radius: 12px;
            padding: 24px;
            background: #ffffff;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #111;
            min-height: 550px;
            display: flex;
            flex-direction: column;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        /* New classes for the highlighted "Best Model" */
        .model-card.best-model {
            border: 2px solid #0078D4;
            box-shadow: 0 8px 20px rgba(0, 120, 212, 0.15);
            transform: scale(1.02);
            position: relative;
            z-index: 10;
        }
        .best-model .model-title {
            color: #0078D4;
        }
        .best-model .card-pill {
            background-color: #0078D4;
            color: #ffffff;
            border-color: #0078D4;
            font-weight: bold;
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 24px;
        }
        .model-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.2;
        }
        .model-sub {
            font-size: 0.85rem;
            color: #666;
            margin-top: 4px;
        }
        .card-pill {
            border: 1px solid #d1d5db;
            border-radius: 16px;
            padding: 4px 10px;
            font-size: 0.7rem;
            color: #4b5563;
            font-weight: 500;
        }
        .section-title {
            font-size: 0.95rem;
            font-weight: 700;
            margin: 20px 0 12px 0;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 4px;
            color: #4b5563;
        }
        .metric-val {
            font-weight: 600;
            color: #111;
        }
        .progress-track {
            background: #f3f4f6;
            height: 3px;
            width: 100%;
            margin-bottom: 12px;
            border-radius: 2px;
        }
        .progress-fill {
            background: #111;
            height: 100%;
            border-radius: 2px;
        }
        .best-model .progress-fill {
            background: #0078D4;
        }
        .capabilities-list {
            list-style: none;
            padding: 0;
            margin: 0;
            flex-grow: 1;
        }
        .capabilities-list li {
            font-size: 0.85rem;
            color: #4b5563;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .card-footer {
            font-size: 0.75rem;
            color: #9ca3af;
            margin-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        models_data = [
            {
                "name": "Collaborative Filtering",
                "sub": "Matrix Factorization (SVD)",
                "pill": "popular",
                "is_best": False,
                "metrics": [
                    ("Accuracy", "89.55%", 89.5), 
                    ("RMSE", "0.2872", 75), 
                    ("Precision@5", "0.45%", 85), 
                    ("Recall@5", "1.21%", 87)
                ],
                "capabilities": [
                    "✓ Learns from user similarities",
                    "✓ High baseline accuracy",
                    "✓ Captures hidden rating patterns"
                ],
                "date": "Aug 2026"
            },
            {
                "name": "Content-Based",
                "sub": "Attribute Matching",
                "pill": "metadata / categories",
                "is_best": False,
                "metrics": [
                    ("Accuracy", "88.95%", 88.9), 
                    ("RMSE", "0.3939", 55), 
                    ("Precision@5", "0.43%", 81), 
                    ("Recall@5", "1.17%", 84)
                ],
                "capabilities": [
                    "✓ Solves cold-start problems",
                    "✓ Recommends niche categories",
                    "✓ Independent of other users"
                ],
                "date": "Aug 2026"
            },
            {
                "name": "Neural Network",
                "sub": "Deep Learning Model",
                "pill": "🏆 Top Performer",
                "is_best": True,  # This flag triggers the styling
                "metrics": [
                    ("Accuracy", "88.95%", 88.9), 
                    ("RMSE", "0.3090", 69), 
                    ("Precision@5", "0.53%", 100), 
                    ("Recall@5", "1.39%", 100)
                ],
                "capabilities": [
                    "✓ Non-linear feature extraction",
                    "✓ Highest top-N precision",
                    "✓ Deep behavioral mapping"
                ],
                "date": "Aug 2026"
            },
            {
                "name": "Hybrid Ensemble",
                "sub": "Multi-Model Architecture",
                "pill": "comprehensive",
                "is_best": False,
                "metrics": [
                    ("Accuracy", "89.63%", 89.6), 
                    ("RMSE", "0.3312", 65), 
                    ("Precision@5", "0.49%", 92), 
                    ("Recall@5", "1.38%", 99)
                ],
                "capabilities": [
                    "✓ Maximum overall accuracy",
                    "✓ Robust edge-case handling",
                    "✓ Balances novelty & popularity"
                ],
                "date": "Aug 2026"
            }
        ]
        
        cols = st.columns(4)
        for i, model in enumerate(models_data):
            with cols[i]:
                metrics_html = ""
                for label, val, bar_width in model["metrics"]:
                    metrics_html += f"""<div class="metric-row"><span>{label}</span><span class="metric-val">{val}</span></div><div class="progress-track"><div class="progress-fill" style="width: {bar_width}%;"></div></div>"""
                    
                caps_html = "".join([f"<li><span style='color:#a1a1aa;'>✓</span> {cap.replace('✓ ', '')}</li>" for cap in model["capabilities"]])
                
                # Apply the CSS class dynamically
                card_class = "model-card best-model" if model["is_best"] else "model-card"
                
                st.markdown(f"""
        <div class="{card_class}">
        <div class="card-header">
        <div>
        <h3 class="model-title">{model["name"]}</h3>
        <div class="model-sub">{model["sub"]}</div>
        </div>
        <div class="card-pill">{model["pill"]}</div>
        </div>
        <div class="section-title">Performance</div>
        {metrics_html}
        <div class="section-title">Capabilities</div>
        <ul class="capabilities-list">
        {caps_html}
        </ul>
        <div class="card-footer">{model["date"]}</div>
        </div>
        """, unsafe_allow_html=True)
                st.divider()
        
        # 1. Initialize states for both buttons
        if "show_top_n" not in st.session_state:
            st.session_state.show_top_n = False
        if "show_rating_pred" not in st.session_state:
            st.session_state.show_rating_pred = False

        # 2. Place buttons side-by-side using columns
        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("🏆 Visualize Top-N Ranking", use_container_width=True):
                st.session_state.show_top_n = not st.session_state.show_top_n

        with btn_col2:
            if st.button("🎯 Visualize Rating Prediction", use_container_width=True):
                st.session_state.show_rating_pred = not st.session_state.show_rating_pred

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Conditionally display Top-N Metrics
        if st.session_state.show_top_n:
            st.markdown("### Top-N Ranking Performance")
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
            st.markdown("### Ranking Metrics Dashboard")
            
            # Set the dataframe index to Model for easy plotting
            # Make sure 'comparison_df' is defined or replace it with 'eval_metrics_df'
            metrics_to_plot = eval_metrics_df.set_index('Algorithm')

            sns.set_theme(style="whitegrid")
            fig = plt.figure(figsize=(16, 12))

            # Plot A: Precision & Recall @5 
            ax1 = plt.subplot2grid((2, 2), (0, 0))
            metrics_to_plot[['Precision@5', 'Recall@5']].plot(kind='barh', ax=ax1, colormap='Blues_r', edgecolor='black')
            ax1.set_title('A. Precision & Recall @ 5', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Score', fontsize=12)
            ax1.set_ylabel('') 
            ax1.invert_yaxis() 
            ax1.legend(loc='lower right')
            ax1.set_xlim(left=0)
            ax1.margins(x=0.15) 
            for container in ax1.containers:
                ax1.bar_label(container, fmt='%.4f', padding=3, fontsize=10)

            # Plot B: F1-Score @5
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            metrics_to_plot[['F1@5']].plot(kind='barh', ax=ax2, color='mediumseagreen', edgecolor='black', legend=False)
            ax2.set_title('B. F1-Score @5', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Score', fontsize=12)
            ax2.set_ylabel('')
            ax2.invert_yaxis()
            ax2.set_xlim(left=0)
            ax2.margins(x=0.15)
            for container in ax2.containers:
                ax2.bar_label(container, fmt='%.4f', padding=3, fontsize=10)

            # Plot C: Hit Rate (HR) @5 
            ax3 = plt.subplot2grid((2, 2), (1, 0))
            metrics_to_plot[['HR@5']].plot(kind='barh', ax=ax3, color='coral', edgecolor='black', legend=False)
            ax3.set_title('C. Hit Rate @5 (Users with ≥1 relevant item)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Score', fontsize=12)
            ax3.set_ylabel('')
            ax3.invert_yaxis()
            ax3.set_xlim(left=0)
            ax3.margins(x=0.15)
            for container in ax3.containers:
                ax3.bar_label(container, fmt='%.4f', padding=3, fontsize=10)

            # Plot D: NDCG @5 
            ax4 = plt.subplot2grid((2, 2), (1, 1))
            metrics_to_plot[['NDCG@5']].plot(kind='barh', ax=ax4, color='mediumpurple', edgecolor='black', legend=False)
            ax4.set_title('D. NDCG @5 (Ranking Quality)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Score', fontsize=12)
            ax4.set_ylabel('')
            ax4.invert_yaxis()
            ax4.set_xlim(left=0)
            ax4.margins(x=0.15)
            for container in ax4.containers:
                ax4.bar_label(container, fmt='%.4f', padding=3, fontsize=10)

            plt.suptitle('Final Model Ranking Evaluation (Top-5 Recommendations)', fontsize=18, fontweight='bold', y=0.98)
            plt.tight_layout(pad=3.0)
            
            # Replace plt.show() with st.pyplot()
            st.pyplot(fig)
            
            # Clear the figure from memory to prevent overlap on reruns
            plt.close(fig)

        # 4. Conditionally display Rating Prediction Metrics
        if st.session_state.show_rating_pred:
            st.markdown("### Rating Prediction & Classification")
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
            st.markdown("### Rating Prediction Dashboard")
            
            # Map your existing Streamlit dataframe for plotting
            rating_metrics_plot = eval_metrics_df.set_index('Algorithm')

            sns.set_theme(style="whitegrid")
            # Adjusted figsize and grid layout so it doesn't leave huge empty spaces
            fig = plt.figure(figsize=(15, 8))

            # Plot A: Error Metrics (Lower is better)
            ax1 = plt.subplot2grid((1, 2), (0, 0))
            rating_metrics_plot[['RMSE', 'MAE']].plot(kind='bar', ax=ax1, colormap='Reds_r', edgecolor='black')
            ax1.set_title('A. Prediction Error (Lower is Better)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Error Score', fontsize=12)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15, ha='right')
            ax1.legend(loc='lower right')
            ax1.margins(y=0.15) 
            for container in ax1.containers:
                ax1.bar_label(container, fmt='%.4f', padding=3, fontsize=10)

            # Plot B: Classification Metrics (Higher is better)
            ax2 = plt.subplot2grid((1, 2), (0, 1))
            # Mapped 'F1-Score' to 'Class F1-Score' to match your dataframe schema
            rating_metrics_plot[['Accuracy', 'Class F1-Score']].plot(kind='bar', ax=ax2, colormap='Greens_r', edgecolor='black')
            ax2.set_title('B. Classification Performance (Higher is Better)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Score', fontsize=12)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15, ha='right')
            ax2.legend(loc='lower right')
            ax2.set_ylim(0, 1.15) 
            for container in ax2.containers:
                ax2.bar_label(container, fmt='%.4f', padding=3, fontsize=10)

            plt.tight_layout()
            
            # Render the plot in Streamlit and clear memory
            st.pyplot(fig)
            plt.close(fig)
        
except Exception as e:
    st.error(f"Application error: {e}")
