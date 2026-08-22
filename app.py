import streamlit as st
import pandas as pd
import pydeck as pdk
import random
from duckduckgo_search import DDGS  

@st.cache_data(show_spinner=False)
def get_attraction_photo(attraction_name, category):
    """Tries multiple search queries on DuckDuckGo, falling back to category-matched photos if needed."""
    
    # Multiple search query variations to maximize finding a real web image
    queries = [
        f"{attraction_name} scenic spot China",
        f"{attraction_name} tourism",
        f"{attraction_name}"
    ]
    
    for query in queries:
        try:
            results = DDGS().images(query, max_results=1)
            if results and 'image' in results[0]:
                return results[0]['image']
        except Exception:
            continue  # Try the next query variation if this one fails
            
    # Ultimate Backup: Category-matched travel photo so the card is NEVER empty
    category_str = str(category).lower()
    if "natural" in category_str or "scenery" in category_str:
        keyword = "nature,mountain"
    elif "ancient" in category_str or "town" in category_str:
        keyword = "ancient,china,town"
    elif "religio" in category_str:
        keyword = "temple,pagoda"
    elif "historic" in category_str or "culture" in category_str:
        keyword = "history,architecture"
    else:
        keyword = "travel,landscape,china"
        
    seed = sum(ord(c) for c in attraction_name)
    return f"https://loremflickr.com/400/300/{keyword}?lock={seed}"

# Set page configuration
st.set_page_config(page_title="Tourism Recommender", layout="wide", page_icon="🗺️")

# Load the raw dataset for demographic filtering
@st.cache_resource
def load_dataset():
    df_raw = pd.read_csv('tourism_recommendation_dataset_en.csv')
    attr_meta = df_raw[['attraction_name', 'attraction_category', 'attraction_level']].drop_duplicates(subset=['attraction_name'])
    return df_raw, attr_meta

try:
    df_raw, attr_meta = load_dataset()

    def recommend_for_demographic(df, age_group, gender, top_n=5):
        filtered_df = df[df['age_group'] == age_group]
        
        if gender != "All":
            filtered_df = filtered_df[filtered_df['gender'] == gender]
            
        if filtered_df.empty:
            return []
            
        popular_spots = filtered_df.groupby('attraction_name').agg(
            avg_rating=('rating', 'mean'),
            visit_count=('rating', 'count')
        ).reset_index()
        
        popular_spots = popular_spots.sort_values(by=['avg_rating', 'visit_count'], ascending=[False, False]).head(top_n)
        
        recommendations = [(row['attraction_name'], row['avg_rating']) for _, row in popular_spots.iterrows()]
        return recommendations

    # Streamlit UI
    st.title("🗺️ Personalized Tourism Recommender")
    st.markdown("Select your demographic profile to see what travelers like you enjoyed the most!")

    # Demographic Dropdowns
    col1, col2 = st.columns(2)
    
    with col1:
        available_ages = sorted(df_raw['age_group'].dropna().unique().tolist())
        selected_age = st.selectbox("Select Your Age Group", options=available_ages)
        
    with col2:
        selected_gender = st.selectbox("Select Your Gender", options=["All", "Female", "Male"])

    recommendations = recommend_for_demographic(df_raw, selected_age, selected_gender, top_n=5)

    if not recommendations:
        st.warning("Not enough data for this specific demographic. Please try adjusting your filters.")
        st.stop()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Top Recommendations", "📍 3D Spatial Map", "📊 Demographic Insights"])

    with tab1:
        st.subheader("Your Personalized Itinerary")
        cols = st.columns(len(recommendations))
        for i, (name, score) in enumerate(recommendations):
            with cols[i]:
                # Fetch metadata category first
                meta = attr_meta[attr_meta['attraction_name'] == name].iloc[0]
                category = meta['attraction_category']
                
                # Fetch image using multi-source search + category fallback
                image_url = get_attraction_photo(name, category)
                st.image(image_url, use_container_width=True)
                
                st.markdown(f"**{name}**")
                st.caption(f"Rating: {score:.2f}⭐ | {meta['attraction_level']}")

    with tab2:
        st.subheader("Attraction Locations")
        st.info("Note: Using simulated coordinates for 3D visualization.")
        
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
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}\nAvg Rating: {score}⭐"}))

    with tab3:
        st.subheader("Why these recommendations?")
        st.write(f"These locations are ranked by the average rating given exclusively by travelers matching your profile (**Age {selected_age}** & **Gender: {selected_gender}**).")

except Exception as e:
    st.error(f"An error occurred: {e}")
