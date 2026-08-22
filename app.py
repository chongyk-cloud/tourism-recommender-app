import streamlit as st
import pandas as pd
import pydeck as pdk
import random
import requests
from duckduckgo_search import DDGS  

@st.cache_data(show_spinner=False)
def get_attraction_photo(attraction_name):
    """Dynamically fetches a real image from the web using DuckDuckGo."""
    try:
        # Optimize the search query for Chinese landmarks
        query = f"{attraction_name} attraction China"
        
        # Search the web and grab the top 1 image result
        results = DDGS().images(query, max_results=1)
        
        if results:
            return results[0]['image']  # Return the image URL
            
    except Exception as e:
        print(f"Image search failed for {attraction_name}: {e}")
        pass
        
    # Fallback to the gray placeholder if the search fails
    return f"https://placehold.co/400x300/e0e0e0/000000?text={attraction_name.replace(' ', '+')}"

# Set page configuration
st.set_page_config(page_title="Tourism Recommender", layout="wide", page_icon="🗺️")

# Load the raw dataset instead of ML matrices for demographic filtering
@st.cache_resource
def load_dataset():
    df_raw = pd.read_csv('tourism_recommendation_dataset_en.csv')
    attr_meta = df_raw[['attraction_name', 'attraction_category', 'attraction_level']].drop_duplicates(subset=['attraction_name'])
    return df_raw, attr_meta

try:
    df_raw, attr_meta = load_dataset()

    # --- NEW RECOMMENDATION LOGIC: Demographic Filtering ---
    def recommend_for_demographic(df, age_group, gender, top_n=5):
        # 1. Filter the dataset by the selected age group
        filtered_df = df[df['age_group'] == age_group]
        
        # 2. Filter by gender (unless 'All' is selected)
        if gender != "All":
            filtered_df = filtered_df[filtered_df['gender'] == gender]
            
        if filtered_df.empty:
            return []
            
        # 3. Find the most highly-rated attractions for this specific demographic
        popular_spots = filtered_df.groupby('attraction_name').agg(
            avg_rating=('rating', 'mean'),
            visit_count=('rating', 'count')
        ).reset_index()
        
        # 4. Sort by highest rating, using visit count to break ties
        popular_spots = popular_spots.sort_values(by=['avg_rating', 'visit_count'], ascending=[False, False]).head(top_n)
        
        recommendations = [(row['attraction_name'], row['avg_rating']) for _, row in popular_spots.iterrows()]
        return recommendations

    # Streamlit UI
    st.title("🗺️ Personalized Tourism Recommender")
    st.markdown("Select your demographic profile to see what travelers like you enjoyed the most!")

    # --- UPGRADE: Replace Tourist ID with Demographic Dropdowns ---
    col1, col2 = st.columns(2)
    
    with col1:
        # Get unique age groups dynamically from the dataset
        available_ages = sorted(df_raw['age_group'].dropna().unique().tolist())
        selected_age = st.selectbox("Select Your Age Group", options=available_ages)
        
    with col2:
        selected_gender = st.selectbox("Select Your Gender", options=["All", "Female", "Male"])

    # Generate recommendations
    recommendations = recommend_for_demographic(df_raw, selected_age, selected_gender, top_n=5)

    if not recommendations:
        st.warning("Not enough data for this specific demographic. Please try adjusting your filters.")
        st.stop()

    # --- Clean Tabs ---
    tab1, tab2, tab3 = st.tabs(["🎯 Top Recommendations", "📍 3D Spatial Map", "📊 Demographic Insights"])

    with tab1:
        st.subheader("Your Personalized Itinerary")
        cols = st.columns(len(recommendations))
        for i, (name, score) in enumerate(recommendations):
            with cols[i]:
                # Fetch image
                image_url = get_attraction_photo(name)
                st.image(image_url, use_container_width=True)
                
                st.markdown(f"**{name}**")
                
                # Fetch metadata
                meta = attr_meta[attr_meta['attraction_name'] == name].iloc[0]
                st.caption(f"Demographic Rating: {score:.2f}⭐ | {meta['attraction_level']}")

    with tab2:
        st.subheader("Attraction Locations")
        st.info("Note: Using simulated coordinates for 3D visualization. Add real 'lat' and 'lon' data to your dataset to map exact locations.")
        
        # --- 3D Spatial Mapping ---
        map_data = []
        for name, score in recommendations:
            # Simulating coordinates around central China for demonstration
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
