import pickle
import random
from duckduckgo_search import DDGS
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st


@st.cache_data(show_spinner=False)
def get_attraction_photo(attraction_name, category):
  """Fetches images using DuckDuckGo first, then falls back to category stock photos."""
  # --- ATTEMPT 1: DuckDuckGo Search ---
  try:
    query = f"{attraction_name} attraction China"
    results = DDGS().images(query, max_results=1)
    if results:
      return results[0]['image']
  except Exception:
    pass

  # --- ATTEMPT 2: Category Stock Photo Fallback ---
  category_str = str(category).lower()
  if 'natural' in category_str or 'scenery' in category_str:
    keyword = 'nature,mountain'
  elif 'ancient' in category_str or 'town' in category_str:
    keyword = 'ancient,china,town'
  elif 'religio' in category_str:
    keyword = 'temple,pagoda'
  elif 'historic' in category_str or 'culture' in category_str:
    keyword = 'history,architecture'
  elif 'sport' in category_str or 'leisure' in category_str:
    keyword = 'skiing,resort'
  else:
    keyword = 'travel,landscape,china'

  seed = sum(ord(c) for c in attraction_name)
  return f'https://loremflickr.com/400/300/{keyword}?lock={seed}'


# Set page configuration
st.set_page_config(
    page_title='Tourism Recommender', layout='wide', page_icon='🗺️'
)


# Load all saved artifacts and dataset
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

  # Load raw dataset for rich filters (Province, Price, Rating, etc.)
  try:
    df_raw = pd.read_csv('tourism_recommendation_dataset_en.csv')
    attr_meta = df_raw[[
        'attraction_name',
        'attraction_category',
        'attraction_level',
        'province',
        'city',
        'ticket_price',
        'rating',
    ]].drop_duplicates(subset=['attraction_name'])
  except Exception:
    attr_meta = pd.read_csv('attraction_metadata.csv')
    if 'province' not in attr_meta.columns:
      attr_meta['province'] = 'All'
    if 'ticket_price' not in attr_meta.columns:
      attr_meta['ticket_price'] = 50
    if 'rating' not in attr_meta.columns:
      attr_meta['rating'] = 4.5

  return (
      pred_cf,
      pred_content,
      pred_nn,
      hybrid,
      user_ids,
      idx_to_item,
      user_to_idx,
      train_seen,
      attr_meta,
  )


try:
  (
      pred_cf,
      pred_content,
      pred_nn,
      hybrid,
      user_ids,
      idx_to_item,
      user_to_idx,
      train_seen,
      attr_meta,
  ) = load_artifacts()

  # Main Dashboard Header
  st.title('🗺️ Personalized Tourism Recommender')
  st.markdown(
      'Filter by your travel style, destination, and budget to create your'
      ' customized itinerary.'
  )

  # --- SIDEBAR: Plan Your Trip ---
  st.sidebar.header('🎯 Plan Your Trip')

  # 1. Location Filter
  provinces = ['All Provinces'] + sorted(
      attr_meta['province'].dropna().unique().tolist()
  )
  selected_prov = st.sidebar.selectbox('📍 Where to?', provinces)

  # 2. Vibe / Category Filter
  categories = ['All Vibes'] + sorted(
      attr_meta['attraction_category'].dropna().unique().tolist()
  )
  selected_cat = st.sidebar.selectbox('✨ What vibe?', categories)

  # 3. Budget Filter
  max_ticket_price = (
      int(attr_meta['ticket_price'].max())
      if 'ticket_price' in attr_meta.columns
      else 400
  )
  max_price = st.sidebar.slider(
      '💰 Max Ticket Price (¥)', 0, max_ticket_price, max_ticket_price
  )

  # 4. Top N selection
  top_n = st.sidebar.slider('🔢 Number of Results', min_value=3, max_value=10, value=5)

  # --- FILTERING LOGIC ---
  filtered_df = attr_meta.copy()

  if selected_prov != 'All Provinces':
    filtered_df = filtered_df[filtered_df['province'] == selected_prov]

  if selected_cat != 'All Vibes':
    filtered_df = filtered_df[
        filtered_df['attraction_category'] == selected_cat
    ]

  if 'ticket_price' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['ticket_price'] <= max_price]

  # Sort by highest rating
  if 'rating' in filtered_df.columns:
    filtered_df = filtered_df.sort_values(by='rating', ascending=False)

  if filtered_df.empty:
    st.warning('No attractions found matching your exact filters. Try loosening your budget or choosing "All".')
  else:
    recommendations = []
    for _, row in filtered_df.head(top_n).iterrows():
      name = row['attraction_name']
      score = row.get('rating', 4.5)
      recommendations.append((name, score))

    # --- TABS DISPLAY ---
    tab1, tab2, tab3 = st.tabs(
        ['🎯 Top Recommendations', '📍 3D Spatial Map', '📋 Full Attraction List']
    )

    with tab1:
      st.subheader('Your Personalized Itinerary')
      cols = st.columns(len(recommendations))
      for i, (name, score) in enumerate(recommendations):
        with cols[i]:
          meta = attr_meta[attr_meta['attraction_name'] == name].iloc[0]
          category = meta['attraction_category']

          # Fetch Image
          image_url = get_attraction_photo(name, category)
          st.image(image_url, use_container_width=True)

          st.markdown(f'**{name}**')
          price_tag = f" | ¥{meta['ticket_price']}" if 'ticket_price' in meta else ''
          st.caption(f"Rating: {score:.1f}⭐ | {meta['attraction_level']}{price_tag}")

    with tab2:
      st.subheader('Attraction Locations')
      st.info('Using simulated coordinates for 3D visualization.')

      map_data = []
      for name, score in recommendations:
        lat = 35.0 + random.uniform(-4, 4)
        lon = 105.0 + random.uniform(-4, 4)
        map_data.append(
            {'name': name, 'lat': lat, 'lon': lon, 'score': float(score)}
        )

      map_df = pd.DataFrame(map_data)

      view_state = pdk.ViewState(
          latitude=35.0, longitude=105.0, zoom=4, pitch=45
      )
      layer = pdk.Layer(
          'ColumnLayer',
          data=map_df,
          get_position=['lon', 'lat'],
          get_elevation='score * 20000',
          elevation_scale=10,
          radius=25000,
          get_fill_color=[255, 75, 75, 200],
          pickable=True,
          auto_highlight=True,
      )
      st.pydeck_chart(
          pdk.Deck(
              layers=[layer],
              initial_view_state=view_state,
              tooltip={'text': '{name}\nRating: {score}⭐'},
          )
      )

    with tab3:
      st.subheader('Matched Attractions Table')
      st.dataframe(filtered_df[['attraction_name', 'attraction_category', 'attraction_level', 'province', 'ticket_price', 'rating']])

except Exception as e:
  st.error(f'An error occurred: {e}')
