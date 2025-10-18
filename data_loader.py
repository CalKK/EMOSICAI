# data_loader.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

@st.cache_data(show_spinner=False)
def load_dataset(path):
    """
    Load CSV dataset and normalize valence/energy/tempo to 0..1 if needed.
    Expected columns: 'name' (or 'title'), 'artist', 'valence', 'energy', 'tempo'
    Optional: 'instrumentation', 'cultural_tags', 'genre'
    """
    df = pd.read_csv(path)
    
    # Preserve raw tempo before normalization
    df['raw_tempo'] = df['tempo']
    
    # Handle name/title column variation
    if 'name' in df.columns and 'title' not in df.columns:
        df['title'] = df['name']
        
    # Add genre if missing
    if 'genre' not in df.columns and 'mood' in df.columns:
        df['genre'] = df['mood']
    elif 'genre' not in df.columns:
        df['genre'] = "Unknown"
        
    required = {'title', 'artist', 'valence', 'energy', 'tempo'}
    if not required.issubset(set(df.columns)):
        st.error(f"Dataset missing required columns. Required: {required}")
        return pd.DataFrame()
    # Normalize if outside 0..1
    for col in ['valence', 'energy', 'tempo']:
        if df[col].max() > 1.0 or df[col].min() < 0.0:
            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            df[[col]] = scaler.fit_transform(df[[col]])
    if 'instrumentation' not in df.columns:
        df['instrumentation'] = ""
    if 'cultural_tags' not in df.columns:
        df['cultural_tags'] = ""
    return df
