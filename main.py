from fastapi import FastAPI, Query
from typing import List, Optional
import pandas as pd

# Load your dataset
df = pd.read_csv("data_moods.csv")

# Define mood frames (example, adjust as needed)
MOOD_FRAMES = {
    "sad": {"valence": (0.0, 0.4)},
    "calm": {"valence": (0.4, 0.6)},
    "joyful": {"valence": (0.6, 1.0)}
}

TEMPO_FRAMES = {
    "slow": (0, 90),
    "medium": (90, 120),
    "fast": (120, 300)
}

app = FastAPI()

def filter_songs(emotion: str, tempo: str):
    # Map emotion to valence range
    mood = emotion.lower()
    tempo = tempo.lower()
    valence_range = MOOD_FRAMES.get(mood)
    tempo_range = TEMPO_FRAMES.get(tempo)
    if not valence_range or not tempo_range:
        return []
    filtered = df[
        (df['valence'] >= valence_range['valence'][0]) &
        (df['valence'] < valence_range['valence'][1]) &
        (df['tempo'] >= tempo_range[0]) &
        (df['tempo'] < tempo_range[1])
    ]
    return filtered.sample(min(5, len(filtered))).to_dict(orient="records")

@app.get("/recommendations")
def get_recommendations(
    emotion: str = Query(..., description="Emotion: sad, calm, joyful"),
    tempo: str = Query(..., description="Tempo: slow, medium, fast")
):
    """
    Get music recommendations based on emotion and tempo.
    """
    results = filter_songs(emotion, tempo)
    return {"recommendations": results}