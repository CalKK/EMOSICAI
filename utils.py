# utils.py
import numpy as np

def to_unit(x):
    """Map 0..10 slider to 0..1"""
    return np.clip(x / 10.0, 0.0, 1.0)

def map_emotion_category(slider_val):
    """Map 0..10 emotion slider to discrete category with enhanced specificity."""
    if slider_val <= 2.0:
        return "Very Sad"
    if 2.1 <= slider_val <= 3.9:
        return "Sad"
    if 4.0 <= slider_val <= 6.0:
        return "Calm"
    if 6.1 <= slider_val <= 8.0:
        return "Joyful"
    return "Very Joyful"

def map_tempo_category(slider_val):
    """Map 0..10 tempo slider to discrete category with enhanced specificity."""
    if slider_val <= 2.0:
        return "Very Slow"
    if 2.1 <= slider_val <= 3.9:
        return "Slow"
    if 4.0 <= slider_val <= 6.0:
        return "Medium"
    if 6.1 <= slider_val <= 8.0:
        return "Fast"
    return "Very Fast"
