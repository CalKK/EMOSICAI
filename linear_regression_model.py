#!/usr/bin/env python3
"""
linear_regression_model.py - Linear Regression Model for EMOSIC AI
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

class LinearRegressionModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self._train_model()

    def _train_model(self):
        """Trains the linear regression model."""
        df = pd.read_csv(self.data_path)

        # Map moods to numerical values
        mood_mapping = {
            'Sad': 1,
            'Calm': 5,
            'Happy': 7,
            'Energetic': 9
        }
        df['mood_value'] = df['mood'].map(mood_mapping)

        # Prepare the data
        X = df[['tempo']]
        y = df['mood_value']

        # Train the model
        self.model = LinearRegression()
        self.model.fit(X, y)

    def predict(self, tempo):
        """Predicts the mood value for a given tempo."""
        return self.model.predict([[tempo]])[0]

    def get_alignment_score(self, tempo, user_slider_value):
        """Calculates the alignment score."""
        predicted_mood = self.predict(tempo)
        lr_fit = 1 - abs(predicted_mood - user_slider_value) / 10
        return lr_fit
