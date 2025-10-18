#!/usr/bin/env python3
"""
rules.py - Consolidated Expert System with Production Rules and Fuzzy Logic
This is the primary inference engine and knowledge base for EMOSIC AI
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# =============================================================================
# KNOWLEDGE BASE DEFINITIONS
# =============================================================================

# FRAMES: Knowledge representation for mood/emotion and music attributes
MOOD_FRAMES = {
    "Happy": {
        "valence_range": (0.6, 1.0),
        "energy_range": (0.5, 1.0),
        "tempo_range": (100, 200),
        "preferred_keys": [0, 2, 4, 5, 7, 9, 11],  # Major keys (C, D, E, F, G, A, B)
        "description": "Joyful, uplifting, positive emotional state",
        "synonyms": ["joyful", "cheerful", "upbeat", "positive"]
    },
    "Sad": {
        "valence_range": (0.0, 0.4),
        "energy_range": (0.0, 0.5),
        "tempo_range": (60, 120),
        "preferred_keys": [1, 3, 6, 8, 10],  # Minor keys (C#, D#, F#, G#, A#)
        "description": "Melancholic, sorrowful, low emotional state",
        "synonyms": ["melancholic", "sorrowful", "depressed", "down"]
    },
    "Calm": {
        "valence_range": (0.2, 0.7),
        "energy_range": (0.0, 0.6),
        "tempo_range": (60, 140),
        "preferred_keys": [0, 2, 4, 5, 7, 9],  # Neutral keys
        "description": "Peaceful, relaxed, tranquil emotional state",
        "synonyms": ["peaceful", "relaxed", "tranquil", "serene"]
    }
}

# SIMPLIFIED PRODUCTION RULES using IF-THEN notation
SIMPLIFIED_RULES = [
    # Happy/Joyful Rules
    {
        "id": "R1",
        "rule": "IF (Tempo IS Fast) AND (Key IS Major) THEN (EmotionalState IS Happy)",
        "conditions": {
            "tempo_min": 120,
            "key_type": "major",
            "valence_min": 0.6
        },
        "conclusion": "Happy",
        "specificity": 3,
        "confidence": 0.9,
        "created_at": datetime.now()
    },
    {
        "id": "R2",
        "rule": "IF (Valence IS High) AND (Energy IS High) THEN (EmotionalState IS Happy)",
        "conditions": {
            "valence_min": 0.7,
            "energy_min": 0.7
        },
        "conclusion": "Happy",
        "specificity": 2,
        "confidence": 0.85,
        "created_at": datetime.now()
    },

    # Sad Rules
    {
        "id": "R3",
        "rule": "IF (Tempo IS Slow) AND (Key IS Minor) THEN (EmotionalState IS Sad)",
        "conditions": {
            "tempo_max": 100,
            "key_type": "minor",
            "valence_max": 0.4
        },
        "conclusion": "Sad",
        "specificity": 3,
        "confidence": 0.9,
        "created_at": datetime.now()
    },
    {
        "id": "R4",
        "rule": "IF (Valence IS Low) AND (Energy IS Low) THEN (EmotionalState IS Sad)",
        "conditions": {
            "valence_max": 0.3,
            "energy_max": 0.4
        },
        "conclusion": "Sad",
        "specificity": 2,
        "confidence": 0.8,
        "created_at": datetime.now()
    },

    # Calm Rules
    {
        "id": "R5",
        "rule": "IF (Energy IS Low) AND (Valence IS Neutral) THEN (EmotionalState IS Calm)",
        "conditions": {
            "energy_max": 0.5,
            "valence_min": 0.3,
            "valence_max": 0.7
        },
        "conclusion": "Calm",
        "specificity": 2,
        "confidence": 0.75,
        "created_at": datetime.now()
    },
    {
        "id": "R6",
        "rule": "IF (Tempo IS Moderate) AND (Energy IS Low) THEN (EmotionalState IS Calm)",
        "conditions": {
            "tempo_min": 80,
            "tempo_max": 130,
            "energy_max": 0.6
        },
        "conclusion": "Calm",
        "specificity": 2,
        "confidence": 0.7,
        "created_at": datetime.now()
    }
]

# =============================================================================
# CONFLICT RESOLUTION SYSTEM
# =============================================================================

class ConflictResolver:
    def __init__(self):
        self.rule_usage_count = {}
        self.rule_last_used = {}
        self.refractory_period = {}

    def resolve_conflicts(self, applicable_rules: List[Dict]) -> Dict:
        """Apply conflict resolution strategies in order of priority"""
        if not applicable_rules:
            return None

        # 1. Specificity - highest specificity wins
        max_specificity = max(rule['specificity'] for rule in applicable_rules)
        high_spec_rules = [r for r in applicable_rules if r['specificity'] == max_specificity]

        if len(high_spec_rules) == 1:
            return high_spec_rules[0]

        # 2. Recency - most recently created rule wins
        most_recent = max(high_spec_rules, key=lambda x: x['created_at'])
        recent_rules = [r for r in high_spec_rules if r['created_at'] == most_recent['created_at']]

        if len(recent_rules) == 1:
            return recent_rules[0]

        # 3. Refractoriness - avoid recently used rules
        available_rules = []
        current_time = datetime.now()
        for rule in recent_rules:
            rule_id = rule['id']
            if rule_id not in self.refractory_period or \
               (current_time - self.rule_last_used.get(rule_id, datetime.min)).seconds > 30:
                available_rules.append(rule)

        if available_rules:
            recent_rules = available_rules

        # 4. Lexical Order - alphabetical by rule ID
        selected_rule = min(recent_rules, key=lambda x: x['id'])

        # 5. Means-End Analysis - select rule with highest confidence
        if len(recent_rules) > 1:
            selected_rule = max(recent_rules, key=lambda x: x['confidence'])

        # Update tracking
        rule_id = selected_rule['id']
        self.rule_usage_count[rule_id] = self.rule_usage_count.get(rule_id, 0) + 1
        self.rule_last_used[rule_id] = current_time

        return selected_rule

# Global conflict resolver instance
conflict_resolver = ConflictResolver()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_major_key(key: int) -> bool:
    """Determine if a key is major based on music theory"""
    # Major keys: C(0), D(2), E(4), F(5), G(7), A(9), B(11)
    # Minor keys: C#(1), D#(3), F#(6), G#(8), A#(10)
    major_keys = [0, 2, 4, 5, 7, 9, 11]
    return key in major_keys

def evaluate_rule_conditions(song: pd.Series, rule: Dict) -> bool:
    """Evaluate if a song matches all conditions of a rule"""
    conditions = rule['conditions']

    # Check tempo conditions
    if 'tempo_min' in conditions and song['tempo'] < conditions['tempo_min']:
        return False
    if 'tempo_max' in conditions and song['tempo'] > conditions['tempo_max']:
        return False

    # Check valence conditions
    if 'valence_min' in conditions and song['valence'] < conditions['valence_min']:
        return False
    if 'valence_max' in conditions and song['valence'] > conditions['valence_max']:
        return False

    # Check energy conditions
    if 'energy_min' in conditions and song['energy'] < conditions['energy_min']:
        return False
    if 'energy_max' in conditions and song['energy'] > conditions['energy_max']:
        return False

    # Check key type conditions
    if 'key_type' in conditions:
        song_is_major = is_major_key(int(song['key']))
        if conditions['key_type'] == 'major' and not song_is_major:
            return False
        if conditions['key_type'] == 'minor' and song_is_major:
            return False

    return True

# =============================================================================
# PRODUCTION RULE SYSTEM
# =============================================================================

def apply_production_rules(df: pd.DataFrame, emotion_cat: str, tempo_cat: str) -> pd.DataFrame:
    """
    Apply simplified production rules with conflict resolution strategies, now incorporating tempo preferences.

    Args:
        df: DataFrame containing songs with audio features
        emotion_cat: Target emotion category (Happy, Sad, Calm)
        tempo_cat: Tempo preference (Very Slow, Slow, Medium, Fast, Very Fast)

    Returns:
        DataFrame with recommended songs (metadata hidden from user)
    """

    # Normalize emotion category
    emotion_mapping = {
        'joyful': 'Happy',
        'very joyful': 'Happy',
        'sad': 'Sad',
        'very sad': 'Sad',
        'calm': 'Calm',
        'very calm': 'Calm'
    }
    target_emotion = emotion_mapping.get(emotion_cat.lower(), emotion_cat)

    # Define tempo ranges based on category
    tempo_ranges = {
        'very slow': (0, 60),
        'slow': (60, 100),
        'medium': (100, 140),
        'fast': (140, 180),
        'very fast': (180, 300)
    }
    target_tempo_range = tempo_ranges.get(tempo_cat.lower(), (0, 300))  # Default to all if unknown

    # Find applicable rules for the target emotion
    applicable_rules = [rule for rule in SIMPLIFIED_RULES if rule['conclusion'] == target_emotion]

    if not applicable_rules:
        return pd.DataFrame()

    # Apply each rule and collect matching songs
    rule_matches = []
    for rule in applicable_rules:
        matching_songs = []
        for idx, song in df.iterrows():
            if evaluate_rule_conditions(song, rule):
                song_data = song.to_dict()
                song_data['matched_rule'] = rule['id']
                song_data['rule_text'] = rule['rule']
                song_data['rule_explanation'] = f"Matches {rule['rule'].replace('IF ', '').replace(' THEN', ', therefore')}"
                song_data['specificity'] = rule['specificity']
                song_data['confidence'] = rule['confidence']
                matching_songs.append(song_data)

        if matching_songs:
            rule_matches.extend(matching_songs)

    if not rule_matches:
        return pd.DataFrame()

    # Convert to DataFrame
    result_df = pd.DataFrame(rule_matches)

    # Remove duplicates, keeping the one with highest specificity
    result_df = result_df.sort_values('specificity', ascending=False)
    result_df = result_df.drop_duplicates(subset=['title'], keep='first')

    # Apply conflict resolution if multiple rules match
    if len(result_df) > 1:
        # Group by song and resolve conflicts
        final_songs = []
        for title, group in result_df.groupby('title'):
            if len(group) > 1:
                # Multiple rules match this song - resolve conflict
                rules_for_song = []
                for _, row in group.iterrows():
                    rule_data = next(r for r in SIMPLIFIED_RULES if r['id'] == row['matched_rule'])
                    rules_for_song.append(rule_data)

                selected_rule = conflict_resolver.resolve_conflicts(rules_for_song)
                if selected_rule:
                    # Keep the song with the selected rule
                    selected_row = group[group['matched_rule'] == selected_rule['id']].iloc[0]
                    final_songs.append(selected_row.to_dict())
            else:
                final_songs.append(group.iloc[0].to_dict())

        result_df = pd.DataFrame(final_songs)

    # Now filter by tempo range using raw_tempo (BPM values)
    min_tempo, max_tempo = target_tempo_range
    if 'raw_tempo' in result_df.columns:
        tempo_filtered_df = result_df[(result_df['raw_tempo'] >= min_tempo) & (result_df['raw_tempo'] <= max_tempo)]
    else:
        tempo_filtered_df = result_df

    # If no songs match the tempo range, fallback to emotion-only results
    if tempo_filtered_df.empty:
        tempo_filtered_df = result_df
        # Update rule_explanation to indicate tempo mismatch
        if 'rule_explanation' in tempo_filtered_df.columns:
            tempo_filtered_df['rule_explanation'] = tempo_filtered_df['rule_explanation'] + f" (Tempo preference: {tempo_cat}, but no exact matches found)"

    # Calculate tempo match score (closer to center of range is better)
    if 'raw_tempo' in tempo_filtered_df.columns:
        tempo_center = (min_tempo + max_tempo) / 2
        tempo_filtered_df['tempo_match_score'] = 1 - abs(tempo_filtered_df['raw_tempo'] - tempo_center) / (max_tempo - min_tempo + 1)
    else:
        tempo_filtered_df['tempo_match_score'] = 0

    # Sort by combined score: confidence + tempo match + specificity
    if 'confidence' in tempo_filtered_df.columns and 'tempo_match_score' in tempo_filtered_df.columns and 'specificity' in tempo_filtered_df.columns:
        tempo_filtered_df['combined_score'] = tempo_filtered_df['confidence'] + tempo_filtered_df['tempo_match_score'] + (tempo_filtered_df['specificity'] / 10)
        tempo_filtered_df = tempo_filtered_df.sort_values('combined_score', ascending=False)

    # Keep rule_explanation for user display, hide other internal metadata
    columns_to_hide = ['matched_rule', 'rule_text', 'specificity', 'tempo_match_score', 'combined_score']
    display_columns = [col for col in tempo_filtered_df.columns if col not in columns_to_hide]

    return tempo_filtered_df[display_columns]

# =============================================================================
# FUZZY LOGIC CONTROLLER
# =============================================================================

class FuzzyLogicController:
    def __init__(self):
        self.setup_fuzzy_system()

    def setup_fuzzy_system(self):
        """Set up the fuzzy logic control system using scikit-fuzzy"""
        # Define input variables
        self.tempo = ctrl.Antecedent(np.arange(50, 201, 1), 'tempo')
        self.valence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'valence')
        self.energy = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'energy')

        # Define output variable
        self.emotion = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'emotion')

        # Define membership functions for tempo
        self.tempo['very_slow'] = fuzz.trimf(self.tempo.universe, [50, 60, 80])
        self.tempo['slow'] = fuzz.trimf(self.tempo.universe, [70, 90, 110])
        self.tempo['moderate'] = fuzz.trimf(self.tempo.universe, [100, 120, 140])
        self.tempo['fast'] = fuzz.trimf(self.tempo.universe, [130, 150, 170])
        self.tempo['very_fast'] = fuzz.trimf(self.tempo.universe, [160, 180, 200])

        # Define membership functions for valence
        self.valence['very_negative'] = fuzz.trimf(self.valence.universe, [0, 0.1, 0.2])
        self.valence['negative'] = fuzz.trimf(self.valence.universe, [0.1, 0.3, 0.4])
        self.valence['neutral'] = fuzz.trimf(self.valence.universe, [0.3, 0.5, 0.7])
        self.valence['positive'] = fuzz.trimf(self.valence.universe, [0.6, 0.7, 0.9])
        self.valence['very_positive'] = fuzz.trimf(self.valence.universe, [0.8, 0.9, 1.0])

        # Define membership functions for energy
        self.energy['very_low'] = fuzz.trimf(self.energy.universe, [0, 0.1, 0.2])
        self.energy['low'] = fuzz.trimf(self.energy.universe, [0.1, 0.3, 0.4])
        self.energy['moderate'] = fuzz.trimf(self.energy.universe, [0.3, 0.5, 0.7])
        self.energy['high'] = fuzz.trimf(self.energy.universe, [0.6, 0.7, 0.9])
        self.energy['very_high'] = fuzz.trimf(self.energy.universe, [0.8, 0.9, 1.0])

        # Define membership functions for emotion output
        self.emotion['sad'] = fuzz.trimf(self.emotion.universe, [0, 0.25, 0.4])
        self.emotion['calm'] = fuzz.trimf(self.emotion.universe, [0.3, 0.5, 0.7])
        self.emotion['happy'] = fuzz.trimf(self.emotion.universe, [0.6, 0.8, 1.0])

        # Define fuzzy rules
        rule1 = ctrl.Rule(self.tempo['very_slow'] & self.valence['very_negative'] & self.energy['very_low'], self.emotion['sad'])
        rule2 = ctrl.Rule(self.tempo['slow'] & self.valence['negative'] & self.energy['low'], self.emotion['sad'])
        rule3 = ctrl.Rule(self.tempo['slow'] & self.valence['neutral'] & self.energy['low'], self.emotion['calm'])
        rule4 = ctrl.Rule(self.tempo['moderate'] & self.valence['neutral'] & self.energy['moderate'], self.emotion['calm'])
        rule5 = ctrl.Rule(self.tempo['moderate'] & self.valence['positive'] & self.energy['moderate'], self.emotion['calm'])
        rule6 = ctrl.Rule(self.tempo['fast'] & self.valence['positive'] & self.energy['high'], self.emotion['happy'])
        rule7 = ctrl.Rule(self.tempo['very_fast'] & self.valence['very_positive'] & self.energy['very_high'], self.emotion['happy'])

        # Create control system
        self.emotion_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
        self.emotion_sim = ctrl.ControlSystemSimulation(self.emotion_ctrl)

    def infer(self, tempo: float, valence: float, energy: float) -> Tuple[str, float]:
        """Perform fuzzy inference on song features"""
        # Input values to the fuzzy system
        self.emotion_sim.input['tempo'] = tempo
        self.emotion_sim.input['valence'] = valence
        self.emotion_sim.input['energy'] = energy

        # Compute the result
        try:
            self.emotion_sim.compute()
            emotion_score = self.emotion_sim.output['emotion']

            # Map the emotion score to a category
            if emotion_score < 0.3:
                return "Sad", emotion_score
            elif emotion_score < 0.7:
                return "Calm", emotion_score
            else:
                return "Happy", emotion_score
        except Exception as e:
            # Fallback to traditional rules if fuzzy computation fails
            if tempo < 90 and valence < 0.3:
                return "Sad", 0.2
            elif tempo < 120 and valence < 0.7:
                return "Calm", 0.5
            else:
                return "Happy", 0.8

    def infer_from_song(self, song: pd.Series) -> Tuple[str, float]:
        """Infer emotion from a song series"""
        return self.infer(song['tempo'], song['valence'], song['energy'])

# =============================================================================
# BACKWARD CHAINING SYSTEM
# =============================================================================

class BackwardChaining:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.knowledge_base = self._initialize_knowledge_base()
        self.frames = MOOD_FRAMES

    def _initialize_knowledge_base(self) -> Dict:
        """Initialize knowledge base for backward chaining"""
        return {
            "is_happy": {
                "rules": [
                    {"conditions": ["has_high_tempo", "has_high_valence"], "confidence": 0.9},
                    {"conditions": ["has_major_key", "has_high_valence"], "confidence": 0.8},
                    {"conditions": ["has_high_energy", "has_high_valence"], "confidence": 0.85}
                ]
            },
            "is_sad": {
                "rules": [
                    {"conditions": ["has_low_tempo", "has_low_valence"], "confidence": 0.9},
                    {"conditions": ["has_minor_key", "has_low_valence"], "confidence": 0.8},
                    {"conditions": ["has_low_energy", "has_low_valence"], "confidence": 0.85}
                ]
            },
            "is_calm": {
                "rules": [
                    {"conditions": ["has_medium_tempo", "has_medium_valence"], "confidence": 0.8},
                    {"conditions": ["has_low_energy", "has_medium_valence"], "confidence": 0.75},
                    {"conditions": ["has_medium_tempo", "has_low_energy"], "confidence": 0.7}
                ]
            },
            "has_high_tempo": {"fact": lambda song: song["tempo"] >= 120},
            "has_medium_tempo": {"fact": lambda song: 80 <= song["tempo"] < 120},
            "has_low_tempo": {"fact": lambda song: song["tempo"] < 80},
            "has_high_valence": {"fact": lambda song: song["valence"] >= 0.7},
            "has_medium_valence": {"fact": lambda song: 0.4 <= song["valence"] < 0.7},
            "has_low_valence": {"fact": lambda song: song["valence"] < 0.4},
            "has_high_energy": {"fact": lambda song: song["energy"] >= 0.7},
            "has_medium_energy": {"fact": lambda song: 0.4 <= song["energy"] < 0.7},
            "has_low_energy": {"fact": lambda song: song["energy"] < 0.4},
            "has_major_key": {"fact": lambda song: song["key"] in [0, 2, 4, 5, 7, 9, 11]},
            "has_minor_key": {"fact": lambda song: song["key"] in [1, 3, 6, 8, 10]}
        }

    def query(self, question: str) -> Dict[str, Any]:
        """Process natural language queries about music recommendations"""
        question = question.lower().strip()

        # Parse the question to extract song name and user state
        song_name = self._extract_song_name(question)
        user_state = self._extract_user_state(question)

        if not song_name:
            return {"error": "Could not identify song name in question"}

        if not user_state:
            return {"error": "Could not identify user emotional state in question"}

        # Find the song in dataset
        song_data = self._find_song(song_name)
        if song_data is None:
            return {"error": f"Song '{song_name}' not found in dataset"}

        # Determine what emotion this song would evoke
        predicted_emotion = self._predict_song_emotion(song_data)

        # Check if predicted emotion matches user state
        recommendation = self._should_recommend(predicted_emotion, user_state)

        return {
            "song": song_name,
            "user_state": user_state,
            "predicted_emotion": predicted_emotion,
            "recommendation": recommendation,
            "reasoning": self._generate_reasoning(song_data, predicted_emotion, user_state),
            "song_attributes": {
                "tempo": song_data['tempo'],
                "valence": song_data['valence'],
                "energy": song_data['energy'],
                "key": song_data['key'],
                "key_type": "Major" if is_major_key(song_data['key']) else "Minor"
            }
        }

    def _extract_song_name(self, question: str) -> Optional[str]:
        """Extract song name from question"""
        import re

        # Try to find text in quotes
        quote_match = re.search(r"['\"]([^'\"]+)['\"]", question)
        if quote_match:
            return quote_match.group(1) or quote_match.group(2)

        # Try to find song name after "should" and before "be"
        should_match = re.search(r"should\s+([^\s]+(?:\s+[^\s]+)*?)\s+be", question) 
        if should_match: # ) was not closed
            return should_match.group(1).strip()

        return None

    def _extract_user_state(self, question: str) -> Optional[str]:
        """Extract user emotional state from question"""
        state_keywords = {
            'anxious': 'Calm',
            'stressed': 'Calm',
            'sad': 'Happy',
            'depressed': 'Happy',
            'happy': 'Happy',
            'joyful': 'Happy',
            'calm': 'Calm',
            'relaxed': 'Calm'
        }

        for keyword, target_state in state_keywords.items():
            if keyword in question:
                return target_state

        return None

    def _find_song(self, song_name: str) -> Optional[pd.Series]:
        """Find song in dataset by name"""
        # Try exact match first
        exact_match = self.dataset[self.dataset['title'].str.lower() == song_name.lower()]
        if not exact_match.empty:
            return exact_match.iloc[0]

        # Try partial match
        partial_match = self.dataset[self.dataset['title'].str.contains(song_name, case=False, na=False)]
        if not partial_match.empty:
            return partial_match.iloc[0]

        return None

    def _predict_song_emotion(self, song_data: pd.Series) -> str:
        """Predict what emotion a song would evoke using production rules"""
        for rule in SIMPLIFIED_RULES:
            if evaluate_rule_conditions(song_data, rule):
                return rule['conclusion']

        # Fallback: use frame-based classification
        for emotion, frame in self.frames.items():
            valence_match = frame['valence_range'][0] <= song_data['valence'] <= frame['valence_range'][1]
            energy_match = frame['energy_range'][0] <= song_data['energy'] <= frame['energy_range'][1]
            tempo_match = frame['tempo_range'][0] <= song_data['tempo'] <= frame['tempo_range'][1]

            if valence_match and energy_match and tempo_match:
                return emotion

        return "Unknown"

    def _should_recommend(self, predicted_emotion: str, user_state: str) -> bool:
        """Determine if song should be recommended based on emotion matching"""
        # For anxious/stressed users, recommend calming music
        if user_state == 'Calm' and predicted_emotion == 'Calm':
            return True
        # For sad users, recommend happy music to lift mood
        elif user_state == 'Happy' and predicted_emotion == 'Happy':
            return True
        # For happy users, maintain the mood
        elif user_state == 'Happy' and predicted_emotion == 'Happy':
            return True

        return False

    def _generate_reasoning(self, song_data: pd.Series, predicted_emotion: str, user_state: str) -> str:
        """Generate human-readable reasoning for the recommendation"""
        key_type = "Major" if is_major_key(song_data['key']) else "Minor"

        reasoning = f"Based on the song's attributes (Tempo: {song_data['tempo']:.0f} BPM, "
        reasoning += f"Valence: {song_data['valence']:.2f}, Energy: {song_data['energy']:.2f}, "
        reasoning += f"Key: {key_type}), it would likely evoke a '{predicted_emotion}' emotional response. "

        if self._should_recommend(predicted_emotion, user_state):
            reasoning += f"This matches well with the desired '{user_state}' state for the user."
        else:
            reasoning += f"This may not be ideal for someone seeking a '{user_state}' state."

        return reasoning

# =============================================================================
# GLOBAL INSTANCES AND INITIALIZATION
# =============================================================================

# Global instances
fuzzy_controller = FuzzyLogicController()
backward_chainer = None

def initialize_backward_chaining(dataset: pd.DataFrame):
    """Initialize the backward chaining system with the dataset"""
    global backward_chainer
    backward_chainer = BackwardChaining(dataset)

# =============================================================================
# CONVENIENCE FUNCTIONS FOR EXTERNAL USE
# =============================================================================

def check_song_emotion(song: pd.Series, target_emotion: str) -> Tuple[bool, float, str]:
    """Check if a song matches an emotion using fuzzy logic"""
    emotion, score = fuzzy_controller.infer_from_song(song)
    return emotion == target_emotion, score, emotion

def is_happy_song(song: pd.Series) -> bool:
    """Check if a song is classified as happy"""
    is_happy, _, _ = check_song_emotion(song, "Happy")
    return is_happy

def is_sad_song(song: pd.Series) -> bool:
    """Check if a song is classified as sad"""
    is_sad, _, _ = check_song_emotion(song, "Sad")
    return is_sad

def is_calm_song(song: pd.Series) -> bool:
    """Check if a song is classified as calm"""
    is_calm, _, _ = check_song_emotion(song, "Calm")
    return is_calm

# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =============================================================================

def apply_production_rules_legacy(df: pd.DataFrame, emotion_slider: float, tempo_slider: float) -> List[Dict]:
    """Legacy function for backward compatibility"""
    # Map slider values to categories
    emotion_category = "Happy" if emotion_slider > 0.66 else "Sad" if emotion_slider < 0.33 else "Calm"
    tempo_category = "Fast" if tempo_slider > 0.66 else "Slow" if tempo_slider < 0.33 else "Medium"

    # Create a list to store recommendations
    recommendations = []

    # Process each song in the dataset
    for _, song in df.iterrows():
        # Apply fuzzy logic controller
        emotion, score = fuzzy_controller.infer_from_song(song)
        confidence = score

        # Check if the recommended emotion matches user's emotion
        if emotion == emotion_category:
            # Create recommendation object
            recommendation = {
                "title": song.get("name", song.get("title", "Unknown")),
                "artist": song.get("artists", song.get("artist", "Unknown")),
                "emotion": emotion,
                "confidence": confidence,
                "tempo": song["tempo"],
                "valence": song["valence"],
                "energy": song["energy"],
                "rule_explanation": f"This song matches your {emotion_category} mood with {confidence:.2f} confidence"
            }

            recommendations.append(recommendation)

    # Sort recommendations by confidence score (descending)
    recommendations.sort(key=lambda x: x["confidence"], reverse=True)

    return recommendations