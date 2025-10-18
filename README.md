# Technical Documentation: EMOSIC AI

## 1. Project Overview

**Purpose:**

EMOSIC AI is an AI-powered music recommendation system that personalizes song suggestions based on user emotions and tempo preferences. It combines expert system techniques (production rules, fuzzy logic, backward chaining), machine learning (linear regression), and retrieval-augmented generation (RAG) to provide explainable, interactive recommendations. The system features a user-friendly Streamlit web interface for direct interaction and a FastAPI backend for programmatic access.

**Core Functionalities:**

*   **Emotion and Tempo-Based Recommendations:** Users input emotions (e.g., "sad," "calm," "joyful") and tempo preferences (e.g., "slow," "medium," "fast") via sliders, and the system recommends songs using production rules with conflict resolution.
*   **Fuzzy Logic Inference:** Employs fuzzy logic to classify songs into emotions based on audio features (tempo, valence, energy).
*   **Expert System with Rules:** Applies IF-THEN production rules for recommendations, resolving conflicts via specificity, recency, refractoriness, lexical order, and confidence.
*   **Backward Chaining Queries:** Allows "what-if" questions about specific songs (e.g., "Should 'Breathe Me' be recommended for anxious users?").
*   **Retrieval-Augmented Generation (RAG):** Enhances explanations with AI insights using a local LLM (DialoGPT) and vectorized knowledge base.
*   **Linear Regression Prediction:** Predicts mood from tempo and computes alignment scores for user feedback.
*   **Interactive UI and API:** Streamlit app for end-users; FastAPI for developers integrating recommendations.

**Key Features:**

*   **Streamlit Frontend:** Interactive sliders, recommendation grids, favorites, feedback collection, session logging, and visualizations (e.g., tempo vs. mood scatter plots).
*   **FastAPI Backend:** RESTful API for emotion/tempo-based recommendations (basic implementation; full engine in Streamlit).
*   **Data-Driven:** Uses a CSV dataset (`data_moods.csv`) with normalized audio features (valence, energy, tempo, key).
*   **Explainable AI:** Provides rule-based explanations for recommendations, fuzzy scores, and RAG-augmented insights.
*   **Modular Architecture:** Separates UI, data loading, rules engine, models, and utilities for maintainability.
*   **Extensible:** Supports custom datasets, rule additions, and model integrations.

**Technology Stack:**

*   **Frontend/UI:** Streamlit (interactive web app), Altair (charts).
*   **Backend/API:** FastAPI, Uvicorn.
*   **Data Processing:** Pandas, NumPy, Scikit-Learn (linear regression).
*   **AI/ML:** Scikit-Fuzzy (fuzzy logic), LangChain (RAG), Transformers (HuggingFace LLM), Sentence-Transformers (embeddings), Chroma (vectorstore).
*   **Testing:** Pytest (implied; manual tests in `test_conflict_resolution.py`).
*   **Other:** Python 3.8+, Matplotlib (plotting).

## 2. Setup & Installation

**Prerequisites:**

*   Python 3.8+ (recommended for compatibility with dependencies like Transformers).
*   pip (Python package installer).
*   Git (for cloning the repository).
*   Optional: Virtual environment tool (e.g., venv or conda) for isolated installations.

**Dependencies:**

The project has two requirement files:
- `requirements.txt`: Full dependencies for the complete system (Streamlit UI, FastAPI, fuzzy logic, RAG, etc.).
- `requirements_api.txt`: Minimal dependencies for the FastAPI backend only.

Key libraries:
*   `streamlit`: For the interactive web UI.
*   `fastapi`, `uvicorn`: For the API server.
*   `pandas`, `numpy`: For data manipulation.
*   `scikit-fuzzy`: For fuzzy logic inference.
*   `scikit-learn`: For linear regression.
*   `langchain`, `chromadb`, `sentence-transformers`, `transformers`, `torch`: For RAG functionality.
*   `altair`, `matplotlib`: For visualizations.
*   `python-multipart`: For file uploads in Streamlit.

**Installation Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Set Up Virtual Environment (Recommended):**
    ```bash
    python -m venv emosic_env
    emosic_env\Scripts\activate  # On Windows; use 'source emosic_env/bin/activate' on macOS/Linux
    ```

3.  **Install Dependencies:**
    - For full system (UI + API + AI features):
      ```bash
      pip install -r requirements.txt
      ```
    - For API-only (lighter setup):
      ```bash
      pip install -r requirements_api.txt
      ```

4.  **Prepare Dataset:**
    - Place `data_moods.csv` in the project root (default path in `config.py`).
    - Ensure the CSV has columns: `title` (or `name`), `artist`, `valence`, `energy`, `tempo`, `key` (optional: `instrumentation`, `cultural_tags`, `genre`).
    - The system normalizes valence/energy/tempo to 0-1 if not already.

5.  **Verify Installation:**
    ```bash
    python -c "import streamlit, fastapi, pandas, skfuzzy; print('Dependencies installed successfully')"
    ```

**Common Issues:**
- If RAG fails (e.g., model download), ensure internet access and sufficient disk space (~1GB for DialoGPT).
- For GPU acceleration, install CUDA-compatible PyTorch if available.
- On Windows, use PowerShell or CMD for commands.

## 3. Usage Documentation

**Running the Streamlit Application (Primary Interface):**

1.  Start the app:
    ```bash
    streamlit run app.py
    ```
    - Opens at `http://localhost:8501`.
    - Upload a custom CSV or use default (`data_moods.csv`).

2.  **User Interaction:**
    - **Emotion Slider:** Move from 0-10 (e.g., 0 = Very Sad, 10 = Very Happy). Maps to categories (Sad, Calm, Happy).
    - **Tempo Slider:** Move from 0-10 (e.g., 0 = Very Slow, 10 = Very Fast). Maps to categories (Slow, Medium, Fast).
    - Click "✨ Find My Perfect Music ✨" to get recommendations.
    - View songs in a grid with details (artist, genre, rule explanation).
    - Add to favorites, request more (up to 2 additional batches), or provide feedback.
    - Use backward chaining: Ask questions like "Should 'Happy' by Pharrell Williams be good when I'm feeling sad?"
    - Explore RAG: Query AI for music insights (e.g., "What songs are good for happy moods?").
    - View linear regression plot: Predict mood from tempo interactively.

3.  **Example Workflow:**
    - Set emotion to 8.5 (Very Joyful), tempo to 7.0 (Fast).
    - Get recommendations like upbeat songs with high valence/tempo.
    - Feedback: Rate as "Perfect match" and explain.

**Running the FastAPI Backend:**

1.  Start the server:
    ```bash
    uvicorn main:app --reload
    ```
    - Available at `http://127.0.0.1:8000`.
    - Interactive docs at `http://127.0.0.1:8000/docs`.

2.  **API Usage:**
    - Endpoint: `GET /recommendations`
    - Parameters: `emotion` (str: "sad", "calm", "joyful"), `tempo` (str: "slow", "medium", "fast").
    - Example Request:
      ```bash
      curl "http://127.0.0.1:8000/recommendations?emotion=joyful&tempo=fast"
      ```
    - Example Response:
      ```json
      {
        "recommendations": [
          {
            "title": "Happy Song 1",
            "artist": "Artist A",
            "valence": 0.8,
            "tempo": 130.0,
            "energy": 0.9,
            "key": 7
          }
        ]
      }
      ```
    - Note: This is a simplified API; full engine logic is in the Streamlit app.

**Input/Output Samples:**

- **Streamlit Input:** Sliders (emotion: 5.0 -> Calm; tempo: 6.0 -> Medium).
- **Streamlit Output:** Song grid with cards (e.g., "Song Title" by Artist, Genre: Pop, Why: Matches high valence rule).
- **API Input:** Query params (emotion=calm&tempo=medium).
- **API Output:** JSON list of songs with metadata.
- **Backward Chaining Input:** "Should 'Calm Waters' be good for relaxed users?"
- **Backward Chaining Output:** Recommendation (Yes/No), reasoning (e.g., "Low tempo matches Calm"), matched songs.
- **RAG Input:** "What makes a song happy?"
- **RAG Output:** AI-generated explanation (e.g., "Happy songs often have high valence and tempo...").

**Configuration Options:**
- Edit `config.py` for dataset path, batch size (5), max requests (2).
- Customize rules in `rules.py` (add new IF-THEN rules).
- Adjust fuzzy membership functions in `FuzzyLogicController`.

## 4. Architecture & Structure

**Directory Organization:**

```
.
├── app.py                          # Streamlit UI application (main entry for users)
├── main.py                         # FastAPI backend (basic API endpoint)
├── config.py                       # Configuration constants (paths, limits)
├── data_loader.py                  # Data loading and normalization utilities
├── utils.py                        # Helper functions (slider mappings, unit conversion)
├── rules.py                        # Expert system (production rules, fuzzy logic, backward chaining, conflict resolution)
├── linear_regression_model.py      # ML model for mood prediction from tempo
├── rag_module.py                   # RAG system (LLM, vectorstore, embeddings)
├── data_moods.csv                  # Default song dataset (valence, energy, tempo, etc.)
├── requirements.txt                # Full dependencies
├── requirements_api.txt            # API-only dependencies
├── test_conflict_resolution.py     # Tests for rule conflict resolution
├── TODO.md                         # Pending tasks and roadmap
├── fuzzy_logic_controller_complete.md  # Notes on fuzzy logic implementation
└── technical_documentation.md      # This documentation file
```

**File-Level Descriptions:**

*   `app.py`: Orchestrates the Streamlit UI, integrates all modules (data loading, rules, models, RAG), handles user interactions, sessions, and visualizations. Core user-facing component.
*   `main.py`: Defines FastAPI app with `/recommendations` endpoint; uses basic filtering (not full rules engine). Suitable for API integrations.
*   `config.py`: Stores constants like dataset path, recommendation batch size, and max requests. Easy to tweak without code changes.
*   `data_loader.py`: Loads and preprocesses CSV data; normalizes features to 0-1; handles missing columns. Ensures data consistency.
*   `utils.py`: Utility functions for mapping sliders to categories (e.g., emotion 0-2 -> Very Sad) and unit conversions. Supports UI explainability.
*   `rules.py`: Heart of the expert system. Includes MOOD_FRAMES (knowledge base), SIMPLIFIED_RULES (IF-THEN), ConflictResolver (resolution strategies), FuzzyLogicController (inference), BackwardChaining (queries). Handles recommendation logic.
*   `linear_regression_model.py`: Trains and uses sklearn LinearRegression to predict mood from tempo; computes alignment scores. Provides predictive insights.
*   `rag_module.py`: Builds RAG pipeline with Chroma vectorstore, SentenceTransformer embeddings, and HuggingFace LLM (DialoGPT). Augments explanations with AI.
*   `data_moods.csv`: Sample dataset with song metadata. Replace with custom CSV for personalization.
*   `test_conflict_resolution.py`: Manual tests for conflict resolution (specificity, recency, etc.). Demonstrates rule engine reliability.
*   `TODO.md`: Lists unfinished features (e.g., full API integration, more tests).
*   `fuzzy_logic_controller_complete.md`: Detailed notes on fuzzy logic setup and tuning.
*   `requirements.txt` / `requirements_api.txt`: Dependency lists for installation.
*   `technical_documentation.md`: Comprehensive docs (this file).

**Architectural Patterns:**
- **Modular Design:** Each file has a single responsibility (e.g., rules.py for inference, app.py for UI).
- **Separation of Concerns:** UI (Streamlit), API (FastAPI), Logic (rules.py), Data (data_loader.py), Models (linear_regression_model.py, rag_module.py).
- **Expert System:** Rules-based with fuzzy extensions for uncertainty handling.
- **Event-Driven UI:** Streamlit reacts to user inputs (sliders, buttons) to update recommendations.

## 5. API Documentation

The API is minimal (FastAPI in `main.py`); full logic is in Streamlit. Use for simple integrations; extend for advanced features.

### Endpoints

**`GET /recommendations`**

Retrieves music recommendations based on emotion and tempo.

*   **Parameters:**
    *   `emotion` (str, required): Emotion category ("sad", "calm", "joyful").
    *   `tempo` (str, required): Tempo category ("slow", "medium", "fast").
*   **Returns:** Dict with "recommendations" key containing list of song dicts (title, artist, valence, tempo, etc.).
*   **Example:**
  ```python
  import requests
  response = requests.get("http://127.0.0.1:8000/recommendations", params={"emotion": "joyful", "tempo": "fast"})
  print(response.json())
  ```
*   **Status Codes:** 200 (success), 422 (invalid params).

### Key Functions (from `main.py`)

**`filter_songs(emotion: str, tempo: str) -> List[dict]`**

Filters dataset by predefined ranges (e.g., joyful: valence 0.6-1.0, fast: tempo 120-300).

*   **Parameters:** emotion (str), tempo (str).
*   **Returns:** List of song dicts (sampled to 5). Empty if no matches.
*   **Notes:** Simple range-based; not using full rules engine.

**`get_recommendations(emotion: str, tempo: str) -> dict`**

API wrapper for `filter_songs`.

*   **Query Params:** Same as above.
*   **Returns:** {"recommendations": list}.

For full API (integrating rules, fuzzy, etc.), modify `main.py` to call `apply_production_rules` from `rules.py`.

## 6. Testing Framework

Testing is limited; `test_conflict_resolution.py` provides manual demos. Use pytest for expansion.

**Overview:**
- **Framework:** pytest (install via pip; run with `pytest`).
- **Current Tests:** `test_conflict_resolution.py` - Tests ConflictResolver strategies (specificity, recency, refractoriness, lexical order, confidence).
- **Coverage:** Rule engine conflict resolution; no UI/API integration tests.
- **Execution:**
  ```bash
  python test_conflict_resolution.py  # Manual run
  pytest test_conflict_resolution.py  # If structured as pytest
  ```
- **Example Output:** Prints test results (e.g., "Selected Rule: R1" for specificity).
- **Expansion Needs:** Add tests for fuzzy inference, RAG queries, UI interactions (e.g., via Streamlit testing tools), API endpoints.

**Best Practices:**
- Write unit tests for functions (e.g., `evaluate_rule_conditions`).
- Use fixtures for dataset loading.
- Aim for 80%+ coverage with tools like coverage.py.

## 7. Development Guidelines

**Contribution Standards:**
- Follow PEP 8 style (use black or flake8 for linting).
- Add docstrings to all functions/classes (e.g., """Brief description. Args: ..., Returns: ...""").
- Write tests for new features; ensure existing tests pass.
- Document changes in commit messages and PRs.

**Coding Practices:**
- Use type hints (e.g., `def func(x: int) -> str:`).
- Keep functions small (<50 lines); use classes for stateful logic (e.g., FuzzyLogicController).
- Handle exceptions gracefully (e.g., try-except in RAG queries).
- Use descriptive names (e.g., `emotion_slider` not `e`).
- Modularize: Import only needed modules; avoid global state except where necessary (e.g., backward_chainer).

**Submission Protocols:**
- Fork repo; create feature branches (e.g., `feature/add-new-rule`).
- Submit PRs with descriptions, screenshots for UI changes.
- Code review required; CI/CD for tests (if set up).
- For data changes, validate CSV schema.

**Environment Setup:**
- Use virtual envs; commit `requirements*.txt` updates.
- Test on multiple Python versions if possible.
- Document breaking changes in this doc.

**Advanced Tips:**
- For fuzzy tuning, experiment with membership functions in `rules.py`.
- Extend RAG with better models (e.g., Llama via HuggingFace).
- Integrate full engine into API for consistency.
