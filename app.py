import streamlit as st
import altair as alt
from config import DEFAULT_DATA_PATH, RECS_PER_BATCH, MAX_MORE_REQUESTS
from data_loader import load_dataset
from utils import map_emotion_category, map_tempo_category, to_unit
from rules import apply_production_rules, initialize_backward_chaining, backward_chainer
from linear_regression_model import LinearRegressionModel
from rag_module import MusicRAG


st.set_page_config(page_title="EMOSIC AI - Your Music Mood Companion", layout="wide")

# Display logo
st.image("https://plus.unsplash.com/premium_photo-1682096492098-8745454126cf?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8bXVzaWMlMjByZWNvbW1lbmRhdGlvbiUyMGV4cGVydCUyMHN5c3RlbXxlbnwwfHwwfHx8MA%3D%3D", width=1000)
st.title("🎵 EMOSIC AI - Your Music Mood Companion 🎵")
st.markdown(
    """Welcome to EMOSIC AI! Tell us how you're feeling and what kind of rhythm you enjoy,
    and we'll find the perfect songs to match your mood. It's as simple as moving the sliders below!"""
)

with st.sidebar:
    st.markdown("### 🎵 Music Dataset")
   
    # Add a nice container for the dataset uploader
    st.markdown("""
    <div style="background-color: #057a30; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
        <p>Upload your own music dataset or use our default collection</p>
    </div>
    """, unsafe_allow_html=True)
   
    data_file = st.file_uploader("Upload music metadata CSV", type=["csv"])
    if data_file is not None:
        DATA_PATH = data_file
        st.success("✅ Custom dataset loaded successfully!")
    else:
        DATA_PATH = DEFAULT_DATA_PATH
        st.info("Using default music collection")
       
    with st.expander("Dataset Information"):
        st.write("If you don't upload a dataset, we'll use the default music collection defined in config.py.")
        st.write("Your dataset should include columns for song name, artist, tempo, key, energy, and valence.")
        st.write("Need help? Check out the documentation for more details.")
   
    # Add a divider
    st.markdown("---")


df = load_dataset(DATA_PATH)  # Using the correct function name from data_loader.py
if df.empty:
    st.warning("No dataset loaded or dataset missing required columns. Upload a dataset to continue.")
    st.stop()

# Initialize linear regression model
lr_model = LinearRegressionModel(DATA_PATH)

# Prepare mood_df for visualization
mood_df = df.copy()
mood_mapping = {
    'Sad': 1,
    'Calm': 5,
    'Happy': 7,
    'Energetic': 9
}
mood_df['mood_value'] = mood_df['mood'].map(mood_mapping)

# Initialize backward chaining system
initialize_backward_chaining(df)

# Initialize RAG system (optional augmented feature)
rag_system = MusicRAG(df)


# Initialize session state
if 'more_requests' not in st.session_state:
    st.session_state.more_requests = 0
if 'recommendation_offset' not in st.session_state:
    st.session_state.recommendation_offset = 0
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'last_query_results' not in st.session_state:
    st.session_state.last_query_results = None
if 'feedback' not in st.session_state:
    st.session_state.feedback = []
if 'backward_chaining_results' not in st.session_state:
    st.session_state.backward_chaining_results = None


# Create more user-friendly UI with descriptive labels and visual cues
st.markdown("### How are you feeling today?")


# Emotion slider with emoticons and descriptive labels
emoji_labels = ["😢", "😔", "😐", "🙂", "😄"]
emoji_html = """
<div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
    <div style="text-align: left; font-size: 24px;">😢</div>
    <div style="text-align: center; font-size: 24px;">😔</div>
    <div style="text-align: center; font-size: 24px;">😐</div>
    <div style="text-align: center; font-size: 24px;">🙂</div>
    <div style="text-align: right; font-size: 24px;">😄</div>
</div>
"""
st.markdown(emoji_html, unsafe_allow_html=True)


text_labels_html = """
<div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
    <div style="text-align: left;">Very Sad<br>Huzuni nyingi</div>
    <div style="text-align: center;">A bit down<br>Huzuni kiasi</div>
    <div style="text-align: center;">Neutral<br>Uko hapo tu</div>
    <div style="text-align: center;">Happy<br>Umefurahi</div>
    <div style="text-align: right;">Very Happy<br>Uko na furaha tele</div>
</div>
"""
st.markdown(text_labels_html, unsafe_allow_html=True)


emotion_slider = st.slider("Move the slider to match your mood", 0.0, 10.0, 6.5, step=0.1, label_visibility="collapsed")


st.markdown("### What kind of music pace do you prefer?")


# Tempo slider with visual analogies
tempo_labels_html = """
<div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
    <div style="text-align: left;">🐢 Relaxed and slow<br>Pole pole sana</div>
    <div style="text-align: center;">Moderate pace<br>Wastani tu</div>
    <div style="text-align: right;">🐇 Energetic and lively<br>Mbio sana</div>
</div>
"""
st.markdown(tempo_labels_html, unsafe_allow_html=True)


tempo_slider = st.slider("Move the slider to set your preferred music pace", 0.0, 10.0, 6.5, step=0.1, label_visibility="collapsed")


# Show mapped categories (explainability) in more friendly language
emotion_cat = map_emotion_category(emotion_slider)
tempo_cat = map_tempo_category(tempo_slider)


# More user-friendly interpretation message
st.info(f"We'll find music that matches your **{emotion_cat.lower()}** mood with a **{tempo_cat.lower()}** rhythm 🎵")


# More engaging button with emoji
if st.button("✨ Find My Perfect Music ✨"):
    # Log initial query parameters
    st.session_state.logs.append({
        "emotion_slider": emotion_slider,
        "tempo_slider": tempo_slider,
        "emotion_cat": emotion_cat,
        "tempo_cat": tempo_cat,
        "action": "initial_query"
    })
    st.session_state.recommendation_offset = 0
    st.session_state.more_requests = 0
    results = apply_production_rules(df, emotion_cat, tempo_cat)
    st.session_state.last_query_results = results


# Initialize session state for favorites
if 'favorites' not in st.session_state:
    st.session_state.favorites = []


# Display results if any
if st.session_state.last_query_results is not None:
    results = st.session_state.last_query_results
    if results.empty:
        st.warning("No songs matched the production rules for the selected categories. Consider widening the dataset or adding softer rules.")
    else:
        offset = st.session_state.recommendation_offset
        batch = results.iloc[offset: offset + RECS_PER_BATCH]
        st.markdown(f"### 🎵 Your Personalized Music Recommendations ({offset+1} to {offset + len(batch)})")
        st.markdown("")
       
        # Create a grid layout for recommendations
        cols = st.columns(3)
        for idx, row in batch.iterrows():
            with cols[idx % 3]:
                # Create a card-like design for each recommendation
                song_title = row.get('title', row.get('name', 'Unknown'))
               
                # Card container with styling
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 10px; background-color: #057a30; margin-bottom: 15px;">
                    <h4 style="margin-top: 0;">🎧 {song_title}</h4>
                    <p><strong>Artist:</strong> {row['artist']}</p>
                </div>
                """, unsafe_allow_html=True)
               
                # Additional song details in an expander
                with st.expander("Song details"):
                    if 'album' in row and row['album']:
                        st.markdown(f"**Album:** {row['album']}")
                    if row.get('genre', 'Unknown') != 'Unknown':
                        st.markdown(f"**Genre:** {row['genre']}")
                   
                    # Simplified explanation without technical metadata
                    st.markdown(f"**Why recommended:** {row['rule_explanation']}")
               
                # Add favorite button with emoji
                if st.button(f"❤️ Add to Favorites", key=f"fav_{idx}"):
                    st.session_state.favorites.append(row)
                    st.success(f"Added {song_title} to favorites!")
               
                st.markdown("")
                st.markdown("")


        # Request more or feedback - improved UI
        st.markdown("---")
        st.markdown("### Want more music suggestions?")
       
        # Show remaining requests with a progress bar
        remaining = MAX_MORE_REQUESTS - st.session_state.more_requests
        st.markdown(f"You have **{remaining}** more recommendation requests available")
        st.progress(remaining/MAX_MORE_REQUESTS)
       
        # More user-friendly input
        more_text = st.text_input("Type 'more' or share what you'd like to hear next:", placeholder="more please" if remaining > 0 else "Tell us what you think...")
        if st.button("🎵 Get More Music"):
            req = more_text.strip().lower()
            if st.session_state.more_requests >= MAX_MORE_REQUESTS:
                st.warning("You have exhausted your additional recommendation requests.")
            else:
                if req in ('more', 'more please', 'more songs', 'another', 'again', 'yes', 'y', 'next'):
                    st.session_state.more_requests += 1
                    st.session_state.recommendation_offset += RECS_PER_BATCH
                    st.session_state.logs.append({
                        "action": "more_request",
                        "text": req,
                        "remaining": MAX_MORE_REQUESTS - st.session_state.more_requests
                    })
                    st.rerun()
                else:
                    # store free-form feedback
                    st.session_state.feedback.append({
                        "text": more_text,
                        "emotion_slider": emotion_slider,
                        "tempo_slider": tempo_slider
                    })
                    st.success("Thanks for the feedback! We will use this to refine recommendations.")
                    st.session_state.logs.append({
                        "action": "feedback",
                        "text": more_text
                    })


        # Batch feedback with improved UI
        st.markdown("---")
        st.markdown("### 💬 How did we do?")
       
        # Create a nicer feedback UI
        feedback_container = st.container()
        with feedback_container:
            st.markdown("""<div style="background-color: #057a30; padding: 15px; border-radius: 10px;">
                <h4 style="margin-top: 0;">Your feedback helps us improve!</h4>
            </div>""", unsafe_allow_html=True)
           
            col_a, col_b = st.columns(2)
            with col_a:
                liked = st.radio("Did these songs match your mood?",
                                ("😃 Perfect match!", "🙂 They were okay", "😕 Not quite right"),
                                index=0,
                                key="liked_radio")
            with col_b:
                reason = st.text_area("Tell us more (optional):",
                                    height=100,
                                    placeholder="What did you like or dislike about these recommendations?",
                                    key="batch_reason")
           
            # More appealing button
            if st.button("📝 Submit Feedback", use_container_width=True):
                st.session_state.feedback.append({
                    "liked": liked,
                    "reason": reason,
                    "emotion_slider": emotion_slider,
                    "tempo_slider": tempo_slider,
                    "offset": offset
                })
                st.success("Thank you — feedback recorded.")
                st.session_state.logs.append({"action": "batch_feedback", "liked": liked, "reason": reason})


# Sidebar: session logs with improved UI
with st.sidebar:
    st.markdown("### 📊 Session Information")
   
    # Show session stats in a nice container
    st.markdown(f"""
    <div style="background-color: #057a30; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
        <p><strong>Music Requests:</strong> {st.session_state.more_requests} of {MAX_MORE_REQUESTS}</p>
        <p><strong>Current Batch:</strong> {st.session_state.recommendation_offset//RECS_PER_BATCH + 1}</p>
    </div>
    """, unsafe_allow_html=True)
   
    # Logs in a cleaner expander
    with st.expander("📝 Session Activity"):
        st.markdown("**Recent Activity:**")
        for entry in st.session_state.logs[-5:]:
            # Format the log entry nicely
            if isinstance(entry, dict):
                action = entry.get('action', 'unknown')
                timestamp = entry.get('timestamp', '')
                if action == 'initial_query':
                    st.markdown(f"🔍 **Initial search** - {timestamp}")
                elif action == 'more_request':
                    st.markdown(f"🔄 **More songs requested** - {entry.get('text', '')}")
                elif action == 'feedback':
                    st.markdown(f"💬 **Feedback provided** - {entry.get('text', '')}")
                elif action == 'batch_feedback':
                    st.markdown(f"📋 **Batch feedback** - {entry.get('liked', '')}")
                else:
                    st.write(entry)
            else:
                st.write(entry)
   
    # Feedback in a cleaner expander
    with st.expander("💬 Collected Feedback"):
        if len(st.session_state.feedback) > 0:
            for fb in st.session_state.feedback[-5:]:
                st.write(fb)


# Display favorite songs
st.subheader("Your Favorite Songs")
if st.session_state.favorites:
    for fav in st.session_state.favorites:
        st.markdown(f"**{fav.get('title', 'Unknown')}** — {fav['artist']}")
else:
    st.write("No favorite songs yet.")




# Backward Chaining Section - Made more user-friendly
st.subheader("🔍 Curious about a specific song?")
st.markdown("Wonder if a particular song would match how you're feeling? Ask us a question and our AI will explain!")


st.markdown("""
<div style="background-color: #057a30; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
    <p><strong>Try questions like:</strong></p>
    <ul>
        <li>Should 'Breathe Me' be recommended for anxious users?</li>
        <li>Would 'Happy' by Pharrell Williams be good when I'm feeling sad?</li>
        <li>Is 'Calm Waters' appropriate for a relaxed mood?</li>
    </ul>
</div>
""", unsafe_allow_html=True)


backward_query = st.text_input("Enter your question about a song:", placeholder="Should 'Breathe Me' be recommended for anxious users?")
if st.button("🔮 Get Answer"):
    if backward_query.strip():
        try:
            result = backward_chainer.query(backward_query)
            st.session_state.backward_chaining_results = result
            st.success("Query processed successfully!")
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
    else:
        st.warning("Please enter a question.")


# Display backward chaining results
if st.session_state.backward_chaining_results:
    with st.expander("🎯 Backward Chaining Results", expanded=True):
        result = st.session_state.backward_chaining_results
       
        # Create a visually appealing results section
        if result.get('recommendation', False):
            st.success("✅ Yes! This song would be a great match for you!")
        else:
            st.error("❌ This song might not be the best match for your current mood.")
       
        # Display reasoning in a nice info box
        st.markdown("#### Why?")
        st.info(result.get('reasoning', 'No explanation available'))

        # Optional RAG enhancement
        if st.checkbox("Enhance explanation with AI insights"):
            try:
                enhanced_reasoning = rag_system.augment_explanation(result['reasoning'], f"Why is '{result.get('song', 'this song')}' {result.get('predicted_emotion', 'recommended')}?")
                st.info(f"Enhanced Reasoning: {enhanced_reasoning}")
            except Exception as e:
                st.error(f"AI enhancement failed: {str(e)}")
       
        # Create two columns for additional details
        col1, col2 = st.columns(2)
       
        # Display matched songs if available in a more structured way
        if result.get('matched_songs'):
            with col1:
                st.markdown("##### 🎧 Matched Songs")
                for song in result['matched_songs']:
                    st.markdown(f"- {song['name']} by {song['artist']} (Mood: {song['mood']})")
       
        # Display fired rules if available in a more structured way
        if result.get('fired_rules'):
            with col2:
                st.markdown("##### ⚙️ Music Selection Logic")
                for rule in result['fired_rules']:
                    st.markdown(f"- {rule}")


st.markdown("---")

st.subheader("🤖 AI-Powered Music Insights (Optional)")
st.markdown("Ask AI questions about music recommendations or emotions.")

rag_query = st.text_input("Enter your question:", placeholder="What songs are good for happy moods?", key="rag_query")
if st.button("🤖 Get AI Answer"):
    if rag_query.strip():
        try:
            answer = rag_system.standalone_query(rag_query)
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question.")

st.markdown("---")

st.subheader("ANN Framework:🔬 Linear Regression Model")
scatter_plot = alt.Chart(mood_df).mark_circle(size=60).encode(
    x='tempo',
    y='mood_value',
    tooltip=['tempo', 'mood_value']
).interactive()

# Create the regression line
regression_line = scatter_plot.transform_regression(
    'tempo', 'mood_value'
).mark_line(color='red')

# Combine the scatter plot and regression line
chart = (scatter_plot + regression_line).properties(
    title='Tempo vs. Mood Value'
)

st.altair_chart(chart, use_container_width=True)

# Interactive prediction
st.markdown("### 🔮 Predict Mood from Tempo")
input_tempo = st.number_input("Enter a tempo value (0-10):", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
if st.button("Predict Mood"):
    predicted_mood = lr_model.predict(input_tempo * 20) # Scale tempo to 0-200 range
    st.markdown(f"**Predicted Mood Value:** {predicted_mood:.2f}")

# Display the predicted mood and alignment score
predicted_mood = lr_model.predict(tempo_slider * 20) # Scale tempo slider to match tempo range
lr_fit = 1 - abs(predicted_mood - emotion_slider) / 10

st.markdown(f"**Predicted Mood Value (from Tempo):** {predicted_mood:.2f}")
st.markdown(f"**Linear Regression Fit Score:** {lr_fit:.2f}")
