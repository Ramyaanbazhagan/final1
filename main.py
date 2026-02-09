import streamlit as st
import json
import numpy as np
import tempfile
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from gtts import gTTS
import speech_recognition as sr

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config("Neural Persona", layout="wide")

# ===============================
# GEMINI API (WORKING MODE)
# ===============================
genai.configure(api_key="AIzaSyDuaKnn1X6sp1kGZ_dtZHYkY-pZeQgfM-4")

# ===============================
# ETHICAL DISCLAIMER
# ===============================
st.info("""
⚠️ **Disclaimer**
This AI is a synthetic research prototype.
It does not represent a real person.
It does not replace human relationships or emotional support.
""")

# ===============================
# LOAD & SAVE MEMORY
# ===============================
MEMORY_FILE = "dataset.json"

def load_memory():
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(new_entry):
    data = load_memory()
    data.setdefault("long_term_memory", []).append(new_entry)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

memory_data = load_memory()

# ===============================
# FLATTEN MEMORY
# ===============================
def flatten_json(data):
    texts = []
    for v in data.values():
        if isinstance(v, list):
            texts.extend([str(x) for x in v])
        elif isinstance(v, dict):
            texts.extend([str(x) for x in v.values()])
        else:
            texts.append(str(v))
    return texts

memory_texts = flatten_json(memory_data)

# ===============================
# EMBEDDINGS
# ===============================
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(memory_texts, convert_to_tensor=True)
    return model, emb

model, embeddings = load_embeddings()

def retrieve_context(query):
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(q_emb, embeddings)[0]
    idx = np.argmax(scores.cpu().numpy())
    return memory_texts[idx]

# ===============================
# EMOTION DETECTION
# ===============================
def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["sad", "lonely", "miss", "cry", "upset"]):
        return "sad"
    if any(w in t for w in ["happy", "excited", "love", "great"]):
        return "happy"
    return "neutral"

# ===============================
# TEXT TO SPEECH
# ===============================
def speak(text, emotion):
    slow = True if emotion == "sad" else False
    tts = gTTS(text, slow=slow)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# ===============================
# VOICE INPUT
# ===============================
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.warning("🎤 Listening...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except:
        return ""

# ===============================
# SIDEBAR – PERSONA MODES
# ===============================
st.sidebar.title("🧠 Persona Mode")

mode = st.sidebar.radio(
    "Choose Mode",
    ["🧠 Memory Mode", "💬 Casual Talk", "🤍 Emotional Support"]
)

# ===============================
# UI
# ===============================
st.title("🤖 Neural Persona – Voice & Emotion AI")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    st.markdown(f"**{role}:** {msg}")

# ===============================
# INPUT OPTIONS
# ===============================
col1, col2 = st.columns(2)

with col1:
    user_text = st.text_input("Type your message")

with col2:
    if st.button("🎤 Speak"):
        user_text = listen()

# ===============================
# PROCESS
# ===============================
if st.button("Send"):
    if user_text.strip():
        st.session_state.chat.append(("You", user_text))
        save_memory(user_text)

        context = retrieve_context(user_text)
        emotion = detect_emotion(user_text)

        persona_instruction = {
            "🧠 Memory Mode": "Focus on recalling memories.",
            "💬 Casual Talk": "Speak lightly and friendly.",
            "🤍 Emotional Support": "Be comforting, but encourage real human support."
        }[mode]

        prompt = f"""
You are a synthetic AI persona for research.
Ethical AI rules:
- Do not claim real identity
- Avoid emotional dependency

Persona Mode:
{persona_instruction}

Emotion detected: {emotion}

Relevant Memory:
{context}

User: {user_text}
AI:
"""

        response = genai.GenerativeModel(
            "models/gemini-2.5-flash"
        ).generate_content(prompt)

        ai_text = response.text.strip()
        st.session_state.chat.append(("AI", ai_text))

        audio = speak(ai_text, emotion)
        st.audio(audio)
