import os
import streamlit as st
from groq import Groq
import json
import io
from tavily import TavilyClient
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from gtts import gTTS
import speech_recognition as sr
from fpdf import FPDF
import sqlite3
import bcrypt
from pydub import AudioSegment

# Import Hugging Face libraries for image generation
from diffusers import StableDiffusionPipeline, PNDMScheduler # Import PNDMScheduler
import torch
from huggingface_hub import HfApi
# Import for Hugging Face ASR model
from transformers import pipeline

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Lion AI Multipurpose App",
    page_icon="🦁",
    layout="wide"
)

st.markdown(
    """
<style>
h1 {
    color: #FF4B4B; /* Change the title color */
    font-size: 3em; /* Increase font size */
    text-align: center;
}
body {
    background-color: #f0f2f6; /* Example: light grey background */
}
</style>
""",
    unsafe_allow_html=True
)

# --- Database Initialization ---
DATABASE_FILE = 'users.db'
def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- Initialize Session State for Authentication ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# --- API Key Handling ---
groq_api_key = os.environ.get('GROQ_API_KEY')
if not groq_api_key:
    st.error("GROQ_API_KEY environment variable not found. Please set it in Colab secrets.")
    st.stop() # Stop the app if API key is missing
client = Groq(api_key=groq_api_key)

tavily_api_key = os.environ.get('TAVILY_API_KEY')
if not tavily_api_key:
    st.error("TAVILY_API_KEY environment variable not found. Please set it in Colab secrets.")
    st.stop() # Stop the app if API key is missing

# Hugging Face Token for Image Generation
hf_token = os.environ.get('HF_TOKEN')
if not hf_token:
    st.info("HF_TOKEN environment variable not found. Image generation will not be available. Please set it in Colab secrets.")
    st.stop()
    
# --- User Management Functions ---
def add_user(username, password):
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # Username already exists
    finally:
        conn.close()

def get_user(username):
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user

# --- Sidebar Navigation and Authentication UI ---
with st.sidebar:
    st.title("Menu")
    if st.session_state.logged_in:
        st.write(f"Welcome, {st.session_state.current_user}!")
        if st.button("Sign Out"):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.success("You have been signed out.")
            st.rerun()

    if not st.session_state.logged_in:
        selected_page = st.sidebar.selectbox(
            "Choose an action",
            ['Sign In', 'Sign Up']
        )
    else:
        selected_page = st.sidebar.selectbox(
            "Choose a page",
            ['Research Assistant', 'StudyMate', 'Convert Audio To Text', 'Convert Text To PDF', 'Prompt Generation']
        )

# --- Main Page Content Logic ---
if not st.session_state.logged_in:
    if selected_page == 'Sign Up':
        st.title("🦁 Sign Up")
        with st.form("signup_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            signup_button = st.form_submit_button("Sign Up")

            if signup_button:
                if not new_username or not new_password:
                    st.error("Username and password cannot be empty.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    if add_user(new_username, new_password):
                        st.success("Account created successfully! Please sign in.")
                        st.session_state.selected_page = 'Sign In' # Redirect to sign in
                        st.rerun()
                    else:
                        st.error("Username already exists.")

    elif selected_page == 'Sign In':
        st.title("🦁 Sign In")
        with st.form("signin_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            signin_button = st.form_submit_button("Sign In")

            if signin_button:
                user = get_user(username)
                if user and bcrypt.checkpw(password.encode('utf-8'), user[1].encode('utf-8')):
                    st.session_state.logged_in = True
                    st.session_state.current_user = username
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

else: # User is logged in
    # --- Protected Pages (only accessible when logged in) ---
    if selected_page == 'Research Assistant':
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Add an initial system message to set the persona for the assistant
            st.session_state.messages.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )

        def chat_with_groq(messages):
            # Call the API with stream=True to get a streaming response
            return client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,  # Pass the entire message history for context
                temperature=0.7,
                stream=True
            )

        def stream_response(current_messages):
            full_response = ""
            placeholder = st.empty()
            for chunk in chat_with_groq(current_messages):
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    # Using st.markdown for potential rich text and blinking cursor
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)  # Display final response
            return full_response  # Return the full response for logging

        st.title("🦁Chat with Lion AI Assistant")

        st.write(
            "This is Lion AI research that provides fast and accurate "
            "solutions to your questions."
        )
        st.write("___")

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            # Don't display the system message directly to the user
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        user_input = st.chat_input("Here is Lion AI, ask anything")

        with st.spinner("Generating response..."):
            if user_input:
                # Add user message to chat history and display it
                st.session_state.messages.append(
                    {"role": "user", "content": user_input}
                )
                with st.chat_message("user"):
                    st.markdown(user_input)

                # Generate and stream assistant response
                with st.chat_message("assistant"):
                    assistant_response = stream_response(
                        st.session_state.messages
                    )
                    # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                )
            else:
                # Only show this message if no user input has been provided yet
                if (len(st.session_state.messages) == 1 and
                        st.session_state.messages[0]["role"] == "system"):
                    st.write("Please enter a message to start the conversation with Lion.")

    elif selected_page == 'StudyMate':
        # ---------- clients (built once, using secrets) ----------
        tavily = TavilyClient(api_key=tavily_api_key)

        st.title("📚 Lion AI is Your StudyMate")
        st.caption(
            "Your research & homework assistant - "
            "upload your notes, then ask anything."
        )
        # Groq-hosted model used for the agent's reasoning + tool calling.
        # openai/gpt-oss-120b is Groq's current recommended model for strong
        # tool-use quality.
        MODEL_NAME = "openai/gpt-oss-120b"

        SYSTEM_PROMPT = (
            "You are StudyMate, a helpful research and homework assistant. "
            "Users will upload their notes via a sidebar, and these notes will be "
            "indexed and available through the 'search_my_notes' tool. "
            "Always use the 'search_my_notes' tool first for any question "
            "that could plausibly be covered in their uploaded (and already "
            "indexed) document. Be clear and concise. If neither tool has the "
            "answer, say so honestly instead of guessing. DO NOT ask the user "
            "to re-upload documents if they have already been indexed. The "
            "'search_my_notes' tool already has access to indexed documents." +
            " If anyone asks about your name, tell them that your name is Lion AI."
        )


        @st.cache_resource
        def get_collection():
            chroma_client = chromadb.Client()
            embed_fn = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=("all-MiniLM-L6-v2")
                )
            )
            return chroma_client.get_or_create_collection(
                name="my_notes",
                embedding_function=embed_fn
            )


        collection = get_collection()


        # ---------- session state ----------
        # The conversation list IS the agent's memory, and its first entry is the
        # system prompt (Groq/OpenAI-style messages put the system prompt in the
        # messages list itself, rather than passing it as a separate argument).
        if "conversation" not in st.session_state:
            st.session_state.conversation = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
        if "doc_indexed" not in st.session_state:
            st.session_state.doc_indexed = False


        # ---------- document ingestion ----------
        def extract_text(uploaded_file):
            if uploaded_file.name.lower().endswith(".pdf"):
                reader = PdfReader(io.BytesIO(uploaded_file.read()))
                return "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            return (uploaded_file.read().decode("utf-8", errors="ignore"))


        def chunk_text(text, chunk_size=800, overlap=100):
            chunks, start = [], 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start = end - overlap
            return [c.strip() for c in chunks if c.strip()]


        with st.sidebar:
            st.write("How to use this page")
            st.caption("1. Upload the document")
            st.caption("2. Click on 'Index the document' button")
            st.caption("3. Go back to chat page and ask questions related to your decument")
            st.write("")
            st.header("Your documents")
            uploaded_file = st.file_uploader(
                "Upload notes (PDF or .txt)",
                type=["pdf", "txt"]
            )
            if uploaded_file is not None and st.button("Index this document"):
                with st.spinner("Reading and indexing..."):
                    text = extract_text(uploaded_file)
                    chunks = chunk_text(text)
                    existing = collection.count()
                    collection.add(
                        documents=chunks,
                        ids=[
                            f"chunk_{existing + i}"
                            for i in range(len(chunks))
                        ],

                    )
                    st.session_state.doc_indexed = True
                st.success(
                    f"Indexed {len(chunks)} chunks from {uploaded_file.name}."
                )
            if st.session_state.doc_indexed:
                st.info(
                    "StudyMate will check your notes first before searching the web."
                )
            if st.button("Clear conversation"):
                st.session_session.conversation = [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ]
                st.rerun()



        # ---------- tools ----------
        def search_my_notes(query: str) -> str:
            if collection.count() == 0:
                return "No documents have been uploaded yet."
            results = collection.query(query_texts=[query], n_results=3)
            if not results["documents"][0]:
                return "No relevant content found in the uploaded document."
            return "\n\n".join(results["documents"][0])


        def web_search(query: str) -> str:
            results = tavily.search(query=query, max_results=3)
            formatted_results = []
            for r in results["results"]:
                title_part = f"- {r['title']}: "
                content_part = f"{r['content'][:200]}"
                formatted_results.append(
                    title_part + content_part
                )
            return "\n".join(formatted_results)


        def calculator(expression: str) -> str:
            try:
                # Using eval is dangerous. For a real app, a safer math parser
                # should be used. For this example, it's sufficient.
                safe_globals = {"__builtins__": {}}
                return str(eval(expression, safe_globals))
            except Exception as e:
                return (
    f"Error evaluating expression: {e}")


        # Groq's API is OpenAI-compatible, so tools are described in OpenAI's
        # "function calling" schema: each tool is wrapped in a
        # {"type": "function", "function": {...}} object,
        # with parameters as JSON Schema.
        TOOLS = [
            {
                "type": "function",
                "function": {
                    "name": "search_my_notes",
                    "description": (
                        "Search the user's own uploaded document/notes for "
                        "relevant content. ALWAYS try this first for anything "
                        "that could be covered in the user's material before "
                        "searching the open web."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to look for in the notes."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the open web for current facts, definitions, "
                        "or general knowledge not found in the user's own notes."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": (
                        "Evaluate a mathematical expression, e.g. for grade "
                        "averages, percentages, or unit conversions."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "A Python-evaluable math "
                                               "expression."
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]


        def run_tool(name, tool_input):
            if name == "search_my_notes":
                return search_my_notes(**tool_input)
            elif name == "web_search":
                return web_search(**tool_input)
            elif name == "calculator":
                return calculator(**tool_input)
            return f"Error: tool '{name}' does not exist."


        def run_agent(messages, max_iterations=6):
            for _ in range(max_iterations):
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    max_tokens=1024,
                    tools=TOOLS,
                    messages=messages,
                )
                message = response.choices[0].message
                # Groq's SDK wants the assistant appended as a plain dict.
                messages.append(
                    message.model_dump(
                        exclude_none=True
                    )
                )

                if not message.tool_calls:
                    return message.content, messages

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_input = json.loads(tool_call.function.arguments)
                    result = run_tool(tool_name, tool_input)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    })
            return (
                "Sorry, I couldn't finish that in time.",
                messages
            )


        # ---------- chat UI ----------
        for msg in st.session_state.conversation:
            if (msg["role"] in ("user", "assistant") and
                    isinstance(msg.get("content"), str) and msg["content"]):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        user_input = st.chat_input("Ask about your notes, or anything else...")
        if user_input:
            st.session_state.conversation.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, st.session_state.conversation = \
                        run_agent(st.session_state.conversation)
                    st.write(answer)

    elif selected_page == 'Convert Audio To Text':
        st.title("🎤 convert Audio To Text With Lion AI")
        st.write("Upload an audio file and get its transcription.")

        uploaded_audio = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

        if uploaded_audio is not None:
            if st.button("Transcribe Audio"):
                with st.spinner("Transcribing audio..."):
                    try:
                        # Cache the model loading for efficiency
                        @st.cache_resource
                        def load_asr_model():
                            # Using a general ASR model like Whisper
                            return pipeline("automatic-speech-recognition", model="openai/whisper-small", token=hf_token)

                        asr_model = load_asr_model()

                        # Read the uploaded file into an in-memory BytesIO object
                        audio_file_like = io.BytesIO(uploaded_audio.getvalue())

                        # Use pydub to load the audio segment, it's more robust with formats
                        audio_segment = AudioSegment.from_file(audio_file_like)

                        # Ensure appropriate sample rate and channels for ASR models
                        audio_segment = audio_segment.set_frame_rate(16000) # Set to 16kHz sample rate
                        audio_segment = audio_segment.set_channels(1) # Set to mono
                        audio_segment = audio_segment.normalize() # Normalize volume for better recognition

                        # Export the processed audio to a temporary WAV file
                        converted_audio_path = "temp_audio_for_asr.wav"
                        audio_segment.export(converted_audio_path, format="wav")

                        # Use the Hugging Face ASR model for transcription
                        # The ASR pipeline directly returns a dictionary with 'text' key
                        transcription_result = asr_model(converted_audio_path, return_timestamps=True)

                        # When return_timestamps=True, the output is a dictionary with 'text' and 'chunks'
                        # We need to concatenate the text from all chunks
                        text = "".join([chunk["text"] for chunk in transcription_result["chunks"]])

                        st.success("Transcription Complete!")
                        st.text_area("Transcribed Text", value=text, height=200)

                        # Clean up temp file
                        if os.path.exists(converted_audio_path):
                            os.remove(converted_audio_path)

                    except Exception as e:
                        st.error(f"An error occurred during transcription: {e}")

    elif selected_page == 'Convert Text To PDF':
        st.title("📄 Convert Text To PDF With Lion AI ")
        st.write("Enter text and generate a PDF document.")

        pdf_text = st.text_area("Enter text for your PDF here:", height=300)
        pdf_filename = st.text_input("PDF Filename (e.g., my_document.pdf):", "document.pdf")

        if st.button("Generate PDF"):
            if not pdf_text:
                st.warning("Please enter some text to generate a PDF.")
            else:
                with st.spinner("Generating PDF..."):
                    try:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        # Encode text to utf-8 if it contains non-ASCII characters
                        pdf.multi_cell(0, 10, txt=pdf_text.encode('latin-1', 'replace').decode('latin-1')) # 0 for full width, 10 for line height

                        # Save PDF to a temporary file
                        pdf_output_path = f"/tmp/{pdf_filename}"
                        pdf.output(pdf_output_path)

                        with open(pdf_output_path, "rb") as f:
                            st.download_button(
                                label="Download PDF",
                                data=f.read(),
                                file_name=pdf_filename,
                                mime="application/pdf"
                            )
                        st.success("PDF generated successfully!")
                        os.remove(pdf_output_path) # Clean up temp file

                    except Exception as e:
                        st.error(f"An error occurred during PDF generation: {e}")
    
