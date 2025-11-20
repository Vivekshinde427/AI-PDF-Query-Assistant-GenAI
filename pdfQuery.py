import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import load_pdf, split_text, create_vector_store

# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()
api_key = os.getenv("gemini_key")
os.environ["GOOGLE_API_KEY"] = api_key

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="📄 GenAI PDF Explorer",
    layout="wide"
)

# -----------------------------
# CSS for menu & styling
# -----------------------------
st.markdown("""
<style>
/* Background Gradient */
body {
    background: linear-gradient(to right, #4B0082, #6A5ACD);
    color: #fff;
}

/* Headers */
h1 {
    text-align: center;
    color: #FFD700;
}
h3 {
    text-align: center;
    color: #EEE;
}

/* Horizontal menu */
.menu {
    display: flex;
    justify-content: center;
    margin-top: 10px;
    margin-bottom: 20px;
    font-weight: bold;
    font-size: 1.1rem;
}
.menu a {
    padding: 10px 25px;
    text-decoration: none;
    color: #fff;
    margin: 0 5px;
    border-bottom: 3px solid transparent;
    transition: all 0.3s;
}
.menu a:hover {
    color: #FFD700;
}
.menu a.active {
    border-bottom: 3px solid #FF4136;  /* red underline for active */
    color: #FFD700;
}

/* Upload box */
.upload-box {
    background-color: #6A5ACD;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: bold;
}

/* Chat bubbles */
.chat-user {
    background-color:#E6E6FA; 
    padding:15px; 
    border-radius:10px; 
    text-align:right;
    color:#000;
    margin-bottom:5px;
}
.chat-ai {
    background-color:#F8F8FF; 
    padding:15px; 
    border-radius:10px; 
    text-align:left;
    color:#000;
    margin-bottom:5px;
}

/* History box */
.history-box {
    background-color:#6A5ACD;
    padding:15px;
    border-radius:10px;
    color:#fff;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("<h1>📄 GenAI PDF Explorer</h1>", unsafe_allow_html=True)
st.markdown("<h3>AI-Powered PDF Q&A | Large Language Model</h3>", unsafe_allow_html=True)

# -----------------------------
# Horizontal Menu
# -----------------------------
# Simple links acting as page selector
page = st.session_state.get("page", "Home")
menu_html = f"""
<div class="menu">
<a href="#" class="{ 'active' if page=='Home' else '' }" onclick="window.location.href='?page=Home'">Home</a>
<a href="#" class="{ 'active' if page=='History' else '' }" onclick="window.location.href='?page=History'">Chat History</a>
</div>
"""
st.markdown(menu_html, unsafe_allow_html=True)

# -----------------------------
# Handle page change
# -----------------------------
query_params = st.experimental_get_query_params()
if "page" in query_params:
    page = query_params["page"][0]
st.session_state["page"] = page

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Home Page
# -----------------------------
if page == "Home":
    uploaded_file = st.file_uploader("📂 Upload PDF to start", type="pdf")

    if uploaded_file:
        # Read PDF
        with st.spinner("🔎 Reading PDF..."):
            raw_text = load_pdf(uploaded_file)
        st.success("✅ PDF loaded successfully!")

        # Split text
        with st.spinner("📄 Splitting text into chunks..."):
            chunks = split_text(raw_text)
        st.info(f"PDF split into {len(chunks)} chunks.")

        # Vector store
        with st.spinner("🧠 Creating embeddings..."):
            vector_store = create_vector_store(chunks)
        st.success("✅ Embeddings ready!")

        # Initialize chat model
        chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", convert_system_message_to_human=True)
        chain = load_qa_chain(chat_model, chain_type="stuff")

        # Q&A Interface
        st.subheader("💬 Ask Questions")
        user_query = st.text_input("Enter your question here:")

        if user_query:
            with st.spinner("🤖 Generating answer..."):
                docs = vector_store.similarity_search(user_query)
                answer = chain.run(input_documents=docs, question=user_query)

                # Save to session
                st.session_state.chat_history.append({"question": user_query, "answer": answer})

                st.markdown(f"<div class='chat-user'><b>You:</b> {user_query}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-ai'><b>AI:</b> {answer}</div>", unsafe_allow_html=True)

    else:
        st.info("📌 Please upload a PDF to start asking questions.")

# -----------------------------
# Chat History Page
# -----------------------------
if page == "History":
    st.markdown("<h2>📝 Chat History</h2>", unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.chat_history:
        for chat in st.session_state.chat_history[::-1]:
            st.markdown(f"<div class='history-box'><b>Q:</b> {chat['question']}<br><b>A:</b> {chat['answer']}</div>", unsafe_allow_html=True)

        # Download all answers
        all_answers = "\n\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in st.session_state.chat_history])
        st.download_button("📥 Download All Answers", data=all_answers, file_name="answers.txt", mime="text/plain")
    else:
        st.info("No questions have been asked yet.")
