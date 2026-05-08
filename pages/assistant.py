import streamlit as st
from query_atlas import build_hybrid_router # Import your router
import sqlite3

st.set_page_config(page_title="Assistant", layout="wide")
st.title("💬 Sepsis Clinical Assistant")

# Use session state to persist the engine so it doesn't reload every click
if "router" not in st.session_state:
    with st.spinner("Initializing AI Brain..."):
        st.session_state.router = build_hybrid_router()

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about lactate, SOFA, mortality performance..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching evidence base..."):
            # Execute Query
            response = st.session_state.router.query(prompt)
            
            # Format Response
            st.markdown(response.response)
            
            # Extra: Show the SQL query used if the SQL tool was triggered
            if "sql_query" in response.metadata:
                with st.expander("🛠️ View SQL Logic"):
                    st.code(response.metadata["sql_query"], language="sql")
            
    st.session_state.messages.append({"role": "assistant", "content": response.response})

# Sidebar: Show current knowledge base
with st.sidebar:
    st.subheader("📚 Documents in Atlas")
    conn = sqlite3.connect("sepsis_atlas.db")
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM papers")
    papers = cursor.fetchall()
    for p in papers:
        st.text(f"• {p[0]}")
    conn.close()