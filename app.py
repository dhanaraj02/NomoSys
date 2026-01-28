import streamlit as st
from chatbot_backend import build_legal_chain

# Initialize chatbot
st.title("⚖️ NyayaSathi – AI Legal Chatbot ")

@st.cache_resource
def load_chain():
    return build_legal_chain()

qa_chain = load_chain()

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a legal question:")

if query:
    result = qa_chain.invoke({"question": query, "chat_history": st.session_state.history})
    st.session_state.history.append((query, result["answer"]))
    st.write("**NyayaSathi:**", result["answer"])

# Display chat history
if st.session_state.history:
    st.markdown("### Chat History")
    for q, a in st.session_state.history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
