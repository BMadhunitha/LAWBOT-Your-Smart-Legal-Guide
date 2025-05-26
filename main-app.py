import os
import streamlit as st
from deep_translator import GoogleTranslator
from langdetect import detect
from dotenv import load_dotenv

# LangChain imports
from langchain_cohere import CohereEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere.chat_models import ChatCohere
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="LAWBOT - Your Smart Legal Guide")
col1, col2, col3 = st.columns([1, 25, 1])
with col2:
    st.title("⚖️ LAWBOT - Your Smart Legal Guide")

# Session management
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def reset_conversation():
    st.session_state["messages"] = []

# Directories
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "data-ingestion-local")
template_dir = os.path.join(current_dir, "legal_templates")

# Embeddings and Chroma vector DB
embedF = CohereEmbeddings(model="embed-english-light-v3.0"
                          ,user_agent="my-legal-app/1.0")
vectorDB = Chroma(embedding_function=embedF, persist_directory=persistent_directory)
kb_retriever = vectorDB.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Chat Models
chatmodel = ChatGroq(
    model="llama-3.1-8b-instant", temperature=0.15, api_key=os.getenv("GROQ_API_KEY")
)
llm = ChatCohere(temperature=0.15, cohere_api_key=os.getenv("COHERE_API_KEY"))

# Rephrasing Prompt
rephrasing_prompt = ChatPromptTemplate.from_messages([
    ("system", "Convert context-dependent questions into standalone ones without answering them."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# History-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm=chatmodel, retriever=kb_retriever, prompt=rephrasing_prompt
)

# QA Prompt
system_prompt_template = (
    "You are LAWBOT, a legal assistant chatbot. "
    "Answer in simple, clear language using a maximum of 4 sentences. "
    "Only respond based on the context below. Do not hallucinate or repeat phrases. "
    "Add the following disclaimer only ONCE at the end: "
    "'⚠️ _This is not legal advice. Please consult a lawyer for legal decisions._'"
    "\nCONTEXT: {context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(chatmodel, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Template handling
def check_for_template_query(query):
    template_keywords = {
        "rental": "Rental_Agreement.txt",
        "power of attorney": "Power_of_Attorney.txt",
        "affidavit": "Affidavit_of_Residence.txt",
        "non-disclosure": "Non_Disclosure_Agreement.txt",
        "employment": "Employment_Contract.txt",
        "ఎన్‌డిఎ": "Non_Disclosure_Agreement.txt",
        "రెంటల్": "Rental_Agreement.txt",
        "అఫిడవిట్": "Affidavit_of_Residence.txt",
    }
    for key, filename in template_keywords.items():
        if key in query.lower():
            return os.path.join(template_dir, filename)
    return None

def get_template_text(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return "⚠️ Sorry, that template format isn't available right now."

# Language detection and translation
def translate_query(user_query):
    try:
        lang = detect(user_query)
        if lang != "en":
            translated = GoogleTranslator(source=lang, target="en").translate(user_query)
            return translated if translated.strip().lower() != user_query.strip().lower() else user_query
        return user_query
    except Exception:
        return user_query

# Display past chat
for msg in st.session_state.messages:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.write(msg.content)

# Chat input
user_query = st.chat_input("Ask me anything ...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    translated_query = translate_query(user_query)
    template_path = check_for_template_query(translated_query)

    if template_path:
        with st.chat_message("assistant"):
            st.markdown("📄 **Here is the requested legal format:**\n")
            st.code(get_template_text(template_path), language="markdown")
    else:
        with st.chat_message("assistant"):
            with st.status("Generating 💡...", expanded=True):
                try:
                    result = rag_chain.invoke({
                        "input": translated_query,
                        "chat_history": st.session_state["messages"]
                    })

                    final_response = (
                        result["answer"].strip() + "\n\n⚠️ _This is not legal advice. Please consult a lawyer for legal decisions._"
                    )
                except Exception as e:
                    final_response = f"❌ Sorry, an error occurred:\n{str(e)}"

            st.markdown(final_response)

        st.session_state.messages.extend([
            HumanMessage(content=user_query),
            AIMessage(content=final_response)
        ])

st.button("Reset Conversation 🗑️", on_click=reset_conversation)
