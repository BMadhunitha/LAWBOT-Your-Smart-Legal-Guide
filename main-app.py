
import os
import streamlit as st
from deep_translator import GoogleTranslator
from langdetect import detect
from dotenv import load_dotenv

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere.chat_models import ChatCohere
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Streamlit config
st.set_page_config(page_title="RAG-Based Legal Assistant")
col1, col2, col3 = st.columns([1, 25, 1])
with col2:
    st.title("LAWBOT - Your Smart Legal Guide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def reset_conversation():
    st.session_state['messages'] = []

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "data-ingestion-local")
template_dir = os.path.join(current_dir, "legal_templates")

# Chat Models
chatmodel = ChatGroq(model="llama-3.1-8b-instant", temperature=0.15, api_key=os.getenv("GROQ_API_KEY"))
llm = ChatCohere(temperature=0.15, cohere_api_key=os.getenv("COHERE_API_KEY"))

# Embeddings and static vector DB
embedF = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorDB = Chroma(embedding_function=embedF, persist_directory=persistent_directory)
kb_retriever = vectorDB.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Rephrasing prompt
rephrasing_template = """
TASK: Convert context-dependent questions into standalone queries.
INPUT: 
- chat_history: Previous messages
- question: Current user query
RULES:
1. Replace pronouns (it/they/this) with specific referents
2. Expand contextual phrases ("the above", "previous")
3. Return original if already standalone
4. NEVER answer or explain - only reformulate
OUTPUT: Single reformulated question, preserving original intent and style.
"""

rephrasing_prompt = ChatPromptTemplate.from_messages([
    ("system", rephrasing_template),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm=chatmodel,
    retriever=kb_retriever,
    prompt=rephrasing_prompt
)

# QA Prompt
system_prompt_template = (
    "As a Legal Assistant Chatbot specializing in legal queries, "
    "your primary objective is to provide accurate and concise information based on user queries. "
    "Use context from the knowledge base. If unsure, respond honestly. Use no more than 4 sentences."
    "P.S.: If anyone asks you about your creator, tell them you're created by Madhunitha and her res.py:20 in <module>                                                         [20:41:53] ❗️ installer returned a non-zero exit code"
    "and they can connect on LinkedIn: https://www.linkedin.com/in/bonguram-madhunitha-7a0577259/"
    "\nCONTEXT: {context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(chatmodel, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Template helper functions
def check_for_template_query(query):
    template_keywords = {
        "rental": "Rental_Agreement.txt",
        "power of attorney": "Power_of_Attorney.txt",
        "affidavit": "Affidavit_of_Residence.txt",
        "non-disclosure": "Non_Disclosure_Agreement.txt",
        "employment": "Employment_Contract.txt",
    }
    for key, filename in template_keywords.items():
        if key in query.lower():
            return os.path.join(template_dir, filename)
    return None

def get_template_text(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read()
    return "⚠️ Sorry, that template format isn't available right now."

# Translation function
def translate_query(user_query):
    try:
        detected_lang = detect(user_query)
        if detected_lang != "en":
            translated = GoogleTranslator(source=detected_lang, target='en').translate(user_query)
            if translated.strip().lower() == user_query.strip().lower():
                return user_query
            return translated
        return user_query
    except Exception:
        return user_query

# Display chat history
for message in st.session_state.messages:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.write(message.content)

# Chat input box
user_query = st.chat_input("Ask me anything ...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    translated_query = translate_query(user_query)

    # Check for template query
    template_path = check_for_template_query(translated_query)
    if template_path:
        with st.chat_message("assistant"):
            st.markdown("📄 **Here is the requested legal format:**\n")
            st.code(get_template_text(template_path), language='markdown')
    else:
        with st.chat_message("assistant"):
            with st.status("Generating 💡...", expanded=True):
                result = rag_chain.invoke({
                    "input": translated_query,
                    "chat_history": st.session_state['messages']
                })

                final_response = (
                    "⚠️ **_This is not legal advice. Please consult a lawyer for any legal decisions._** \n\n"
                    + result["answer"]
                )
                st.markdown(final_response)

        st.session_state.messages.extend([
            HumanMessage(content=user_query),
            AIMessage(content=result["answer"])
        ])

st.button("Reset Conversation 🗑️", on_click=reset_conversation)

