import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

os.environ["LANGCHAIN_TRACING_V2"] = "true"

def get_vectorstore_from_url(urls, api_key):
    with st.sidebar:
        with st.spinner('Loading the document...'):
            loader = WebBaseLoader(urls)
            document = loader.load()
        st.success('Document loaded!', icon="✅")

    with st.sidebar:
        with st.spinner('Splitting the document into chunks...'):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
            document_chunks = text_splitter.split_documents(document)
        st.success(f'Document chunking completed! {len(document_chunks)} chunks', icon="✅")

    with st.sidebar:
        with st.spinner('Creating vectorstore from document chunks...'):
            embeddings = OpenAIEmbeddings(api_key=api_key)
            vector_store = Chroma.from_documents(document_chunks, embeddings)
        st.success('Embeddings created and saved to vector store!', icon="✅")
        st.info("The vector store will take care of storing embedded data and perform vector search.")

    return vector_store

def get_context_retriever_chain(vector_store, api_key):
    llm = ChatOpenAI(temperature=0, api_key=api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    return create_history_aware_retriever(llm=llm, retriever=retriever, prompt=prompt)

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(temperature=0, api_key=st.session_state.api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the below context:\n\n{context}."),
        ("system", "Also return the sources of your answer from the response metadata."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_query):
    if 'api_key' not in st.session_state:
        st.warning("API Key is not set. Please enter the API Key.")
        return "No API key provided."
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store, st.session_state.api_key)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    return response['answer']

# App configuration
st.set_page_config(page_title="Chat with URLs", page_icon="💬")
st.title("Chat with URLs")

# Initialize chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar configuration
with st.sidebar:
    st.title("Chat with URLs")
    st.header("Settings")
    api_key = st.text_input("Enter API Key", type="password")
    if api_key:
        st.session_state.api_key = api_key
    option = st.selectbox('Select number of URLs to chat with...', ('1', '2', '3', '4', '5'))
    urls = [st.text_input(f"URL {i+1}") for i in range(int(option))]

# Main content area
if 'api_key' not in st.session_state or not st.session_state.api_key:
    st.info("Please enter the API Key to use the application.")
elif any(not url for url in urls):
    st.info("Please enter the website URLs to proceed.")
else:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(urls, st.session_state.api_key)
    user_query = st.chat_input("Type your message here...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        response = get_response(user_query)
        st.session_state.chat_history.append(AIMessage(content=response))
    
    # Display the chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)