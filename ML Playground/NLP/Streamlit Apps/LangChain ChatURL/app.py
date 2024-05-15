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
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
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
    llm = ChatOpenAI(temperature=0, api_key=api_key, model_name="gpt-4o")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    st.write(f"Using model: {llm.model_name}")  # Display the model name in the app
    return create_history_aware_retriever(llm=llm, retriever=retriever, prompt=prompt)

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(temperature=0, api_key=st.session_state.api_key, model_name="gpt-4o")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the below context:\n\n{context}."),
        ("system", "Also return the sources of your answer from the response metadata."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_query, code_mode=False):
    if 'api_key' not in st.session_state:
        st.warning("API Key is not set. Please enter the API Key.")
        return "No API key provided."
    
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store, st.session_state.api_key)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    if code_mode:
        user_query = f"Provide a comprehensive production-ready code solution for the following query according to the provided documentation: {user_query}"
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    return response['answer']

def display_chat_history():
    if 'chat_history' in st.session_state:
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)

def display_all_history():
    st.header("All Chat History")
    if 'all_chat_history' in st.session_state:
        for message in st.session_state.all_chat_history:
            if isinstance(message, AIMessage):
                st.markdown(f"**AI:** {message.content}")
            elif isinstance(message, HumanMessage):
                st.markdown(f"**Human:** {message.content}")
    else:
        st.markdown("No chat history available.")

# App configuration
st.set_page_config(page_title="Chat with URLs", page_icon="💬")
st.title("Chat with URLs")

# Initialize chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'all_chat_history' not in st.session_state:
    st.session_state.all_chat_history = []

# Initialize URL inputs if they don't exist
if 'urls' not in st.session_state:
    st.session_state.urls = [""] * 5  # Initialize with 5 empty URLs

# Sidebar configuration
with st.sidebar:
    st.title("Chat with URLs")
    st.header("Settings")
    api_key = st.text_input("Enter API Key", type="password")
    if api_key:
        st.session_state.api_key = api_key
    option = st.selectbox('Select number of URLs to chat with...', ('1', '2', '3', '4', '5'))
    num_urls = int(option)
    urls = [st.text_input(f"URL {i+1}", value=st.session_state.urls[i]) for i in range(num_urls)]
    st.session_state.urls = urls

    if 'show_all_history' not in st.session_state:
        st.session_state.show_all_history = False

    code_mode = st.checkbox("Code Assistant Mode")

    if st.session_state.show_all_history:
        if st.button("Hide History"):
            st.session_state.show_all_history = False
            st.rerun()
    else:
        if st.button("Show History"):
            st.session_state.show_all_history = True
            st.rerun()

    # Clear Knowledge button
    if st.button("Clear Knowledge"):
        st.session_state.pop("vector_store", None)
        st.session_state.pop("chat_history", None)
        st.session_state.urls = [""] * num_urls  # Clear the URLs
        st.rerun()

# Main content area
if st.session_state.show_all_history:
    display_all_history()
else:
    if 'api_key' not in st.session_state or not st.session_state.api_key:
        st.info("Please enter the API Key to use the application.")
    elif any(not url for url in st.session_state.urls[:num_urls]):
        st.info("Please enter the website URLs to proceed.")
    else:
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vectorstore_from_url(st.session_state.urls[:num_urls], st.session_state.api_key)
        user_query = st.chat_input("Type your message here...")
        if user_query:
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.all_chat_history.append(HumanMessage(content=user_query))
            response = get_response(user_query, code_mode=code_mode)
            st.session_state.chat_history.append(AIMessage(content=response))
            st.session_state.all_chat_history.append(AIMessage(content=response))

        display_chat_history()