# ChatURL: LangChain Streamlit App

Welcome to the **ChatURL** project! This Streamlit app leverages LangChain and OpenAI to allow users to interact with content from a given URL in a conversational manner. Below you'll find detailed information on the project, its purpose, and how to use it.

![ChatURL](https://i.imgur.com/64VTTb6.png)

## Problem Statement

**Design a conversational interface to enable users to interact with web content in real-time.**

**Target Users:** Researchers, students, and professionals who need to extract and understand information from web pages quickly andaccurately.

**Context:** Users often need to gather insights from lengthy web documents and want to ask specific questions or request summaries without manually sifting through the text.

**Activity:** Provide an interactive, AI-driven chat interface that allows users to ask questions about the content of a given URL and receive relevant responses based on the content.

**Target Performance:** Users should be able to input a URL, have the document processed and split into chunks, and interact with the content using natural language queries with a high degree of accuracy and relevance.

## Project Design

### System Architecture

The system integrates multiple components to achieve its goal:

1. **Document Loader:** Fetches the document content from the provided URL.
2. **Text Splitter:** Divides the document into manageable chunks.
3. **Embeddings and Vector Store:** Converts text chunks into embeddings and stores them in a vector store for efficient retrieval.
4. **Retrieval Chain:** Uses a retriever to find relevant chunks based on user queries.
5. **Conversational Chain:** Generates responses based on retrieved content and maintains conversational context.

### Implementation

The app is built using Streamlit for the frontend interface and LangChain for NLP and AI functionalities. It uses OpenAI's GPT models for generating responses and embeddings.

## Usage Instructions

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/martintmv-git/ml-playground.git
   cd ML Playground/NLP/Streamlit Apps/LangChain ChatURL
   ```
2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

### Running the App

1. **Start the Streamlit app:**

   ```bash
   streamlit run app.py
   ```
2. **Open your browser and navigate to the local address provided by Streamlit.**

### Using the App

1. **Enter the OpenAI API Key and LangChain API Key (optional) in the sidebar.**
2. **Input the URL of the web page you want to interact with.**
3. **Type your questions or queries in the chat box.**
4. **View the responses generated based on the document content.**

### Features

- **Conversational Interface:** Interact with the content of a URL using natural language queries.
- **Document Chunking:** Automatically splits the document into chunks for efficient processing.
- **Vector Store:** Stores embedded document chunks for fast retrieval.
- **Contextual Responses:** Maintains conversation history to provide context-aware answers.
- **Code Mode:** Enables users to request production-ready code solutions based on the documentation.

## Code Documentation

### Main Functions

- `get_vectorstore_from_url(url, api_key)`: Loads the document from the URL, splits it into chunks, and creates a vector store from the chunks.
- `get_context_retriever_chain(vector_store, api_key)`: Creates a history-aware retriever chain using the vector store.
- `get_conversational_rag_chain(retriever_chain)`: Creates a conversational RAG chain for generating responses.
- `get_response(user_query, code_mode=False)`: Retrieves and generates a response based on the user query.
- `display_chat_history()`: Displays the chat history in the Streamlit app.
- `display_all_history()`: Displays all chat history in a separate section.

### App Configuration

The app is configured with a sidebar for API key input and URL input. It also includes options to view chat history and switch to Code Assistant Mode for generating code solutions.

### Example Query

- **User Query:** "What is the main topic of the document?"
- **URL:** "https://some.random.document"
- **Response:** The AI will provide a summary or relevant information from the document based on the input URL.

---

- **User Query:** "Generate a 3 page blog about NLP - ML"
- **URL:** "https://docusaurus.io/docs/blog"
- **Response:** To generate a 3-page blog about Natural Language Processing (NLP) and Machine Learning (ML) using Docusaurus, you can follow the steps below. This solution will create three separate Markdown files, each representing a page of the blog.

## Deployment

The app is hosted on Hugging Face Spaces and can be accessed via the following link:
[ChatURL on Hugging Face Spaces](https://huggingface.co/spaces/martintmv/ChatURL)

## License

This project is licensed under the **Apache License Version 2.0**.
