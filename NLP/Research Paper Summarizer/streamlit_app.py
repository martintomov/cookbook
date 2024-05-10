import streamlit as st
import requests
from io import BytesIO
import fitz  # PyMuPDF
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def extract_text_from_pdf(pdf_stream):
    try:
        with fitz.open("pdf", pdf_stream) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return extract_text_from_pdf(BytesIO(response.content))
    except requests.RequestException as e:
        return f"Failed to fetch PDF from URL: {str(e)}"
    except Exception as e:
        return f"Error reading PDF from URL: {str(e)}"

def generate_toc(pdf_stream):
    try:
        doc = fitz.open("pdf", pdf_stream)
        toc = []
        for i, page in enumerate(doc, 1):
            for block in page.get_text("blocks"):
                if block[4].startswith("Chapter") or block[4].startswith("Section") or any(char.isdigit() for char in block[4][:10]):
                    toc.append(f"Page {i}: {block[4].strip()}")
        return toc if toc else ["No table of contents items detected."]
    except Exception as e:
        return [f"Error generating table of contents: {str(e)}"]

def split_text_into_chunks(text, chunk_size=4000):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if sum(len(w) for w in current_chunk) > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

@st.cache_data
def generate_response(txt, api_key):
    try:
        llm = OpenAI(temperature=0, openai_api_key=api_key)
        chunks = split_text_into_chunks(txt)
        summaries = []

        for chunk in chunks:
            text_splitter = CharacterTextSplitter()
            texts = text_splitter.split_text(chunk)
            docs = [Document(page_content=t) for t in texts]
            chain = load_summarize_chain(llm, chain_type='map_reduce')
            summary = chain.run(docs)
            summaries.append(summary)

        return ' '.join(summaries)
    except Exception as e:
        return f"An error occurred: {str(e)}"

st.set_page_config(page_title='🦜🔗 Research Paper Summarizer')
st.title('🦜🔗 Research Paper Summarizer')

st.markdown('### Instructions')
st.info('Enter your OpenAI API key first, then upload a PDF file or enter a URL to a PDF file of the research paper you want to summarize. Ensure that your OpenAI API key is correct to enable processing.')

# OpenAI API key input at the top
openai_api_key = st.text_input('OpenAI API Key', type='password')

# Choose the input method and optionally generate ToC
st.subheader('Input your research paper')
input_option = st.radio("Choose your input method:", ('URL', 'Upload PDF'))
show_toc = st.checkbox('Generate Table of Contents')

paper_text = ''
toc = ''
if input_option == 'URL':
    paper_url = st.text_input('Enter the URL of the research paper:')
    if paper_url and st.button('Fetch and Process PDF'):
        pdf_content = requests.get(paper_url).content
        if show_toc:
            toc = generate_toc(BytesIO(pdf_content))
            st.markdown("## Table of Contents")
            for item in toc:
                st.markdown(item)
        paper_text = extract_text_from_pdf(BytesIO(pdf_content))
        if "Error" not in paper_text:
            st.success('PDF fetched and processed successfully.')
        else:
            st.error(paper_text)
elif input_option == 'Upload PDF':
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file and st.button('Process Uploaded PDF'):
        pdf_content = uploaded_file.getvalue()
        if show_toc:
            toc = generate_toc(BytesIO(pdf_content))
            st.markdown("## Table of Contents")
            for item in toc:
                st.markdown(item)
        paper_text = extract_text_from_pdf(BytesIO(pdf_content))
        if "Error" not in paper_text:
            st.success('PDF processed successfully.')
        else:
            st.error(paper_text)

if openai_api_key and paper_text:
    with st.spinner('Generating summary...'):
        response = generate_response(paper_text, openai_api_key)
        if 'Error' not in response:
            st.markdown("## Summary")
            st.markdown(f"<div style='color: orange;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.error(response)