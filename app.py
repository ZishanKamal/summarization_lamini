import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# File loader and pre-processing 
def process_file(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, separators=["\n\n", "\n"])
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts

# LLM Pipeline 
def llm_pipeline(filepath):
    summary_pipeline = pipeline(
        'summarization',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 500,
        min_length = 100
    )
    input_text = process_file(filepath)
    result_summary = summary_pipeline(input_text)[0]['summary_text']
    return result_summary

@st.cache_data
#Function to display the PDF file 
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# streamlit page layout 
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization using Language Model")
    uploaded_file = st.file_uploader("Upload your PDF file to be summarized", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            pdf_file_path = "data/" + uploaded_file.name
            with open(pdf_file_path, "wb") as temp_pdf_file:
                temp_pdf_file.write(uploaded_file.read())
            with col1: 
                st.info("Uploaded PDF File")
                displayPDF(pdf_file_path)
            with col2: 
                st.info("Summary")
                result_summary = llm_pipeline(pdf_file_path)
                st.write(result_summary)
            

if __name__ == "__main__":
    # Model and Tokenizer 
    checkpoint = "LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = "auto", torch_dtype = torch.float32)

    main()