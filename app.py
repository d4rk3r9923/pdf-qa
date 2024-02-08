import torch
import base64
import textwrap
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import chroma_settings

checkpoint = 'LaMini-T5-738M'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline('text2text-generation', model=base, tokenizer=tokenizer, max_length=256, do_sample=True, temperature=0.3, top_p=0.95)
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    db = Chroma(persist_directory='db', embedding_function=embeddings, client_settings=chroma_settings)
    retriever = db.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True)

def qa_process(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    gen = qa(instruction)
    return gen['result'], gen

def main():
    st.title("Search PDF...")
    with st.expander("About..."):
        st.markdown("""This is...""")
    
    question = st.text_area("Question: ")
    if st.button("Search"):
        st.info("Question: " + question)
        st.info("Answer:")
        answer, metadata = qa_process(question)
        st.write(answer)
        st.write(metadata)

if __name__ == "__main__":
    main()
