from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from constants import chroma_settings
import os

def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith('.pdf'):
                print(file)
                pdf_path = os.path.join(root, file)
                loader = PDFMinerLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embeddings, persist_directory='db', client_settings=chroma_settings)
    db.persist()
    db = None

if __name__ == "__main__":
    main()
