import streamlit as st
from stqdm import stqdm
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os
import hashlib
import re
import requests

def clean_response(text: str) -> str:
    # Traži prvo veliko slovo u tekstu
    match = re.search(r'[A-Z]', text)
    if match:
        # Vraća deo teksta koji počinje od prvog velikog slova
        return text[match.start():].strip()
    else:
        return text.strip()

def clean_text(text: str) -> str:
    # Uklanjanje novih redova i višestrukih razmaka
    cleaned_text = re.sub(r'\s+', ' ', text.replace("\n", " ")).strip()
    return cleaned_text

class PDFProcessor:

    def __init__(self, data_path: str = './data'):
        self.data_path = data_path
        self.chunks = []


    def load_and_split_pdfs(self):
        loader = PyPDFDirectoryLoader(self.data_path)
        documents = loader.load()

        self.chunks = self.split_text(documents)

    def split_text(self, documents: list[Document]) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
class Embedding:
    def __init__(self, model_name="all-mpnet-base-v2", persist_directory="./chroma_langchain_db"):
            self.model_name = model_name
            self.persist_directory = persist_directory
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)



    def embed_documents(self, documents: list[Document]):
        for document in stqdm(documents, desc="Embedovanje dokumenata", unit="dok"):
            self.vector_store.add_documents([document])

    def get_relevant_chunks(self, query: str, n_resources_to_return: int = 2):
        relevant_texts_with_scores = self.vector_store.similarity_search_with_score(query, k=n_resources_to_return)
        return relevant_texts_with_scores

    def query(self, query: str, max_tokens: int =100    , n_resources_to_return: int = 2,
              lm_studio_url: str = "http://localhost:1234/v1/completions") -> str:

        relevant_texts = self.vector_store.similarity_search(query, k=n_resources_to_return)
        context = " ".join([text.page_content for text in relevant_texts])
        cleaned_context = clean_text(context)

        input_text = f"YOU HAVE TO RETURN A SHORT AND CONCISE ANSWER TO THE QUESTION  {query}  BASED EXLUSIVELY ON THE FOLLOWING TEXT:  {cleaned_context}"

        payload = {
            "prompt": input_text,
            "max_tokens": max_tokens
        }

        response = requests.post(lm_studio_url, json=payload)

        if response.status_code == 200:
            result = response.json()
            raw_response = result['choices'][0]['text']
            cleaned_response = clean_response(raw_response)
            return cleaned_response
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

def check_if_already_embedded(file_path, persist_directory="./chroma_langchain_db"):
    # Izračunaj hash vrednost PDF-a
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    # Proveri da li je hash već sačuvan u bazi
    existing_hashes = os.listdir(persist_directory)
    if file_hash in existing_hashes:
        return True
    else:
        # Ako hash ne postoji, sačuvaj ga
        with open(os.path.join(persist_directory, file_hash), "w") as f:
            f.write(file_hash)
        return False

# Streamlit aplikacija
st.title("Primena velikih jezičkih modela  za odgovaranje na pitanja iz dokumenata")

uploaded_files = st.file_uploader("Učitajte PDF", type=["pdf"], accept_multiple_files=True)

if st.button("Obrada i embeding dokumenta"):
    if uploaded_files:
        data_path = './data'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        embedding_system = Embedding()

        # Čuvanje svakog aploudovanog PDF-a
        for uploaded_file in uploaded_files:
            file_path = os.path.join(data_path, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Provera da li je PDF već embedovan
            if check_if_already_embedded(file_path):
                st.info(f"{uploaded_file.name} je već embedovan.")
            else:
                # Obrada i embedovanje PDF-a
                pdf_processor = PDFProcessor(data_path=data_path)
                pdf_processor.load_and_split_pdfs()

                embedding_system.embed_documents(pdf_processor.chunks)
                st.success(f"{uploaded_file.name} je uspešno obrađen i dodat u vektorsku bazu!")

    else:
        st.error("Učitajte PDF dokument.")

query = st.text_input("Unesite upit za LLM")

# Dugme za prikaz relevantnih chunkova sa sličnostima
if st.button("Prikaz relevantnih chunkova"):
    if query:
        embedding_system = Embedding()
        with st.spinner("Pretraga relevantnih chunkova..."):
            relevant_chunks = embedding_system.get_relevant_chunks(query)

        # Prikaz chunkova sa sličnostima
        st.subheader("Relevantni chunkovi sa sličnostima:")
        for i, (chunk, score) in enumerate(relevant_chunks, 1):
            st.write(f"**Chunk {i}:**")
            st.write(f"Sličnost: {score}")
            st.write(f"Tekst: {chunk.page_content}")
    else:
        st.error("Unesite upit.")

# Dugme za izvršenje LLM upita
if st.button("Izvrši upit"):
    if query:
        embedding_system = Embedding()
        with st.spinner("Učitavanje odgovora..."):
            response = embedding_system.query(query)
        st.text_area("Odgovor LLM-a", response, height=300)
    else:
        st.error("Unesite upit.")
