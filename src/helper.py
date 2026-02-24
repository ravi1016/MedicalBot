from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf_file():
    loader = DirectoryLoader(
        "E:\Ravi\GenAI_projects\MedicalBot\data",
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    return docs

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )

    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk

def download_hugging_face_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings

embedding = download_hugging_face_embeddings()
