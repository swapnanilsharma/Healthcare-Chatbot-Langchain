from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


#Extract data from the PDF
def load_pdf(data):
    """
    Loads PDF files from the specified directory using DirectoryLoader and PyPDFLoader.

    Parameters:
    data: The directory path containing the PDF files.

    Returns:
    The loaded documents from the PDF files.
    """
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents



#Create text chunks
def text_split(extracted_data):
    """
    Splits the extracted data into chunks using a RecursiveCharacterTextSplitter object.

    Parameters:
    extracted_data: The data to be split into chunks.

    Returns:
    A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks



#download embedding model
def download_hugging_face_embeddings():
    """
    Downloads Hugging Face embeddings with the specified model name and returns the embeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings