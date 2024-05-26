from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import pinecone # type: ignore
from dotenv import load_dotenv
import os
from tqdm import tqdm
import hashlib

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medical-chatbot")

# vectors = [
#     {
#         "id": hashlib.sha256(t.page_content.encode("utf-8")).hexdigest(),
#         "values": emb,
#         "metadata": {"text": t.page_content.replace("\r\n", " ").replace("\n", " ")},
#     }
#     for t, emb in zip(
#         text_chunks,
#         embeddings.embed_documents([doc.page_content for doc in text_chunks]),
#     )
# ]
# index.upsert(vectors=vectors)


for text in tqdm(text_chunks):
    emb = embeddings.embed_query(text.page_content)
    # Clean up the page content to create a unique ID
    doc_id = hashlib.sha256(text.page_content.encode("utf-8")).hexdigest()
    text = text.page_content.replace("\r\n", " ").replace("\n", " ")

    # Create the vector
    vector = {"id": doc_id, "values": emb, "metadata": {"text": text}}

    # Upsert the vector one by one
    try:
        index.upsert(vectors=[vector])
    except Exception as e:
        print(vector)
        print(f"Error while upserting vector: {str(e)}")
