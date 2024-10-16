from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


documents = SimpleDirectoryReader(input_files=['D:\\1. Desenvolvimento-estudos\\3. IA Generativa\llama-index-project\data\\annotation-data.txt']).load_data()

embed_model = HuggingFaceEmbedding()

db = chromadb.PersistentClient(path="./matheus_chromadb")
chroma_collection = db.get_or_create_collection("matheus_collection")


print(documents)