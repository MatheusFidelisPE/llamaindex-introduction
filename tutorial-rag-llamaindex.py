from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.gemini import Gemini
import google.generativeai as genai
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, StorageContext, \
  load_index_from_storage, get_response_synthesizer
from llama_index.embeddings.gemini import GeminiEmbedding

#Adicionar a chave para gemini
google_api_key = ""
genai.configure(api_key=google_api_key)

# documents = SimpleDirectoryReader(input_files=['./data/Parecer_Previo_23100645-7.pdf'])
documents = SimpleDirectoryReader(input_dir="./data/")

doc = documents.load_data()

model = Gemini(models="gemini-1.5-pro", api_key=google_api_key)

gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=google_api_key)
service_context = ServiceContext.from_defaults(llm=model, embed_model=gemini_embed_model, chunk_size=800, chunk_overlap=20)

index = VectorStoreIndex.from_documents(doc, service_context=service_context)

index.storage_context.persist()

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)
# response_synthesizer = get_response_synthesizer()
query_engine = RetrieverQueryEngine(
    retriever=retriever
)
# query_engine = index.as_query_engine()

while True:
  prompt = input("prompt: ")
  # busca = query_engine.retrieve(prompt)
  response = query_engine.query(prompt)
  # print("busca: "+ )
  print(type(response))
  cont = 0
  for x in response.source_nodes:
      print(response.source_nodes[cont].text)
      print("--")
      cont += 1

  print(response)