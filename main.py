# 1. Install necessary libraries
# pip install langchain-community langchain-core pymupdf faiss-cpu sentence-transformers rank_bm25
# pip install --upgrade --quiet  langchain-cohere # Or use a local cross-encoder
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_cohere import ChatCohere
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere.rerank import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


import time

# --- SETUP ---
# Load and split the document as before
loader = PyMuPDFLoader("Entrepreneurship-The Startup Owner_s Manual.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# --- 1. HYBRID SEARCH SETUP ---
print("Setting up Hybrid Search...")

# Vector Store Retriever (Semantic Search)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, embedding_model)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Keyword Retriever (BM25 Search)
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 5

# Combine them with EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5] # Give them equal importance
)
print("✅ Hybrid Search Ready!")


# --- 2. RERANKING SETUP ---
# You can get a free Cohere API key for development.
# Alternatively, use a local cross-encoder model with HuggingFace.
print("\nSetting up Reranker...")
# Make sure to set your COHERE_API_KEY as an environment variable
# Initialize huggingface cross-encoder reranker (choose device as needed)
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base", model_kwargs={"device": "cpu"}) # or "cuda" for GPU

compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)
print("✅ Reranker!")
# --- 3. QA CHAIN SETUP ---
# Use the final, most powerful retriever in your chain
final_retriever = compression_retriever

llm = Ollama(model="mistral")
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context provided. Be concise.

Context: {context}
Question: {input}
""")
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(final_retriever, document_chain)
# --- ASK A QUESTION ---
while(1):
    question =input("what is your question")
    if question=="exit":
        break
    else:
        print(f"\n❓ Asking: {question}")
        response = retrieval_chain.invoke({"input": question})
        print(f"\n🤖 Answer: {response['answer']}")
start = time.time()
results = ensemble_retriever.get_relevant_documents(question)
end = time.time()
print(f"Retrieval Time: {end - start:.4f}s")  # Retrieval latency

start = time.time()
reranked = cross_encoder.predict(query, results)
end = time.time()
print(f"Reranking Time: {end - start:.4f}s")  # Reranking latency
