from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import json
from llama_index.core import StorageContext


def persist_embed_llamaIndex(root_path, 
                             model_name, api_key, base_url, dim,
                             save_path="/llama_index_qw1024"):


    with open(root_path+"/chunk_json/rag_chunks.json", "r") as f:
        items = json.load(f)
    
    documents = [
        Document(
            text=item["text"],
            metadata=item["metadata"]
        ) for item in items
    ]

    embed_model = OpenAILikeEmbedding(
        model_name= model_name, #"text-embedding-v4"
        dimensions=dim, #1024,
        api_key=api_key, 
        api_base=base_url,
    )
    chroma_client = chromadb.PersistentClient(
        path=root_path+save_path
    )
    chroma_collection = chroma_client.get_or_create_collection("llama_index")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        storage_context=storage_context
    )
    return "完成文档向量化"

