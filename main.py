from deepdoc.vision.pdf2chunks import get_chunks
from deepdoc.vision.chunks2embeddings import persist_embed_llamaIndex
from deepdoc.vision.query import retriever
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class ChunkRequest(BaseModel):
    inputs: str = "/root/RAGappv0/example_data"
    root_path: str = "/root/myData4"
    api_key: str = "sk-73e16ce211754324828584932b9f1caf"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vlm_model: str = "qwen2.5-vl-7b-instruct"

class EmbeddingRequest(BaseModel):
    root_path: str = "/root/myData4"
    api_key: str = "sk-73e16ce211754324828584932b9f1caf"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "text-embedding-v4"
    save_path: str ="/llama_index_qw1024"
    dim: int = 1024

class RetrieverRequest(BaseModel):
    root_path: str = "/root/myData4"
    embedding_path: str = "/llama_index_qw1024"
    model_name: str = "text-embedding-v4"
    dim: int = 1024
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = "sk-73e16ce211754324828584932b9f1caf"
    vlm_model: str = "qwen-vl-max-latest"   
    translate_model: str = "qwen2.5-72b-instruct"  
    prompt: str = "表格2中， Llama-3-70B在各个测试集上的表现怎么样？", 
    topk: int = 3, 
    en:bool = True


@app.post("/generate-chunks")
async def generate_chunks(request: ChunkRequest):
    try:
        result = get_chunks(
            inputs=request.inputs,
            root_path=request.root_path,
            api_key=request.api_key,
            base_url=request.base_url,
            vlm_model=request.vlm_model
        )
        return {"result": result}
    except Exception as e:
        import traceback
        traceback.print_exc()  # 打印完整错误堆栈
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-embeddings")
async def generate_embeddings(request: EmbeddingRequest):
    try:
        result = persist_embed_llamaIndex(request.root_path, 
                             request.model_name, request.api_key, request.base_url, request.dim,
                             request.save_path)
        return {"result": result}
    except Exception as e:
        import traceback
        traceback.print_exc()  # 打印完整错误堆栈
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-retriever")
async def generate_retriever(request: RetrieverRequest):
    try:
        result = retriever(request.root_path, request.embedding_path,
              request.model_name, request.dim, request.base_url, request.api_key, request.vlm_model, request.translate_model,
              request.prompt, request.topk, request.en)
        return {"result": result}
    except Exception as e:
        import traceback
        traceback.print_exc()  # 打印完整错误堆栈
        raise HTTPException(status_code=500, detail=str(e))
