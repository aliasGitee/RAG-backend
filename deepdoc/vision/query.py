
import re
from typing import List
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json
import os
from openai import OpenAI
import os
import base64
import logging
from langchain_openai import OpenAIEmbeddings


def extract_tables_figures(text: str) -> List[str]:
    """
    从文本中提取Table x和Figure x的引用
    
    :param text: 输入文本
    :return: 提取到的引用列表
    """
    # 匹配Table x或Figure x，x可以是数字或数字+字母(如Table 1a)
    pattern = r'\b(?:Table|Figure)\s+\d+[a-zA-Z]?\b'
    return re.findall(pattern, text, flags=re.IGNORECASE)

def read_para_files(source_indices: List[int], base_dir: str = "/root/myData2/para_txt", m: int = 3):
    """
    根据source_idx读取para_txt文件夹下的文件内容
    
    :param source_indices: 需要读取的source_idx列表
    :param base_dir: para_txt文件夹路径
    :param m: 需要额外读取的后续文件数量
    :return: 去重后的文件内容列表
    """
    all_contents = []
    processed_files = set()  # 用于去重
    tables_figures = [] 
    
    # 获取文件夹下所有文件并排序
    all_files = sorted(
        [f for f in os.listdir(base_dir) if f.startswith("para_txt_") and f.endswith(".txt")],
        key=lambda x: int(x[9:-4])  # 按n排序
    )
    
    # 创建文件名到路径的映射
    file_dict = {f: os.path.join(base_dir, f) for f in all_files}
    
    number = 1
    for idx in source_indices:
        target_file = f"para_txt_{idx}.txt"
        
        if target_file not in file_dict:
            continue
            
        if target_file in processed_files:
            continue
            
        # 读取当前文件
        with open(file_dict[target_file], 'r', encoding='utf-8') as f:
            content = f.read().strip()
            extracted_refs = extract_tables_figures(content)
            tables_figures.extend(extracted_refs)

            all_contents.append(f"文档{number}:"+"\n"+content)
            processed_files.add(target_file)
            number+=1
            # 检查是否需要读取后续m个文件
            if (content == "ABSTRACT" or 
                re.match(r'^\d+[ .]', content)):  # 以数字+空格或数字+.开头
                
                # 找到当前文件在列表中的位置
                current_pos = all_files.index(target_file)
                
                # 读取后续m个文件
                for i in range(1, m+1):
                    if current_pos + i >= len(all_files):
                        break
                        
                    next_file = all_files[current_pos + i]
                    if next_file in processed_files:
                        continue
                        
                    with open(file_dict[next_file], 'r', encoding='utf-8') as nf:
                        next_content = nf.read().strip()

                        extracted_refs = extract_tables_figures(next_content)
                        tables_figures.extend(extracted_refs)

                        all_contents.append(next_content)
                        processed_files.add(next_file)
    
    return all_contents, list(set(tables_figures))

def filter_by_tables_figures(json_path: str, tables_figures: list) -> list:
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 筛选 idx 在 target_indices 中的字典
    filtered_data = [d for d in data if d.get('idx') in tables_figures]
    return filtered_data

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def retriever(root_path, embedding_path,
              model_name, dim, base_url, api_key, vlm_model, translate_model,
              prompt=None, topk=3, en=True):

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)


    with open(root_path+"/chunk_json/rag_chunks.json", "r") as f:
        items = json.load(f)
    documents = [
        Document(
            page_content=item["text"],
            metadata=item["metadata"]
        ) for item in items
    ]
    top_k = 3 if topk==None else topk

    bm25_retriever = BM25Retriever.from_documents(documents=documents)
    bm25_retriever.k = top_k

    embedding = OpenAIEmbeddings(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            dimensions=dim,
            check_embedding_ctx_length=False
        )
    
    vector_db = Chroma(
        persist_directory=root_path+embedding_path,
        embedding_function=embedding,
        collection_name="llama_index"
    )
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": top_k})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.0, 1.0]  # 调整权重偏向
    )

    prompt = "Based on the picture, can you tell us about the specific data in Table 1?" if prompt==None else prompt
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    if en:
        prompt_trans = client.chat.completions.create(
            model=translate_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Only return the English translation of the input text, no extra explanation or content."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        prompt = prompt_trans.choices[0].message.content.strip()
        logger.info(f"翻译成英文输入: {prompt}\n")
    
    results = ensemble_retriever.invoke(prompt)

    logger.info('检索到子chunk')
    for i, doc in enumerate(results):
        logger.info(doc.page_content)
    logger.info('\n')

    source_idx = [r.metadata['source_idx'] for r in results]
    print(source_idx)

    # 文档列表，图像id列表
    contents, ft = read_para_files(source_idx, root_path + "/para_txt")

    # 组合文档列表为字符串
    prompt_docs = ''
    for i in contents:
        prompt_docs = prompt_docs + i + '\n'


    logger.info(f"共读取 {len(contents)} 个chunk(段落)内容:")
    for i, content in enumerate(contents):
        logger.info(content[:200] + "..." if len(content) > 200 else content)  # 只打印前200字符
    logger.info(f'检索到的图片:\n{ft}\n')


    # 拿到召回的文档中出现的图像信息
    tables_figures = filter_by_tables_figures(root_path+"/pic_json/pic.json", ft)

    img_content = []
    for i in tables_figures:
        img_base64 = i["content"]
        img_content.append({"type": "text", "text": i["title"]})
        img_content.append({"type": "image_url","image_url": {"url": f"data:image/png;base64,{img_base64}"}, })

    messages=[
            {
                "role": "user",
                "content": img_content
            },
            {
                "role": "user",
                "content": f'''
                        You are an assistant for question-answering tasks. Use the following pieces of retrieved context, and the above pictures or tables to answer the question. If you don't know the answer, just say that you don't know.
                        Question: {prompt} 
                        Context: {prompt_docs} 
                        Answer:
                        '''
            },
        ]
    completion = client.chat.completions.create(
        model= vlm_model, # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/model-studio/getting-started/model
        messages = messages,
    )
    logger.info('LLM:\n'+completion.choices[0].message.content)
    return completion.choices[0].message.content

if __name__ =="__main__":
    # response = retriever_serve_llamaIndex(prompt="表格2中， Llama-3-70B在各个测试集上的表现怎么样？",en=True)
    # response = retriever_serve_llamaIndex(prompt="这篇文章的摘要的内容是什么？",en=True)
    pass