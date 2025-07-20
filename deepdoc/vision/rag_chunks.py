import os
import re
import json
from typing import List, Dict

def split_sentences(text: str) -> List[str]:
    if len(text) >= 2:
        first_char, second_char = text[0], text[1]
        if (first_char.isdigit() and second_char == ' ') or \
           (first_char.isdigit() and second_char == '.'):
            return [text.strip()]
        
    pattern = r'(?<!\$\$)[。](?!\$\$)|\.\s'
    sentences = re.split(pattern, text)
    processed = []
    for i, s in enumerate(sentences):
        s = s.strip()
        if not s:
            continue
        if i < len(sentences)-1 and text.find(s + '. ') != -1:
            s += '.'
        processed.append(s)
    return processed

def prepare_rag_data(input_folder: str, output_json: str):
    rag_data = []
    for filename in os.listdir(input_folder):
        if filename.startswith('para_txt_') and filename.endswith('.txt'):
            try:
                n = int(filename[9:-4])
            except ValueError:
                continue
            
            filepath = os.path.join(input_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            sentences = split_sentences(content)
            rag_data.extend([
                {"text": s, "metadata": {"source_idx": n}} 
                for s in sentences
            ])
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(rag_data, f, ensure_ascii=False, indent=2)

# 生成结构化数据
# prepare_rag_data("/root/myData4/para_txt", "/root/rag_chunks.json")