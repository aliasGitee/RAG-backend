import os
import re
import json
import base64
from pathlib import Path
'''
    合并标题与图片
'''

def process_image_title_pairs(pic_img_dir, pic_title_dir, output_json):
    """
    处理pic_img和pic_title文件夹中的文件，生成JSON输出
    
    参数:
        pic_img_dir: 存放pic_img_n.png的文件夹路径
        pic_title_dir: 存放pic_title_m.txt的文件夹路径
        output_json: 输出的JSON文件路径
    """
    # 获取并排序图片文件
    img_files = sorted(
        [f for f in os.listdir(pic_img_dir) if f.startswith('pic_img_') and f.endswith('.png')],
        key=lambda x: int(re.search(r'pic_img_(\d+)\.png', x).group(1)))
    
    # 获取并排序标题文件
    title_files = sorted(
        [f for f in os.listdir(pic_title_dir) if f.startswith('pic_title_') and f.endswith('.txt')],
        key=lambda x: int(re.search(r'pic_title_(\d+)\.txt', x).group(1)))
    
    # 检查文件数量是否一致
    if len(img_files) != len(title_files):
        raise ValueError(f"文件数量不匹配: pic_img有{len(img_files)}个文件，pic_title有{len(title_files)}个文件")
    
    result = []
    
    for img_file, title_file in zip(img_files, title_files):
        # 处理标题文件
        title_path = os.path.join(pic_title_dir, title_file)
        with open(title_path, 'r', encoding='utf-8') as f:
            title_content = f.read().strip()
        
        # 从文件内容中提取idx部分（第一个数字及其前面的所有字符串）
        idx_match = re.search(r'^([^\d]*\d+)', title_content)
        if idx_match:
            idx = idx_match.group(1).strip()  # 提取匹配部分并去除两端空白
        else:
            idx = "未找到编号"  # 如果没有找到数字，使用默认值
        
        # 处理图片文件
        img_path = os.path.join(pic_img_dir, img_file)
        with open(img_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # 添加到结果
        result.append({
            "idx": idx,
            "title": title_content,
            "content": img_base64,
            "img_filename": img_file,
            "title_filename": title_file
        })
    
    # 保存为JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"成功处理 {len(result)} 对文件，结果已保存到 {output_json}")

def extract_content_from_json(json_file_path, output_dir=None):
    """
    从JSON文件中提取所有content字段(Base64编码的图片数据)
    
    参数:
        json_file_path: JSON文件路径
        output_dir: 可选，指定输出目录（如果要把content保存为文件）
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有content
    contents = [item['content'] for item in data]
    contents = contents[0]
    
    # 如果 base64 数据是以 data:image/png;base64, 开头，请先去除前缀
    if contents.startswith("data:image"):
        base64_data = contents.split(",")[1]
    else:
        base64_data = contents
    # 将 base64 解码为二进制数据
    image_data = base64.b64decode(base64_data)
    # 保存为 PNG 文件
    with open("output.png", "wb") as f:
        f.write(image_data)


if __name__ == "__main__":
    pic_img_dir = "/root/myData2/pic_img"
    pic_title_dir = "/root/myData2/pic_title"
    output_json = "/root/myData2/pic_json/pic.json"

    # Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    process_image_title_pairs(pic_img_dir, pic_title_dir, output_json)

    # extract_content_from_json("/root/myData/output.json")
