#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import os
import sys
import pytesseract
from openai import OpenAI

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            '../../')))

from deepdoc.vision.seeit import draw_box
from deepdoc.vision import LayoutRecognizer, init_in_out
from deepdoc.vision.p_title import process_image_title_pairs
from deepdoc.vision.rag_chunks import prepare_rag_data
import argparse
import re
import numpy as np
from tqdm import tqdm
import base64
import json
import os

def remove_nested_boxes(boxes):
    """
    专门处理type为figure和table的box:
    1. 去除完全相同的box
    2. 删除被其他box完全包含的嵌套box
    其他type的box保持不变
    """
    # 分离出需要处理的box和不需要处理的box
    to_process = []
    others = []
    
    for box in boxes:
        if box["type"] in ["figure", "table"]:
            to_process.append(box)
        else:
            others.append(box)
    
    # 如果没有需要处理的box，直接返回原列表
    if not to_process:
        return boxes
    
    # 1. 首先对完全相同的box去重
    unique_boxes = []
    seen_boxes = set()
    
    for box in to_process:
        bbox_tuple = tuple(box["bbox"])
        if bbox_tuple not in seen_boxes:
            seen_boxes.add(bbox_tuple)
            unique_boxes.append(box)
    
    # 2. 然后处理嵌套box
    to_keep = []
    n = len(unique_boxes)
    
    for i in range(n):
        box_i = unique_boxes[i]
        bbox_i = box_i["bbox"]
        is_outer = True  # 假设当前box是最外部的
        
        for j in range(n):
            if i == j:
                continue
                
            box_j = unique_boxes[j]
            bbox_j = box_j["bbox"]
            
            # 检查box_i是否完全被box_j包含
            if (bbox_j[0] <= bbox_i[0] and  # left
                bbox_j[1] <= bbox_i[1] and  # top
                bbox_j[2] >= bbox_i[2] and  # right
                bbox_j[3] >= bbox_i[3]):    # bottom
                is_outer = False
                break
        
        if is_outer:
            to_keep.append(box_i)
    
    # 合并处理后的box和其他box
    return others + to_keep

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def find_and_append_largest_para_txt(folder_path, content_to_add):
    """
    找到文件夹中编号最大的 para_txt_n.txt 文件，并在末尾添加内容
    
    参数:
        folder_path: 文件夹路径
        content_to_add: 要添加的内容字符串
    """
    # 获取文件夹内所有 para_txt_*.txt 文件
    files = [f for f in os.listdir(folder_path) if f.startswith('para_txt_') and f.endswith('.txt')]
    
    if not files:
        print("没有找到 para_txt_*.txt 文件")
        return
    
    # 提取数字并找到最大的n
    max_num = -1
    max_file = None
    
    for file in files:
        # 使用正则表达式提取数字部分
        match = re.search(r'para_txt_(\d+)\.txt', file)
        if match:
            current_num = int(match.group(1))
            if current_num > max_num:
                max_num = current_num
                max_file = file
    
    if max_file is None:
        print("未找到有效编号的文件")
        return
    
    # 构建完整文件路径
    file_path = os.path.join(folder_path, max_file)
    
    # 以追加模式打开文件并添加内容
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write('\n'+content_to_add+'\n')

def extract_text_with_formulas(client,img_path,model_name) -> str:
    """
    识别图片中的文字和公式，将公式转换为Markdown格式
    
    :param image_url: 图片的URL地址
    :return: 包含Markdown格式公式的识别结果
    """
    
    # 构造包含格式要求的提示词
    prompt = """请识别图片中的文字内容，特别注意：
        1. 如果内容包含数学公式、化学式等科学表达式：
        - 将公式用Markdown语法标注，例如：`$E=mc^2$`
        - 对于段落标题，比如 3. title ，请勿识别成公式，正常输出即可
        2. 对于普通文字内容，保持原样输出
        3. 直接输出识别结果，不要添加任何引导语或总结语！"""
    image_base64 = image_to_base64(img_path)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    )
    
    # 后处理：确保公式被正确标记
    text = response.choices[0].message.content
    return text

def has_digits_in_parentheses(text):
    """
    检查字符串是否符合以下格式：
    以0个或若干个空格 + (或（开始，
    中间是1个或者若干数字，
    以）或) + 0个或若干个空格结尾

    例如：
    (123)  （42）  (1)    （4567）   等
    """
    pattern = r'^\s*[\(（]\d+[\)）]\s*$'  # 匹配整个字符串符合指定格式
    return bool(re.fullmatch(pattern, text))


def calculate_iou(box1, box2):
    """计算两个边界框的IoU（交并比）"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def nms(lyt, iou_threshold=0.5):
    """非极大值抑制（NMS）去重"""
    # 先按 score 降序排序
    lyt.sort(key=lambda x: -x["score"])
    
    keep = []
    while lyt:
        current = lyt.pop(0)
        keep.append(current)
        
        # 移除与当前框 IoU 超过阈值的框
        lyt = [r for r in lyt if calculate_iou(current["bbox"], r["bbox"]) < iou_threshold]
    
    return keep

class A():
    def __init__(self,inputs,output_dir,root_path):
        self.mode = "layout"
        self.threshold = 0.2
        self.output_dir = output_dir
        self.inputs = inputs
        self.root_path = root_path

class model_config():
    def __init__(self, api_key,
                 base_url,
                 vlm_model,
                 ):
        self.api_key = api_key
        self.base_url = base_url
        self.vlm_model = vlm_model

def get_chunks(inputs="/root/test_qw", root_path="/root/myData4",
               api_key = "api_key",
               base_url="base_url",
               vlm_model="qwen2.5-vl-7b-instruct"
               ):
    
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    #ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)  # 设置处理器的日志级别
    # 给 logger 添加处理器
    ## logger.addHandler(ch)

    logger.info("该服务用于解析pdf的布局，并存储每一个布局块")

    os.makedirs(root_path, exist_ok=True)
    logger.info(f"创建文件夹{root_path}")

    os.makedirs(root_path+"/para_img", exist_ok=True)
    logger.info(f"创建文件夹{root_path}"+"/para_img")

    os.makedirs(root_path+"/para_txt", exist_ok=True)
    logger.info(f"创建文件夹{root_path}"+"/para_txt")

    os.makedirs(root_path+"/pic_title", exist_ok=True)
    logger.info(f"创建文件夹{root_path}"+"/pic_title")

    os.makedirs(root_path+"/pic_img", exist_ok=True)
    logger.info(f"创建文件夹{root_path}"+"/pic_img")

    os.makedirs(root_path+"/layout_all", exist_ok=True)
    logger.info(f"创建文件夹{root_path}"+"/layout_all")


    args = A(inputs, root_path+"/layout_all", root_path)
    model_args = model_config(api_key, base_url, vlm_model)
    logger.info(f"{model_args.api_key},{model_args.base_url},{model_args.vlm_model}")
    client = OpenAI(
        api_key=model_args.api_key,
        base_url=model_args.base_url,
    )
    images, outputs = init_in_out(args)
    num_page = len(images)
    logger.info("完成 pdf2imgs或图片读取")

    if args.mode.lower() == "layout":
        detr = LayoutRecognizer("layout")
        logger.info("完成 载入布局模型")
        layouts = detr.forward(images, thr=float(args.threshold))
        logger.info("完成 pdf布局分析")
    number = 0
    for i, lyt in enumerate(layouts):
        logger.info(f"共{num_page}页，开始处理第{i+1}页")
        lyt = [r for r in lyt if r["score"] >= float(args.threshold)]
        lyt = nms(lyt, iou_threshold=0.5)
        lyt.sort(key=lambda x: x["bbox"][1])
        for j in range(len(lyt)):
            if lyt[j]["type"] in ["text","title"]:
                cropped_image = images[i].crop(lyt[j]["bbox"])
                cropped_image.save(root_path+f"/para_img/para_img_{j + number}.png")
                text = extract_text_with_formulas(client,root_path+f"/para_img/para_img_{j + number}.png",model_args.vlm_model)
                # text = pytesseract.image_to_string(cropped_image, lang='chi_sim+eng')
                with open(root_path+f"/para_txt/para_txt_{j + number}.txt", "w", encoding="utf-8") as f:
                    f.write(text)
            elif lyt[j]["type"] in ["table", "figure", "table caption", "figure caption", "equation"]:
                cropped_image = images[i].crop(lyt[j]["bbox"])
                # text = pytesseract.image_to_string(cropped_image, lang='chi_sim+eng')
                if lyt[j]["type"] in ["table caption", "figure caption"]:
                    cropped_image.save(root_path+f"/pic_img/pic_img_{j + number}.png")
                    text = extract_text_with_formulas(client,root_path+f"/pic_img/pic_img_{j + number}.png",model_args.vlm_model)
                    os.remove(root_path+f"/pic_img/pic_img_{j + number}.png")
                    if has_digits_in_parentheses(text):
                        logger.info(f"检测到公式编号: {text}")
                        find_and_append_largest_para_txt(root_path+"/para_txt",text)
                    else:
                        logger.info(f"检测到标题: {text}")
                        with open(root_path+f"/pic_title/pic_title_{j + number}.txt", "w", encoding="utf-8") as f:
                            f.write(text)
                elif lyt[j]["type"] in ["table", "figure"]:
                    logger.info("检测到图片或表格")
                    cropped_image.save(root_path+f"/pic_img/pic_img_{j + number}.png")
                elif lyt[j]["type"] in ["equation"]:
                    cropped_image.save(root_path+f"/pic_img/pic_img_{j + number}.png")
                    text = extract_text_with_formulas(client,root_path+f"/pic_img/pic_img_{j + number}.png",model_args.vlm_model)
                    logger.info(f"检测到公式: {text}, 将其放入最新的一个para_txt里")
                    os.remove(root_path+f"/pic_img/pic_img_{j + number}.png")
                    find_and_append_largest_para_txt(root_path+"/para_txt",text)
        img = draw_box(images[i], lyt, detr.labels, float(args.threshold))
        img.save(outputs[i], quality=95)
        number = number + len(lyt)
    

    os.makedirs(root_path+"/pic_json", exist_ok=True)
    logger.info(f"创建文件夹{root_path}"+"/pic_json")
    process_image_title_pairs(pic_img_dir = root_path+"/pic_img",
                                pic_title_dir = root_path+"/pic_title",
                                output_json = root_path+"/pic_json/pic.json")
    logger.info("完成 图像和标题的匹配")

    os.makedirs(root_path+"/chunk_json", exist_ok=True)
    prepare_rag_data(root_path+"/para_txt", root_path+"/chunk_json/rag_chunks.json")
    logger.info("完成 子chunk的获取")
    
    logger.info(f"完成 对{inputs}的处理")
    return "完成切分和存储"


# python deepdoc/vision/rec.py --inputs=/root/test_qw --threshold=0.2 --mode=layout --output_dir=/root/myData2/layout_all


if __name__ == "__main__":

    get_chunks(inputs="/root/test_qw", root_path="/root/myData4",)

